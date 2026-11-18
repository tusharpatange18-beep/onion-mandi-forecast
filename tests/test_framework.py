"""Tests for the reusable backtesting interface (framework), the tail-risk metrics and the long/flat trading evaluation."""
import numpy as np
import pandas as pd
import pytest

from mandi import risk, trading
from mandi.framework import Backtester, BacktestResult, ColumnModel, Model, refit_blocks, refit_schedule


# --------------------------------------------------------------------------- risk
def test_cvar_is_the_mean_of_the_worst_share():
    x = np.arange(1, 101, dtype=float)               # losses 1..100
    assert risk.cvar(x, 0.05) == pytest.approx(np.mean([96, 97, 98, 99, 100]))
    assert risk.cvar(np.array([3.0, 1.0]), 0.05) == 3.0   # at least one observation is always used
    assert np.isnan(risk.cvar(np.array([])))


def test_tail_risk_counts_over_and_under_forecasts_alike():
    e = np.array([-10, 10, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8, -8, 9, -100.0])
    t = risk.tail_risk(e, 0.05)
    assert t["n"] == 20 and t["max_abs_error"] == 100 and t["cvar5_abs_error"] == 100       # worst 5% of 20 = 1 value
    assert t["MAE"] == pytest.approx(np.abs(e).mean())
    assert t["cvar5_abs_error"] >= t["p95_abs_error"] >= t["MAE"]
    assert 0 < t["worst5_share_of_total_error"] < 1


# --------------------------------------------------------------------------- trading
def _frame(rows):
    """rows: (market, origin, h, p0_price, target_price, forecast_price)."""
    return pd.DataFrame({"market": [r[0] for r in rows], "origin": pd.to_datetime([r[1] for r in rows]),
                         "target_date": pd.to_datetime([r[1] for r in rows]) + pd.to_timedelta([r[2] for r in rows], unit="D"),
                         "p0": np.log([r[3] for r in rows]), "y_log": np.log([r[4] for r in rows]), "f_log": np.log([r[5] for r in rows])})


def test_long_flat_trades_only_when_the_forecast_beats_today_by_the_threshold():
    f = _frame([("A", "2020-01-01", 7, 100, 110, 105), ("A", "2020-02-01", 7, 100, 90, 101), ("A", "2020-03-01", 7, 100, 80, 102.9)])
    r = trading.backtest_long_flat(f, threshold=0.03)
    assert len(r.trades) == 1 and r.trades["ret"].iloc[0] == pytest.approx(0.10)   # only the +5% forecast; +1% and +2.9% stay flat


def test_one_position_per_market_at_a_time_and_next_entry_at_the_exit_date():
    f = _frame([("A", "2020-01-01", 7, 100, 110, 120), ("A", "2020-01-04", 7, 100, 120, 120),   # inside the first trade: skipped
                ("A", "2020-01-08", 7, 100, 90, 120),                                           # origin == previous exit: allowed
                ("B", "2020-01-04", 7, 100, 105, 120)])                                          # another market is independent
    r = trading.backtest_long_flat(f, threshold=0.03)
    assert sorted(zip(r.trades["market"], r.trades["entry"].dt.strftime("%m-%d"))) == [("A", "01-01"), ("A", "01-08"), ("B", "01-04")]


def test_costs_are_charged_per_trade_and_summary_metrics_on_a_known_case():
    f = _frame([("A", "2020-01-01", 7, 100, 110, 120), ("A", "2020-02-01", 7, 100, 90, 120)])   # +10%, then -10%
    s = trading.backtest_long_flat(f, threshold=0.03, cost_bps=100).summary()
    assert s["trades"] == 2 and s["win_rate_pct"] == 50.0
    assert s["total_return_pct"] == pytest.approx(100 * (0.09 - 0.11))              # one market: P&L is the sum of net trade returns
    assert s["worst_trade_pct"] == pytest.approx(-11.0)
    assert s["max_drawdown_pct"] == pytest.approx(-11.0)                            # the second trade is the whole drawdown
    assert s["cvar5_trade_return_pct"] == pytest.approx(-11.0)


def test_pnl_is_added_not_compounded():
    f = _frame([("A", f"2020-0{m}-01", 7, 100, 110, 120) for m in range(1, 7)])      # six +10% trades
    assert trading.backtest_long_flat(f).summary()["total_return_pct"] == pytest.approx(60.0)


def test_no_trades_gives_flat_equity_and_nan_ratios():
    f = _frame([("A", "2020-01-01", 7, 100, 110, 100)])
    s = trading.backtest_long_flat(f).summary()
    assert s["trades"] == 0 and s["total_return_pct"] == 0 and np.isnan(s["win_rate_pct"])


def test_entry_lag_trades_at_the_next_quote_and_keeps_the_signal_from_the_origin():
    f = _frame([("A", "2020-01-01", 7, 100, 120, 110), ("A", "2020-01-03", 7, 105, 120, 100), ("A", "2020-01-20", 7, 100, 100, 100)])
    r = trading.backtest_long_flat(f, threshold=0.03, entry_lag=True)
    assert len(r.trades) == 1 and r.trades["ret"].iloc[0] == pytest.approx(120 / 105 - 1)   # signal on day 1 (+10%), filled at day 3's quote of 105
    lagged = trading.with_entry_lag(f)
    # day 3's next quote (day 20) is after its target date (day 10) and day 20 has no next quote: both rows are dropped
    assert len(lagged) == 1 and lagged["origin"].iloc[0] == pd.Timestamp("2020-01-01")


def test_concentration_and_adverse_excursion_on_known_trades():
    trades = pd.DataFrame({"market": "A", "entry": pd.to_datetime(["2020-01-01", "2020-02-01"]), "exit": pd.to_datetime(["2020-01-08", "2020-02-08"]), "ret": [0.50, 0.10]})
    c = trading.concentration(trades, n_markets=1, share=0.5)
    assert c["top_trades"] == 1 and c["top_share_of_pnl_pct"] == pytest.approx(100 * 0.5 / 0.6) and c["pnl_without_top_pct_of_capital"] == pytest.approx(10.0)
    days = pd.to_datetime(["2020-01-01", "2020-01-04", "2020-01-08"])
    frame = pd.DataFrame({"market": "A", "origin": days, "p0": np.log([100.0, 70.0, 100.0])})       # fell 30% inside the trade, then recovered
    t = pd.DataFrame({"market": "A", "entry": days[:1], "exit": days[2:], "ret": [0.0]})
    assert trading.adverse_excursion(t, frame).iloc[0] == pytest.approx(-0.30)


def test_momentum_signal_uses_only_past_prices():
    days = pd.date_range("2020-01-01", periods=30)
    f = pd.DataFrame({"market": "A", "origin": days, "target_date": days + pd.Timedelta(days=7), "p0": np.log(100.0 + np.arange(30)),
                      "y_log": np.log(200.0), "f_log": np.log(100.0)})
    base = trading.past_return(f, 7)
    f2 = f.copy()
    f2.loc[20:, "p0"] = np.log(1.0)                               # change the prices from day 20 on ...
    assert np.allclose(trading.past_return(f2, 7)[:20], base[:20], equal_nan=True)   # ... the earlier signals do not move
    assert np.isnan(base[:7]).all() and base[7] == pytest.approx(107 / 100 - 1)


# --------------------------------------------------------------------------- framework
class _Spy(Model):
    name = "spy"

    def __init__(self):
        self.max_targets = []

    def fit(self, train):
        self.max_targets.append(train["target_date"].max())
        return self

    def predict(self, test):
        return test["p0"].to_numpy() + 0.01


def _table(h=30, start="2018-01-01", end="2020-06-30"):
    origins = pd.date_range(start, end)
    t = pd.DataFrame({"market": "A", "market_idx": 0, "origin": origins, "h": h, "p0": 7.0})
    t["target_date"] = t["origin"] + pd.Timedelta(days=h)
    t["y_log"] = 7.1
    return t


def test_backtester_embargo_blocks_and_output_shape():
    spy = _Spy()
    res = Backtester(_table(), ("2020-01-01", "2020-06-30"), min_train_rows=10).run(spy)
    assert res.n_fits == 6 == len(spy.max_targets)
    for refit, m in zip(refit_schedule("2020-01-01", "2020-06-30"), spy.max_targets):
        assert m < refit                                           # label embargo
    assert res.frame["origin"].min() == pd.Timestamp("2020-01-01") and res.frame["origin"].max() == pd.Timestamp("2020-06-30")
    assert (res.frame["f_log"] == 7.01).all() and set(res.frame.columns) >= {"market", "origin", "h", "target_date", "y_log", "p0", "f_log"}
    assert refit_blocks(refit_schedule("2020-01-01", "2020-03-31"), pd.Timestamp("2020-03-31"))[-1][1] == pd.Timestamp("2020-03-31")


def test_backtester_needs_one_horizon_and_skips_blocks_with_too_little_training_data():
    both = pd.concat([_table(7), _table(30)])
    with pytest.raises(ValueError):
        Backtester(both, ("2020-01-01", "2020-06-30"))
    assert Backtester(_table(), ("2020-01-01", "2020-06-30"), min_train_rows=10 ** 9).run(_Spy()).frame.empty


def test_compare_scores_models_on_exactly_the_same_forecasts():
    t = _table()
    t["base_seasonal"] = np.where(t["origin"].dt.month == 3, np.nan, 7.0)      # this baseline cannot forecast March origins
    t["base_naive"] = t["p0"]
    bt = Backtester(t, ("2020-01-01", "2020-06-30"), min_train_rows=10)
    seasonal, naive = bt.compare([ColumnModel("base_seasonal"), ColumnModel("base_naive")])
    assert len(seasonal.frame) == len(naive.frame) and (seasonal.frame[["origin"]].values == naive.frame[["origin"]].values).all()
    assert not (seasonal.frame["origin"].dt.month == 3).any()


def test_result_metrics_and_from_columns_agree_with_hand_calculation():
    tab = pd.DataFrame({"market": ["A"] * 4, "origin": pd.date_range("2020-01-01", periods=4), "h": 7,
                        "target_date": pd.date_range("2020-01-08", periods=4), "y_log": np.log([100, 100, 100, 100.0]),
                        "base_naive": np.log([100, 100, 100, 100.0]), "m": np.log([110, 90, 100, 200.0])})
    r = BacktestResult.from_columns(tab, "m")
    assert r.point_metrics()["MAE"] == pytest.approx((10 + 10 + 0 + 100) / 4)
    t = r.tail_risk()
    assert t["max_abs_error"] == pytest.approx(100) and t["cvar5_abs_error"] == pytest.approx(100)
    assert t["max_abs_error_pct"] == pytest.approx(100.0)        # error as a share of the realised price
