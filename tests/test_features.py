import numpy as np
import pandas as pd
import pytest

from mandi import features as F
from mandi.config import ROOT, load_config

CFG = load_config()
FP = {**CFG["forecast"]["features"], **CFG["forecast"]["baselines"]}
START, END = "2018-01-01", "2020-06-30"


def _synthetic_panel(seed: int = 0, n_markets: int = 5, closed_weekday: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.date_range(START, END)
    frames = []
    for m in range(n_markets):
        lp = 6.5 + np.cumsum(rng.normal(0, 0.03, len(days)))
        price = np.exp(lp).round(-1)
        price[rng.random(len(days)) < 0.12] = np.nan
        price[days.dayofweek == closed_weekday] = np.nan
        frames.append(pd.DataFrame({"market_key": f"M{m}", "date": days, "modal_price": price,
                                    "arrivals_tonnes": rng.uniform(5, 50, len(days)), "is_stale": False}))
    return pd.concat(frames, ignore_index=True)


def _table(panel: pd.DataFrame, h: int = 7) -> tuple[F.Arrays, pd.DataFrame]:
    a = F.build_arrays(panel, START, END)
    return a, F.build_origin_table(a, h, {m: 6 for m in a.markets}, FP)


def _feature_cols():
    return F.FEATURE_COLUMNS + ["base_naive", "base_mean3", "p0"]


@pytest.mark.parametrize("h", [7, 30])
def test_no_leakage_changing_the_future_does_not_change_features_at_the_origin(h):
    panel = _synthetic_panel()
    cutoff = pd.Timestamp("2019-06-30")
    _, base = _table(panel, h)
    future = panel.copy()
    mask = future["date"] > cutoff
    rng = np.random.default_rng(1)
    future.loc[mask, "modal_price"] = rng.uniform(100, 5000, mask.sum())      # change every later price ...
    future.loc[mask, "arrivals_tonnes"] = rng.uniform(1, 500, mask.sum())     # ... and arrival, and fill the gaps
    _, changed = _table(future, h)
    cols = _feature_cols()
    a = base[base["origin"] <= cutoff].sort_values(["market_idx", "origin"]).reset_index(drop=True)
    b = changed[changed["origin"] <= cutoff].sort_values(["market_idx", "origin"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(a[["market_idx", "origin"] + cols], b[["market_idx", "origin"] + cols])
    # the target is the only thing that is allowed to differ
    assert not np.allclose(a["y_log"].fillna(0), b["y_log"].fillna(0))


def test_no_leakage_on_real_panel():
    path = ROOT / CFG["paths"]["panel"]
    if not path.exists():
        pytest.skip("panel not built")
    p = pd.read_parquet(path)
    cutoff = pd.Timestamp("2021-12-31")
    a = F.build_arrays(p, "2014-01-01", "2024-12-31", CFG["markets"])
    clo = F.closure_weekdays(p, CFG["eda"]["train_end"])
    base = F.build_origin_table(a, 7, clo, FP)
    b_arr = F.build_arrays(p, "2014-01-01", "2024-12-31", CFG["markets"])
    j = np.searchsorted(b_arr.days, cutoff) + 1
    rng = np.random.default_rng(0)
    b_arr.logp[:, j:] = rng.normal(7, 1, b_arr.logp[:, j:].shape)
    b_arr.larr[:, j:] = rng.normal(4, 1, b_arr.larr[:, j:].shape)
    changed = F.build_origin_table(b_arr, 7, clo, FP)
    cols = _feature_cols()
    x = base[base["origin"] <= cutoff].reset_index(drop=True)
    y = changed[changed["origin"] <= cutoff].reset_index(drop=True)
    pd.testing.assert_frame_equal(x[["market_idx", "origin"] + cols], y[["market_idx", "origin"] + cols])


def test_arrivals_feature_never_uses_the_origin_day():
    panel = _synthetic_panel()
    _, base = _table(panel)
    day = pd.Timestamp("2019-03-12")
    alt = panel.copy()
    alt.loc[alt["date"] == day, "arrivals_tonnes"] = 9999.0  # change the arrivals on the origin day only
    _, changed = _table(alt)
    x = base[base["origin"] == day].sort_values("market_idx")
    y = changed[changed["origin"] == day].sort_values("market_idx")
    np.testing.assert_allclose(x[["arr_prev", "arr_mean14"]].to_numpy(), y[["arr_prev", "arr_mean14"]].to_numpy(), equal_nan=True)


def test_target_is_the_quote_h_calendar_days_later_and_missing_when_absent():
    panel = _synthetic_panel()
    a, t = _table(panel, 7)
    r = t[t["market_idx"] == 2].sample(60, random_state=0)
    for row in r.itertuples():
        j = row.origin_idx + 7
        expected = a.logp[2, j] if j < a.logp.shape[1] else np.nan
        assert (np.isnan(row.y_log) and np.isnan(expected)) or np.isclose(row.y_log, expected)
        assert row.target_date == a.days[row.origin_idx] + pd.Timedelta(days=7)
    assert t["y_log"].isna().any() and t["y_log"].notna().any()


def test_origins_are_exactly_the_observed_days():
    panel = _synthetic_panel()
    a, t = _table(panel)
    for m in range(5):
        assert set(t.loc[t["market_idx"] == m, "origin_idx"]) == set(np.flatnonzero(~np.isnan(a.logp[m])))


def _single_market_panel(prices: dict) -> pd.DataFrame:
    days = pd.date_range("2018-01-01", "2020-06-30")
    s = pd.Series(np.nan, index=days)
    for d, v in prices.items():
        s[pd.Timestamp(d)] = v
    return pd.DataFrame({"market_key": "M0", "date": days, "modal_price": s.to_numpy(), "arrivals_tonnes": 10.0, "is_stale": False})


def _with_others(panel: pd.DataFrame) -> pd.DataFrame:
    """Add four other markets so the cross-market features exist; they do not affect the checks below."""
    others = [panel.assign(market_key=f"M{k}", modal_price=np.where(np.arange(len(panel)) % 2 == 0, 500.0 + k, np.nan)) for k in range(1, 5)]
    return pd.concat([panel] + others, ignore_index=True)


def _row(panel: pd.DataFrame, origin: str) -> pd.Series:
    a = F.build_arrays(panel, START, END)
    t = F.build_origin_table(a, 7, {m: None for m in a.markets}, FP)
    return t[(t["market_idx"] == 0) & (t["origin"] == origin)].iloc[0]


def test_baseline_mean3_is_the_mean_of_the_last_three_observed_quotes_in_rupees():
    panel = _with_others(_single_market_panel({"2019-01-02": 100.0, "2019-01-05": 400.0, "2019-01-20": 1000.0, "2019-01-25": 300.0}))
    row = _row(panel, "2019-01-25")
    assert np.isclose(np.exp(row["base_mean3"]), (400 + 1000 + 300) / 3)
    assert np.isclose(np.exp(row["base_naive"]), 300)


def test_lag_means_last_observed_quote_at_or_before_origin_minus_k_not_the_exact_day():
    panel = _with_others(_single_market_panel({"2019-01-01": 100.0, "2019-01-10": 200.0, "2019-01-20": 300.0}))
    row = _row(panel, "2019-01-20")
    # 7 days before the origin is 2019-01-13 (no quote); the last quote at or before it is 2019-01-10 (200)
    assert np.isclose(row["chg_7"], np.log(300) - np.log(200))
    assert np.isclose(row["chg_1"], np.log(300) - np.log(200))
    assert row["gap_prev"] == 10


def test_seasonal_naive_takes_the_nearest_quote_within_three_days_and_prefers_the_earlier_on_a_tie():
    origin = pd.Timestamp("2019-05-10")
    ref = origin + pd.Timedelta(days=7) - pd.Timedelta(days=365)
    fmt = lambda d: d.strftime("%Y-%m-%d")  # noqa: E731
    prices = {fmt(origin): 500.0, fmt(ref - pd.Timedelta(days=2)): 111.0, fmt(ref + pd.Timedelta(days=2)): 222.0, fmt(ref + pd.Timedelta(days=5)): 333.0}
    row = _row(_with_others(_single_market_panel(prices)), fmt(origin))
    assert np.isclose(np.exp(row["base_seasonal"]), 111.0)  # -2 and +2 tie: the earlier date wins; +5 is outside the tolerance
    only_far = _with_others(_single_market_panel({fmt(origin): 500.0, fmt(ref + pd.Timedelta(days=5)): 333.0}))
    assert np.isnan(_row(only_far, fmt(origin))["base_seasonal"])


def test_expected_closure_flag_marks_closure_weekday_and_fixed_holidays():
    panel = _synthetic_panel()
    a = F.build_arrays(panel, START, END)
    t = F.build_origin_table(a, 7, {m: 6 for m in a.markets}, FP)
    tgt = t["origin"] + pd.Timedelta(days=7)
    assert (t.loc[tgt.dt.dayofweek == 6, "target_expected_closed"] == 1).all()
    hol = (tgt.dt.month == 1) & (tgt.dt.day == 26)
    assert hol.any() and (t.loc[hol, "target_expected_closed"] == 1).all()


def test_closure_weekday_is_learned_from_training_period_only():
    days = pd.date_range("2016-01-01", "2021-12-31")
    df = pd.DataFrame({"market_key": "A", "date": days, "was_missing": False})
    df.loc[(df["date"] <= "2019-12-31") & (df["date"].dt.dayofweek == 6), "was_missing"] = True   # closed Sundays until 2019
    df.loc[(df["date"] > "2019-12-31") & (df["date"].dt.dayofweek == 2), "was_missing"] = True    # closed Wednesdays afterwards
    assert F.closure_weekdays(df, "2019-12-31") == {"A": 6}
