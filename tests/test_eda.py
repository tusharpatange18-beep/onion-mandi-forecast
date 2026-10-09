import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.stattools import acf

from mandi import eda
from mandi.config import ROOT, load_config

CFG = load_config()


def _panel(prices_by_date: dict[str, float], market: str = "A", closed_weekday: int | None = None) -> pd.DataFrame:
    idx = pd.date_range(min(prices_by_date), max(prices_by_date), freq="D")
    s = pd.Series(prices_by_date, dtype=float)
    s.index = pd.to_datetime(s.index)
    s = s.reindex(idx)
    df = pd.DataFrame({"date": idx, "m": market, "modal_price": s.values, "arrivals_tonnes": 10.0})
    df["was_missing"] = df["modal_price"].isna()
    df["weekday"] = df["date"].dt.dayofweek
    return df


def test_weekly_series_requires_min_observed_days():
    # week 1 (Mon 2020-01-06 .. Sun 12th): 3 observed days; week 2: only 2
    obs = {"2020-01-06": 100, "2020-01-07": 100, "2020-01-08": 100, "2020-01-13": 200, "2020-01-14": 200, "2020-01-20": 300}
    w = eda.weekly_log_series(_panel(obs), "modal_price", min_days=3)
    assert np.isclose(np.exp(w["A"].iloc[0]), 100)
    assert np.isnan(w["A"].iloc[1])  # 2 observed days -> not a valid week


def test_weekly_mean_uses_observed_days_only():
    obs = {"2020-01-06": 100, "2020-01-07": 400, "2020-01-08": 100}  # missing days do not count as zero or filled
    w = eda.weekly_log_series(_panel(obs), "modal_price", min_days=3)
    assert np.isclose(w["A"].iloc[0], np.mean(np.log([100, 400, 100])))


def test_fill_weekly_gaps_respects_limit():
    w = pd.DataFrame({"A": [1.0, np.nan, 2.0, np.nan, np.nan, 3.0]}, index=pd.date_range("2020-01-05", periods=6, freq="W-SUN"))
    out = eda.fill_weekly_gaps(w, max_weeks=1)["A"]
    assert out.iloc[1] == 1.0  # one-week gap filled
    assert out.iloc[3:5].isna().all()  # two-week gap left alone


def test_longest_segment():
    s = pd.Series([1, 2, np.nan, 3, 4, 5, np.nan, 6.0])
    assert eda.longest_segment(s).tolist() == [3, 4, 5]


def test_closure_weekday_detected():
    idx = pd.date_range("2020-01-01", "2020-03-31")
    df = pd.DataFrame({"date": idx, "m": "A", "weekday": idx.dayofweek})
    df["was_missing"] = df["weekday"] == 6  # always closed on Sunday
    assert eda.closure_weekday(df)["A"] == 6


def test_pairwise_acf_equals_statsmodels_without_gaps():
    x = pd.Series(np.random.default_rng(0).normal(size=200)).cumsum().diff().dropna()
    t = eda.pairwise_acf_pacf(x, 5)
    assert np.allclose(t["acf"], acf(x.values, nlags=5, fft=False), atol=0.02)
    assert np.isclose(t["pacf"].iloc[1], t["acf"].iloc[1])  # PACF(1) == ACF(1)
    from statsmodels.tsa.stattools import pacf as sm_pacf
    assert np.allclose(t["pacf"], sm_pacf(x.values, nlags=5, method="yw"), atol=0.02)  # same Yule-Walker PACF as statsmodels


def test_weekday_effect_zero_for_constant_prices_and_skips_rare_weekdays():
    idx = pd.date_range("2018-01-01", "2019-12-30")
    df = pd.DataFrame({"date": idx, "m": "A", "modal_price": 100.0, "arrivals_tonnes": 5.0})
    df.loc[df["date"].dt.dayofweek == 6, "modal_price"] = np.nan  # closed Sundays
    df.loc[df.index[6], "modal_price"] = 100.0  # one exceptional Sunday opening
    df["weekday"] = df["date"].dt.dayofweek
    t = eda.weekday_effect(df)
    assert (t["price_effect_pct"].abs() < 1e-9).all()
    assert "Sun" not in set(t["weekday"])  # fewer than 30 days: not reported as an effect


def test_stationarity_table_flags_random_walk_and_white_noise():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2015-01-04", periods=250, freq="W-SUN")
    w = pd.DataFrame({"walk": rng.normal(size=250).cumsum(), "noise": rng.normal(size=250)}, index=idx)
    t = eda.stationarity_table(w)
    lvl = t[(t["series"] == "longest segment") & (t["transform"] == "level")].set_index("market")
    assert lvl.loc["noise", "ADF_p"] < 0.05
    assert lvl.loc["walk", "ADF_p"] > 0.05
    diff = t[(t["series"] == "longest segment") & (t["transform"] == "first difference")].set_index("market")
    assert diff.loc["walk", "ADF_p"] < 0.05


@pytest.mark.local_data("panel")
def test_modelling_evidence_slice_ends_before_test_period():
    panel_path = ROOT / CFG["paths"]["panel"]
    if not panel_path.exists():
        pytest.skip("panel not built")
    p = eda.load_panel(panel_path)
    train = p[p["date"] <= CFG["eda"]["train_end"]]
    assert train["date"].max() < pd.Timestamp(CFG["backtest"]["first_test_date"])
    # weekly series built from the train slice cannot contain a week that is mostly in the test period
    w = eda.weekly_log_series(train, "modal_price", CFG["eda"]["min_days_per_week"])
    assert w.index.max() <= pd.Timestamp("2020-01-05")
    assert w.loc[w.index > "2019-12-31"].isna().all().all()  # the partial last week has < 3 train days
