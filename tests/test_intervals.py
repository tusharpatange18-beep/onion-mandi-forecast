import numpy as np
import pandas as pd
import pytest

from mandi import intervals as IV

DAY = pd.Timedelta(days=1)


def _series(n=1200, h=7, seed=0, scale=0.1, start="2018-01-01"):
    """Daily forecasts for one market and horizon with iid errors; target = origin + h days."""
    rng = np.random.default_rng(seed)
    origin = pd.date_range(start, periods=n)
    e = rng.laplace(0, scale, n)
    return pd.DataFrame({"market": "A", "h": h, "origin": origin, "target_date": origin + h * DAY, "y_log": 7.0 + e, "f_lgbm": 7.0,
                         "f_naive": 7.0 + rng.normal(0, 0.05, n), "gap_prev": 1.0, "vol_scale": scale, "source": "final"})


def test_finite_sample_quantile_is_the_k_th_smallest_with_k_ceil_n_plus_1_times_level():
    n = 100
    score = np.arange(1, n + 1, dtype=float)          # 1..100
    origin = np.array(["2020-06-01"], dtype="datetime64[ns]")
    target = pd.date_range("2020-01-01", periods=n).to_numpy()          # all before the origin and inside 365 days
    q, n_cal = IV.conformal_halfwidths(origin, target, np.r_[score], [0.8, 0.95], 365, 100)
    # a single forecast row: scores/targets are the calibration pool, origin is the forecast's origin
    # (rebuild with the pool as rows and one extra later row)
    o = np.r_[target, origin].astype("datetime64[ns]")
    t = np.r_[target, (origin + np.timedelta64(7, "D"))].astype("datetime64[ns]")
    q, n_cal = IV.conformal_halfwidths(o, t, np.r_[score, 0.0], [0.8, 0.95], 365, 100)
    assert n_cal[-1] == 100
    assert q[0.95][-1] == 96.0                    # k = ceil(101 * 0.95) = 96
    assert q[0.8][-1] == 81.0                     # k = ceil(101 * 0.80) = 81 (80.8 -> 81)


def test_calibration_set_is_strictly_before_the_origin_and_inside_the_trailing_window():
    origin = pd.to_datetime(["2020-01-01", "2020-06-01", "2021-07-01"]).to_numpy()
    target = pd.to_datetime(["2020-01-08", "2020-06-08", "2021-07-08"]).to_numpy()
    # forecast 1 sees nothing; forecast 2 (origin 2020-06-01) sees the 2020-01-08 target (known, 145 days earlier);
    # forecast 3 (origin 2021-07-01) sees neither: the 2020-06-08 target is 388 days earlier, outside the 365-day window
    q, n_cal = IV.conformal_halfwidths(origin, target, np.array([1.0, 2.0, 3.0]), [0.8], 365, 1)
    assert list(n_cal) == [0, 1, 0]
    # forecast 2 sees the first target (2020-01-08 < 2020-06-01, within 365 days) only
    origin2 = pd.to_datetime(["2020-01-01", "2020-06-01"]).to_numpy()
    target2 = pd.to_datetime(["2020-01-08", "2020-06-08"]).to_numpy()
    _, n2 = IV.conformal_halfwidths(origin2, target2, np.array([1.0, 2.0]), [0.8], 365, 1)
    assert list(n2) == [0, 1]
    # a target on the origin day itself is NOT yet known: strictly before
    o3 = pd.to_datetime(["2020-01-01", "2020-01-08"]).to_numpy()
    t3 = pd.to_datetime(["2020-01-08", "2020-01-15"]).to_numpy()
    _, n3 = IV.conformal_halfwidths(o3, t3, np.array([1.0, 2.0]), [0.8], 365, 1)
    assert list(n3) == [0, 0]


def test_no_interval_when_fewer_than_100_errors():
    f = _series(n=300)
    out = IV.add_intervals(f, [0.8, 0.95], 365, 100)
    first_ok = out["lgbm_abs_lo_80"].first_valid_index()
    # the first interval needs 100 resolved errors: origin index >= 100 + h (targets resolve h days after the origin)
    assert out["n_calib"].iloc[first_ok] >= 100 and out["n_calib"].iloc[first_ok - 1] < 100
    assert out["lgbm_abs_lo_80"].iloc[:first_ok].isna().all()


def test_empirical_coverage_is_close_to_nominal_for_exchangeable_errors():
    f = _series(n=3000, seed=2)
    out = IV.add_intervals(f, [0.8, 0.95], 365, 100)
    out = out.dropna(subset=["lgbm_abs_lo_80"])
    for lv in (0.8, 0.95):
        s = IV.summarise(out, "lgbm_abs", lv)
        assert abs(s["coverage"] - lv) < 0.03, (lv, s["coverage"])


def test_interval_at_t_does_not_depend_on_outcomes_resolved_at_or_after_t():
    f = _series(n=600)
    base = IV.add_intervals(f, [0.8], 365, 100)
    i = 450
    t0 = f["origin"].iloc[i]
    g = f.copy()
    later = g["target_date"] >= t0                       # change every outcome that is not yet known at origin t0
    g.loc[later, "y_log"] = 99.0
    changed = IV.add_intervals(g, [0.8], 365, 100)
    # the forecast at origin t0 only uses errors with target < t0, which are unchanged
    assert np.isclose(base["lgbm_abs_hi_80"].iloc[i], changed["lgbm_abs_hi_80"].iloc[i])
    assert np.isclose(base["lgbm_abs_lo_80"].iloc[i], changed["lgbm_abs_lo_80"].iloc[i])


def test_volatility_scaled_intervals_cover_better_across_volatility_regimes():
    rng = np.random.default_rng(5)
    n = 4000
    origin = pd.date_range("2016-01-01", periods=n)
    regime = np.where(np.sin(np.arange(n) / 90.0) > 0, 0.25, 0.05)       # slow calm / turbulent regimes known at the origin
    e = rng.normal(0, 1, n) * regime
    f = pd.DataFrame({"market": "A", "h": 7, "origin": origin, "target_date": origin + 7 * DAY, "y_log": 7.0 + e, "f_lgbm": 7.0,
                      "f_naive": 7.0, "gap_prev": 1.0, "vol_scale": regime, "source": "final"})
    out = IV.add_intervals(f, [0.8], 365, 100).dropna(subset=["lgbm_abs_lo_80", "lgbm_vol_lo_80"])
    calm = out["vol_scale"] < 0.1
    cov_abs_calm = IV.summarise(out[calm], "lgbm_abs", 0.8)["coverage"]
    cov_vol_calm = IV.summarise(out[calm], "lgbm_vol", 0.8)["coverage"]
    assert abs(cov_vol_calm - 0.8) < abs(cov_abs_calm - 0.8)             # the unscaled interval over-covers when calm
    assert cov_abs_calm > 0.9
    w_abs = IV.summarise(out[calm], "lgbm_abs", 0.8)["avg_width_rs"]
    w_vol = IV.summarise(out[calm], "lgbm_vol", 0.8)["avg_width_rs"]
    assert w_vol < w_abs                                                 # and the scaled one is narrower when calm


def test_interval_scores_hand_check():
    y = np.log(np.array([100.0, 100.0, 100.0]))
    lo = np.log(np.array([90.0, 110.0, 80.0]))
    hi = np.log(np.array([120.0, 130.0, 95.0]))
    s = IV.interval_scores(y, np.log(np.array([105.0, 120.0, 90.0])), lo, hi, alpha=0.2)
    assert list(s["covered"]) == [1.0, 0.0, 0.0]
    assert np.allclose(s["width"], [30, 20, 15])
    # Winkler: width + (2/alpha) * shortfall; alpha = 0.2 -> factor 10
    assert np.allclose(s["interval_score"], [30, 20 + 10 * 10, 15 + 10 * 5])
    assert np.allclose(s["rel_width"], [30 / 105, 20 / 120, 15 / 90])


def test_summary_counts_forecasts_without_an_interval_and_ignores_them_in_coverage():
    f = _series(n=300)
    out = IV.add_intervals(f, [0.8, 0.95], 365, 100)
    s = IV.summarise(out, "lgbm_abs", 0.8)
    assert s["n_forecasts"] == 300 and s["n_without_interval"] > 0 and s["n_with_interval"] + s["n_without_interval"] == 300
    assert 0.5 < s["coverage"] <= 1.0


def test_volatility_scale_uses_only_quotes_up_to_the_origin_and_needs_a_full_window():
    days = pd.date_range("2018-01-01", periods=120)
    rng = np.random.default_rng(0)
    price = np.exp(7 + np.cumsum(rng.normal(0, 0.03, 120)))
    price[rng.random(120) < 0.2] = np.nan
    panel = pd.DataFrame({"market_key": "M", "date": days, "modal_price": price, "arrivals_tonnes": 10.0, "is_stale": False})
    a = IV.volatility_scale_table(panel, ["M"], "2018-01-01", "2018-04-30", 30)
    changed = panel.copy()
    changed.loc[changed["date"] > "2018-03-01", "modal_price"] = 12345.0
    b = IV.volatility_scale_table(changed, ["M"], "2018-01-01", "2018-04-30", 30)
    early = a["origin"] <= "2018-03-01"
    pd.testing.assert_series_equal(a.loc[early, "vol_scale"].reset_index(drop=True), b.loc[b["origin"] <= "2018-03-01", "vol_scale"].reset_index(drop=True))
    assert a["vol_scale"].iloc[:30].isna().all() and a["vol_scale"].iloc[30:].notna().all()    # 30 changes needed
    q = np.log(price[~np.isnan(price)])
    assert np.isclose(a["vol_scale"].iloc[30], np.mean(np.abs(np.diff(q))[:30]))


def test_bootstrap_interval_contains_the_point_coverage_and_is_reproducible():
    f = _series(n=1500, seed=3)
    out = IV.add_intervals(f, [0.8], 365, 100).dropna(subset=["lgbm_abs_lo_80"])
    start, end = out["origin"].min(), out["origin"].max()
    a = IV.coverage_bootstrap(out, "lgbm_abs", 0.8, ["A"], start, end, 30, 300, seed=1)
    b = IV.coverage_bootstrap(out, "lgbm_abs", 0.8, ["A"], start, end, 30, 300, seed=1)
    assert a == b
    cov = IV.summarise(out, "lgbm_abs", 0.8)["coverage"]
    assert a["ci_low_A"] <= cov <= a["ci_high_A"] and a["ci_high_A"] - a["ci_low_A"] > 0
