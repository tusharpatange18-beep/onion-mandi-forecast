import copy

import numpy as np
import pandas as pd
import pytest

from mandi import anomalies as AN
from mandi.config import load_config

CFG = load_config()


def _weekly(n=400, seed=0, spike_at=None, spike=0.0, gap=None):
    """Weekly log prices with a yearly cycle and noise; optional multiplicative spike and a run of unobserved weeks."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2014-01-05", periods=n, freq="W-SUN", name="week_end")
    lp = 7 + 0.3 * np.sin(2 * np.pi * np.arange(n) / 52) + np.cumsum(rng.normal(0, 0.01, n)) + rng.normal(0, 0.05, n)
    if spike_at is not None:
        lp[spike_at] += np.log1p(spike)
    s = pd.Series(lp, index=idx)
    if gap:
        s.iloc[gap[0]:gap[1]] = np.nan
    return pd.DataFrame({"lp": s, "n": 6, "la": 4.0})


def test_robust_flags_use_median_and_mad_over_observed_points_only():
    resid = pd.Series(np.r_[np.random.default_rng(0).normal(0, 1, 200), 12.0])
    observed = pd.Series(True, index=resid.index)
    out = AN.robust_flags(resid, observed, 3.5, 1.4826)
    assert out["flag"].iloc[-1] and out["flag"].sum() <= 3                       # the planted outlier, and (almost) nothing else
    assert out["score"].iloc[-1] > 3.5
    observed.iloc[-1] = False                                                    # an unobserved week is never flagged
    assert not AN.robust_flags(resid, observed, 3.5, 1.4826)["flag"].iloc[-1]
    flat = AN.robust_flags(pd.Series(np.ones(50)), pd.Series(True, index=range(50)), 3.5, 1.4826)   # MAD = 0: no score, no flags
    assert not flat["flag"].any() and flat["score"].isna().all()


def test_stl_flags_a_planted_spike_in_the_right_direction_and_leaves_interpolated_weeks_unflagged():
    w = _weekly(spike_at=250, spike=0.8, gap=(100, 108))
    d = AN.stl_detect(w, CFG)
    assert d["flag"].loc[w.index[250]] and d.loc[w.index[250], "resid"] > 0
    gap_weeks = w.index[100:108]
    assert not d.loc[gap_weeks, "observed"].any() and not d.loc[gap_weeks, "flag"].any()   # bridged only so STL can run
    low = AN.stl_detect(_weekly(spike_at=250, spike=-0.6), CFG)
    assert low["flag"].iloc[250] and low["resid"].iloc[250] < 0


def test_a_series_without_spikes_still_gets_some_flags_because_robust_stl_residuals_are_heavy_tailed():
    """Measured, not assumed: with robust=True the STL residuals of pure Gaussian noise are heavy-tailed (kurtosis about 4) and
    1.4826 * MAD underestimates their spread, so the '3.5 robust sigma' threshold flags roughly 5% of clean weeks (1-11% across seeds)."""
    rates = [AN.stl_detect(_weekly(seed=sd), CFG)["flag"].mean() for sd in range(8)]
    assert 0.0 < np.mean(rates) < 0.12 and max(rates) < 0.2


def test_weekly_series_needs_enough_observed_days_and_ignores_missing_days():
    days = pd.date_range("2020-01-06", periods=14)                           # two Monday-Sunday weeks
    price = np.full(14, np.nan)
    price[[0, 1, 2]] = [100.0, 400.0, 100.0]                                 # week 1: 3 observed days
    price[[7, 8]] = [200.0, 200.0]                                           # week 2: only 2
    panel = pd.DataFrame({"date": days, "modal_price": price, "arrivals_tonnes": 10.0})
    w = AN.weekly_series(panel, 3)
    assert np.isclose(w["lp"].iloc[0], np.mean(np.log([100, 400, 100]))) and np.isnan(w["lp"].iloc[1])


def test_surprise_flags_are_exactly_the_forecasts_outside_the_95_interval():
    n = 10
    df = pd.DataFrame({"market": "A", "h": 7, "target_date": pd.date_range("2021-01-01", periods=n), "y_log": 7.0 + np.array([0, 0, .3, 0, -.3, 0, 0, .1, 0, 0]),
                       "f_lgbm": 7.0, "lgbm_abs_lo_95": 6.8, "lgbm_abs_hi_95": 7.2})
    out = AN.surprise_flags(df, 0.95, [7])
    assert list(out["date"]) == [pd.Timestamp("2021-01-03"), pd.Timestamp("2021-01-05")]
    assert list(out["direction"]) == ["up", "down"]
    assert np.allclose(out["expected"], np.exp(7.0)) and np.allclose(out["price"], np.exp([7.3, 6.7]))
    assert np.allclose(out["score"], [0.3 / 0.2, -0.3 / 0.2])                # in units of the interval half-width
    assert AN.surprise_flags(df.assign(lgbm_abs_lo_95=np.nan, lgbm_abs_hi_95=np.nan), 0.95, [7]).empty   # no interval, no flag


def test_arrivals_z_is_measured_against_the_trailing_window_before_the_period():
    days = pd.date_range("2020-01-01", periods=60)
    rng = np.random.default_rng(1)
    arr = np.exp(5 + rng.normal(0, 0.2, 60))
    arr[45] = np.exp(3.0)                                                    # a collapse on one day
    d = pd.DataFrame({"arrivals_tonnes": arr}, index=days)
    z = AN.arrivals_z(d, days[45], days[45], 30)
    ref = np.log(arr[15:45])
    assert np.isclose(z, (np.log(arr[45]) - np.median(ref)) / (1.4826 * np.median(np.abs(ref - np.median(ref)))))
    assert z < -3
    assert np.isnan(AN.arrivals_z(d, days[3], days[3], 30))                  # fewer than 8 reference observations


def test_event_labels_use_only_high_confidence_periods_and_the_verified_mep_period_is_one_of_them():
    d = pd.Series(pd.to_datetime(["2020-04-10", "2020-10-15", "2023-11-15", "2022-06-01", "2024-01-10", "2023-12-07", "2023-12-08"]))
    lab = AN.label_periods(d, CFG)
    assert list(lab) == ["covid_lockdown_2020", "export_ban_2020_21", "mep_oct_dec_2023", "none", "export_ban_2023_24", "mep_oct_dec_2023", "export_ban_2023_24"]
    periods = {p["name"]: p for p in CFG["forecast"]["event_periods"]}
    assert periods["mep_oct_dec_2023"]["end"] < periods["export_ban_2023_24"]["start"]          # no overlap with the ban
    assert "mep_oct_dec_2023" in CFG["anomalies"]["high_confidence_periods"]


def test_week_end_is_the_sunday_of_the_monday_to_sunday_week():
    d = pd.Series(pd.to_datetime(["2021-03-01", "2021-03-03", "2021-03-07", "2021-03-08"]))
    assert list(AN.week_end(d)) == [pd.Timestamp("2021-03-07")] * 3 + [pd.Timestamp("2021-03-14")]


def test_flags_in_other_markets_the_same_week_are_counted_per_detector():
    base = pd.DataFrame({"market": ["A", "B", "A"], "date": pd.to_datetime(["2021-03-03", "2021-03-05", "2021-05-05"]),
                         "detector": ["stl_residual"] * 3})
    base["week_end"] = AN.week_end(base["date"])
    n = base.groupby(["detector", "week_end"])["market"].nunique().rename("n")
    t = base.merge(n, on=["detector", "week_end"])
    assert list(t["n"] - 1) == [1, 1, 0]


def test_counts_table_adds_up():
    tab = pd.DataFrame({"market": ["A"] * 4, "date": pd.to_datetime(["2020-01-05", "2020-02-02", "2021-03-07", "2021-04-04"]),
                        "detector": "stl_residual", "direction": ["up", "down", "up", "up"]})
    c = AN.counts_table(tab)
    assert c["n_flags"].sum() == 4 and (c["n_flags"] == c["n_up"] + c["n_down"]).all()
    assert c[c["year"] == 2021]["n_up"].iloc[0] == 2


def test_spike_placement_is_separated_and_injection_works_on_an_in_memory_copy_only():
    rng = np.random.default_rng(0)
    cand = pd.date_range("2020-01-01", "2024-12-01")
    settings = AN.make_settings(CFG["anomalies"]["synthetic"])
    spikes = AN.place_spikes(cand, settings, 45, rng)
    assert len(spikes) == len(settings) == 12
    starts = sorted(s["start"] for s in spikes)
    assert min(b - a for a, b in zip(starts[:-1], starts[1:])) >= pd.Timedelta(days=45)
    days = pd.date_range("2020-01-01", periods=200)
    panel = pd.DataFrame({"date": days, "modal_price": 1000.0, "arrivals_tonnes": 10.0})
    before = panel.copy()
    sp = [{"magnitude": 0.4, "direction": "up", "duration": 3, "days": pd.date_range("2020-02-01", periods=3)},
          {"magnitude": 0.25, "direction": "down", "duration": 1, "days": pd.date_range("2020-03-01", periods=1)}]
    out = AN.inject_stl(panel, sp)
    pd.testing.assert_frame_equal(panel, before)                              # the real series is untouched
    assert out is not panel
    assert np.allclose(out.loc[out["date"].isin(sp[0]["days"]), "modal_price"], 1400.0)
    assert np.allclose(out.loc[out["date"].isin(sp[1]["days"]), "modal_price"], 750.0)
    assert (out["modal_price"] == 1000.0).sum() == 200 - 4


def test_synthetic_stl_detects_large_spikes_far_more_often_than_tiny_ones_and_keeps_false_positives_low():
    rng = np.random.default_rng(2)
    days = pd.date_range("2016-01-01", "2024-12-31")
    lp = 7 + 0.3 * np.sin(2 * np.pi * np.arange(len(days)) / 365.25) + np.cumsum(rng.normal(0, 0.004, len(days))) + rng.normal(0, 0.02, len(days))
    panel = pd.DataFrame({"market_key": "Maharashtra_Pune_Pune", "date": days, "modal_price": np.exp(lp), "arrivals_tonnes": 10.0})
    cfg = copy.deepcopy(CFG)
    cfg["anomalies"]["synthetic"].update({"magnitudes": [0.02, 0.9], "runs": 4, "durations_days": [3], "directions": ["up", "down"]})
    det, fp = AN.synthetic_stl(cfg, panel)
    rec = det.groupby("magnitude")["detected"].mean()
    assert rec[0.9] > 0.8 and rec[0.02] < 0.3           # not 1.0: robust STL can absorb a large isolated spike (see the next test)
    assert fp[fp["run"] >= 0]["false_positive_rate"].mean() < 0.12


def test_synthetic_surprise_detects_big_shocks_and_its_false_positive_rate_is_near_nominal():
    rng = np.random.default_rng(4)
    n = 2200
    origin = pd.date_range("2017-01-01", periods=n)
    frame = pd.DataFrame({"market": "A", "h": 7, "origin": origin, "target_date": origin + pd.Timedelta(days=7),
                          "y_log": 7.0 + rng.laplace(0, 0.08, n), "f_lgbm": 7.0, "f_naive": 7.0, "gap_prev": 1.0, "vol_scale": 0.05,
                          "source": np.where(origin < "2020-01-01", "seed", "final")})
    cfg = copy.deepcopy(CFG)
    cfg["anomalies"]["synthetic"].update({"magnitudes": [0.02, 0.9], "runs": 4, "durations_days": [1]})
    cfg["anomalies"]["surprise"]["horizons"] = [7]
    det, fp = AN.synthetic_surprise(cfg, frame)
    rec = det.groupby("magnitude")["detected"].mean()
    assert rec[0.9] > 0.95 and rec[0.02] < 0.2
    assert abs(fp[fp["run"] >= 0]["false_positive_rate"].mean() - 0.05) < 0.03


def test_robust_stl_can_absorb_a_large_isolated_down_spike_into_the_seasonal_part_while_non_robust_stl_leaves_it_in_the_residual():
    """Documented behaviour of the specified detector (robust=True): a 90% one-week drop gets robustness weight 0, yet the residual is near
    zero because the seasonal component takes the dip; non-robust STL leaves it in the residual. Found by the synthetic validation."""
    rng = np.random.default_rng(2)
    days = pd.date_range("2016-01-01", "2024-12-31")
    lp = 7 + 0.3 * np.sin(2 * np.pi * np.arange(len(days)) / 365.25) + np.cumsum(rng.normal(0, 0.004, len(days))) + rng.normal(0, 0.02, len(days))
    panel = pd.DataFrame({"date": days, "modal_price": np.exp(lp), "arrivals_tonnes": 10.0})
    sp = [{"magnitude": 0.9, "direction": "down", "duration": 3, "days": pd.date_range("2020-03-06", periods=3)}]
    w = AN.weekly_series(AN.inject_stl(panel, sp), 3)
    wk = pd.Timestamp("2020-03-08")
    robust, plain = AN.stl_detect(w, CFG, robust=True), AN.stl_detect(w, CFG, robust=False)
    assert plain.loc[wk, "flag"] and plain.loc[wk, "resid"] < -0.5
    assert abs(robust.loc[wk, "resid"]) < 0.1 and not robust.loc[wk, "flag"]



# ---------------------------------------------------------------- detector 3: causal daily
def _daily(n=400, seed=0, level=7.0, noise=0.05):
    rng = np.random.default_rng(seed)
    days = pd.date_range("2018-01-01", periods=n)
    return pd.DataFrame({"date": days, "modal_price": np.exp(level + rng.normal(0, noise, n)), "arrivals_tonnes": 10.0})


def test_causal_daily_score_matches_a_hand_computation():
    g = _daily(n=130, seed=1)
    d = AN.causal_daily_detect(g, CFG)
    lp = np.log(g["modal_price"].to_numpy())
    t = 110                                                          # the 111th quote
    base = np.median(lp[t - 28:t])                                   # previous 28 observed quotes
    changes = np.diff(lp)[t - 91:t - 1]                              # the 90 observed daily log changes up to the change into quote t-1
    assert len(changes) == 90
    scale = 1.4826 * np.median(np.abs(changes - np.median(changes)))
    assert np.isclose(d["score"].iloc[t], (lp[t] - base) / scale)
    assert np.isclose(d["expected"].iloc[t], np.exp(base))
    assert d["score"].iloc[:91].isna().all() and d["score"].iloc[91:].notna().all()       # needs 90 prior changes


def test_causal_daily_uses_only_earlier_quotes():
    g = _daily(n=300, seed=2)
    a = AN.causal_daily_detect(g, CFG)
    h = g.copy()
    h.loc[h.index >= 200, "modal_price"] = 99999.0                  # change every quote from the 201st on
    b = AN.causal_daily_detect(h, CFG)
    pd.testing.assert_series_equal(a["score"].iloc[:200].reset_index(drop=True), b["score"].iloc[:200].reset_index(drop=True))
    assert not np.isclose(a["score"].iloc[250], b["score"].iloc[250], equal_nan=True)


def test_causal_daily_skips_days_without_a_quote_and_flags_a_planted_spike_in_the_right_direction():
    g = _daily(n=400, seed=3)
    g.loc[g.index % 7 == 6, "modal_price"] = np.nan                  # a weekly closure: not quotes, so they never enter the windows
    g.loc[250, "modal_price"] *= 1.6
    g.loc[301, "modal_price"] *= 0.4
    d = AN.causal_daily_detect(g, CFG)
    up = d[d["date"] == g.loc[250, "date"]].iloc[0]
    down = d[d["date"] == g.loc[301, "date"]].iloc[0]
    assert up["flag"] and up["direction"] == "up" and down["flag"] and down["direction"] == "down"
    assert len(d) == int(g["modal_price"].notna().sum())
    assert d["flag"].mean() < 0.03                                   # iid noise: very few false flags


def test_synthetic_causal_detects_large_spikes_and_rarely_flags_ordinary_days():
    rng = np.random.default_rng(5)
    days = pd.date_range("2018-01-01", "2024-12-31")
    panel = pd.DataFrame({"market_key": "Maharashtra_Pune_Pune", "date": days, "modal_price": np.exp(7 + rng.normal(0, 0.03, len(days))), "arrivals_tonnes": 10.0})
    cfg = copy.deepcopy(CFG)
    cfg["anomalies"]["synthetic"].update({"magnitudes": [0.02, 0.9], "runs": 3, "durations_days": [1, 3]})
    det, fp = AN.synthetic_causal(cfg, panel)
    rec = det.groupby("magnitude")["detected"].mean()
    assert rec[0.9] > 0.95 and rec[0.02] < 0.1
    assert fp[fp["run"] >= 0]["false_positive_rate"].mean() < 0.03


def test_synthetic_causal_places_spikes_on_the_same_dates_as_the_stl_test():
    """Same seed, candidate days and loop order: the placements of the two tests coincide, so the detectors are compared on the same spikes."""
    rng1, rng2 = np.random.default_rng(7), np.random.default_rng(7)
    cand = pd.date_range("2020-01-01", "2024-12-01")
    settings = AN.make_settings(CFG["anomalies"]["synthetic"])
    a = AN.place_spikes(cand, settings, 45, rng1)
    b = AN.place_spikes(cand, settings, 45, rng2)
    assert [x["start"] for x in a] == [x["start"] for x in b]


def test_rates_by_period_divides_flags_by_the_comparable_dates_of_each_period():
    days = pd.date_range("2020-01-01", "2020-12-31")
    rng = np.random.default_rng(0)
    panel = pd.DataFrame({"market_key": "Maharashtra_Pune_Pune", "date": days, "modal_price": np.exp(7 + rng.normal(0, 0.05, len(days))), "arrivals_tonnes": 10.0})
    mk = AN.F.short_name("Maharashtra_Pune_Pune")
    det = AN.causal_daily_detect(panel, CFG)
    flags = det[det["flag"]].assign(market=mk, detector="causal_daily")
    tab = pd.DataFrame({"market": flags["market"], "date": flags["date"], "detector": flags["detector"]})
    tab["event_period"] = AN.label_periods(tab["date"], CFG)
    weeks = pd.DataFrame({"observed": [True] * 3}, index=pd.DatetimeIndex(["2020-04-05", "2020-04-12", "2020-06-07"], name="week_end"))
    iv = pd.DataFrame({"h": [7], "target_date": [pd.Timestamp("2020-04-15")], "lgbm_abs_lo_95": [1.0]})
    out = AN.rates_by_period(tab, weeks, panel, iv, CFG)
    c = out[(out["detector"] == "causal_daily")]
    assert c["comparable_dates"].sum() == det["score"].notna().sum()                    # every scored day belongs to exactly one label
    assert np.isclose(c["flags"].sum(), len(flags)) and np.allclose(c["flags_per_100_dates"], 100 * c["flags"] / c["comparable_dates"])
    s = out[out["detector"] == "stl_residual"].set_index("event_period")
    assert s.loc["covid_lockdown_2020", "comparable_dates"] == 2 and s.loc["none", "comparable_dates"] == 1   # weeks ending 5 and 12 April are in the lockdown
