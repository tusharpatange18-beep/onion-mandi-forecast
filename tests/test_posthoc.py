import numpy as np
import pandas as pd
import pytest

from mandi import posthoc as PH
from mandi.backtest import BASELINES

START, END = pd.Timestamp("2020-01-01"), pd.Timestamp("2022-12-31")
MARKETS = ["A", "B", "C", "D", "E"]


def _fair(model_scale: float, seed: int = 0, h: int = 7) -> pd.DataFrame:
    """Synthetic fair table: every day, every market, one triple. Model error = model_scale x baseline error."""
    rng = np.random.default_rng(seed)
    days = pd.date_range(START, END)
    rows = []
    for m in MARKETS:
        y = np.log(rng.uniform(500, 1500, len(days)))
        e = rng.normal(0, 0.15, len(days))
        rows.append(pd.DataFrame({"market": m, "h": h, "origin": days, "target_date": days + pd.Timedelta(days=h), "y_log": y,
                                  "base_naive": y + e, "base_mean3": y + e * 1.2, "base_seasonal": y + e * 4, "lgbm": y + e * model_scale,
                                  "ets": y + e, "sarima": y + e, "prophet": y + e * 3, "prophet_anchored": y + e * 1.5}))
    return pd.concat(rows, ignore_index=True)


def test_block_indices_are_runs_of_consecutive_days_inside_the_range():
    rng = np.random.default_rng(0)
    idx = PH.block_indices(D=1000, block=30, n_resamples=50, rng=rng)
    assert idx.shape == (50, 1000) and idx.min() >= 0 and idx.max() <= 999
    row = idx[0]
    for b in range(0, 990, 30):                      # inside a block the days are consecutive
        assert (np.diff(row[b:b + 30]) == 1).all()


def test_same_seed_gives_the_same_interval_and_a_different_seed_a_different_one():
    f = _fair(0.9)
    a = PH.bootstrap_ci(f, 7, MARKETS, "lgbm", START, END, 30, 300, seed=1)
    b = PH.bootstrap_ci(f, 7, MARKETS, "lgbm", START, END, 30, 300, seed=1)
    c = PH.bootstrap_ci(f, 7, MARKETS, "lgbm", START, END, 30, 300, seed=2)
    assert a == b and a["ci95_low_strongest"] != c["ci95_low_strongest"]


def test_interval_excludes_zero_for_a_clearly_better_model_and_straddles_it_for_an_equal_one():
    better = PH.bootstrap_ci(_fair(0.8), 7, MARKETS, "lgbm", START, END, 30, 500, seed=3)
    assert better["ci95_low_strongest"] > 0 and better["point_estimate_pct"] > 0
    assert better["ci95_low_strongest"] <= better["point_estimate_pct"] <= better["ci95_high_strongest"]
    equal = _fair(1.0)
    equal["lgbm"] = equal["base_naive"]                  # identical to the strongest baseline
    same = PH.bootstrap_ci(equal, 7, MARKETS, "lgbm", START, END, 30, 300, seed=3)
    assert abs(same["point_estimate_pct"]) < 1e-9 and abs(same["ci95_low_strongest"]) < 1e-9 and abs(same["ci95_high_strongest"]) < 1e-9


def test_point_estimate_equals_the_plain_macro_improvement():
    f = _fair(0.9)
    out = PH.bootstrap_ci(f, 7, MARKETS, "lgbm", START, END, 30, 100, seed=0)
    imps = []
    for m, g in f.groupby("market"):
        y = np.exp(g["y_log"])
        mae = {c: np.abs(y - np.exp(g[c])).mean() for c in BASELINES + ["lgbm"]}
        best = min(BASELINES, key=lambda c: mae[c])  # per-market strongest baseline
        imps.append(100 * (mae[best] - mae["lgbm"]) / mae[best])
    # the plain macro uses each market's strongest baseline, as day_sums does
    assert np.isclose(out["point_estimate_pct"], np.mean(imps))


def test_wider_blocks_do_not_make_the_interval_narrower_when_errors_are_serially_correlated():
    f = _fair(0.9)
    # make the model-vs-baseline advantage persistent in time (slow regimes)
    reg = np.sign(np.sin(np.arange(len(f["origin"].unique())) / 40.0))
    f["lgbm"] = f["y_log"] + (f["base_naive"] - f["y_log"]) * np.tile(np.where(reg > 0, 0.7, 1.3), len(MARKETS))
    short = PH.bootstrap_ci(f, 7, MARKETS, "lgbm", START, END, 1, 400, seed=5)
    long = PH.bootstrap_ci(f, 7, MARKETS, "lgbm", START, END, 60, 400, seed=5)
    w = lambda r: r["ci95_high_strongest"] - r["ci95_low_strongest"]  # noqa: E731
    assert w(long) > w(short)


def test_holm_matches_a_hand_computation():
    dm = pd.DataFrame({"which": "headline", "market": list("abcd"), "h": 7, "model": "lgbm", "strongest_baseline": "base_naive",
                       "DM": 1.0, "p_value": [0.01, 0.04, 0.03, 0.005]})
    t = PH.holm_table(dm).set_index("market")
    # sorted p: .005, .01, .03, .04 -> x4, x3, x2, x1 -> .02, .03, .06, .04 -> running max -> .02, .03, .06, .06
    assert np.allclose(t.loc[["d", "a", "c", "b"], "p_holm"], [0.02, 0.03, 0.06, 0.06])
    assert t.loc["d", "significant_holm_5pct"] and t.loc["a", "significant_holm_5pct"] and not t.loc["c", "significant_holm_5pct"]
    assert t["bonferroni_threshold"].iloc[0] == 0.05 / 4
    assert list(t.index[:1]) == ["d"]  # sorted by raw p-value


def test_loss_year_breakdown_counts_add_up_and_flags_the_losing_month():
    f = _fair(0.9)
    f["target_date"] = f["origin"] + pd.Timedelta(days=7)
    jun = (f["target_date"].dt.year == 2021) & (f["target_date"].dt.month == 6)
    f.loc[jun, "lgbm"] = f.loc[jun, "y_log"] + (f.loc[jun, "base_naive"] - f.loc[jun, "y_log"]) * 2.0   # worse than baseline in June 2021
    summ = pd.DataFrame({"market": MARKETS, "h": 7, "strongest_baseline": "base_naive"})
    bm, bk = PH.loss_year_breakdown(f, summ, {7: "lgbm"}, 2021)
    assert bm["n"].sum() == int((f["target_date"].dt.year == 2021).sum())
    assert (bm[bm["month"] == 6]["impr_vs_strongest_pct"] < 0).all()
    assert (bm[bm["month"] == 5]["impr_vs_strongest_pct"] > 0).all()
    assert set(bk["period"]) == {"2021", "other years"}


def test_gain_concentration_attributes_the_saving_to_the_large_moves():
    rng = np.random.default_rng(0)
    n = 1000
    y = np.log(np.full(n, 1000.0))
    e = rng.normal(0, 0.02, n)
    big = np.arange(n) < 100                              # 10% of forecasts are large moves for the baseline
    e[big] = 0.5
    f = pd.DataFrame({"market": "A", "h": 7, "origin": pd.date_range("2020-01-01", periods=n), "y_log": y, "base_naive": y + e,
                      "base_mean3": y + e * 3, "base_seasonal": y + e * 9, "lgbm": y + np.where(big, 0.1, e), "ets": y, "sarima": y, "prophet": y, "prophet_anchored": y})
    summ = pd.DataFrame({"market": ["A"], "h": [7], "strongest_baseline": ["base_naive"]})
    g = PH.gain_concentration(f, summ, {7: "lgbm"})
    r = g[g["market"] == "A"].iloc[0]
    assert r["share_of_total_saving_from_top_10pct_largest_baseline_errors_pct"] > 99      # all the gain is in the large moves
    assert np.isclose(r["median_saving_rs"], 0.0)                                          # typical forecasts are unchanged
    assert np.isclose(r["mean_saving_rs_excluding_top"], 0.0)
    assert (g["market"] == "ALL (mean of markets)").sum() == 1


def test_gain_concentration_all_row_share_is_pooled_not_an_average_of_ratios():
    """Two markets: in A the whole saving is in the top 10% (share 100%); in B the total saving is tiny and negative
    elsewhere. An average of the two ratios would be meaningless; the pooled share is sum(top savings) / sum(all savings)."""
    rng = np.random.default_rng(1)
    n = 1000
    frames = []
    for m, top_gain, rest_gain in [("A", 0.10, 0.0), ("B", 0.02, -0.001)]:
        y = np.log(np.full(n, 1000.0))
        e = rng.normal(0, 0.02, n)
        big = np.arange(n) < 100
        e[big] = 0.5
        lg = np.where(big, e - top_gain * np.sign(e), e + rest_gain)
        frames.append(pd.DataFrame({"market": m, "h": 7, "origin": pd.date_range("2020-01-01", periods=n), "y_log": y, "base_naive": y + e,
                                    "base_mean3": y + e * 3, "base_seasonal": y + e * 9, "lgbm": y + lg, "ets": y, "sarima": y, "prophet": y, "prophet_anchored": y}))
    f = pd.concat(frames, ignore_index=True)
    summ = pd.DataFrame({"market": ["A", "B"], "h": [7, 7], "strongest_baseline": ["base_naive", "base_naive"]})
    g = PH.gain_concentration(f, summ, {7: "lgbm"})
    col = "share_of_total_saving_from_top_10pct_largest_baseline_errors_pct"
    # recompute the pooled share by hand
    top_sum = tot_sum = 0.0
    for m, d in f.groupby("market"):
        y = np.exp(d["y_log"].to_numpy())
        eb, em = np.abs(y - np.exp(d["base_naive"].to_numpy())), np.abs(y - np.exp(d["lgbm"].to_numpy()))
        s_ = eb - em
        top = np.argsort(-eb)[:100]
        top_sum += s_[top].sum()
        tot_sum += s_.sum()
    all_row = g[g["market"] == "ALL (mean of markets)"].iloc[0]
    assert np.isclose(all_row[col], 100 * top_sum / tot_sum)
    per_market = g[g["market"].isin(["A", "B"])][col]
    assert not np.isclose(all_row[col], per_market.mean())          # it is not the average of the market ratios
    assert np.isclose(all_row["mean_saving_rs"], g[g["market"].isin(["A", "B"])]["mean_saving_rs"].mean())  # other columns stay means of markets
