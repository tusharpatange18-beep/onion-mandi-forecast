"""Phase 3, analysis only (no model, selection or headline is changed by anything here).

1. Block-bootstrap confidence interval for the macro improvement of the headline model over the strongest baseline.
2. Holm adjustment of the ten Diebold-Mariano p-values of the headline model.
3. A descriptive breakdown of 2021, the year in which the headline model loses to the strongest baseline.

Usage: python -m mandi.posthoc
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import yaml
from statsmodels.stats.multitest import multipletests

from mandi.backtest import BASELINES, git_state
from mandi.config import ROOT, load_config

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- 1. block bootstrap
def day_sums(fair: pd.DataFrame, h: int, markets: list[str], model: str, start: pd.Timestamp, end: pd.Timestamp):
    """Per origin day and market: summed absolute rupee errors of the model, of each baseline, and the number of triples.

    Returns (days, model_sums[D, M], baseline_sums[3, D, M], strongest_sums[D, M], counts[D, M]); `strongest` uses the
    strongest baseline of each market fixed from the full window, exactly as in the headline table.
    """
    days = pd.date_range(start, end)
    d_idx = {d: i for i, d in enumerate(days)}
    D, M = len(days), len(markets)
    g = fair[fair["h"] == h]
    di = g["origin"].map(d_idx).to_numpy()
    mi = g["market"].map({m: i for i, m in enumerate(markets)}).to_numpy()
    y = np.exp(g["y_log"].to_numpy())
    err = {c: np.abs(y - np.exp(g[c].to_numpy())) for c in [model] + BASELINES}
    counts = np.zeros((D, M))
    np.add.at(counts, (di, mi), 1)
    msum = np.zeros((D, M))
    np.add.at(msum, (di, mi), err[model])
    bsum = np.zeros((len(BASELINES), D, M))
    for k, c in enumerate(BASELINES):
        np.add.at(bsum[k], (di, mi), err[c])
    # strongest baseline per market from the full window
    tot = bsum.sum(axis=1)  # [3, M]
    best = tot.argmin(axis=0)
    ssum = np.stack([bsum[best[m], :, m] for m in range(M)], axis=1)
    return days, msum, bsum, ssum, counts


def block_indices(D: int, block: int, n_resamples: int, rng: np.random.Generator) -> np.ndarray:
    """Moving-block resamples of the day index 0..D-1: shape (n_resamples, D). Each row is a concatenation of
    randomly placed runs of `block` consecutive days, trimmed to D days."""
    n_blocks = int(np.ceil(D / block))
    starts = rng.integers(0, D - block + 1, size=(n_resamples, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_resamples, -1)
    return idx[:, :D]


def macro_improvement(msum: np.ndarray, bsum_strong: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Macro % improvement in rupee MAE: equal-weight mean over markets of 100 * (MAE_base - MAE_model) / MAE_base."""
    mae_m, mae_b = msum.sum(axis=-2) / counts.sum(axis=-2), bsum_strong.sum(axis=-2) / counts.sum(axis=-2)
    return (100 * (mae_b - mae_m) / mae_b).mean(axis=-1)


def bootstrap_ci(fair: pd.DataFrame, h: int, markets: list[str], model: str, start: pd.Timestamp, end: pd.Timestamp,
                 block: int, n_resamples: int, seed: int, chunk: int = 250) -> dict[str, float]:
    """95% block-bootstrap interval for the macro improvement over the strongest baseline, and over naive.

    Origin days are resampled in blocks of `block` days and the same days are used for all markets, so correlation
    between markets (and between neighbouring days, and the overlap of the h-day horizons) is kept inside each block.
    The strongest baseline of each market is held fixed at its full-window identity; the `reselect` variant re-chooses it
    in every resample, which is more conservative.
    """
    days, msum, bsum, ssum, counts = day_sums(fair, h, markets, model, start, end)
    point = float(macro_improvement(msum, ssum, counts))
    point_naive = float(macro_improvement(msum, bsum[BASELINES.index("base_naive")], counts))
    rng = np.random.default_rng(seed)
    idx = block_indices(len(days), block, n_resamples, rng)
    fixed, naive, resel = [], [], []
    for i in range(0, n_resamples, chunk):
        ix = idx[i:i + chunk]
        m, c = msum[ix], counts[ix]                       # (chunk, D, M)
        fixed.append(macro_improvement(m, ssum[ix], c))
        naive.append(macro_improvement(m, bsum[BASELINES.index("base_naive")][ix], c))
        tot = np.stack([bsum[k][ix].sum(axis=1) for k in range(len(BASELINES))])   # (3, chunk, M)
        best = tot.argmin(axis=0)                                                   # (chunk, M)
        pick = np.stack([bsum[k][ix] for k in range(len(BASELINES))])               # (3, chunk, D, M)
        chosen = np.take_along_axis(pick, best[None, :, None, :], axis=0)[0]
        resel.append(macro_improvement(m, chosen, c))
    out = {"h": h, "point_estimate_pct": point, "point_estimate_vs_naive_pct": point_naive}
    for name, arr in [("strongest", np.concatenate(fixed)), ("vs_naive", np.concatenate(naive)), ("strongest_reselected", np.concatenate(resel))]:
        out[f"ci95_low_{name}"], out[f"ci95_high_{name}"] = (float(v) for v in np.percentile(arr, [2.5, 97.5]))
        out[f"share_resamples_above_0_{name}"] = float((arr > 0).mean())
    out.update({"block_days": block, "resamples": n_resamples, "seed": seed, "origin_days": len(days)})
    return out


# --------------------------------------------------------------------------- 2. Holm
def holm_table(dm: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Holm step-down adjustment of the headline model's Diebold-Mariano p-values (one per market and horizon)."""
    t = dm[dm["which"] == "headline"][["market", "h", "model", "strongest_baseline", "DM", "p_value"]].copy().reset_index(drop=True)
    reject, p_adj, _, _ = multipletests(t["p_value"].to_numpy(), alpha=alpha, method="holm")
    t["p_holm"] = p_adj
    t["significant_raw_5pct"] = t["p_value"] < alpha
    t["significant_holm_5pct"] = reject
    t["bonferroni_threshold"] = alpha / len(t)
    t["significant_bonferroni"] = t["p_value"] < alpha / len(t)
    return t.sort_values("p_value").reset_index(drop=True)


# --------------------------------------------------------------------------- 3. 2021
def loss_year_breakdown(fair: pd.DataFrame, summ: pd.DataFrame, headline: dict[int, str], year: int = 2021) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Descriptive breakdown of a loss year, by target month and market.

    Returns (by_month, by_market). `by_month` has the MAE of the headline model, the strongest baseline and naive,
    the improvement, and the 'excess' absolute error (headline minus strongest baseline, mean per forecast, in rupees).
    `by_market` compares the year with the other years: improvement, realised absolute log change over the horizon
    (how much prices moved) and the mean signed log error (a persistent over- or under-forecast shows up here).
    Nothing here explains why; it only shows where.
    """
    sb = summ.set_index(["market", "h"])["strongest_baseline"].to_dict()
    t = fair.copy()
    t["year"], t["month"] = t["target_date"].dt.year, t["target_date"].dt.month
    y = np.exp(t["y_log"])
    t["f_head"] = [r[headline[h]] for r, h in zip(t[["ets", "sarima", "prophet", "prophet_anchored", "lgbm"]].to_dict("records"), t["h"])]
    t["ae_model"] = np.abs(y - np.exp(t["f_head"]))
    t["f_base"] = [r[sb[(m, h)]] for r, m, h in zip(t[BASELINES].to_dict("records"), t["market"], t["h"])]
    t["ae_base"] = np.abs(y - np.exp(t["f_base"]))
    t["ae_naive"] = np.abs(y - np.exp(t["base_naive"]))
    t["excess"] = t["ae_model"] - t["ae_base"]
    t["move"] = (t["y_log"] - t["base_naive"]).abs()
    t["signed_model"], t["signed_base"] = t["f_head"] - t["y_log"], t["f_base"] - t["y_log"]
    yr = t[t["year"] == year]
    bm = yr.groupby(["h", "market", "month"]).agg(n=("y_log", "size"), headline_MAE=("ae_model", "mean"), strongest_baseline_MAE=("ae_base", "mean"),
                                                  naive_MAE=("ae_naive", "mean"), excess_AE_per_forecast=("excess", "mean"), mean_abs_log_move=("move", "mean")).reset_index()
    bm["impr_vs_strongest_pct"] = 100 * (bm["strongest_baseline_MAE"] - bm["headline_MAE"]) / bm["strongest_baseline_MAE"]
    t["is_year"] = t["year"] == year
    rows = []
    for (h, m, is_y), g in t.groupby(["h", "market", "is_year"]):
        rows.append({"h": h, "market": m, "period": str(year) if is_y else f"other years", "n": len(g),
                     "impr_vs_strongest_pct": 100 * (g["ae_base"].mean() - g["ae_model"].mean()) / g["ae_base"].mean(),
                     "mean_abs_log_move": g["move"].mean(), "mean_signed_log_error_headline": g["signed_model"].mean(),
                     "mean_signed_log_error_strongest_baseline": g["signed_base"].mean()})
    return bm, pd.DataFrame(rows)


def gain_concentration(fair: pd.DataFrame, summ: pd.DataFrame, headline: dict[int, str], top_share: float = 0.10) -> pd.DataFrame:
    """Where does the headline model's mean gain over the strongest baseline come from?

    Per market and horizon: the saving on each forecast is |baseline error| - |model error| (rupees). The table gives the
    share of the total saving that comes from the `top_share` of forecasts with the largest baseline errors (the large
    moves), the share of forecasts on which the model is better, the median saving, and the mean saving without the large moves.

    In the "ALL (mean of markets)" row every column is the mean of the five market values EXCEPT the share column, which
    is pooled: the sum over all markets of the savings in each market's top forecasts divided by the sum over all markets
    of all savings (averaging ratios of totals that can be near zero or negative is not meaningful).
    """
    sb = summ.set_index(["market", "h"])["strongest_baseline"].to_dict()
    rows, sums = [], {}
    for (m, h), g in fair.groupby(["market", "h"]):
        y = np.exp(g["y_log"].to_numpy())
        e_b = np.abs(y - np.exp(g[sb[(m, h)]].to_numpy()))
        e_m = np.abs(y - np.exp(g[headline[h]].to_numpy()))
        saving = e_b - e_m
        order = np.argsort(-e_b)
        k = int(np.ceil(top_share * len(g)))
        top, rest = order[:k], order[k:]
        total = saving.sum()
        sums[(h, m)] = (float(saving[top].sum()), float(total))
        rows.append({"market": m, "h": h, "n": len(g), "mean_saving_rs": saving.mean(), "median_saving_rs": float(np.median(saving)),
                     "share_of_forecasts_model_better_pct": 100 * float((saving > 0).mean()),
                     f"share_of_total_saving_from_top_{int(top_share * 100)}pct_largest_baseline_errors_pct": 100 * float(saving[top].sum() / total) if total > 0 else np.nan,
                     "mean_saving_rs_excluding_top": float(saving[rest].mean()), "mean_saving_rs_top": float(saving[top].mean())})
    t = pd.DataFrame(rows)
    macro = t.drop(columns="market").groupby("h").mean().reset_index()
    macro.insert(0, "market", "ALL (mean of markets)")
    col = f"share_of_total_saving_from_top_{int(top_share * 100)}pct_largest_baseline_errors_pct"
    for h in macro["h"]:
        top_sum = sum(v[0] for (hh, _), v in sums.items() if hh == h)
        all_sum = sum(v[1] for (hh, _), v in sums.items() if hh == h)
        macro.loc[macro["h"] == h, col] = 100 * top_sum / all_sum if all_sum > 0 else np.nan
    return pd.concat([t, macro], ignore_index=True)


def main(only: str | None = None) -> None:
    """Write results/bootstrap_ci.csv, holm_dm.csv, loss_2021_by_*.csv and gain_concentration.csv (only the last with only='gain')."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config()
    bs = cfg["forecast"]["bootstrap"]
    sel = yaml.safe_load((ROOT / cfg["forecast"]["selected_models_file"]).read_text(encoding="utf-8"))
    headline = {int(h): m for h, m in sel["headline_model_by_horizon"].items()}
    fair = pd.read_parquet(ROOT / "results" / "fair_forecasts.parquet")
    summ = pd.read_csv(ROOT / "results" / "summary_table.csv")
    dm = pd.read_csv(ROOT / "results" / "dm_tests.csv")
    if only == "gain":
        gain_concentration(fair, summ, headline).to_csv(ROOT / "results" / "gain_concentration.csv", index=False)
        logger.info("wrote results/gain_concentration.csv only")
        return
    markets = sorted(fair["market"].unique())
    start, end = pd.Timestamp(cfg["backtest"]["first_test_date"]), pd.Timestamp(cfg["backtest"]["last_test_date"])
    rows = [bootstrap_ci(fair, h, markets, headline[h], start, end, bs["block_days"], bs["resamples"], bs["seed"]) for h in sorted(headline)]
    ci = pd.DataFrame(rows)
    ci.to_csv(ROOT / "results" / "bootstrap_ci.csv", index=False)
    holm_table(dm).to_csv(ROOT / "results" / "holm_dm.csv", index=False)
    gain_concentration(fair, summ, headline).to_csv(ROOT / "results" / "gain_concentration.csv", index=False)
    bm, bk = loss_year_breakdown(fair, summ, headline, 2021)
    bm.to_csv(ROOT / "results" / "loss_2021_by_month.csv", index=False)
    bk.to_csv(ROOT / "results" / "loss_2021_by_market.csv", index=False)
    (ROOT / "results" / "posthoc_meta.json").write_text(
        json.dumps({"git": git_state(), "bootstrap": bs, "note": "analysis only; no model, selection or headline changed"}, indent=2), encoding="utf-8")
    logger.info("bootstrap:\n%s", ci[["h", "point_estimate_pct", "ci95_low_strongest", "ci95_high_strongest", "share_resamples_above_0_strongest"]].round(2).to_string(index=False))


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else None)
