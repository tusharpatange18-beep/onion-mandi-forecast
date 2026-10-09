"""Phase 3: model selection on the tuning folds, final metrics, Diebold-Mariano tests and error analysis.

Conventions (declared in the brief and at the top of docs/model_report.md):
* Primary metric: MAE in rupees per quintal. Log-scale MAE is secondary.
* Strongest baseline per market and horizon = the baseline (naive, mean of last 3 quotes, 365-day seasonal-naive) with the
  lowest rupee MAE on the final evaluation window. This is a conservative, hindsight choice (it makes the baseline as hard as possible).
* Headline model = the family with the best pooled tuning score at each horizon (fixed before 2020-2024 was run).
  The per-market "per-cell tuning-selected" model and the "test-best" model are secondary rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mandi import metrics as MET
from mandi.backtest import BASELINES

FAMILIES = ["ets", "sarima", "prophet", "prophet_anchored", "lgbm"]  # prophet = off the shelf; prophet_anchored = declared variant


def family_of(col: str) -> str:
    """'sarima|111' -> 'sarima'; baselines are returned unchanged."""
    return col.split("|")[0] if "|" in col else col


# --------------------------------------------------------------------------- selection on the tuning folds (2017-2019 only)
def tuning_scores(tab: pd.DataFrame, spec_cols: list[str]) -> pd.DataFrame:
    """Score of every candidate: log-scale MAE averaged over markets, then over fold years, per horizon.

    `tab` must already be the fair set of triples (every candidate and baseline has a forecast).
    """
    t = tab.copy()
    t["fold"] = t["origin"].dt.year
    rows = []
    for col in spec_cols + BASELINES:
        t["_e"] = (t["y_log"] - t[col]).abs()
        g = t.groupby(["h", "fold", "market"])["_e"].mean().groupby(["h", "fold"]).mean()  # equal weight per market
        for h, s in g.groupby(level=0):
            rows.append({"col": col, "family": family_of(col), "h": h, "score_log_MAE": s.mean(), **{f"fold_{f}": v for (_, f), v in s.items()}})
    return pd.DataFrame(rows)


def select_specs(scores: pd.DataFrame) -> dict[int, dict[str, str]]:
    """Per horizon and model family, the candidate with the lowest tuning score (ties go to the first listed)."""
    out: dict[int, dict[str, str]] = {}
    for h, g in scores[scores["family"].isin(FAMILIES)].groupby("h"):
        out[int(h)] = {fam: gg.loc[gg["score_log_MAE"].idxmin(), "col"] for fam, gg in g.groupby("family")}
    return out


def best_model_by_validation(tab: pd.DataFrame, chosen: dict[int, dict[str, str]]) -> dict[tuple[str, int], str]:
    """Per market and horizon, the family with the lowest rupee MAE on the tuning folds (the secondary 'per-cell' choice)."""
    best = {}
    for (m, h), g in tab.groupby(["market", "h"]):
        maes = {fam: np.abs(np.exp(g["y_log"]) - np.exp(g[col])).mean() for fam, col in chosen[h].items()}
        best[(m, h)] = min(maes, key=maes.get)
    return best


# --------------------------------------------------------------------------- final evaluation
def attach_model_columns(tab: pd.DataFrame, chosen: dict[int, dict[str, str]]) -> pd.DataFrame:
    """Add one forecast column per family (named by family) holding that horizon's selected spec's forecast."""
    t = tab.copy()
    for fam in FAMILIES:
        t[fam] = np.nan
        for h, d in chosen.items():
            if fam in d:
                sel = t["h"] == h
                t.loc[sel, fam] = t.loc[sel, d[fam]]
    return t


def final_metrics(tab: pd.DataFrame, scales: dict[tuple[str, int], float]) -> pd.DataFrame:
    """Metrics per market, horizon and model (3 baselines plus the five families) on the shared triples."""
    return MET.metrics_table(tab, BASELINES + FAMILIES, scales)


def _impr(base: float, model: float) -> float:
    return 100 * (base - model) / base


def summary_table(met: pd.DataFrame, headline: dict[int, str], per_cell: dict[tuple[str, int], str]) -> pd.DataFrame:
    """One row per market and horizon, every family shown.

    Columns: strongest baseline (lowest rupee MAE on the final window), the headline model's MAE and improvements over
    the strongest baseline and over naive, the secondary per-cell tuning-selected model, the optimistic test-best model,
    and the improvement of every family over the strongest baseline and over naive.
    """
    rows = []
    for (m, h), g in met.groupby(["market", "h"]):
        g = g.set_index("model")
        bb = g.loc[BASELINES, "MAE"].idxmin()
        bmae, nmae = g.loc[bb, "MAE"], g.loc["base_naive", "MAE"]
        hm, pc = headline[h], per_cell[(m, h)]
        tb = g.loc[FAMILIES, "MAE"].idxmin()
        row = {"market": m, "h": h, "n": int(g.loc[bb, "n"]), "strongest_baseline": bb, "baseline_MAE": bmae, "naive_MAE": nmae,
               "headline_model": hm, "headline_MAE": g.loc[hm, "MAE"],
               "headline_impr_vs_strongest_pct": _impr(bmae, g.loc[hm, "MAE"]), "headline_impr_vs_naive_pct": _impr(nmae, g.loc[hm, "MAE"]),
               "per_cell_selected_model": pc, "per_cell_impr_vs_strongest_pct": _impr(bmae, g.loc[pc, "MAE"]),
               "test_best_model_(optimistic)": tb, "test_best_impr_vs_strongest_pct": _impr(bmae, g.loc[tb, "MAE"])}
        for fam in FAMILIES:
            row[f"impr_{fam}_vs_strongest_pct"] = _impr(bmae, g.loc[fam, "MAE"])
        for fam in FAMILIES:
            row[f"impr_{fam}_vs_naive_pct"] = _impr(nmae, g.loc[fam, "MAE"])
        rows.append(row)
    return pd.DataFrame(rows)


def macro_average(summ: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight average over markets of every improvement percentage, per horizon."""
    cols = [c for c in summ.columns if c.endswith("_pct")]
    return summ.groupby("h")[cols].mean().reset_index()


def _loss(g: pd.DataFrame, col: str) -> np.ndarray:
    return np.abs(np.exp(g["y_log"]) - np.exp(g[col])).to_numpy()


def dm_table(tab: pd.DataFrame, summ: pd.DataFrame) -> pd.DataFrame:
    """Diebold-Mariano tests (absolute rupee error) against the strongest baseline, per market and horizon, for the
    headline model, the per-cell tuning-selected model and the test-best model. The last is optimistic: it was picked
    after seeing the test period, so its p-value ignores the selection."""
    rows = []
    for r in summ.to_dict("records"):
        g = tab[(tab["market"] == r["market"]) & (tab["h"] == r["h"])].sort_values("origin")
        base = _loss(g, r["strongest_baseline"])
        for label, mod in [("headline", r["headline_model"]), ("per_cell_selected", r["per_cell_selected_model"]),
                           ("test_best_(optimistic)", r["test_best_model_(optimistic)"])]:
            rows.append({"market": r["market"], "h": r["h"], "which": label, "model": mod, "strongest_baseline": r["strongest_baseline"],
                         **MET.diebold_mariano(base, _loss(g, mod), int(r["h"]))})
    return pd.DataFrame(rows)


def dm_all_models(tab: pd.DataFrame, summ: pd.DataFrame) -> pd.DataFrame:
    """DM test of every family against the strongest baseline (no selection involved)."""
    rows = []
    for r in summ.to_dict("records"):
        g = tab[(tab["market"] == r["market"]) & (tab["h"] == r["h"])].sort_values("origin")
        base = _loss(g, r["strongest_baseline"])
        for fam in FAMILIES:
            rows.append({"market": r["market"], "h": r["h"], "model": fam, **MET.diebold_mariano(base, _loss(g, fam), int(r["h"]))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- robustness: per year, win rate, median error
def per_year_table(tab: pd.DataFrame, summ: pd.DataFrame, headline: dict[int, str]) -> pd.DataFrame:
    """MAE by target year, per market and horizon: strongest baseline, naive, headline model, and the improvements."""
    sb = summ.set_index(["market", "h"])["strongest_baseline"].to_dict()
    t = tab.copy()
    t["year"] = t["target_date"].dt.year
    rows = []
    for (m, h, y), g in t.groupby(["market", "h", "year"]):
        base, hm = sb[(m, h)], headline[h]
        b, n, mo = _loss(g, base).mean(), _loss(g, "base_naive").mean(), _loss(g, hm).mean()
        rows.append({"market": m, "h": h, "year": y, "n": len(g), "strongest_baseline_MAE": b, "naive_MAE": n, "headline_MAE": mo,
                     "impr_vs_strongest_pct": _impr(b, mo), "impr_vs_naive_pct": _impr(n, mo)})
    return pd.DataFrame(rows)


def robustness_table(tab: pd.DataFrame, summ: pd.DataFrame) -> pd.DataFrame:
    """Win rate and median absolute error of every family and baseline against the strongest baseline and against naive.

    Win rate = share of triples where the model's absolute rupee error is smaller (ties count half).
    """
    sb = summ.set_index(["market", "h"])["strongest_baseline"].to_dict()
    rows = []
    for (m, h), g in tab.groupby(["market", "h"]):
        base, naive = _loss(g, sb[(m, h)]), _loss(g, "base_naive")
        for fam in FAMILIES + BASELINES:
            e = _loss(g, fam)
            row = {"market": m, "h": h, "model": fam, "n": len(g), "median_AE": float(np.median(e)),
                   "median_AE_strongest_baseline": float(np.median(base)), "median_AE_naive": float(np.median(naive)),
                   "win_rate_vs_strongest_pct": 100 * float(np.mean((e < base) + 0.5 * (e == base))),
                   "win_rate_vs_naive_pct": 100 * float(np.mean((e < naive) + 0.5 * (e == naive)))}
            row["median_AE_impr_vs_strongest_pct"] = _impr(row["median_AE_strongest_baseline"], row["median_AE"])
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- error analysis
def assign_period(dates: pd.Series, periods: list[dict[str, str]]) -> pd.Series:
    """Label each date with the event period it falls in ('normal' if none). Periods must not overlap."""
    out = pd.Series("normal", index=dates.index)
    for p in periods:
        out[(dates >= p["start"]) & (dates <= p["end"])] = p["name"]
    return out


def gap_bucket(gap: pd.Series) -> pd.Series:
    """Days since the previous observed quote before the origin, in buckets."""
    return pd.cut(gap, [0, 1, 2, 6, 10_000], labels=["1 day (consecutive)", "2 days", "3-6 days", "7+ days"]).astype(str)


def segment_table(tab: pd.DataFrame, summ: pd.DataFrame, headline: dict[int, str], seg: pd.Series, name: str) -> pd.DataFrame:
    """Pooled error by segment: strongest baseline, naive and headline model, per horizon.

    Pooling is done on the log scale so markets with different price levels carry equal weight per forecast; rupee MAE
    (dominated by the high-priced markets) is shown alongside. The segment is that of the *target* date.
    """
    t = tab.copy()
    t["segment"] = seg
    sb = summ.set_index(["market", "h"])["strongest_baseline"].to_dict()
    t["_base"] = [r[sb[(m, h)]] for r, m, h in zip(t[BASELINES].to_dict("records"), t["market"], t["h"])]
    t["_model"] = [r[headline[h]] for r, h in zip(t[FAMILIES].to_dict("records"), t["h"])]
    t["e_base"] = (t["y_log"] - t["_base"]).abs()
    t["e_naive"] = (t["y_log"] - t["base_naive"]).abs()
    t["e_model"] = (t["y_log"] - t["_model"]).abs()
    t["ae_base"] = np.abs(np.exp(t["y_log"]) - np.exp(t["_base"]))
    t["ae_model"] = np.abs(np.exp(t["y_log"]) - np.exp(t["_model"]))
    g = t.groupby(["h", "segment"]).agg(n=("y_log", "size"), logMAE_strongest_baseline=("e_base", "mean"), logMAE_naive=("e_naive", "mean"),
                                       logMAE_headline=("e_model", "mean"), rupeeMAE_strongest_baseline=("ae_base", "mean"),
                                       rupeeMAE_headline=("ae_model", "mean")).reset_index()
    g["log_impr_vs_strongest_pct"] = 100 * (g["logMAE_strongest_baseline"] - g["logMAE_headline"]) / g["logMAE_strongest_baseline"]
    g["log_impr_vs_naive_pct"] = 100 * (g["logMAE_naive"] - g["logMAE_headline"]) / g["logMAE_naive"]
    g.insert(0, "split_by", name)
    return g


def stale_sensitivity(tab: pd.DataFrame, scales: dict[tuple[str, int], float], headline: dict[int, str],
                      per_cell: dict[tuple[str, int], str]) -> pd.DataFrame:
    """Sensitivity (b): metrics with all triples vs without triples whose origin or target date is flagged is_stale.

    The strongest baseline is recomputed on each scoring set. This is a sensitivity check; it is never used to choose models.
    """
    rows = []
    for label, t in [("all dates", tab), ("excluding is_stale dates", tab[~(tab["stale_origin"] | tab["stale_target"])])]:
        s = summary_table(final_metrics(t, scales), headline, per_cell)
        s.insert(0, "scoring_set", label)
        rows.append(s)
    return pd.concat(rows, ignore_index=True)
