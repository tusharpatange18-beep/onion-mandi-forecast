"""Phase 3: model selection on the tuning folds, final metrics, Diebold-Mariano tests and error analysis."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mandi import metrics as MET
from mandi.backtest import BASELINES, Spec

FAMILIES = ["ets", "sarima", "prophet", "lgbm"]


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
    """Per market and horizon, the model family with the lowest rupee MAE on the tuning folds.

    This fixes the headline model before the 2020-2024 period is looked at.
    """
    best = {}
    for (m, h), g in tab.groupby(["market", "h"]):
        maes = {fam: np.abs(np.exp(g["y_log"]) - np.exp(g[col])).mean() for fam, col in chosen[h].items()}
        best[(m, h)] = min(maes, key=maes.get)
    return best


# --------------------------------------------------------------------------- final evaluation
def long_names(chosen: dict[int, dict[str, str]]) -> dict[tuple[int, str], str]:
    """(h, family) -> column name."""
    return {(h, fam): col for h, d in chosen.items() for fam, col in d.items()}


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
    """Metrics per market, horizon and model (baselines plus the four families) on the shared triples."""
    return MET.metrics_table(tab, BASELINES + FAMILIES, scales)


def summary_table(tab: pd.DataFrame, met: pd.DataFrame, best_val: dict[tuple[str, int], str]) -> pd.DataFrame:
    """Per market and horizon: strongest baseline, validation-chosen model, its improvement in MAE, and every model's improvement."""
    rows = []
    for (m, h), g in met.groupby(["market", "h"]):
        g = g.set_index("model")
        bb = g.loc[BASELINES, "MAE"].idxmin()
        bb_mae = g.loc[bb, "MAE"]
        vm = best_val[(m, h)]
        test_best = g.loc[FAMILIES, "MAE"].idxmin()
        row = {"market": m, "h": h, "n": int(g.loc[bb, "n"]), "strongest_baseline": bb, "baseline_MAE": bb_mae,
               "validation_chosen_model": vm, "model_MAE": g.loc[vm, "MAE"],
               "improvement_pct": 100 * (bb_mae - g.loc[vm, "MAE"]) / bb_mae,
               "test_best_model_(optimistic)": test_best, "test_best_improvement_pct": 100 * (bb_mae - g.loc[test_best, "MAE"]) / bb_mae}
        for fam in FAMILIES:
            row[f"impr_{fam}_pct"] = 100 * (bb_mae - g.loc[fam, "MAE"]) / bb_mae
        rows.append(row)
    return pd.DataFrame(rows)


def _loss(g: pd.DataFrame, col: str) -> np.ndarray:
    """Absolute rupee error of forecast column `col`."""
    return np.abs(np.exp(g["y_log"]) - np.exp(g[col])).to_numpy()


def dm_table(tab: pd.DataFrame, summ: pd.DataFrame) -> pd.DataFrame:
    """Diebold-Mariano tests (absolute rupee error loss) of the validation-chosen model and of the test-best model
    against the strongest baseline, per market and horizon. The test-best row is optimistic: that model was picked
    after seeing the test period, so its p-value ignores the selection."""
    rows = []
    for r in summ.to_dict("records"):
        g = tab[(tab["market"] == r["market"]) & (tab["h"] == r["h"])].sort_values("origin")
        base = _loss(g, r["strongest_baseline"])
        for label, mod in [("validation_chosen", r["validation_chosen_model"]), ("test_best_(optimistic)", r["test_best_model_(optimistic)"])]:
            rows.append({"market": r["market"], "h": r["h"], "which": label, "model": mod, "strongest_baseline": r["strongest_baseline"],
                         **MET.diebold_mariano(base, _loss(g, mod), int(r["h"]))})
    return pd.DataFrame(rows)


def dm_all_models(tab: pd.DataFrame, summ: pd.DataFrame) -> pd.DataFrame:
    """DM test of every family against the strongest baseline (no model selection involved)."""
    rows = []
    for r in summ.to_dict("records"):
        g = tab[(tab["market"] == r["market"]) & (tab["h"] == r["h"])].sort_values("origin")
        base = _loss(g, r["strongest_baseline"])
        for fam in FAMILIES:
            rows.append({"market": r["market"], "h": r["h"], "model": fam, **MET.diebold_mariano(base, _loss(g, fam), int(r["h"]))})
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


def segment_table(tab: pd.DataFrame, summ: pd.DataFrame, best_val: dict[tuple[str, int], str], seg: pd.Series, name: str) -> pd.DataFrame:
    """Pooled error by segment: log-scale MAE of the strongest baseline and of the validation-chosen model, per horizon.

    Pooling is done on the log scale so markets with different price levels carry equal weight per forecast.
    """
    t = tab.copy()
    t["segment"] = seg
    sb = summ.set_index(["market", "h"])["strongest_baseline"].to_dict()
    t["_base"] = [r[sb[(m, h)]] for r, m, h in zip(t[BASELINES].to_dict("records"), t["market"], t["h"])]
    t["_model"] = [r[best_val[(m, h)]] for r, m, h in zip(t[FAMILIES].to_dict("records"), t["market"], t["h"])]
    t["e_base"] = (t["y_log"] - t["_base"]).abs()
    t["e_model"] = (t["y_log"] - t["_model"]).abs()
    t["ae_base"] = np.abs(np.exp(t["y_log"]) - np.exp(t["_base"]))
    t["ae_model"] = np.abs(np.exp(t["y_log"]) - np.exp(t["_model"]))
    g = t.groupby(["h", "segment"]).agg(n=("y_log", "size"), logMAE_baseline=("e_base", "mean"), logMAE_model=("e_model", "mean"),
                                       rupeeMAE_baseline=("ae_base", "mean"), rupeeMAE_model=("ae_model", "mean")).reset_index()
    g["log_improvement_pct"] = 100 * (g["logMAE_baseline"] - g["logMAE_model"]) / g["logMAE_baseline"]
    g.insert(0, "split_by", name)
    return g


def stale_sensitivity(tab: pd.DataFrame, scales: dict[tuple[str, int], float], chosen_best: dict[tuple[str, int], str]) -> pd.DataFrame:
    """Metrics with all triples vs without triples whose origin or target date is flagged is_stale."""
    rows = []
    for label, t in [("all dates", tab), ("excluding is_stale dates", tab[~(tab["stale_origin"] | tab["stale_target"])])]:
        met = final_metrics(t, scales)
        s = summary_table(t, met, chosen_best)
        s.insert(0, "scoring_set", label)
        rows.append(s)
    return pd.concat(rows, ignore_index=True)


def macro_average(summ: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight average over markets of the improvement percentages, per horizon."""
    cols = ["improvement_pct", "test_best_improvement_pct"] + [f"impr_{f}_pct" for f in FAMILIES]
    return summ.groupby("h")[cols].mean().reset_index()
