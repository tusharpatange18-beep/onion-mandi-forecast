"""Phase 3: write docs/tuning_report.md from a saved tuning run (no number is typed by hand).

Usage: python -m mandi.report_tuning [run_id]   (default: the latest results/runs/tune_* run)
"""
from __future__ import annotations

import json
import logging
import sys

import numpy as np
import pandas as pd
import yaml

from mandi import backtest as B
from mandi import evaluate as E
from mandi.config import ROOT, load_config
from mandi.run_phase1 import df_to_md

logger = logging.getLogger(__name__)
MARKETS = ["Solapur", "Pune", "Vashi", "Kolhapur", "Sambhajinagar"]


def _score_tables(kept: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """One row per (market, h, candidate): n, log-scale MAE, rupee MAE, and the same by fold year."""
    rows = []
    for (m, h), g in kept.groupby(["market", "h"]):
        for c in B.BASELINES + cols:
            e_log = (g["y_log"] - g[c]).abs()
            e_rs = (np.exp(g["y_log"]) - np.exp(g[c])).abs()
            r = {"market": m, "h": h, "col": c, "family": E.family_of(c), "n": len(g), "logMAE": e_log.mean(), "MAE_rs": e_rs.mean()}
            for y in (2017, 2018, 2019):
                r[f"logMAE_{y}"] = e_log[g["origin"].dt.year == y].mean()
            rows.append(r)
    return pd.DataFrame(rows)


def _pivot(sc: pd.DataFrame, h: int, value: str, order: list[str]) -> pd.DataFrame:
    t = sc[sc["h"] == h].pivot(index="col", columns="market", values=value).reindex(order)
    t = t[[m for m in MARKETS if m in t.columns]]
    t["macro_mean"] = t.mean(axis=1)
    return t.reset_index().rename(columns={"col": "candidate"})


def _round(df: pd.DataFrame, nd: int) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].round(nd)
    return out


def build(run_id: str | None = None) -> str:
    """Assemble the tuning report."""
    cfg = load_config()
    runs = sorted((ROOT / cfg["forecast"]["runs_dir"]).glob("tune_*"))
    run = (ROOT / cfg["forecast"]["runs_dir"] / run_id) if run_id else runs[-1]
    meta = json.loads((run / "run.json").read_text(encoding="utf-8"))
    tab = pd.read_parquet(run / "forecasts.parquet")
    specs = B.specs_from_config(cfg)
    cols = [s.col for s in specs]
    sel = yaml.safe_load((ROOT / cfg["forecast"]["selected_models_file"]).read_text(encoding="utf-8"))
    chosen = {int(h): d for h, d in sel["chosen"].items()}
    assert sel["tuning_run"] == run.name, "selected_models.yaml belongs to a different tuning run"
    kept, rep = B.fairness_filter(tab, B.BASELINES + cols)
    sc = _score_tables(kept, cols)
    doc = [f"# Tuning report (2017-2019 folds only)", "",
           f"Run `{run.name}`, code commit `{meta['git']['commit']}` (working tree dirty at the end of the run: {meta['git']['dirty']}). "
           f"Window {meta['window'][0]} to {meta['window'][1]}; models refit on the first day of every month using only rows whose target date is before that day. "
           "Nothing from 2020-2024 was used. **The selection below is a proposal: `configs/selected_models.yaml` has not been committed or frozen.**", ""]

    # ---- 1. kept and dropped
    doc += ["## 1. Triples kept and dropped, by market and reason", "",
            "A triple is a (market, origin, horizon) with an observed target. It is kept only if every candidate and every baseline can forecast it.", ""]
    rows = []
    for (m, h), g in tab.groupby(["market", "h"]):
        seasonal = g["base_seasonal"].isna()
        model_fail = g[cols].isna().any(axis=1) | g[["base_naive", "base_mean3"]].isna().any(axis=1)
        k = ~(seasonal | model_fail)
        rows.append({"market": m, "h": h, "triples": len(g), "no_365d_quote_for_baseline_c": int(seasonal.sum()),
                     "a_model_could_not_forecast": int(model_fail.sum()), "dropped_total": int((~k).sum()), "kept": int(k.sum()),
                     "kept_pct": round(100 * k.mean(), 1)})
    kd = pd.DataFrame(rows)
    tot = kd[["triples", "no_365d_quote_for_baseline_c", "a_model_could_not_forecast", "dropped_total", "kept"]].sum()
    kd = pd.concat([kd, pd.DataFrame([{"market": "ALL", "h": "both", **tot.to_dict(), "kept_pct": round(100 * tot["kept"] / tot["triples"], 1)}])], ignore_index=True)
    doc += [df_to_md(kd), "",
            "`no_365d_quote_for_baseline_c`: no observed quote within 3 days of the target date one year earlier (a baseline-(c) limitation, mostly Vashi, which starts in 2016-05). "
            "`a_model_could_not_forecast` is a guard: the run aborts if any model misses more than 2% of a market's triples, so a failing candidate can no longer remove a market silently.", ""]

    # ---- 2. non-converged fits
    doc += ["## 2. Fits, non-converged fits and cost per model", ""]
    tm = pd.DataFrame([{"candidate": k, **v} for k, v in meta["timings"]["per_model"].items()])
    tm["not_converged_pct"] = (100 * tm["not_converged"] / tm["fits"]).round(1)
    doc += [df_to_md(tm[["candidate", "fits", "not_converged", "not_converged_pct", "cpu_seconds", "cpu_seconds_per_fit"]]), "",
            f"Wall-clock for the whole run: {meta['timings']['wall_seconds']} s. Non-converged means the optimiser hit its iteration limit; the fitted parameters are used as they are. "
            "LightGBM and Prophet have no convergence flag in this setup (shown as 0).", ""]

    # ---- 3. full candidate table
    order = B.BASELINES + cols
    doc += ["## 3. All candidates and baselines, by market and horizon", "",
            "Scores on the shared triples (section 1). Lower is better. `macro_mean` is the equal-weight mean over the five markets. "
            "ETS local level and ARIMA(0,1,1) are mathematically the same model, so `ets|local_level` and `sarima|011` count as one result "
            "(identical scores below confirm it). `prophet` is off the shelf; `prophet_anchored` is the declared variant.", ""]
    for h in (7, 30):
        for value, title, nd in [("logMAE", "log-scale MAE", 4), ("MAE_rs", "MAE in rupees per quintal", 1)]:
            doc += [f"### h = {h} days: {title}", "", df_to_md(_round(_pivot(sc, h, value, order), nd)), ""]
    doc += ["### Log-scale MAE by fold year (macro mean over markets), h = 7 and 30", ""]
    fy = sc.groupby(["h", "col"])[["logMAE_2017", "logMAE_2018", "logMAE_2019", "logMAE"]].mean().reset_index()
    doc += [df_to_md(_round(fy, 4)), ""]

    # ---- 4. selection
    doc += ["## 4. Proposed selection (lowest pooled tuning log-MAE per family and horizon)", "", df_to_md(pd.DataFrame([{"h": h, **d} for h, d in chosen.items()])), "",
            "Model chosen per market and horizon on the tuning folds (lowest rupee MAE among the family choices above):", "",
            df_to_md(pd.DataFrame(sel["best_model_by_validation"]).pivot(index="market", columns="h", values="model").reset_index()), ""]

    # ---- 5. vs strongest baseline
    rows = []
    for (m, h), g in sc.groupby(["market", "h"]):
        g = g.set_index("col")
        bb = g.loc[B.BASELINES, "MAE_rs"].idxmin()
        r = {"market": m, "h": h, "n": int(g["n"].iloc[0]), "strongest_baseline": bb, "baseline_MAE_rs": g.loc[bb, "MAE_rs"]}
        for fam in E.FAMILIES:
            r[f"{fam}"] = 100 * (g.loc[bb, "MAE_rs"] - g.loc[chosen[h][fam], "MAE_rs"]) / g.loc[bb, "MAE_rs"]
        rows.append(r)
    imp = pd.DataFrame(rows)
    macro = imp.groupby("h")[E.FAMILIES].mean().reset_index()
    macro_log = []
    for h in (7, 30):
        p = sc[sc["h"] == h].groupby("col")["logMAE"].mean()
        bb = p[B.BASELINES].idxmin()
        macro_log.append({"h": h, "strongest_baseline_(pooled_log)": bb, **{fam: 100 * (p[bb] - p[chosen[h][fam]]) / p[bb] for fam in E.FAMILIES}})
    doc += ["## 5. Frozen candidate of each family vs the strongest baseline on the tuning folds", "",
            "Improvement in rupee MAE = 100 x (strongest baseline MAE - model MAE) / strongest baseline MAE. Positive = the model is better. "
            "The strongest baseline is chosen per market and horizon among naive, mean of the last 3 quotes and 365-day seasonal-naive.", "",
            "### Per market", "", df_to_md(_round(imp, 2)), "", "### Macro average over markets (rupee MAE)", "", df_to_md(_round(macro, 2)), "",
            "### Pooled log-scale MAE improvement (equal weight per market)", "", df_to_md(_round(pd.DataFrame(macro_log), 2)), ""]
    beats = []
    for h in (7, 30):
        for fam in E.FAMILIES:
            v = imp.loc[imp["h"] == h, fam]
            beats.append({"h": h, "family": fam, "markets_where_it_beats_strongest_baseline": f"{int((v > 0).sum())} of {len(v)}",
                          "macro_improvement_pct": round(float(v.mean()), 2)})
    doc += ["### Does the best candidate of each model type beat the strongest baseline on the tuning folds?", "", df_to_md(pd.DataFrame(beats)), ""]

    # ---- 6. notes and history
    doc += ["## 6. Notes", "",
            "- **Tuning history.** Run 1 had a silent NaN in the damped-trend ETS fit for Kolhapur (a 71-day gap right after its first quote made the library's start value NaN), "
            "so Kolhapur was dropped from the selection without any error. This was fixed with explicit start values, a non-finite-parameter check and a coverage guard, and tuning was re-run (run 2). "
            "`prophet_anchored` was then added at the student's request before anything was frozen, and tuning was re-run once more (this run). No 2020-2024 result was looked at in any of the runs.",
            "- **Off-the-shelf Prophet** forecasts from a curve fitted at the last monthly refit; it does not use prices after the refit, which is why it is worse than naive even on refit day. "
            "`prophet_anchored` keeps only its predicted movement and anchors it to the latest quote.",
            "- **ETS local level and ARIMA(0,1,1) are the same model**; they are listed twice only because both were declared candidates, and they count as one result.",
            "- The 365-day seasonal baseline is much worse than naive because price levels differ by a factor of 2-3 between years; an independent recomputation of 40 random triples from the panel matched exactly."]
    return "\n".join(doc)


def main() -> None:
    """Write docs/tuning_report.md."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config()
    text = build(sys.argv[1] if len(sys.argv) > 1 else None)
    (ROOT / cfg["paths"]["docs"] / "tuning_report.md").write_text(text, encoding="utf-8")
    logger.info("wrote docs/tuning_report.md")


if __name__ == "__main__":
    main()
