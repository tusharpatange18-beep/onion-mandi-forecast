"""Run Phase 3.

Usage (from the project root, with the project venv):
    python -m mandi.run_phase3 tune                    # 2017-2019 tuning folds -> configs/selected_models.yaml
    python -m mandi.run_phase3 final                   # 2020-2024 evaluation with the frozen selection
    python -m mandi.run_phase3 final --variant error   # sensitivity (a): 'likely error' flagged points set to missing
    python -m mandi.run_phase3 ablation                # post-hoc feature-group ablation of the frozen LightGBM
    python -m mandi.run_phase3 analyse <run_id> [--prefix sens_error_]

`final` and `ablation` refuse to start unless configs/selected_models.yaml is committed and unchanged AND the source,
config and test files have no uncommitted changes, so a run is always tied to a commit.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from mandi import backtest as B
from mandi import evaluate as E
from mandi import features as F
from mandi.config import ROOT, load_config

logger = logging.getLogger(__name__)


def _inputs(cfg: dict[str, Any], variant: str | None = None):
    """Panel, arrays, origin tables; `variant='error'` sets 'likely error' flagged points to missing."""
    panel = pd.read_parquet(ROOT / cfg["paths"]["panel"])
    extra: dict[str, Any] = {}
    if variant == "error":
        rev = pd.read_csv(ROOT / cfg["paths"]["interim"] / "outlier_review.csv", parse_dates=["date"])
        bad = rev[rev["guess"] == "likely error"]
        short = panel["market_key"].str.split("_").str[-1]
        newly = []
        for r in bad.itertuples():
            sel = (short == r.market) & (panel["date"] == r.date) & panel["modal_price"].notna()
            if sel.any():
                newly.append(f"{r.market} {r.date.date()}")
            panel.loc[sel, "modal_price"] = np.nan
        extra = {"likely_error_points_in_review": len(bad), "newly_set_to_missing": newly}
        logger.info("variant error: %d likely-error points in the review table, newly set to missing: %s", len(bad), newly)
    arrays = F.build_arrays(panel, cfg["data"]["start_date"], cfg["data"]["end_date"], cfg["markets"])
    closure = F.closure_weekdays(pd.read_parquet(ROOT / cfg["paths"]["panel"]), cfg["eda"]["train_end"])
    tables = B.build_tables(arrays, closure, cfg)
    return panel, arrays, tables, extra


def _selected_path(cfg: dict[str, Any]) -> Path:
    return ROOT / cfg["forecast"]["selected_models_file"]


def cmd_tune(cfg: dict[str, Any], workers: int) -> None:
    """Fit every candidate on the 2017-2019 folds, pick one spec per family and horizon, and write the proposed selection."""
    _, arrays, tables, _ = _inputs(cfg)
    specs = B.specs_from_config(cfg)
    folds = cfg["forecast"]["tuning_folds"]
    window = (f"{folds[0]}-01-01", f"{folds[-1]}-12-31")
    assert pd.Timestamp(window[1]) <= pd.Timestamp(cfg["eda"]["train_end"]), "tuning must stay inside 2014-2019"
    tab, timings = B.run_backtest(arrays, tables, specs, window, cfg, workers)
    cols = [s.col for s in specs]
    kept, rep = B.fairness_filter(tab, B.BASELINES + cols)
    scores = E.tuning_scores(kept, cols)
    chosen = E.select_specs(scores)
    best_val = E.best_model_by_validation(kept, chosen)
    run_id = "tune_" + pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    B.log_run("tune", run_id, cfg, specs, tab, timings, {"window": window, "triples_kept": len(kept), "triples_total": len(tab)})
    out = ROOT / "results" / "tuning"
    out.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out / "tuning_scores.csv", index=False)
    rep.to_csv(out / "tuning_dropped_triples.csv", index=False)
    sel = {"tuning_run": run_id, "window": list(window), "chosen": {int(h): d for h, d in chosen.items()},
           "best_model_by_validation": [{"market": m, "h": int(h), "model": mod} for (m, h), mod in sorted(best_val.items())]}
    _selected_path(cfg).write_text("# Written by `python -m mandi.run_phase3 tune`. Commit this file BEFORE running `final`.\n" + yaml.safe_dump(sel, sort_keys=False), encoding="utf-8")
    logger.info("tuning done: %d of %d triples kept; selection written to %s", len(kept), len(tab), _selected_path(cfg))


def _load_selection(cfg: dict[str, Any]) -> dict[str, Any]:
    """The frozen selection. Refuses to run unless it is committed and unchanged and the code is clean."""
    p = _selected_path(cfg)
    if not p.exists():
        raise SystemExit("configs/selected_models.yaml not found: run `tune` first")
    rel = str(p.relative_to(ROOT))
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=ROOT, capture_output=True, text=True).returncode == 0
    dirty_sel = bool(subprocess.run(["git", "status", "--porcelain", rel], cwd=ROOT, capture_output=True, text=True).stdout.strip())
    if not tracked or dirty_sel:
        raise SystemExit("configs/selected_models.yaml must be committed (and unchanged) before the final run")
    if B.git_state()["dirty"]:
        raise SystemExit("src/, configs/, tests/ or project.md have uncommitted changes: commit them so the run log is dirty=False")
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def cmd_final(cfg: dict[str, Any], workers: int, variant: str | None) -> str:
    """Run the frozen selection over 2020-2024 and analyse it."""
    sel = _load_selection(cfg)
    chosen = {int(h): d for h, d in sel["chosen"].items()}
    all_specs = {s.col: s for s in B.specs_from_config(cfg)}
    specs = [all_specs[c] for c in sorted({c for d in chosen.values() for c in d.values()})]
    _, arrays, tables, extra = _inputs(cfg, variant)
    window = (cfg["backtest"]["first_test_date"], cfg["backtest"]["last_test_date"])
    tab, timings = B.run_backtest(arrays, tables, specs, window, cfg, workers)
    run_id = f"final{'_' + variant if variant else ''}_" + pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    B.log_run("final" + (f"_{variant}" if variant else ""), run_id, cfg, specs, tab, timings,
              {"window": window, "selection_run": sel["tuning_run"], "variant": variant, **extra})
    logger.info("final run %s done in %.0fs", run_id, timings["wall_seconds"])
    cmd_analyse(cfg, run_id, "sens_error_" if variant == "error" else "")
    return run_id


def cmd_analyse(cfg: dict[str, Any], run_id: str, prefix: str) -> None:
    """Compute every table from a saved run (no refitting) and write results/*.csv."""
    sel = yaml.safe_load(_selected_path(cfg).read_text(encoding="utf-8"))
    chosen = {int(h): d for h, d in sel["chosen"].items()}
    headline = {int(h): m for h, m in sel["headline_model_by_horizon"].items()}
    per_cell = {(r["market"], int(r["h"])): r["model"] for r in sel["best_model_by_validation"]}
    run_dir = ROOT / cfg["forecast"]["runs_dir"] / run_id
    tab = pd.read_parquet(run_dir / "forecasts.parquet")
    meta = yaml.safe_load((run_dir / "run.json").read_text(encoding="utf-8"))
    chosen_cols = sorted({c for d in chosen.values() for c in d.values()})
    kept, rep = B.fairness_filter(tab, B.BASELINES + chosen_cols)
    kept = E.attach_model_columns(kept, chosen)
    _, _, tables, _ = _inputs(cfg, "error" if prefix == "sens_error_" else None)
    scales = B.mase_scales(tables, cfg["eda"]["train_end"])  # in-sample naive error, fixed from 2014-2019
    met = E.final_metrics(kept, scales)
    spec_of = {(h, fam): col for h, d in chosen.items() for fam, col in d.items()}
    met["spec"] = [spec_of.get((h, m), "") for h, m in zip(met["h"], met["model"])]
    summ = E.summary_table(met, headline, per_cell)
    periods = cfg["forecast"]["event_periods"]
    per = E.assign_period(kept["target_date"], periods)
    grp = per.map({p["name"]: p["group"] for p in periods}).fillna("normal")
    seg = pd.concat([E.segment_table(kept, summ, headline, per, "event_period"), E.segment_table(kept, summ, headline, grp, "event_group"),
                     E.segment_table(kept, summ, headline, E.gap_bucket(kept["gap_prev"]), "gap_before_origin")], ignore_index=True)
    runtime = pd.DataFrame([{"model": k, **v} for k, v in meta["timings"]["per_model"].items()])
    outputs = [("backtest_metrics", met), ("summary_table", summ), ("summary_macro", E.macro_average(summ)), ("dm_tests", E.dm_table(kept, summ)),
               ("dm_all_models", E.dm_all_models(kept, summ)), ("per_year", E.per_year_table(kept, summ, headline)),
               ("robustness", E.robustness_table(kept, summ)), ("error_analysis", seg), ("runtime", runtime), ("dropped_triples", rep)]
    if not prefix:
        outputs.append(("sensitivity_stale", E.stale_sensitivity(kept, scales, headline, per_cell)))
    res = ROOT / "results"
    for name, df in outputs:
        df.to_csv(res / f"{prefix}{name}.csv", index=False)
    kept.to_parquet(res / f"{prefix}fair_forecasts.parquet", index=False)
    logger.info("analysis written: %d fair triples of %d (%d dropped); run commit %s dirty=%s selection commit %s", len(kept), len(tab),
                len(tab) - len(kept), meta["git"]["commit"], meta["git"]["dirty"], meta["git"]["selection_commit"])


def cmd_ablation(cfg: dict[str, Any], workers: int) -> None:
    """POST-HOC, not used for any choice: refit the frozen LightGBM with one feature group dropped at a time.

    Same hyperparameters, same monthly refits, same triples as the final run; reports the change in rupee MAE.
    """
    sel = _load_selection(cfg)
    chosen = {int(h): d for h, d in sel["chosen"].items()}
    groups = cfg["forecast"]["ablation_groups"]
    all_specs = {s.col: s for s in B.specs_from_config(cfg)}
    fair = pd.read_parquet(ROOT / "results" / "fair_forecasts.parquet")
    _, arrays, tables, _ = _inputs(cfg)
    window = (cfg["backtest"]["first_test_date"], cfg["backtest"]["last_test_date"])
    keys = ["market_idx", "origin", "h"]
    out = fair[["market", *keys, "y_log", "lgbm", "base_naive", "base_mean3", "base_seasonal"]].rename(columns={"lgbm": "full"})
    all_timings, spec_list, parts = {}, [], []
    for h in tables:
        base = all_specs[chosen[h]["lgbm"]]
        specs = [B.Spec("lgbm", f"{base.name}__drop_{g}", base.params + (("drop", tuple(cols)),)) for g, cols in groups.items()]
        spec_list += specs
        tab, timings = B.run_backtest(arrays, {h: tables[h]}, specs, window, cfg, workers)
        all_timings[h] = timings
        parts.append(tab[keys + [s.col for s in specs]].rename(columns={s.col: f"drop_{g}" for s, g in zip(specs, groups)}))
    out = out.merge(pd.concat(parts, ignore_index=True), on=keys, how="left")
    variants = ["full"] + [f"drop_{g}" for g in groups]
    out = out.dropna(subset=variants)
    rows = []
    for (m, h), g in out.groupby(["market", "h"]):
        bl = min(B.BASELINES, key=lambda c: np.abs(np.exp(g["y_log"]) - np.exp(g[c])).mean())
        bmae = np.abs(np.exp(g["y_log"]) - np.exp(g[bl])).mean()
        full_mae = np.abs(np.exp(g["y_log"]) - np.exp(g["full"])).mean()
        for v in variants:
            mae = np.abs(np.exp(g["y_log"]) - np.exp(g[v])).mean()
            rows.append({"market": m, "h": h, "variant": v, "n": len(g), "MAE": mae, "change_vs_full_MAE": mae - full_mae,
                         "change_vs_full_pct": 100 * (mae - full_mae) / full_mae, "strongest_baseline": bl,
                         "impr_vs_strongest_pct": 100 * (bmae - mae) / bmae})
    abl = pd.DataFrame(rows)
    abl.to_csv(ROOT / "results" / "ablation_lgbm.csv", index=False)
    run_id = "ablation_" + pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    timings = {"wall_seconds": round(sum(t["wall_seconds"] for t in all_timings.values()), 1),
               "per_model": {k: v for t in all_timings.values() for k, v in t["per_model"].items()}}
    B.log_run("ablation_post_hoc", run_id, cfg, spec_list, out, timings, {"window": window, "groups": groups})
    macro = abl.groupby(["h", "variant"])[["change_vs_full_pct", "impr_vs_strongest_pct"]].mean().reset_index()
    logger.info("ablation done:\n%s", macro.round(2).to_string(index=False))


def main() -> None:
    """Command-line entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["tune", "final", "analyse", "ablation"])
    ap.add_argument("run_id", nargs="?")
    ap.add_argument("--variant", choices=["error"], default=None)
    ap.add_argument("--prefix", default="")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    cfg = load_config()
    if args.command == "tune":
        cmd_tune(cfg, args.workers)
    elif args.command == "final":
        cmd_final(cfg, args.workers, args.variant)
    elif args.command == "ablation":
        cmd_ablation(cfg, args.workers)
    else:
        cmd_analyse(cfg, args.run_id, args.prefix)


if __name__ == "__main__":
    main()
