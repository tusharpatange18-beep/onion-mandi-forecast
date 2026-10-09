"""Run Phase 3.

Usage (from the project root, with the project venv):
    python -m mandi.run_phase3 tune                 # 2017-2019 tuning folds -> configs/selected_models.yaml
    python -m mandi.run_phase3 final                # 2020-2024 evaluation with the frozen selection
    python -m mandi.run_phase3 final --variant error   # sensitivity (a): 'likely error' flagged points set to missing
    python -m mandi.run_phase3 analyse <run_id> [--prefix sens_error_]

The final run refuses to start unless configs/selected_models.yaml exists and is committed, so nothing can be
changed after the 2020-2024 period has been looked at without leaving a trace in git.
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
from mandi.run_phase1 import df_to_md

logger = logging.getLogger(__name__)


def _inputs(cfg: dict[str, Any], variant: str | None = None):
    """Panel, arrays, closure weekdays and origin tables; `variant='error'` sets 'likely error' flagged points to missing."""
    panel = pd.read_parquet(ROOT / cfg["paths"]["panel"])
    extra: dict[str, Any] = {}
    if variant == "error":
        rev = pd.read_csv(ROOT / cfg["paths"]["interim"] / "outlier_review.csv", parse_dates=["date"])
        bad = rev[rev["guess"] == "likely error"]
        short = panel["market_key"].str.split("_").str[-1]
        n_changed = 0
        for r in bad.itertuples():
            sel = (short == r.market) & (panel["date"] == r.date) & panel["modal_price"].notna()
            n_changed += int(sel.sum())
            panel.loc[sel, "modal_price"] = np.nan
        extra = {"likely_error_points_in_review": len(bad), "newly_set_to_missing": n_changed}
        logger.info("variant error: %d likely-error points in review table, %d newly set to missing", len(bad), n_changed)
    arrays = F.build_arrays(panel, cfg["data"]["start_date"], cfg["data"]["end_date"], cfg["markets"])
    closure = F.closure_weekdays(pd.read_parquet(ROOT / cfg["paths"]["panel"]), cfg["eda"]["train_end"])
    tables = B.build_tables(arrays, closure, cfg)
    return panel, arrays, tables, extra


def _selected_path(cfg: dict[str, Any]) -> Path:
    return ROOT / cfg["forecast"]["selected_models_file"]


def cmd_tune(cfg: dict[str, Any], workers: int) -> None:
    """Fit every candidate on the 2017-2019 folds, pick one spec per family and horizon, and freeze the choice."""
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
    print(scores[scores["family"].isin(E.FAMILIES)].sort_values(["h", "family", "score_log_MAE"]).round(4).to_string(index=False))
    print(scores[scores["family"] == "base_naive"].round(4).to_string(index=False))


def _load_selection(cfg: dict[str, Any]) -> dict[str, Any]:
    p = _selected_path(cfg)
    if not p.exists():
        raise SystemExit("configs/selected_models.yaml not found: run `tune` first")
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(p.relative_to(ROOT))], cwd=ROOT, capture_output=True, text=True).returncode == 0
    dirty = bool(subprocess.run(["git", "status", "--porcelain", str(p.relative_to(ROOT))], cwd=ROOT, capture_output=True, text=True).stdout.strip())
    if not tracked or dirty:
        raise SystemExit("configs/selected_models.yaml must be committed (and unchanged) before the final run")
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def cmd_final(cfg: dict[str, Any], workers: int, variant: str | None) -> str:
    """Run the frozen selection over 2020-2024 and analyse it."""
    sel = _load_selection(cfg)
    chosen = {int(h): d for h, d in sel["chosen"].items()}
    all_specs = {s.col: s for s in B.specs_from_config(cfg)}
    cols = sorted({c for d in chosen.values() for c in d.values()})
    specs = [all_specs[c] for c in cols]
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
    """Compute every table from a saved run (no refitting) and write results/*.csv and docs/model_numbers.md."""
    sel = yaml.safe_load(_selected_path(cfg).read_text(encoding="utf-8"))
    chosen = {int(h): d for h, d in sel["chosen"].items()}
    best_val = {(r["market"], int(r["h"])): r["model"] for r in sel["best_model_by_validation"]}
    run_dir = ROOT / cfg["forecast"]["runs_dir"] / run_id
    tab = pd.read_parquet(run_dir / "forecasts.parquet")
    timings = yaml.safe_load((run_dir / "run.json").read_text(encoding="utf-8"))["timings"]
    chosen_cols = sorted({c for d in chosen.values() for c in d.values()})
    kept, rep = B.fairness_filter(tab, B.BASELINES + chosen_cols)
    kept = E.attach_model_columns(kept, chosen)
    # the in-sample naive scale is fixed from 2014-2019 data
    _, _, tables, _ = _inputs(cfg, "error" if prefix == "sens_error_" else None)
    scales = B.mase_scales(tables, cfg["eda"]["train_end"])
    met = E.final_metrics(kept, scales)
    spec_of = {(h, fam): col for h, d in chosen.items() for fam, col in d.items()}
    met["spec"] = [spec_of.get((h, m), "") for h, m in zip(met["h"], met["model"])]
    summ = E.summary_table(kept, met, best_val)
    macro = E.macro_average(summ)
    dm = E.dm_table(kept, summ)
    dm_all = E.dm_all_models(kept, summ)
    periods = cfg["forecast"]["event_periods"]
    per = E.assign_period(kept["target_date"], periods)
    grp = per.map({p["name"]: p["group"] for p in periods}).fillna("normal")
    seg = pd.concat([E.segment_table(kept, summ, best_val, per, "event_period"),
                     E.segment_table(kept, summ, best_val, grp, "event_group"),
                     E.segment_table(kept, summ, best_val, E.gap_bucket(kept["gap_prev"]), "gap_before_origin")], ignore_index=True)
    stale = E.stale_sensitivity(kept, scales, best_val)
    runtime = pd.DataFrame([{"model": k, **v} for k, v in timings["per_model"].items()])
    res = ROOT / "results"
    res.mkdir(exist_ok=True)
    for name, df in [("backtest_metrics", met), ("summary_table", summ), ("summary_macro", macro), ("dm_tests", dm), ("dm_all_models", dm_all),
                     ("error_analysis", seg), ("sensitivity_stale", stale), ("runtime", runtime), ("dropped_triples", rep)]:
        df.to_csv(res / f"{prefix}{name}.csv", index=False)
    kept.to_parquet(res / f"{prefix}fair_forecasts.parquet", index=False)
    logger.info("analysis written: %d fair triples of %d (%d dropped)", len(kept), len(tab), len(tab) - len(kept))
    print(summ.round(3).to_string(index=False))
    print(macro.round(2).to_string(index=False))


def main() -> None:
    """Command-line entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["tune", "final", "analyse"])
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
    else:
        cmd_analyse(cfg, args.run_id, args.prefix)


if __name__ == "__main__":
    main()
