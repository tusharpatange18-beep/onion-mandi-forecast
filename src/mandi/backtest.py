"""Phase 3: rolling-origin, expanding-window backtest.

* Models are refit on the first day of every month (the retrain interval, see config) using only data before that
  day; the refit applies to origins in that month. For LightGBM, a training row is used only if its *target date* is
  before the refit date, so no unseen target leaks into training.
* Every model is scored on the same (market, origin, horizon) triples; a triple any model cannot forecast is dropped
  for all and counted.
"""
from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mandi import features as F
from mandi import models as M
from mandi.config import ROOT

logger = logging.getLogger(__name__)

BASELINES = ["base_naive", "base_mean3", "base_seasonal"]
MIN_TRAIN_OBS = 150  # observed days needed before a statistical model is fitted


@dataclass(frozen=True)
class Spec:
    """One model configuration, e.g. family 'sarima', name '111_s7'."""

    family: str
    name: str
    params: tuple  # sorted (key, value) pairs so the spec is hashable

    @property
    def col(self) -> str:
        return f"{self.family}|{self.name}"

    @property
    def p(self) -> dict[str, Any]:
        return dict(self.params)


def specs_from_config(cfg: dict[str, Any]) -> list[Spec]:
    """All candidate model specs from forecast.models (LightGBM candidates are the grid product)."""
    out = []
    mc = cfg["forecast"]["models"]
    for fam in ("ets", "sarima", "prophet"):
        for s in mc[fam]:
            p = {k: (tuple(v) if isinstance(v, list) else v) for k, v in s.items() if k != "name"}
            out.append(Spec(fam, s["name"], tuple(sorted(p.items()))))
    g = mc["lgbm"]["grid"]
    for obj in g["objective"]:
        for nl in g["num_leaves"]:
            for mcs in g["min_child_samples"]:
                for ne in g["n_estimators"]:
                    p = {"objective": obj, "num_leaves": nl, "min_child_samples": mcs, "n_estimators": ne, "learning_rate": mc["lgbm"]["learning_rate"]}
                    out.append(Spec("lgbm", f"{obj}_nl{nl}_mc{mcs}_n{ne}", tuple(sorted(p.items()))))
    return out


def refit_schedule(start: str, end: str) -> list[pd.Timestamp]:
    """First day of every month from `start` to `end` (the retrain dates)."""
    return list(pd.date_range(start, end, freq="MS"))


def _blocks(refits: list[pd.Timestamp], end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """(refit date, last day of the block) pairs; origins in the block use the model fitted at the refit date."""
    ends = [r - pd.Timedelta(days=1) for r in refits[1:]] + [end]
    return list(zip(refits, ends))


# --------------------------------------------------------------------------- worker tasks (top level so they can be pickled)
def _task_state_space(args: tuple) -> tuple[str, int, pd.DataFrame, float, int]:
    """Statistical state-space model for one market across all refit blocks."""
    spec, m, y, days, blocks, horizons = args
    t0 = time.time()
    kind = spec.family
    ys = pd.Series(y, index=days)
    first = ys.first_valid_index()
    rows, params, n_fits = [], None, 0
    for r, b_end in blocks:
        if first is None or r <= first:
            continue
        ytr = ys.loc[first: r - pd.Timedelta(days=1)]
        if ytr.notna().sum() < MIN_TRAIN_OBS:
            continue
        try:
            params = M.fit_state_space(kind, spec.p, ytr, start_params=params)
            n_fits += 1
            yrun = ys.loc[first:b_end]
            fc = M.state_space_forecasts(kind, spec.p, yrun, params, horizons)
        except Exception as exc:  # a failed fit leaves the forecasts missing; the triple is then dropped for all models
            logger.warning("%s market %d refit %s failed: %s", spec.col, m, r.date(), exc)
            params = None
            continue
        pos = np.flatnonzero((yrun.index >= r) & ~np.isnan(yrun.to_numpy()))
        for h in horizons:
            idx = yrun.index[pos]
            rows.append(pd.DataFrame({"market_idx": m, "origin": idx, "h": h, "f": fc[h][pos]}))
    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["market_idx", "origin", "h", "f"])
    return spec.col, m, df, time.time() - t0, n_fits


def _task_prophet(args: tuple) -> tuple[str, int, pd.DataFrame, float, int]:
    """Prophet for one market across all refit blocks."""
    spec, m, y, days, blocks, horizons = args
    t0 = time.time()
    ys = pd.Series(y, index=days)
    first = ys.first_valid_index()
    rows, n_fits = [], 0
    for r, b_end in blocks:
        if first is None or r <= first:
            continue
        ytr = ys.loc[first: r - pd.Timedelta(days=1)]
        if ytr.notna().sum() < MIN_TRAIN_OBS:
            continue
        origins = ys.loc[r:b_end].dropna().index
        if len(origins) == 0:
            continue
        targets = sorted({o + pd.Timedelta(days=h) for o in origins for h in horizons})
        try:
            pred = pd.Series(M.prophet_forecast(ytr, pd.DatetimeIndex(targets), spec.p), index=targets)
            n_fits += 1
        except Exception as exc:
            logger.warning("%s market %d refit %s failed: %s", spec.col, m, r.date(), exc)
            continue
        for h in horizons:
            rows.append(pd.DataFrame({"market_idx": m, "origin": origins, "h": h, "f": pred.reindex(origins + pd.Timedelta(days=h)).to_numpy()}))
    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["market_idx", "origin", "h", "f"])
    return spec.col, m, df, time.time() - t0, n_fits


def _task_lgbm(args: tuple) -> tuple[str, int, pd.DataFrame, float, int]:
    """Pooled LightGBM for one horizon across all refit blocks."""
    spec, h, table, blocks, seed = args
    t0 = time.time()
    rows, n_fits = [], 0
    labelled = table[table["y_log"].notna()]
    for r, b_end in blocks:
        train = labelled[labelled["target_date"] < r]
        test = table[(table["origin"] >= r) & (table["origin"] <= b_end) & table["y_log"].notna()]
        if len(train) < 500 or len(test) == 0:
            continue
        pred = M.lgbm_fit_predict(train, test, F.FEATURE_COLUMNS, spec.p, seed)
        n_fits += 1
        rows.append(pd.DataFrame({"market_idx": test["market_idx"].to_numpy(), "origin": test["origin"].to_numpy(), "h": h,
                                  "f": test["p0"].to_numpy() + pred}))
    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["market_idx", "origin", "h", "f"])
    return spec.col, h, df, time.time() - t0, n_fits


# --------------------------------------------------------------------------- orchestration
def build_tables(arrays: F.Arrays, closure: dict[str, int | None], cfg: dict[str, Any]) -> dict[int, pd.DataFrame]:
    """Origin tables (features, baselines, targets) for every horizon."""
    fp = dict(cfg["forecast"]["features"])
    fp.update(cfg["forecast"]["baselines"])
    return {h: F.build_origin_table(arrays, h, closure, fp) for h in cfg["backtest"]["horizons_days"]}


def run_backtest(arrays: F.Arrays, tables: dict[int, pd.DataFrame], specs: list[Spec], window: tuple[str, str],
                 cfg: dict[str, Any], workers: int = 12) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run all `specs` over origins in `window` and return (triple table with one log-forecast column per model, timings).

    The triple table has one row per (market, origin, horizon) whose target quote exists, before the fairness filter.
    """
    start, end = pd.Timestamp(window[0]), pd.Timestamp(window[1])
    blocks = _blocks(refit_schedule(window[0], window[1]), end)
    horizons = list(tables)
    seed = cfg["seed"]
    jobs: list[tuple[Any, tuple]] = []
    for s in specs:
        if s.family in ("ets", "sarima"):
            jobs += [(_task_state_space, (s, m, arrays.logp[m], arrays.days, blocks, horizons)) for m in range(len(arrays.markets))]
        elif s.family == "prophet":
            jobs += [(_task_prophet, (s, m, arrays.logp[m], arrays.days, blocks, horizons)) for m in range(len(arrays.markets))]
        else:
            jobs += [(_task_lgbm, (s, h, tables[h], blocks, seed)) for h in horizons]
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(fn, a) for fn, a in jobs]
        for f in futs:
            results.append(f.result())
    wall = time.time() - t0
    base = pd.concat([t[(t["origin"] >= start) & (t["origin"] <= end) & t["y_log"].notna()] for t in tables.values()], ignore_index=True)
    out = base[["market", "market_idx", "origin", "h", "target_date", "y_log", "gap_prev", "stale_origin", "stale_target",
                "base_naive", "base_mean3", "base_seasonal"]].copy()
    timings: dict[str, Any] = {"wall_seconds": round(wall, 1), "per_model": {}}
    by_col: dict[str, list[pd.DataFrame]] = {}
    for col, _, df, secs, n_fits in results:
        by_col.setdefault(col, []).append(df)
        d = timings["per_model"].setdefault(col, {"cpu_seconds": 0.0, "fits": 0})
        d["cpu_seconds"] = round(d["cpu_seconds"] + secs, 1)
        d["fits"] += n_fits
    for col, dfs in by_col.items():
        f = pd.concat(dfs, ignore_index=True).rename(columns={"f": col})
        out = out.merge(f, on=["market_idx", "origin", "h"], how="left")
    for col, d in timings["per_model"].items():
        d["cpu_seconds_per_fit"] = round(d["cpu_seconds"] / max(d["fits"], 1), 2)
    return out, timings


def fairness_filter(tab: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep only triples for which every column in `columns` has a forecast. Returns (kept, drop report).

    The report counts, per column, the triples that column could not forecast (a triple can be missing for several).
    """
    miss = tab[columns].isna()
    rep = pd.DataFrame({"column": columns, "cannot_forecast": miss.sum().to_numpy()})
    rep["total_triples"] = len(tab)
    kept = tab[~miss.any(axis=1)].copy()
    rep["dropped_for_all_models_total"] = len(tab) - len(kept)
    return kept, rep


def mase_scales(tables: dict[int, pd.DataFrame], train_end: str) -> dict[tuple[str, int], float]:
    """In-sample naive error per market and horizon: mean |quote(origin+h) - quote(origin)| over training triples."""
    from mandi.metrics import naive_scale

    out = {}
    for h, t in tables.items():
        tt = t[(t["target_date"] <= train_end) & t["y_log"].notna()]
        for m, g in tt.groupby("market"):
            out[(m, h)] = naive_scale(g["y_log"].to_numpy(), g["p0"].to_numpy())
    return out


# --------------------------------------------------------------------------- run logging
def git_state() -> dict[str, Any]:
    """Current commit and whether the working tree has uncommitted changes."""
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip())
    except Exception:
        sha, dirty = "unknown", None
    return {"commit": sha, "dirty": dirty}


def log_run(kind: str, run_id: str, cfg: dict[str, Any], specs: list[Spec], tab: pd.DataFrame, timings: dict[str, Any], extra: dict[str, Any]) -> Path:
    """Write results/runs/<run_id>/ (config snapshot, forecasts, timings) and append a row to results/run_log.csv."""
    d = ROOT / cfg["forecast"]["runs_dir"] / run_id
    d.mkdir(parents=True, exist_ok=True)
    meta = {"run_id": run_id, "kind": kind, "git": git_state(), "specs": [s.col for s in specs], "timings": timings,
            "config_sha1": hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:10], **extra}
    (d / "run.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    tab.to_parquet(d / "forecasts.parquet", index=False)
    log_path = ROOT / "results" / "run_log.csv"
    row = pd.DataFrame([{"run_id": run_id, "kind": kind, "when": pd.Timestamp.now().isoformat(timespec="seconds"), "commit": meta["git"]["commit"],
                         "dirty": meta["git"]["dirty"], "config_sha1": meta["config_sha1"], "models": len(specs), "triples": len(tab),
                         "wall_seconds": timings["wall_seconds"]}])
    row.to_csv(log_path, mode="a", header=not log_path.exists(), index=False)
    return d
