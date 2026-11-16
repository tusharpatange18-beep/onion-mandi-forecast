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
from mandi.framework import Backtester, LGBMModel, refit_blocks as _blocks, refit_schedule  # noqa: F401 (re-exported: the schedule now lives in framework)

logger = logging.getLogger(__name__)

BASELINES = ["base_naive", "base_mean3", "base_seasonal"]
MIN_TRAIN_OBS = 150  # observed days needed before a statistical model is fitted
FORECAST_COLUMNS = ["market_idx", "origin", "h", "f"]  # columns every worker task returns


@dataclass(frozen=True)
class Spec:
    """One model configuration, e.g. family 'sarima', name '111_s7'."""

    family: str
    name: str
    params: tuple  # sorted (key, value) pairs so the spec is hashable

    @property
    def col(self) -> str:
        # Column name used for this model's forecasts, e.g. "sarima|111_s7".
        return f"{self.family}|{self.name}"

    @property
    def p(self) -> dict[str, Any]:
        # The parameters as a normal dictionary.
        return dict(self.params)


def specs_from_config(cfg: dict[str, Any]) -> list[Spec]:
    """All candidate model specs from forecast.models (LightGBM candidates are the grid product)."""
    specs = []
    model_config = cfg["forecast"]["models"]

    # Statistical models: one spec per entry in the config.
    for family in ("ets", "sarima", "prophet", "prophet_anchored"):
        for entry in model_config[family]:
            params = {}
            for key, value in entry.items():
                if key == "name":
                    continue  # the name is stored separately
                # Lists become tuples so the spec can be hashed.
                if isinstance(value, list):
                    value = tuple(value)
                params[key] = value
            sorted_params = tuple(sorted(params.items()))
            specs.append(Spec(family, entry["name"], sorted_params))

    # LightGBM: every combination of the grid values.
    grid = model_config["lgbm"]["grid"]
    for objective in grid["objective"]:
        for num_leaves in grid["num_leaves"]:
            for min_child_samples in grid["min_child_samples"]:
                for n_estimators in grid["n_estimators"]:
                    params = {
                        "objective": objective,
                        "num_leaves": num_leaves,
                        "min_child_samples": min_child_samples,
                        "n_estimators": n_estimators,
                        "learning_rate": model_config["lgbm"]["learning_rate"],
                    }
                    name = f"{objective}_nl{num_leaves}_mc{min_child_samples}_n{n_estimators}"
                    specs.append(Spec("lgbm", name, tuple(sorted(params.items()))))
    return specs


def _join_parts(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """Stack the per-block forecast tables, or return an empty table if there are none."""
    if parts:
        return pd.concat(parts, ignore_index=True)
    return pd.DataFrame(columns=FORECAST_COLUMNS)


# --------------------------------------------------------------------------- worker tasks (top level so they can be pickled)
def _task_state_space(args: tuple) -> tuple[str, int, pd.DataFrame, float, int, int]:
    """Statistical state-space model for one market across all refit blocks."""
    spec, market, log_prices, days, blocks, horizons = args
    start_time = time.time()
    kind = spec.family  # "ets" or "sarima"

    # Log prices of this market on the calendar, and the first day with a price.
    prices = pd.Series(log_prices, index=days)
    first_day = prices.first_valid_index()

    parts = []
    params = None  # last fitted parameters, reused as start values for the next fit
    n_fits = 0
    n_not_converged = 0

    for refit_date, block_end in blocks:
        # Nothing to fit before the market's first quote.
        if first_day is None or refit_date <= first_day:
            continue

        # Training data: everything before the refit date.
        train_prices = prices.loc[first_day: refit_date - pd.Timedelta(days=1)]
        if train_prices.notna().sum() < MIN_TRAIN_OBS:
            continue  # not enough quotes yet

        try:
            # Estimate the parameters on the training data.
            params, converged = M.fit_state_space(kind, spec.p, train_prices, start_params=params, return_info=True)
            n_fits = n_fits + 1
            if not converged:
                n_not_converged = n_not_converged + 1

            # Run the model up to the end of the block and get forecasts at every day.
            run_prices = prices.loc[first_day:block_end]
            forecasts = M.state_space_forecasts(kind, spec.p, run_prices, params, horizons)
        except Exception as exc:  # a failed fit leaves the forecasts missing; the triple is then dropped for all models
            logger.warning("%s market %d refit %s failed: %s", spec.col, market, refit_date.date(), exc)
            params = None
            continue

        # Keep only origins inside this block that have a quote.
        in_block = run_prices.index >= refit_date
        has_quote = ~np.isnan(run_prices.to_numpy())
        positions = np.flatnonzero(in_block & has_quote)
        origin_dates = run_prices.index[positions]

        for h in horizons:
            parts.append(pd.DataFrame({"market_idx": market, "origin": origin_dates, "h": h, "f": forecasts[h][positions]}))

    seconds = time.time() - start_time
    return spec.col, market, _join_parts(parts), seconds, n_fits, n_not_converged


def anchored_forecast(last_log_price: np.ndarray, pred_target: np.ndarray, pred_origin: np.ndarray) -> np.ndarray:
    """prophet_anchored: the last observed log price plus the change Prophet predicts between the origin and the target.

    Plain Prophet ignores recent prices; this keeps only its predicted *movement* (trend plus yearly shape) and anchors
    it to the latest quote, the same way the naive baseline does.
    """
    predicted_move = pred_target - pred_origin
    return last_log_price + predicted_move


def _task_prophet(args: tuple) -> tuple[str, int, pd.DataFrame, float, int, int]:
    """Prophet (plain, or anchored to the last quote when spec.family == 'prophet_anchored') for one market across all refit blocks."""
    spec, market, log_prices, days, blocks, horizons = args
    anchored = spec.family == "prophet_anchored"
    start_time = time.time()

    prices = pd.Series(log_prices, index=days)
    first_day = prices.first_valid_index()

    parts = []
    n_fits = 0
    for refit_date, block_end in blocks:
        if first_day is None or refit_date <= first_day:
            continue

        # Training data: everything before the refit date.
        train_prices = prices.loc[first_day: refit_date - pd.Timedelta(days=1)]
        if train_prices.notna().sum() < MIN_TRAIN_OBS:
            continue

        # Origins: days in this block that have a quote.
        origins = prices.loc[refit_date:block_end].dropna().index
        if len(origins) == 0:
            continue

        # All dates Prophet must predict: every origin + h. The anchored version also needs the origins themselves.
        dates_needed = set()
        for origin in origins:
            for h in horizons:
                dates_needed.add(origin + pd.Timedelta(days=h))
        if anchored:
            for origin in origins:
                dates_needed.add(origin)
        target_dates = sorted(dates_needed)

        try:
            predicted = M.prophet_forecast(train_prices, pd.DatetimeIndex(target_dates), spec.p)
            predictions = pd.Series(predicted, index=target_dates)
            n_fits = n_fits + 1
        except Exception as exc:
            logger.warning("%s market %d refit %s failed: %s", spec.col, market, refit_date.date(), exc)
            continue

        for h in horizons:
            # Prophet's prediction for each origin's target date.
            forecast = predictions.reindex(origins + pd.Timedelta(days=h)).to_numpy()
            if anchored:
                # Keep only the predicted movement and add it to today's price.
                price_now = prices.loc[origins].to_numpy()
                predicted_at_origin = predictions.reindex(origins).to_numpy()
                forecast = anchored_forecast(price_now, forecast, predicted_at_origin)
            parts.append(pd.DataFrame({"market_idx": market, "origin": origins, "h": h, "f": forecast}))

    seconds = time.time() - start_time
    return spec.col, market, _join_parts(parts), seconds, n_fits, 0


def _task_lgbm(args: tuple) -> tuple[str, int, pd.DataFrame, float, int, int]:
    """Pooled LightGBM for one horizon across all refit blocks (the rolling-origin loop itself lives in framework.Backtester)."""
    spec, h, table, blocks, seed = args
    start_time = time.time()

    # Feature names removed for the post-hoc ablation (empty for every normal run).
    dropped = set(spec.p.get("drop", ()))
    features = []
    for name in F.FEATURE_COLUMNS:
        if name not in dropped:
            features.append(name)

    # Run the rolling backtest over the given blocks.
    window = (str(blocks[0][0].date()), str(blocks[-1][1].date()))
    backtester = Backtester(table, window, h=h, blocks=blocks)
    model = LGBMModel(spec.p, features, seed, name=spec.col)
    result = backtester.run(model)

    # Same column layout as the other tasks.
    forecasts = result.frame.rename(columns={"f_log": "f"})
    forecasts = forecasts[FORECAST_COLUMNS]

    seconds = time.time() - start_time
    return spec.col, h, forecasts, seconds, result.n_fits, 0


# --------------------------------------------------------------------------- orchestration
def build_tables(arrays: F.Arrays, closure: dict[str, int | None], cfg: dict[str, Any]) -> dict[int, pd.DataFrame]:
    """Origin tables (features, baselines, targets) for every horizon."""
    # Feature and baseline settings in one dictionary.
    feature_params = dict(cfg["forecast"]["features"])
    feature_params.update(cfg["forecast"]["baselines"])

    tables = {}
    for h in cfg["backtest"]["horizons_days"]:
        tables[h] = F.build_origin_table(arrays, h, closure, feature_params)
    return tables


def run_backtest(arrays: F.Arrays, tables: dict[int, pd.DataFrame], specs: list[Spec], window: tuple[str, str],
                 cfg: dict[str, Any], workers: int = 12) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run all `specs` over origins in `window` and return (triple table with one log-forecast column per model, timings).

    The triple table has one row per (market, origin, horizon) whose target quote exists, before the fairness filter.
    """
    start = pd.Timestamp(window[0])
    end = pd.Timestamp(window[1])
    blocks = _blocks(refit_schedule(window[0], window[1]), end)
    horizons = list(tables)
    seed = cfg["seed"]
    n_markets = len(arrays.markets)

    # Make a list of jobs: (function, arguments).
    jobs = []
    for spec in specs:
        if spec.family in ("ets", "sarima"):
            # One job per market.
            for market in range(n_markets):
                jobs.append((_task_state_space, (spec, market, arrays.logp[market], arrays.days, blocks, horizons)))
        elif spec.family in ("prophet", "prophet_anchored"):
            # One job per market.
            for market in range(n_markets):
                jobs.append((_task_prophet, (spec, market, arrays.logp[market], arrays.days, blocks, horizons)))
        else:
            # LightGBM: one job per horizon (all markets together).
            for h in horizons:
                jobs.append((_task_lgbm, (spec, h, tables[h], blocks, seed)))

    # Run the jobs in parallel processes and collect the results in job order.
    start_time = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = []
        for function, arguments in jobs:
            futures.append(executor.submit(function, arguments))
        for future in futures:
            results.append(future.result())
    wall_seconds = time.time() - start_time

    # The triples to score: origins inside the window whose target exists, for every horizon.
    scored_rows = []
    for table in tables.values():
        in_window = (table["origin"] >= start) & (table["origin"] <= end)
        has_target = table["y_log"].notna()
        scored_rows.append(table[in_window & has_target])
    base = pd.concat(scored_rows, ignore_index=True)
    out = base[["market", "market_idx", "origin", "h", "target_date", "y_log", "gap_prev", "stale_origin", "stale_target",
                "base_naive", "base_mean3", "base_seasonal"]].copy()

    # Group the forecast tables by model column, and add up the timings.
    timings: dict[str, Any] = {"wall_seconds": round(wall_seconds, 1), "per_model": {}}
    tables_by_column: dict[str, list[pd.DataFrame]] = {}
    for column, _, forecast_table, seconds, n_fits, n_not_converged in results:
        if column not in tables_by_column:
            tables_by_column[column] = []
        tables_by_column[column].append(forecast_table)

        if column not in timings["per_model"]:
            timings["per_model"][column] = {"cpu_seconds": 0.0, "fits": 0, "not_converged": 0}
        model_timing = timings["per_model"][column]
        model_timing["cpu_seconds"] = round(model_timing["cpu_seconds"] + seconds, 1)
        model_timing["fits"] = model_timing["fits"] + n_fits
        model_timing["not_converged"] = model_timing["not_converged"] + n_not_converged

    # Add one forecast column per model to the triple table.
    for column, forecast_tables in tables_by_column.items():
        forecasts = pd.concat(forecast_tables, ignore_index=True)
        forecasts = forecasts.rename(columns={"f": column})
        out = out.merge(forecasts, on=["market_idx", "origin", "h"], how="left")

    # Average CPU seconds per fit.
    for column, model_timing in timings["per_model"].items():
        model_timing["cpu_seconds_per_fit"] = round(model_timing["cpu_seconds"] / max(model_timing["fits"], 1), 2)

    check_forecast_coverage(out, list(tables_by_column))
    return out, timings


def check_forecast_coverage(tab: pd.DataFrame, model_cols: list[str], max_missing: float = 0.02, seasonal_max_missing: float = 0.5) -> None:
    """Abort if any model silently failed to forecast (part of) a market.

    The fairness filter drops a triple for every model when one model cannot forecast it, so one broken candidate can
    erase a whole market without any error. This guard turns that into a hard failure: no model column may be missing
    more than `max_missing` of the triples in any (market, horizon); the 365-day baseline, which is legitimately missing
    when no quote exists a year earlier, is allowed `seasonal_max_missing`.
    """
    problems = []
    for group_key, group in tab.groupby(["market", "h"]):
        market = group_key[0]
        h = group_key[1]

        # Share of missing forecasts for each model.
        for column in model_cols:
            missing_share = group[column].isna().mean()
            if missing_share > max_missing:
                problems.append(f"{column} missing {missing_share:.1%} of {len(group)} triples for {market} h={h}")

        # The seasonal baseline is allowed more missing values.
        missing_share = group["base_seasonal"].isna().mean()
        if missing_share > seasonal_max_missing:
            problems.append(f"base_seasonal missing {missing_share:.1%} for {market} h={h}")

    if problems:
        # Show at most 40 problems, one per line.
        message = "forecast coverage guard failed:\n  " + "\n  ".join(problems[:40])
        raise RuntimeError(message)


def fairness_filter(tab: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep only triples for which every column in `columns` has a forecast. Returns (kept, drop report).

    The report counts, per column, the triples that column could not forecast (a triple can be missing for several).
    """
    # True where a model has no forecast.
    is_missing = tab[columns].isna()

    # Report: how many triples each model could not forecast.
    report = pd.DataFrame({"column": columns, "cannot_forecast": is_missing.sum().to_numpy()})
    report["total_triples"] = len(tab)

    # Keep a row only if no model is missing.
    any_missing = is_missing.any(axis=1)
    kept = tab[~any_missing].copy()
    report["dropped_for_all_models_total"] = len(tab) - len(kept)
    return kept, report


def mase_scales(tables: dict[int, pd.DataFrame], train_end: str) -> dict[tuple[str, int], float]:
    """In-sample naive error per market and horizon: mean |quote(origin+h) - quote(origin)| over training triples."""
    from mandi.metrics import naive_scale

    scales = {}
    for h, table in tables.items():
        # Training triples: target inside the training period and observed.
        in_training = (table["target_date"] <= train_end) & table["y_log"].notna()
        training_rows = table[in_training]
        for market, market_rows in training_rows.groupby("market"):
            scales[(market, h)] = naive_scale(market_rows["y_log"].to_numpy(), market_rows["p0"].to_numpy())
    return scales


# --------------------------------------------------------------------------- run logging
CODE_PATHS = ["src", "configs", "tests", "project.md"]  # "dirty" is judged on these only; results/ and docs/ are outputs


def _git(*args: str) -> str:
    """Run a git command in the project folder and return its output."""
    command = ["git"]
    command.extend(args)
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
    return completed.stdout.strip()


def git_state(selection_file: str = "configs/selected_models.yaml") -> dict[str, Any]:
    """Current commit, whether source/config/test files have uncommitted changes, and the commit that last touched the selection file.

    results/ and docs/ are ignored for `dirty` because running the pipeline rewrites them.
    """
    try:
        commit = _git("rev-parse", "--short", "HEAD")  # current commit id
        changed_files = _git("status", "--porcelain", "--", *CODE_PATHS)  # uncommitted changes in code
        dirty = bool(changed_files)
        selection_commit = _git("log", "-1", "--format=%h", "--", selection_file)  # last commit of the selection file
        if not selection_commit:
            selection_commit = None
    except Exception:
        # Not a git repository, or git is not installed.
        commit = "unknown"
        dirty = None
        selection_commit = None
    return {"commit": commit, "dirty": dirty, "selection_commit": selection_commit}


def log_run(kind: str, run_id: str, cfg: dict[str, Any], specs: list[Spec], tab: pd.DataFrame, timings: dict[str, Any], extra: dict[str, Any]) -> Path:
    """Write results/runs/<run_id>/ (config snapshot, forecasts, timings) and append a row to results/run_log.csv."""
    run_folder = ROOT / cfg["forecast"]["runs_dir"] / run_id
    run_folder.mkdir(parents=True, exist_ok=True)

    # A short fingerprint of the config, so we can tell later which settings produced this run.
    config_text = json.dumps(cfg, sort_keys=True, default=str)
    config_sha1 = hashlib.sha1(config_text.encode()).hexdigest()[:10]

    spec_names = []
    for spec in specs:
        spec_names.append(spec.col)

    # Everything we want to remember about this run.
    meta = {"run_id": run_id, "kind": kind, "git": git_state(), "specs": spec_names, "timings": timings, "config_sha1": config_sha1}
    for key, value in extra.items():
        meta[key] = value

    # Save the run description and the forecasts.
    (run_folder / "run.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    tab.to_parquet(run_folder / "forecasts.parquet", index=False)

    # Add one line to the run log (write the header only if the file is new).
    log_path = ROOT / "results" / "run_log.csv"
    log_row = {
        "run_id": run_id,
        "kind": kind,
        "when": pd.Timestamp.now().isoformat(timespec="seconds"),
        "commit": meta["git"]["commit"],
        "dirty": meta["git"]["dirty"],
        "selection_commit": meta["git"]["selection_commit"],
        "config_sha1": meta["config_sha1"],
        "models": len(specs),
        "triples": len(tab),
        "wall_seconds": timings["wall_seconds"],
    }
    pd.DataFrame([log_row]).to_csv(log_path, mode="a", header=not log_path.exists(), index=False)
    return run_folder
