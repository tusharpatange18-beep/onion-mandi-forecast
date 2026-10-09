"""Phase 5: generate the serving forecasts.

CONSTRAINT: the dataset ends 2024-12-31, so "latest" forecasts are as of the last data date, not live.

Usage:  python -m mandi.forecast [--as-of YYYY-MM-DD]

* Trains the frozen LightGBM specs from configs/selected_models.yaml (7 and 30 days) on all rows whose TARGET date is strictly
  before the as-of date (default: the day after the last observed quote).
* Forecasts from each market's latest observed origin, and from its last N origins so the dashboard can draw recent history.
  Forecasts for origins whose target quote is already known were fitted on that very row (`in_sample` = True); they show the
  model's recent track, not out-of-sample accuracy. (A target date that has passed without a quote has no actual price and is not in sample.)
* Intervals: the Phase 4 rolling split-conformal procedure (80% and 95%) calibrated on the resolved out-of-sample errors of the
  previous 365 days (the 2020-2024 backtest errors, seeded with the 2017-2019 tuning-fold errors); no interval if fewer than
  `min_errors` errors are available.
* Writes results/forecasts.parquet (not in git) and one row to results/run_log.csv.
"""
from __future__ import annotations

import argparse
import logging
import time
from typing import Any

import numpy as np
import pandas as pd
import yaml

from mandi import backtest as B
from mandi import features as F
from mandi import intervals as IV
from mandi import models as M
from mandi.config import ROOT, load_config

logger = logging.getLogger(__name__)
OUTPUT_COLUMNS = ["market", "horizon", "origin_date", "target_date", "last_price", "point", "lo80", "hi80", "lo95", "hi95", "naive",
                  "n_calibration", "model_spec", "trained_through", "generated_at", "git_commit",
                  "as_of", "data_through", "is_latest", "in_sample", "actual_price", "gap_before_origin_days"]


def training_rows(table: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Rows with an observed target whose TARGET date is strictly before `as_of` (so every label was known before as_of)."""
    t = table[table["y_log"].notna()]
    return t[t["target_date"] < as_of]


def load_inputs(cfg: dict[str, Any]) -> tuple[F.Arrays, dict[int, pd.DataFrame], pd.Timestamp]:
    """Arrays, origin tables for the served horizons and the date of the last observed quote."""
    panel = pd.read_parquet(ROOT / cfg["paths"]["panel"])
    arrays = F.build_arrays(panel, cfg["data"]["start_date"], cfg["data"]["end_date"], cfg["markets"])
    closure = F.closure_weekdays(panel, cfg["eda"]["train_end"])
    tables = B.build_tables(arrays, closure, cfg)
    observed = ~np.isnan(arrays.logp).all(axis=0)
    return arrays, tables, pd.Timestamp(arrays.days[observed][-1])


def frozen_specs(cfg: dict[str, Any]) -> dict[int, B.Spec]:
    """The frozen LightGBM spec for each horizon, from the committed selection file."""
    sel = yaml.safe_load((ROOT / cfg["forecast"]["selected_models_file"]).read_text(encoding="utf-8"))
    by_col = {s.col: s for s in B.specs_from_config(cfg)}
    return {int(h): by_col[d["lgbm"]] for h, d in sel["chosen"].items() if int(h) in cfg["serving"]["horizons"]}


def predict_origins(cfg: dict[str, Any], tables: dict[int, pd.DataFrame], as_of: pd.Timestamp, select: dict[int, pd.Series]) -> tuple[pd.DataFrame, dict[int, pd.Timestamp]]:
    """Train each horizon's frozen model on rows with target < as_of and predict the origins chosen by `select[h]` (a boolean mask).

    Returns a frame with the log point forecast (`f_log`) and the table columns, and the training cut-off per horizon.
    """
    specs = frozen_specs(cfg)
    parts, trained_through = [], {}
    for h, spec in specs.items():
        tr = training_rows(tables[h], as_of)
        test = tables[h][select[h]]
        pred = M.lgbm_fit_predict(tr, test, F.FEATURE_COLUMNS, spec.p, cfg["seed"])
        out = test.copy()
        out["f_log"] = test["p0"].to_numpy() + pred
        out["h"] = h
        out["model_spec"] = spec.col
        parts.append(out)
        trained_through[h] = pd.Timestamp(tr["target_date"].max())
    return pd.concat(parts, ignore_index=True), trained_through


def calibration_pool(cfg: dict[str, Any]) -> pd.DataFrame:
    """Resolved out-of-sample errors of the headline LightGBM: the 2020-2024 backtest plus the 2017-2019 tuning-fold seed."""
    fr = IV.build_frame(cfg)
    fr = fr.assign(score=(fr["y_log"] - fr["f_lgbm"]).abs())
    return fr.sort_values(["market", "h", "target_date"]).reset_index(drop=True)


def add_intervals(preds: pd.DataFrame, pool: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """80% and 95% conformal bounds and the calibration-set size for every forecast row (NaN where fewer than min_errors)."""
    ic = cfg["intervals"]
    levels = ic["levels"]
    out = preds.copy()
    for lv in levels:
        out[f"q{int(lv * 100)}"] = np.nan
    out["n_calibration"] = 0
    for (m, h), g in out.groupby(["market", "h"]):
        c = pool[(pool["market"] == m) & (pool["h"] == h)]
        q, n = IV.quantiles_for(g["origin"].to_numpy(), c["target_date"].to_numpy(), c["score"].to_numpy(), levels, ic["calibration_window_days"], ic["min_errors"])
        for lv in levels:
            out.loc[g.index, f"q{int(lv * 100)}"] = q[lv]
        out.loc[g.index, "n_calibration"] = n
    return out


def build_forecasts(cfg: dict[str, Any], as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """The serving table for one as-of date (see the module docstring)."""
    arrays, tables, last_quote = load_inputs(cfg)
    as_of = pd.Timestamp(as_of) if as_of is not None else last_quote + pd.Timedelta(days=1)
    n_hist = cfg["serving"]["history_origins"]
    select = {}
    for h, t in tables.items():
        if h not in cfg["serving"]["horizons"]:
            continue
        ok = t["origin"] < as_of                                           # an origin must lie before the as-of date
        rank = t[ok].groupby("market_idx")["origin"].rank(method="first", ascending=False)
        mask = pd.Series(False, index=t.index)
        mask[rank[rank <= n_hist].index] = True
        select[h] = mask
    preds, trained = predict_origins(cfg, tables, as_of, select)
    preds = add_intervals(preds, calibration_pool(cfg), cfg)
    git = B.git_state()
    now = pd.Timestamp.now().isoformat(timespec="seconds")
    latest = preds.groupby(["market", "h"])["origin"].transform("max")
    f = preds["f_log"]
    out = pd.DataFrame({
        "market": preds["market"], "horizon": preds["h"], "origin_date": preds["origin"], "target_date": preds["target_date"],
        "last_price": np.exp(preds["p0"]), "point": np.exp(f),
        "lo80": np.exp(f - preds["q80"]), "hi80": np.exp(f + preds["q80"]), "lo95": np.exp(f - preds["q95"]), "hi95": np.exp(f + preds["q95"]),
        "naive": np.exp(preds["p0"]), "n_calibration": preds["n_calibration"], "model_spec": preds["model_spec"],
        "trained_through": preds["h"].map(trained), "generated_at": now, "git_commit": git["commit"],
        "as_of": as_of, "data_through": last_quote, "is_latest": preds["origin"] == latest,
        "in_sample": (preds["target_date"] < as_of) & preds["y_log"].notna(),       # the row was in the training set
        "actual_price": np.where(preds["target_date"] < as_of, np.exp(preds["y_log"]), np.nan),
        "gap_before_origin_days": preds["gap_prev"],
    })
    return out.sort_values(["market", "horizon", "origin_date"]).reset_index(drop=True)[OUTPUT_COLUMNS]


def main() -> None:
    """Write results/forecasts.parquet and log the run."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Generate serving forecasts (as of the last data date, not live).")
    ap.add_argument("--as-of", default=None, help="train on rows whose target date is before this date (default: day after the last quote)")
    args = ap.parse_args()
    cfg = load_config()
    t0 = time.time()
    out = build_forecasts(cfg, pd.Timestamp(args.as_of) if args.as_of else None)
    path = ROOT / cfg["serving"]["forecasts_file"]
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    secs = round(time.time() - t0, 1)
    specs = list(frozen_specs(cfg).values())
    run_id = "forecast_" + pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    B.log_run("forecast", run_id, cfg, specs, out, {"wall_seconds": secs, "per_model": {}},
              {"as_of": str(out["as_of"].iloc[0].date()), "data_through": str(out["data_through"].iloc[0].date()), "rows": len(out)})
    lat = out[out["is_latest"]]
    logger.info("wrote %s: %d rows (%d latest) as of %s, data through %s, in %.1fs", path, len(out), len(lat), out["as_of"].iloc[0].date(), out["data_through"].iloc[0].date(), secs)


if __name__ == "__main__":
    main()
