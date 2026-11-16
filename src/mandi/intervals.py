"""Phase 4A: prediction intervals for the headline LightGBM by rolling split-conformal prediction.

Method (declared in configs/config.yaml before any result was seen):
* Errors are on the log scale, e = y_log - forecast. Intervals are [f - q, f + q] on the log scale and are exponentiated to rupees.
* For a forecast made at origin t (market m, horizon h) the calibration set is the errors of earlier forecasts of the same
  market and horizon whose TARGET date is strictly before t (so every error was known at t), within the trailing
  `calibration_window_days`, and there must be at least `min_errors` of them; otherwise no interval is issued.
* Primary method: q is the k-th smallest absolute error with k = ceil((n + 1) * level) (the finite-sample correction).
* Sensitivity method: errors are first divided by a volatility scale known at the forecast's origin (mean absolute log
  change over the last `vol_window_quotes` observed quotes), the quantile is taken on the scaled errors, and q is rescaled
  by the volatility scale of the current origin.
* Comparator: the same procedure on naive's errors.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # draw figures to files, no window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from mandi import evaluate as E  # noqa: E402
from mandi import features as F  # noqa: E402
from mandi.config import ROOT, load_config  # noqa: E402
from mandi.posthoc import block_indices  # noqa: E402

logger = logging.getLogger(__name__)

# Interval methods: name -> (forecast column, divide errors by volatility?).
TAGS = {"lgbm_abs": ("f_lgbm", False), "lgbm_vol": ("f_lgbm", True), "naive_abs": ("f_naive", False)}
NS_DAY = 86_400 * 10**9  # nanoseconds in one day


def _column_names(tag: str, level: float) -> tuple[str, str]:
    """Names of the lower and upper bound columns, e.g. ("lgbm_abs_lo_80", "lgbm_abs_hi_80")."""
    percent = int(level * 100)
    return f"{tag}_lo_{percent}", f"{tag}_hi_{percent}"


# --------------------------------------------------------------------------- inputs
def volatility_scale_table(panel: pd.DataFrame, markets: list[str], start: str, end: str, window: int) -> pd.DataFrame:
    """Per (market, origin quote): mean absolute log change over the last `window` observed quotes, known at that origin.

    Uses only quotes up to and including the origin. Needs a full window of `window` changes, else NaN.
    """
    arrays = F.build_arrays(panel, start, end, markets)
    parts = []
    for i in range(len(arrays.short)):
        market_name = arrays.short[i]

        # Days with a quote, and the log prices on those days.
        quote_days = np.flatnonzero(~np.isnan(arrays.logp[i]))
        log_prices = arrays.logp[i, quote_days]

        # Size of each quote-to-quote change (the first quote has no previous one, so NaN).
        changes = np.abs(np.diff(log_prices))
        abs_changes = np.concatenate([[np.nan], changes])

        # Average over the last `window` changes (NaN until a full window exists).
        rolling_mean = pd.Series(abs_changes).rolling(window, min_periods=window).mean().to_numpy()

        parts.append(pd.DataFrame({"market": market_name, "origin": arrays.days[quote_days], "vol_scale": rolling_mean}))
    return pd.concat(parts, ignore_index=True)


def calibration_frame(cfg: dict[str, Any], fair: pd.DataFrame, tuning: pd.DataFrame | None, headline_cols: dict[int, str], vol: pd.DataFrame) -> pd.DataFrame:
    """One frame of out-of-sample forecasts: the final window (source 'final') and, optionally, the 2017-2019 tuning-fold
    forecasts of the same frozen LightGBM (source 'seed'), with the volatility scale at each origin."""
    # Final-window forecasts, with the model columns renamed to f_lgbm / f_naive.
    final = fair[["market", "h", "origin", "target_date", "y_log", "lgbm", "base_naive", "gap_prev"]]
    final = final.rename(columns={"lgbm": "f_lgbm", "base_naive": "f_naive"})
    final = final.assign(source="final")
    parts = [final]

    # Optional: earlier tuning-run forecasts of the same model, used only to "warm up" the calibration.
    if tuning is not None:
        seed_parts = []
        for h, column in headline_cols.items():
            rows_h = tuning[tuning["h"] == h]
            rows_h = rows_h[["market", "h", "origin", "target_date", "y_log", column, "base_naive", "gap_prev"]]
            rows_h = rows_h.rename(columns={column: "f_lgbm", "base_naive": "f_naive"})
            seed_parts.append(rows_h)
        seed = pd.concat(seed_parts, ignore_index=True)
        seed = seed.dropna(subset=["y_log", "f_lgbm"])
        seed = seed.assign(source="seed")
        parts.append(seed)

    # Join everything and add the volatility scale at each origin.
    frame = pd.concat(parts, ignore_index=True)
    frame = frame.merge(vol, on=["market", "origin"], how="left")
    frame = frame.sort_values(["market", "h", "origin"]).reset_index(drop=True)
    return frame


# --------------------------------------------------------------------------- conformal core
def quantiles_for(origin: np.ndarray, cal_target: np.ndarray, cal_score: np.ndarray, levels: list[float],
                  window_days: int, min_errors: int) -> tuple[dict[float, np.ndarray], np.ndarray]:
    """Conformal quantiles for forecasts made at `origin`, from a pool of resolved errors.

    The pool is (cal_target, cal_score), with `cal_target` (the dates the errors became known) sorted ascending. For each origin the
    calibration set is the scores whose target is STRICTLY before the origin and within `window_days` before it; with fewer than
    `min_errors` usable scores no interval is issued (NaN). The quantile is the k-th smallest with k = ceil((n + 1) x level).
    Returns ({level: q per origin}, number of calibration errors per origin).
    """
    # Dates as whole numbers (nanoseconds) so they are easy to compare.
    origin_ns = np.asarray(origin).astype("datetime64[ns]").astype("int64")
    target_ns = np.asarray(cal_target).astype("datetime64[ns]").astype("int64")
    n_origins = len(origin_ns)

    # For each origin: position of the first error inside the window, and of the first error NOT before the origin.
    window_start_ns = origin_ns - window_days * NS_DAY
    first_position = np.searchsorted(target_ns, window_start_ns, side="left")
    end_position = np.searchsorted(target_ns, origin_ns, side="left")  # strictly before the origin

    # Start with "no interval" everywhere.
    quantiles = {}
    for level in levels:
        quantiles[level] = np.full(n_origins, np.nan)
    n_calibration = np.zeros(n_origins, dtype=int)

    for i in range(n_origins):
        # The errors known at this origin (skip missing ones).
        window_scores = cal_score[first_position[i]:end_position[i]]
        window_scores = window_scores[~np.isnan(window_scores)]
        n_scores = len(window_scores)
        n_calibration[i] = n_scores

        # Too few errors: no interval.
        if n_scores < min_errors:
            continue

        sorted_scores = np.sort(window_scores)  # smallest first
        for level in levels:
            # k-th smallest error, with the finite-sample correction (n + 1).
            k = int(math.ceil((n_scores + 1) * level))
            if k <= n_scores:
                quantiles[level][i] = sorted_scores[k - 1]
    return quantiles, n_calibration


def conformal_halfwidths(origin: np.ndarray, target: np.ndarray, score: np.ndarray, levels: list[float],
                         window_days: int, min_errors: int) -> tuple[dict[float, np.ndarray], np.ndarray]:
    """Rolling split-conformal quantiles for one (market, horizon) series sorted by origin: each forecast is calibrated on the
    errors of the earlier forecasts of the same series (see `quantiles_for`). `target` must be non-decreasing."""
    return quantiles_for(origin, target, score, levels, window_days, min_errors)


def add_intervals(fr: pd.DataFrame, levels: list[float], window_days: int, min_errors: int) -> pd.DataFrame:
    """Add lower/upper bounds (log scale) for every method tag and level, and the calibration-set size of the primary method."""
    out = fr.copy()

    # Create the empty bound columns first.
    for tag in TAGS:
        for level in levels:
            lower_name, upper_name = _column_names(tag, level)
            out[lower_name] = np.nan
            out[upper_name] = np.nan
    out["n_calib"] = 0

    for _, group in out.groupby(["market", "h"], sort=False):
        group = group.sort_values("origin")
        row_labels = group.index.to_numpy()  # where to write the results back

        for tag in TAGS:
            forecast_column = TAGS[tag][0]
            use_volatility = TAGS[tag][1]

            # Absolute log error of each forecast.
            abs_error = (group["y_log"] - group[forecast_column]).abs().to_numpy()
            volatility = group["vol_scale"].to_numpy()

            # The score: the error itself, or the error divided by the volatility (only where volatility > 0).
            if use_volatility:
                score = np.full(len(abs_error), np.nan)
                valid = (volatility > 0) & ~np.isnan(volatility)
                score[valid] = abs_error[valid] / volatility[valid]
            else:
                score = abs_error

            quantiles, n_calibration = conformal_halfwidths(group["origin"].to_numpy(), group["target_date"].to_numpy(),
                                                            score, levels, window_days, min_errors)

            # Store the calibration-set size of the primary method.
            if tag == "lgbm_abs":
                out.loc[row_labels, "n_calib"] = n_calibration

            forecast = group[forecast_column].to_numpy()
            for level in levels:
                # Half-width of the interval (scaled back by volatility if needed).
                if use_volatility:
                    half_width = quantiles[level] * volatility
                else:
                    half_width = quantiles[level]
                lower_name, upper_name = _column_names(tag, level)
                out.loc[row_labels, lower_name] = forecast - half_width
                out.loc[row_labels, upper_name] = forecast + half_width
    return out


# --------------------------------------------------------------------------- scoring
def interval_scores(y_log: np.ndarray, f_log: np.ndarray, lo_log: np.ndarray, hi_log: np.ndarray, alpha: float) -> dict[str, np.ndarray]:
    """Per-forecast coverage indicator, width and Winkler interval score in rupees (NaN where no interval was issued)."""
    # Back to rupees.
    actual = np.exp(y_log)
    forecast = np.exp(f_log)
    lower = np.exp(lo_log)
    upper = np.exp(hi_log)

    # 1 if the actual price fell inside the interval, 0 if not, NaN if there was no interval.
    covered = ((actual >= lower) & (actual <= upper)).astype(float)
    no_interval = np.isnan(lower) | np.isnan(upper)
    covered[no_interval] = np.nan

    # Width of the interval in rupees.
    width = upper - lower

    # Winkler score: width plus a penalty when the actual price falls outside.
    below_penalty = (2 / alpha) * np.maximum(lower - actual, 0)
    above_penalty = (2 / alpha) * np.maximum(actual - upper, 0)
    score = width + below_penalty + above_penalty

    return {"covered": covered, "width": width, "rel_width": width / forecast, "interval_score": score}


def _scores_for(g: pd.DataFrame, tag: str, level: float) -> dict[str, np.ndarray]:
    """Interval scores of one method and level for the rows of `g`."""
    forecast_column = TAGS[tag][0]
    lower_name, upper_name = _column_names(tag, level)
    return interval_scores(g["y_log"].to_numpy(), g[forecast_column].to_numpy(),
                           g[lower_name].to_numpy(), g[upper_name].to_numpy(), 1 - level)


def summarise(g: pd.DataFrame, tag: str, level: float) -> dict[str, float]:
    """Coverage, widths and interval score of one method and level over the rows of `g`."""
    scores = _scores_for(g, tag, level)
    has_interval = ~np.isnan(scores["covered"])
    n_with = int(has_interval.sum())
    n_without = int((~has_interval).sum())

    # Averages over forecasts that have an interval (NaN if none has one).
    if has_interval.any():
        coverage = float(np.nanmean(scores["covered"]))
        avg_width = float(np.nanmean(scores["width"]))
        avg_rel_width = float(np.nanmean(scores["rel_width"]))
        avg_score = float(np.nanmean(scores["interval_score"]))
    else:
        coverage = np.nan
        avg_width = np.nan
        avg_rel_width = np.nan
        avg_score = np.nan

    return {"n_forecasts": len(g), "n_with_interval": n_with, "n_without_interval": n_without,
            "coverage": coverage, "avg_width_rs": avg_width, "avg_rel_width": avg_rel_width, "interval_score_rs": avg_score}


def coverage_bootstrap(g: pd.DataFrame, tag: str, level: float, markets: list[str], start: pd.Timestamp, end: pd.Timestamp,
                       block: int, n_resamples: int, seed: int, chunk: int = 250) -> dict[str, float]:
    """Block-bootstrap 95% interval for the coverage of each market and for the macro (equal-weight) coverage.

    Origin days are resampled in blocks (same days for every market, as in the Phase 3 bootstrap).
    """
    scores = _scores_for(g, tag, level)
    covered = scores["covered"]
    has_interval = ~np.isnan(covered)

    # Day number (0 = start) and market number of every forecast.
    n_days = len(pd.date_range(start, end))
    n_markets = len(markets)
    day_number = (g["origin"].dt.normalize() - start).dt.days.to_numpy()
    market_position = {}
    for i in range(n_markets):
        market_position[markets[i]] = i
    market_number = g["market"].map(market_position).to_numpy()

    # Table [day, market]: how many forecasts were covered, and how many had an interval.
    covered_count = np.zeros((n_days, n_markets))
    interval_count = np.zeros((n_days, n_markets))
    for row in range(len(covered)):
        if has_interval[row]:
            covered_count[day_number[row], market_number[row]] += covered[row]
            interval_count[day_number[row], market_number[row]] += 1

    # Each resample is a list of day numbers, drawn in blocks of consecutive days.
    resampled_days = block_indices(n_days, block, n_resamples, np.random.default_rng(seed))

    per_market_parts = []
    macro_parts = []
    # Work in chunks of resamples to keep memory small.
    for chunk_start in range(0, n_resamples, chunk):
        chunk_days = resampled_days[chunk_start:chunk_start + chunk]  # shape (chunk, n_days)

        # Covered and total counts per resample and market (sum over the drawn days).
        covered_sum = covered_count[chunk_days].sum(axis=1)
        total_sum = interval_count[chunk_days].sum(axis=1)

        # Coverage per resample and market (NaN if no forecasts).
        with np.errstate(invalid="ignore"):
            coverage = np.where(total_sum > 0, covered_sum / np.maximum(total_sum, 1), np.nan)
        per_market_parts.append(coverage)

        # Macro coverage: plain average over the markets.
        macro_parts.append(np.nanmean(coverage, axis=1))

    per_market = np.concatenate(per_market_parts)
    macro = np.concatenate(macro_parts)

    # 95% interval = 2.5th and 97.5th percentiles over the resamples.
    result = {}
    for j in range(n_markets):
        result[f"ci_low_{markets[j]}"] = float(np.nanpercentile(per_market[:, j], 2.5))
    for j in range(n_markets):
        result[f"ci_high_{markets[j]}"] = float(np.nanpercentile(per_market[:, j], 97.5))
    result["ci_low_macro"] = float(np.nanpercentile(macro, 2.5))
    result["ci_high_macro"] = float(np.nanpercentile(macro, 97.5))
    return result


# --------------------------------------------------------------------------- orchestration
def build_frame(cfg: dict[str, Any]) -> pd.DataFrame:
    """The calibration frame: final-window forecasts plus (if configured) the 2017-2019 tuning-run forecasts of the same frozen LightGBM."""
    interval_cfg = cfg["intervals"]

    # Which LightGBM column was chosen for each horizon.
    selection_text = (ROOT / cfg["forecast"]["selected_models_file"]).read_text(encoding="utf-8")
    selection = yaml.safe_load(selection_text)
    headline_cols = {}
    for h, chosen in selection["chosen"].items():
        headline_cols[int(h)] = chosen["lgbm"]

    # Saved forecasts and the price panel.
    fair = pd.read_parquet(ROOT / "results" / "fair_forecasts.parquet")
    panel = pd.read_parquet(ROOT / cfg["paths"]["panel"])
    vol = volatility_scale_table(panel, cfg["markets"], cfg["data"]["start_date"], cfg["data"]["end_date"], interval_cfg["vol_window_quotes"])

    tuning = None
    if interval_cfg["seed_with_tuning_forecasts"]:
        tuning = pd.read_parquet(ROOT / cfg["forecast"]["runs_dir"] / selection["tuning_run"] / "forecasts.parquet")
        logger.info("seeding calibration with %d tuning-run forecasts from %s", len(tuning), selection["tuning_run"])
    return calibration_frame(cfg, fair, tuning, headline_cols, vol)


def _volatility_tercile(vol_scale: pd.Series):
    """Label each value low / mid / high by thirds (NaN when the market has 30 or fewer values)."""
    if vol_scale.notna().sum() > 30:
        return pd.qcut(vol_scale, 3, labels=["low", "mid", "high"])
    return np.nan


def _count_missing(values: pd.Series) -> int:
    """Number of missing values."""
    return int(values.isna().sum())


def run(cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Build intervals for the final window and write all tables to results/."""
    interval_cfg = cfg["intervals"]
    bootstrap_cfg = cfg["forecast"]["bootstrap"]

    # All levels we need (reported levels plus the calibration-curve levels), sorted.
    levels = sorted(set(interval_cfg["levels"]) | set(interval_cfg["calibration_levels"]))

    frame = build_frame(cfg)
    full = add_intervals(frame, levels, interval_cfg["calibration_window_days"], interval_cfg["min_errors"])

    # Only the final window is scored (the "seed" rows are only for calibration).
    fin = full[full["source"] == "final"].reset_index(drop=True)
    markets = sorted(fin["market"].unique())
    start = pd.Timestamp(cfg["backtest"]["first_test_date"])
    end = pd.Timestamp(cfg["backtest"]["last_test_date"])
    results_dir = ROOT / "results"

    # ---- per market / horizon / method / level, with block-bootstrap intervals
    rows = []
    for group_key, group in fin.groupby(["market", "h"]):
        market = group_key[0]
        h = group_key[1]
        for tag in TAGS:
            for level in interval_cfg["levels"]:
                row = {"market": market, "h": h, "method": tag, "level": level}
                row.update(summarise(group, tag, level))
                rows.append(row)
    cov = pd.DataFrame(rows)

    boots = {}
    macro_rows = []
    for h, group_h in fin.groupby("h"):
        for tag in TAGS:
            for level in interval_cfg["levels"]:
                boot = coverage_bootstrap(group_h, tag, level, markets, start, end,
                                          bootstrap_cfg["block_days"], bootstrap_cfg["resamples"], bootstrap_cfg["seed"])
                boots[(h, tag, level)] = boot

                # The per-market rows of this horizon, method and level.
                same_setting = (cov["h"] == h) & (cov["method"] == tag) & (cov["level"] == level)
                sub = cov[same_setting]
                macro_rows.append({
                    "h": h, "method": tag, "level": level,
                    "macro_coverage": sub["coverage"].mean(),
                    "ci_low": boot["ci_low_macro"],
                    "ci_high": boot["ci_high_macro"],
                    "macro_avg_width_rs": sub["avg_width_rs"].mean(),
                    "macro_avg_rel_width": sub["avg_rel_width"].mean(),
                    "macro_interval_score_rs": sub["interval_score_rs"].mean(),
                    "n_with_interval": int(sub["n_with_interval"].sum()),
                    "n_without_interval": int(sub["n_without_interval"].sum()),
                })

    # Add the per-market bootstrap interval to each coverage row.
    ci_low = []
    ci_high = []
    for row in cov.itertuples():
        boot = boots[(row.h, row.method, row.level)]
        ci_low.append(boot[f"ci_low_{row.market}"])
        ci_high.append(boot[f"ci_high_{row.market}"])
    cov["ci_low"] = ci_low
    cov["ci_high"] = ci_high
    macro = pd.DataFrame(macro_rows)

    # ---- breakdowns (pooled over markets): year, event group, gap bucket, volatility tercile
    periods = cfg["forecast"]["event_periods"]
    period_to_group = {}
    for period in periods:
        period_to_group[period["name"]] = period["group"]
    fin["year"] = fin["target_date"].dt.year
    fin["event_group"] = E.assign_period(fin["target_date"], periods).map(period_to_group).fillna("normal")
    fin["gap_bucket"] = E.gap_bucket(fin["gap_prev"])
    fin["vol_tercile"] = fin.groupby("market")["vol_scale"].transform(_volatility_tercile).astype(str)

    breakdown_rows = []
    for group_type in ["year", "event_group", "gap_bucket", "vol_tercile"]:
        for group_key, group in fin.groupby(["h", group_type]):
            h = group_key[0]
            value = group_key[1]
            for tag in TAGS:
                for level in interval_cfg["levels"]:
                    row = {"group_type": group_type, "group": value, "h": h, "method": tag, "level": level}
                    row.update(summarise(group, tag, level))
                    breakdown_rows.append(row)
    by = pd.DataFrame(breakdown_rows)

    # ---- calibration curve (pooled), comparator table, burn-in
    calibration_rows = []
    for h, group_h in fin.groupby("h"):
        for tag in TAGS:
            for level in interval_cfg["calibration_levels"]:
                summary = summarise(group_h, tag, level)
                calibration_rows.append({"h": h, "method": tag, "nominal": level, "empirical": summary["coverage"],
                                         "n_with_interval": summary["n_with_interval"]})
    cal = pd.DataFrame(calibration_rows)

    # Comparator: one row per market/horizon/level, one column per (measure, method).
    comp = cov.pivot_table(index=["market", "h", "level"], columns="method",
                           values=["coverage", "avg_width_rs", "avg_rel_width", "interval_score_rs"]).reset_index()
    # Flatten the two-level column names, e.g. ("coverage", "lgbm_abs") -> "coverage|lgbm_abs".
    flat_names = []
    for measure, method in comp.columns:
        if method:
            flat_names.append(f"{measure}|{method}")
        else:
            flat_names.append(measure)
    comp.columns = flat_names
    comp["width_ratio_lgbm_over_naive"] = comp["avg_width_rs|lgbm_abs"] / comp["avg_width_rs|naive_abs"]
    comp["interval_score_ratio_lgbm_over_naive"] = comp["interval_score_rs|lgbm_abs"] / comp["interval_score_rs|naive_abs"]

    # Burn-in: how many forecasts had no interval because too few errors were known yet.
    burn = fin.groupby(["market", "h"]).agg(
        forecasts=("y_log", "size"),
        without_interval=("lgbm_abs_lo_80", _count_missing),
        min_calibration_errors=("n_calib", "min"),
        median_calibration_errors=("n_calib", "median"),
    ).reset_index()
    burn["seeded_with_tuning_forecasts"] = bool(interval_cfg["seed_with_tuning_forecasts"])

    # Save all tables and figures.
    tables_to_save = [("interval_coverage", cov), ("interval_coverage_macro", macro), ("interval_breakdowns", by),
                      ("interval_calibration", cal), ("interval_comparator", comp), ("interval_burnin", burn)]
    for name, table in tables_to_save:
        table.to_csv(results_dir / f"{name}.csv", index=False)
    fin.to_parquet(results_dir / "interval_forecasts.parquet", index=False)
    figures_dir = ROOT / cfg["paths"]["results"] / "figures"
    plot_calibration(cal, figures_dir)
    plot_fans(fin, cfg, figures_dir)
    return {"coverage": cov, "macro": macro, "by": by, "cal": cal, "comp": comp, "burn": burn}


# --------------------------------------------------------------------------- figures
def plot_calibration(cal: pd.DataFrame, out_dir: Path) -> Path:
    """Nominal vs empirical coverage, pooled over markets, for the three methods at each horizon."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)

    # Line style of each method: (colour, line style, marker, label).
    style = {"lgbm_abs": ("#0072B2", "-", "o", "LightGBM, conformal (primary)"),
             "lgbm_vol": ("#009E73", "--", "s", "LightGBM, volatility-scaled"),
             "naive_abs": ("#D55E00", ":", "^", "naive, conformal (comparator)")}

    panel_number = 0
    for h, group in cal.groupby("h"):
        if panel_number >= len(axes):
            break  # only two panels
        ax = axes[panel_number]
        panel_number = panel_number + 1

        # Diagonal = perfect calibration.
        ax.plot([0.45, 1], [0.45, 1], color="black", lw=0.8)
        for tag in style:
            colour, line_style, marker, label = style[tag]
            method_rows = group[group["method"] == tag].sort_values("nominal")
            ax.plot(method_rows["nominal"], method_rows["empirical"], color=colour, ls=line_style, marker=marker, ms=4, label=label)
        ax.set_title(f"{h}-day horizon", fontsize=10)
        ax.set_xlabel("nominal coverage")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("empirical coverage (2020-2024)")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Calibration of the prediction intervals (pooled over the five markets; the diagonal is perfect calibration)", fontsize=9)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "interval_calibration.png"
    fig.savefig(path, bbox_inches="tight", dpi=130)
    plt.close(fig)
    return path


def plot_fans(fin: pd.DataFrame, cfg: dict[str, Any], out_dir: Path) -> Path:
    """Fan charts: realised quote with the forecast and its 80% and 95% intervals, for chosen markets and periods."""
    fan_cfg = cfg["intervals"]["fan_chart"]
    h = fan_cfg["h"]
    chart_markets = fan_cfg["markets"]
    chart_periods = fan_cfg["periods"]

    # One row of panels per market, one column per period.
    fig, axes = plt.subplots(len(chart_markets), len(chart_periods),
                             figsize=(5 * len(chart_periods), 3.3 * len(chart_markets)), squeeze=False)
    for i in range(len(chart_markets)):
        market = chart_markets[i]
        for j in range(len(chart_periods)):
            period = chart_periods[j]
            ax = axes[i, j]

            # Forecasts of this market and horizon whose target falls in the period, with a 95% interval.
            in_panel = ((fin["market"] == market) & (fin["h"] == h)
                        & (fin["target_date"] >= period["start"]) & (fin["target_date"] <= period["end"]))
            rows = fin[in_panel].sort_values("target_date")
            rows = rows.dropna(subset=["lgbm_abs_lo_95"])
            x = rows["target_date"]

            # Shaded intervals and the forecast line (in rupees).
            ax.fill_between(x, np.exp(rows["lgbm_abs_lo_95"]), np.exp(rows["lgbm_abs_hi_95"]), color="#9ecae1", alpha=0.55, lw=0, label="95% interval")
            ax.fill_between(x, np.exp(rows["lgbm_abs_lo_80"]), np.exp(rows["lgbm_abs_hi_80"]), color="#3182bd", alpha=0.55, lw=0, label="80% interval")
            ax.plot(x, np.exp(rows["f_lgbm"]), color="#08519c", lw=1, label="LightGBM forecast")

            # Actual prices: black inside the 95% interval, red outside.
            outside_95 = (rows["y_log"] < rows["lgbm_abs_lo_95"]) | (rows["y_log"] > rows["lgbm_abs_hi_95"])
            ax.scatter(x[~outside_95], np.exp(rows["y_log"][~outside_95]), s=6, color="black", zorder=3, label="realised quote")
            ax.scatter(x[outside_95], np.exp(rows["y_log"][outside_95]), s=14, color="#d7191c", zorder=4, label="outside 95%")

            ax.set_title(f"{market}: {period['name']}", fontsize=9)
            ax.tick_params(axis="x", labelrotation=30, labelsize=7)
            if j == 0:
                ax.set_ylabel(f"Rs/quintal ({h}-day-ahead)")

    axes[0, 0].legend(fontsize=6, loc="upper left")
    fig.suptitle(f"{h}-day-ahead forecasts with conformal intervals; x-axis = the date being forecast", fontsize=9, y=1.0)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "interval_fan_charts.png"
    fig.savefig(path, bbox_inches="tight", dpi=130)
    plt.close(fig)
    return path


def main() -> None:
    """Write results/interval_*.csv, results/interval_forecasts.parquet and the figures."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config()
    out = run(cfg)

    # Print the macro coverage of the primary method and the naive comparator.
    macro = out["macro"]
    shown = macro[macro["method"].isin(["lgbm_abs", "naive_abs"])]
    shown = shown[["h", "method", "level", "macro_coverage", "ci_low", "ci_high", "macro_avg_rel_width", "n_without_interval"]]
    logger.info("macro coverage:\n%s", shown.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
