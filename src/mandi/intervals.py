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
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from mandi import evaluate as E  # noqa: E402
from mandi import features as F  # noqa: E402
from mandi.config import ROOT, load_config  # noqa: E402
from mandi.posthoc import block_indices  # noqa: E402

logger = logging.getLogger(__name__)
TAGS = {"lgbm_abs": ("f_lgbm", False), "lgbm_vol": ("f_lgbm", True), "naive_abs": ("f_naive", False)}
NS_DAY = 86_400 * 10**9


# --------------------------------------------------------------------------- inputs
def volatility_scale_table(panel: pd.DataFrame, markets: list[str], start: str, end: str, window: int) -> pd.DataFrame:
    """Per (market, origin quote): mean absolute log change over the last `window` observed quotes, known at that origin.

    Uses only quotes up to and including the origin. Needs a full window of `window` changes, else NaN.
    """
    a = F.build_arrays(panel, start, end, markets)
    rows = []
    for i, name in enumerate(a.short):
        idx = np.flatnonzero(~np.isnan(a.logp[i]))
        lp = a.logp[i, idx]
        absd = np.r_[np.nan, np.abs(np.diff(lp))]
        s = pd.Series(absd).rolling(window, min_periods=window).mean().to_numpy()
        rows.append(pd.DataFrame({"market": name, "origin": a.days[idx], "vol_scale": s}))
    return pd.concat(rows, ignore_index=True)


def calibration_frame(cfg: dict[str, Any], fair: pd.DataFrame, tuning: pd.DataFrame | None, headline_cols: dict[int, str], vol: pd.DataFrame) -> pd.DataFrame:
    """One frame of out-of-sample forecasts: the final window (source 'final') and, optionally, the 2017-2019 tuning-fold
    forecasts of the same frozen LightGBM (source 'seed'), with the volatility scale at each origin."""
    parts = [fair[["market", "h", "origin", "target_date", "y_log", "lgbm", "base_naive", "gap_prev"]]
             .rename(columns={"lgbm": "f_lgbm", "base_naive": "f_naive"}).assign(source="final")]
    if tuning is not None:
        seed = []
        for h, col in headline_cols.items():
            g = tuning[tuning["h"] == h]
            seed.append(g[["market", "h", "origin", "target_date", "y_log", col, "base_naive", "gap_prev"]]
                        .rename(columns={col: "f_lgbm", "base_naive": "f_naive"}))
        parts.append(pd.concat(seed, ignore_index=True).dropna(subset=["y_log", "f_lgbm"]).assign(source="seed"))
    fr = pd.concat(parts, ignore_index=True).merge(vol, on=["market", "origin"], how="left")
    return fr.sort_values(["market", "h", "origin"]).reset_index(drop=True)


# --------------------------------------------------------------------------- conformal core
def conformal_halfwidths(origin: np.ndarray, target: np.ndarray, score: np.ndarray, levels: list[float],
                         window_days: int, min_errors: int) -> tuple[dict[float, np.ndarray], np.ndarray]:
    """Rolling split-conformal quantiles for one (market, horizon) series sorted by origin.

    For forecast i the calibration scores are those j with target_j strictly before origin_i and target_j within
    `window_days` before origin_i. Returns ({level: q_i}, n_calibration_i); q is NaN when fewer than `min_errors`
    usable scores exist. `target` must be non-decreasing (true for a fixed horizon sorted by origin).
    """
    o, t = origin.astype("datetime64[ns]").astype("int64"), target.astype("datetime64[ns]").astype("int64")
    lo = np.searchsorted(t, o - window_days * NS_DAY, side="left")
    hi = np.searchsorted(t, o, side="left")           # strictly before the origin
    out = {lv: np.full(len(o), np.nan) for lv in levels}
    n_cal = np.zeros(len(o), dtype=int)
    for i in range(len(o)):
        w = score[lo[i]:hi[i]]
        w = w[~np.isnan(w)]
        n = len(w)
        n_cal[i] = n
        if n < min_errors:
            continue
        for lv in levels:
            k = int(np.ceil((n + 1) * lv))
            if k <= n:
                out[lv][i] = np.partition(w, k - 1)[k - 1]
    return out, n_cal


def add_intervals(fr: pd.DataFrame, levels: list[float], window_days: int, min_errors: int) -> pd.DataFrame:
    """Add lower/upper bounds (log scale) for every method tag and level, and the calibration-set size of the primary method."""
    out = fr.copy()
    for tag, (fcol, scaled) in TAGS.items():
        for lv in levels:
            out[f"{tag}_lo_{int(lv * 100)}"] = np.nan
            out[f"{tag}_hi_{int(lv * 100)}"] = np.nan
    out["n_calib"] = 0
    for (m, h), g in out.groupby(["market", "h"], sort=False):
        g = g.sort_values("origin")
        ix = g.index.to_numpy()
        for tag, (fcol, scaled) in TAGS.items():
            e = (g["y_log"] - g[fcol]).abs().to_numpy()
            s = g["vol_scale"].to_numpy()
            score = np.where((s > 0) & ~np.isnan(s), e / s, np.nan) if scaled else e
            q, n_cal = conformal_halfwidths(g["origin"].to_numpy(), g["target_date"].to_numpy(), score, levels, window_days, min_errors)
            if tag == "lgbm_abs":
                out.loc[ix, "n_calib"] = n_cal
            for lv in levels:
                half = q[lv] * s if scaled else q[lv]
                out.loc[ix, f"{tag}_lo_{int(lv * 100)}"] = g[fcol].to_numpy() - half
                out.loc[ix, f"{tag}_hi_{int(lv * 100)}"] = g[fcol].to_numpy() + half
    return out


# --------------------------------------------------------------------------- scoring
def interval_scores(y_log: np.ndarray, f_log: np.ndarray, lo_log: np.ndarray, hi_log: np.ndarray, alpha: float) -> dict[str, np.ndarray]:
    """Per-forecast coverage indicator, width and Winkler interval score in rupees (NaN where no interval was issued)."""
    y, f, lo, hi = np.exp(y_log), np.exp(f_log), np.exp(lo_log), np.exp(hi_log)
    covered = ((y >= lo) & (y <= hi)).astype(float)
    covered[np.isnan(lo) | np.isnan(hi)] = np.nan
    width = hi - lo
    score = width + (2 / alpha) * np.maximum(lo - y, 0) + (2 / alpha) * np.maximum(y - hi, 0)
    return {"covered": covered, "width": width, "rel_width": width / f, "interval_score": score}


def summarise(g: pd.DataFrame, tag: str, level: float) -> dict[str, float]:
    """Coverage, widths and interval score of one method and level over the rows of `g`."""
    fcol = TAGS[tag][0]
    p = int(level * 100)
    sc = interval_scores(g["y_log"].to_numpy(), g[fcol].to_numpy(), g[f"{tag}_lo_{p}"].to_numpy(), g[f"{tag}_hi_{p}"].to_numpy(), 1 - level)
    has = ~np.isnan(sc["covered"])
    return {"n_forecasts": len(g), "n_with_interval": int(has.sum()), "n_without_interval": int((~has).sum()),
            "coverage": float(np.nanmean(sc["covered"])) if has.any() else np.nan,
            "avg_width_rs": float(np.nanmean(sc["width"])) if has.any() else np.nan,
            "avg_rel_width": float(np.nanmean(sc["rel_width"])) if has.any() else np.nan,
            "interval_score_rs": float(np.nanmean(sc["interval_score"])) if has.any() else np.nan}


def coverage_bootstrap(g: pd.DataFrame, tag: str, level: float, markets: list[str], start: pd.Timestamp, end: pd.Timestamp,
                       block: int, n_resamples: int, seed: int, chunk: int = 250) -> dict[str, float]:
    """Block-bootstrap 95% interval for the coverage of each market and for the macro (equal-weight) coverage.

    Origin days are resampled in blocks (same days for every market, as in the Phase 3 bootstrap).
    """
    p = int(level * 100)
    sc = interval_scores(g["y_log"].to_numpy(), g[TAGS[tag][0]].to_numpy(), g[f"{tag}_lo_{p}"].to_numpy(), g[f"{tag}_hi_{p}"].to_numpy(), 1 - level)
    ok = ~np.isnan(sc["covered"])
    days = pd.date_range(start, end)
    di = (g["origin"].dt.normalize() - start).dt.days.to_numpy()
    mi = g["market"].map({m: i for i, m in enumerate(markets)}).to_numpy()
    D, M = len(days), len(markets)
    cov, cnt = np.zeros((D, M)), np.zeros((D, M))
    np.add.at(cov, (di[ok], mi[ok]), sc["covered"][ok])
    np.add.at(cnt, (di[ok], mi[ok]), 1)
    idx = block_indices(D, block, n_resamples, np.random.default_rng(seed))
    per_market, macro = [], []
    for i in range(0, n_resamples, chunk):
        ix = idx[i:i + chunk]
        c, n = cov[ix].sum(axis=1), cnt[ix].sum(axis=1)       # (chunk, M)
        cm = np.where(n > 0, c / np.maximum(n, 1), np.nan)
        per_market.append(cm)
        macro.append(np.nanmean(cm, axis=1))
    pm, mc = np.concatenate(per_market), np.concatenate(macro)
    out = {f"ci_low_{m}": float(np.nanpercentile(pm[:, j], 2.5)) for j, m in enumerate(markets)}
    out |= {f"ci_high_{m}": float(np.nanpercentile(pm[:, j], 97.5)) for j, m in enumerate(markets)}
    out |= {"ci_low_macro": float(np.nanpercentile(mc, 2.5)), "ci_high_macro": float(np.nanpercentile(mc, 97.5))}
    return out


# --------------------------------------------------------------------------- orchestration
def run(cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Build intervals for the final window and write all tables to results/."""
    ic, bs = cfg["intervals"], cfg["forecast"]["bootstrap"]
    levels = sorted(set(ic["levels"]) | set(ic["calibration_levels"]))
    sel = yaml.safe_load((ROOT / cfg["forecast"]["selected_models_file"]).read_text(encoding="utf-8"))
    headline_cols = {int(h): d["lgbm"] for h, d in sel["chosen"].items()}
    fair = pd.read_parquet(ROOT / "results" / "fair_forecasts.parquet")
    panel = pd.read_parquet(ROOT / cfg["paths"]["panel"])
    vol = volatility_scale_table(panel, cfg["markets"], cfg["data"]["start_date"], cfg["data"]["end_date"], ic["vol_window_quotes"])
    tuning = None
    if ic["seed_with_tuning_forecasts"]:
        run_dir = ROOT / cfg["forecast"]["runs_dir"] / sel["tuning_run"]
        tuning = pd.read_parquet(run_dir / "forecasts.parquet")
        logger.info("seeding calibration with %d tuning-run forecasts from %s", len(tuning), sel["tuning_run"])
    fr = calibration_frame(cfg, fair, tuning, headline_cols, vol)
    full = add_intervals(fr, levels, ic["calibration_window_days"], ic["min_errors"])
    fin = full[full["source"] == "final"].reset_index(drop=True)
    markets = sorted(fin["market"].unique())
    start, end = pd.Timestamp(cfg["backtest"]["first_test_date"]), pd.Timestamp(cfg["backtest"]["last_test_date"])
    res = ROOT / "results"

    # ---- per market / horizon / method / level, with block-bootstrap intervals
    rows = []
    for (m, h), g in fin.groupby(["market", "h"]):
        for tag in TAGS:
            for lv in ic["levels"]:
                rows.append({"market": m, "h": h, "method": tag, "level": lv, **summarise(g, tag, lv)})
    cov = pd.DataFrame(rows)
    boots, macro_rows = {}, []
    for h, gh in fin.groupby("h"):
        for tag in TAGS:
            for lv in ic["levels"]:
                b = coverage_bootstrap(gh, tag, lv, markets, start, end, bs["block_days"], bs["resamples"], bs["seed"])
                boots[(h, tag, lv)] = b
                sub = cov[(cov["h"] == h) & (cov["method"] == tag) & (cov["level"] == lv)]
                macro_rows.append({"h": h, "method": tag, "level": lv, "macro_coverage": sub["coverage"].mean(), "ci_low": b["ci_low_macro"], "ci_high": b["ci_high_macro"],
                                   "macro_avg_width_rs": sub["avg_width_rs"].mean(), "macro_avg_rel_width": sub["avg_rel_width"].mean(),
                                   "macro_interval_score_rs": sub["interval_score_rs"].mean(), "n_with_interval": int(sub["n_with_interval"].sum()),
                                   "n_without_interval": int(sub["n_without_interval"].sum())})
    cov["ci_low"] = [boots[(r.h, r.method, r.level)][f"ci_low_{r.market}"] for r in cov.itertuples()]
    cov["ci_high"] = [boots[(r.h, r.method, r.level)][f"ci_high_{r.market}"] for r in cov.itertuples()]
    macro = pd.DataFrame(macro_rows)

    # ---- breakdowns (pooled over markets): year, event group, gap bucket, volatility tercile
    periods = cfg["forecast"]["event_periods"]
    fin["year"] = fin["target_date"].dt.year
    fin["event_group"] = E.assign_period(fin["target_date"], periods).map({p["name"]: p["group"] for p in periods}).fillna("normal")
    fin["gap_bucket"] = E.gap_bucket(fin["gap_prev"])
    fin["vol_tercile"] = fin.groupby("market")["vol_scale"].transform(lambda s: pd.qcut(s, 3, labels=["low", "mid", "high"]) if s.notna().sum() > 30 else np.nan).astype(str)
    by = []
    for gtype in ["year", "event_group", "gap_bucket", "vol_tercile"]:
        for (h, val), g in fin.groupby(["h", gtype]):
            for tag in TAGS:
                for lv in ic["levels"]:
                    by.append({"group_type": gtype, "group": val, "h": h, "method": tag, "level": lv, **summarise(g, tag, lv)})
    by = pd.DataFrame(by)

    # ---- calibration curve (pooled), comparator table, burn-in
    cal = []
    for h, gh in fin.groupby("h"):
        for tag in TAGS:
            for lv in ic["calibration_levels"]:
                s = summarise(gh, tag, lv)
                cal.append({"h": h, "method": tag, "nominal": lv, "empirical": s["coverage"], "n_with_interval": s["n_with_interval"]})
    cal = pd.DataFrame(cal)
    comp = cov.pivot_table(index=["market", "h", "level"], columns="method", values=["coverage", "avg_width_rs", "avg_rel_width", "interval_score_rs"]).reset_index()
    comp.columns = [f"{a}|{b}" if b else a for a, b in comp.columns]
    comp["width_ratio_lgbm_over_naive"] = comp["avg_width_rs|lgbm_abs"] / comp["avg_width_rs|naive_abs"]
    comp["interval_score_ratio_lgbm_over_naive"] = comp["interval_score_rs|lgbm_abs"] / comp["interval_score_rs|naive_abs"]
    burn = fin.groupby(["market", "h"]).agg(forecasts=("y_log", "size"), without_interval=("lgbm_abs_lo_80", lambda s: int(s.isna().sum())),
                                            min_calibration_errors=("n_calib", "min"), median_calibration_errors=("n_calib", "median")).reset_index()
    burn["seeded_with_tuning_forecasts"] = bool(ic["seed_with_tuning_forecasts"])
    for name, df in [("interval_coverage", cov), ("interval_coverage_macro", macro), ("interval_breakdowns", by), ("interval_calibration", cal),
                     ("interval_comparator", comp), ("interval_burnin", burn)]:
        df.to_csv(res / f"{name}.csv", index=False)
    fin.to_parquet(res / "interval_forecasts.parquet", index=False)
    plot_calibration(cal, ROOT / cfg["paths"]["results"] / "figures")
    plot_fans(fin, cfg, ROOT / cfg["paths"]["results"] / "figures")
    return {"coverage": cov, "macro": macro, "by": by, "cal": cal, "comp": comp, "burn": burn}


# --------------------------------------------------------------------------- figures
def plot_calibration(cal: pd.DataFrame, out_dir: Path) -> Path:
    """Nominal vs empirical coverage, pooled over markets, for the three methods at each horizon."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    style = {"lgbm_abs": ("#0072B2", "-", "o", "LightGBM, conformal (primary)"), "lgbm_vol": ("#009E73", "--", "s", "LightGBM, volatility-scaled"),
             "naive_abs": ("#D55E00", ":", "^", "naive, conformal (comparator)")}
    for ax, (h, g) in zip(axes, cal.groupby("h")):
        ax.plot([0.45, 1], [0.45, 1], color="black", lw=0.8)
        for tag, (c, ls, mk, lab) in style.items():
            t = g[g["method"] == tag].sort_values("nominal")
            ax.plot(t["nominal"], t["empirical"], color=c, ls=ls, marker=mk, ms=4, label=lab)
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
    fc = cfg["intervals"]["fan_chart"]
    h = fc["h"]
    fig, axes = plt.subplots(len(fc["markets"]), len(fc["periods"]), figsize=(5 * len(fc["periods"]), 3.3 * len(fc["markets"])), squeeze=False)
    for i, m in enumerate(fc["markets"]):
        for j, per in enumerate(fc["periods"]):
            ax = axes[i, j]
            g = fin[(fin["market"] == m) & (fin["h"] == h) & (fin["target_date"] >= per["start"]) & (fin["target_date"] <= per["end"])].sort_values("target_date")
            g = g.dropna(subset=["lgbm_abs_lo_95"])
            x = g["target_date"]
            ax.fill_between(x, np.exp(g["lgbm_abs_lo_95"]), np.exp(g["lgbm_abs_hi_95"]), color="#9ecae1", alpha=0.55, lw=0, label="95% interval")
            ax.fill_between(x, np.exp(g["lgbm_abs_lo_80"]), np.exp(g["lgbm_abs_hi_80"]), color="#3182bd", alpha=0.55, lw=0, label="80% interval")
            ax.plot(x, np.exp(g["f_lgbm"]), color="#08519c", lw=1, label="LightGBM forecast")
            out95 = (g["y_log"] < g["lgbm_abs_lo_95"]) | (g["y_log"] > g["lgbm_abs_hi_95"])
            ax.scatter(x[~out95], np.exp(g["y_log"][~out95]), s=6, color="black", zorder=3, label="realised quote")
            ax.scatter(x[out95], np.exp(g["y_log"][out95]), s=14, color="#d7191c", zorder=4, label="outside 95%")
            ax.set_title(f"{m}: {per['name']}", fontsize=9)
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
    m = out["macro"]
    logger.info("macro coverage:\n%s", m[m["method"].isin(["lgbm_abs", "naive_abs"])][["h", "method", "level", "macro_coverage", "ci_low", "ci_high", "macro_avg_rel_width", "n_without_interval"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
