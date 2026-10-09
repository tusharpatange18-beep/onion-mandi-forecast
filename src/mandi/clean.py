"""Phase 1: clean the raw rows and build the daily panel per market.

Principles: never invent data, flag instead of correcting, log every decision.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PRICE_COLS = ["min_price_rs_qtl", "max_price_rs_qtl", "modal_price_rs_qtl"]


# --------------------------------------------------------------------------- logging
@dataclass
class CleaningLog:
    """Collects one entry per cleaning rule, with the number of rows it affected."""

    entries: list[dict[str, Any]] = field(default_factory=list)

    def add(self, step: str, rule: str, rows: int, action: str, detail: str = "") -> None:
        """Record a rule. `action` says what happened to the rows: dropped, flagged, filled, none."""
        self.entries.append({"step": step, "rule": rule, "rows": int(rows), "action": action, "detail": detail})
        logger.info("%s | %s | rows=%d | %s", step, rule, rows, action)

    def to_markdown(self, title: str, intro: str = "") -> str:
        """Render the log as a markdown table."""
        lines = [f"# {title}", "", intro, "", "| # | Step | Rule | Rows affected | Action | Detail |", "|---|---|---|---|---|---|"]
        for i, e in enumerate(self.entries, 1):
            lines.append(f"| {i} | {e['step']} | {e['rule']} | {e['rows']:,} | {e['action']} | {e['detail']} |")
        return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- row-level rules
def normalise_names(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    """Strip whitespace from text columns and check that market_key maps to one market name."""
    df = df.copy()
    changed = 0
    for col in ["state", "district", "market", "market_key", "variety"]:
        stripped = df[col].str.strip()
        changed += int((stripped != df[col]).sum())
        df[col] = stripped
    log.add("names", "strip leading/trailing whitespace in text columns", changed, "changed" if changed else "none")
    n_multi = int((df.groupby("market_key")["market"].nunique() > 1).sum())
    n_case = int(df.groupby(df["market_key"].str.casefold())["market_key"].nunique().gt(1).sum())
    log.add("names", "market_key maps to more than one market name, or differs only by case", n_multi + n_case, "none" if not (n_multi + n_case) else "REVIEW")
    return df


def drop_exact_duplicates(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    """Drop rows identical in every column."""
    n = int(df.duplicated().sum())
    log.add("duplicates", "fully identical rows", n, "dropped" if n else "none")
    return df.drop_duplicates().reset_index(drop=True)


def choose_primary_variety(df: pd.DataFrame) -> dict[str, str]:
    """For each market, the single most frequent variety (by number of rows)."""
    return df.groupby("market_key")["variety"].agg(lambda s: s.value_counts().index[0]).to_dict()


def select_primary_rows(df: pd.DataFrame, primary: dict[str, str], log: CleaningLog) -> pd.DataFrame:
    """Keep only each market's primary-variety rows. Other varieties are never substituted."""
    keep = df["variety"] == df["market_key"].map(primary)
    for mk, var in primary.items():
        sub = df[df["market_key"] == mk]
        n_other = int((sub["variety"] != var).sum())
        log.add("variety", f"{mk}: keep primary variety '{var}' only", n_other, "dropped (other varieties)",
                f"{len(sub) - n_other} rows kept of {len(sub)}")
    out = df[keep].copy()
    dup = int(out.duplicated(["market_key", "date"]).sum())
    if dup:
        raise ValueError(f"{dup} duplicate (market, date) rows remain within the primary variety; needs a rule")
    log.add("duplicates", "more than one row per (market, date) within the primary variety", dup, "none")
    return out.sort_values(["market_key", "date"]).reset_index(drop=True)


def drop_invalid_prices(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    """Rows with a missing or non-positive price are dropped (that day becomes missing).

    Rows where min <= modal <= max does not hold are only flagged, never changed.
    """
    bad = (df[PRICE_COLS].isna() | (df[PRICE_COLS] <= 0)).any(axis=1)
    log.add("prices", "missing or non-positive min/max/modal price", int(bad.sum()), "dropped (day becomes missing)" if bad.any() else "none")
    df = df[~bad].copy()
    bad_arr = df["arrivals_tonnes"].isna() | (df["arrivals_tonnes"] <= 0)
    log.add("arrivals", "missing or non-positive arrivals", int(bad_arr.sum()), "arrivals set to missing" if bad_arr.any() else "none")
    df.loc[bad_arr, "arrivals_tonnes"] = np.nan
    df["is_order_violation"] = (df["min_price_rs_qtl"] > df["modal_price_rs_qtl"]) | (df["modal_price_rs_qtl"] > df["max_price_rs_qtl"])
    log.add("prices", "violates min <= modal <= max", int(df["is_order_violation"].sum()), "flagged only (is_order_violation)" if df["is_order_violation"].any() else "none")
    return df.reset_index(drop=True)


def flag_stale_runs(df: pd.DataFrame, min_len: int, log: CleaningLog) -> pd.DataFrame:
    """Flag runs of `min_len`+ identical (min, max, modal) quotes on consecutive observed days.

    The values are kept; only `is_stale` is set.
    """
    df = df.sort_values(["market_key", "date"]).copy()
    key = df[PRICE_COLS]
    new_run = (key.diff().abs().sum(axis=1) != 0) | (df["market_key"] != df["market_key"].shift())
    run_id = new_run.cumsum()
    run_len = run_id.map(run_id.value_counts())
    df["is_stale"] = run_len >= min_len
    n_runs = int(run_id[df["is_stale"]].nunique())
    log.add("stale", f"runs of >= {min_len} identical (min, max, modal) quotes on consecutive observed days",
            int(df["is_stale"].sum()), "flagged only (is_stale)", f"{n_runs} runs")
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- outlier flagging
def rolling_median_mad(log_price: pd.Series, window_days: int, min_obs: int) -> pd.DataFrame:
    """Centred calendar-day rolling median and MAD of a log-price series on a daily index.

    Missing days stay NaN, so a window with gaps simply has fewer observations. Windows with
    fewer than `min_obs` observations give NaN (no flag). Centred windows use future days; that
    is acceptable here because this is only used to *flag* history, never as a model feature.
    """
    roll = log_price.rolling(window=window_days, center=True, min_periods=min_obs)
    med = roll.median()
    mad = log_price.rolling(window=window_days, center=True, min_periods=min_obs).apply(
        lambda w: np.nanmedian(np.abs(w - np.nanmedian(w))), raw=True
    )
    return pd.DataFrame({"log_median": med, "log_mad": mad})


def flag_outliers(panel: pd.DataFrame, params: dict[str, Any], log: CleaningLog) -> pd.DataFrame:
    """Add rolling median, robust score and `is_outlier_flag` to the daily panel. Flag only."""
    pieces = []
    for mk, g in panel.groupby("market_key", sort=False):
        s = np.log(g.set_index("date")["modal_price"])
        stats = rolling_median_mad(s, params["window_days"], params["min_obs"])
        scale = np.maximum(1.4826 * stats["log_mad"], params["mad_floor_log"])
        dev = s - stats["log_median"]
        out = g.set_index("date").copy()
        out["rolling_median"] = np.exp(stats["log_median"])
        out["log_dev"] = dev
        out["robust_score"] = dev / scale
        out["is_outlier_flag"] = (
            (out["robust_score"].abs() > params["mad_threshold"])
            & (dev.abs() > params["min_abs_log_dev"])
            & out["modal_price"].notna()
        )
        pieces.append(out.reset_index())
    res = pd.concat(pieces, ignore_index=True)
    for mk, g in res.groupby("market_key"):
        log.add("outliers", f"{mk}: |log(price / rolling median)| > {params['mad_threshold']} x robust scale and > {params['min_abs_log_dev']} "
                f"({params['window_days']}-day centred window, min {params['min_obs']} obs)",
                int(g["is_outlier_flag"].sum()), "flagged only (is_outlier_flag), no value changed",
                f"{int(g['modal_price'].notna().sum())} observed days")
    return res


def build_review_table(panel: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """One row per flagged point, with the same-day prices in the other chosen markets.

    Guess rule (a hint for the human reviewer, not a decision): with at least two other markets
    quoting that day, the flagged price is "likely real" if it lies inside the range of their
    prices widened by `peer_range_tolerance`, otherwise "likely error". With fewer than two
    peers the guess is "uncertain".
    """
    wide_px = panel.pivot(index="date", columns="market_key", values="modal_price")
    short = {mk: mk.split("_")[-1] for mk in wide_px.columns}
    tol = params["peer_range_tolerance"]
    rows = []
    for r in panel[panel["is_outlier_flag"]].itertuples():
        peers = {m: wide_px.at[r.date, m] for m in wide_px.columns if m != r.market_key and pd.notna(wide_px.at[r.date, m])}
        if len(peers) < 2:
            guess = f"uncertain ({len(peers)} other market(s) quoted that day)"
        else:
            lo, hi = min(peers.values()) * (1 - tol), max(peers.values()) * (1 + tol)
            guess = "likely real" if lo <= r.modal_price <= hi else "likely error"
        rows.append({"market": short[r.market_key], "date": r.date.date(), "modal_price": r.modal_price,
                     "rolling_median": round(r.rolling_median), "ratio_to_median": round(r.modal_price / r.rolling_median, 2),
                     "robust_score": round(r.robust_score, 1),
                     "other_markets_same_day": "; ".join(f"{short[m]} {v:.0f}" for m, v in peers.items()) or "none observed",
                     "guess": guess})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- daily panel
def fill_short_gaps(s: pd.Series, max_gap: int) -> tuple[pd.Series, pd.Series]:
    """Forward-fill only runs of missing days no longer than `max_gap`; longer runs stay missing.

    Returns the filled series and a boolean mask of the days that were filled. Pandas'
    `ffill(limit=n)` is not used because it would fill the first n days of a longer gap.
    """
    isna = s.isna()
    run_id = (isna != isna.shift()).cumsum()
    run_len = isna.groupby(run_id).transform("sum")
    # a leading gap has nothing to fill from; ffill leaves it NaN, so it is never marked filled
    fillable = isna & (run_len <= max_gap)
    filled = s.ffill().where(fillable, s)
    return filled, fillable & filled.notna()


def build_daily_grid(df: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Build the daily grid per market (first to last observed date), keeping missing days missing.

    `*_raw` columns hold the original quotes and are never modified afterwards. `modal_price`,
    `min_price`, `max_price` and `arrivals_tonnes` are the target-side columns: missing days stay NaN.
    """
    out = []
    for mk, g in df.groupby("market_key", sort=False):
        g = g.sort_values("date").set_index("date")
        grid = pd.date_range(g.index.min(), g.index.max(), freq="D", name="date")
        p = pd.DataFrame(index=grid)
        p["market_key"] = mk
        p["district"] = g["district"].iloc[0]
        p["market"] = g["market"].iloc[0]
        p["variety"] = g["variety"].iloc[0]
        p["modal_price_raw"] = g["modal_price_rs_qtl"]
        p["min_price_raw"] = g["min_price_rs_qtl"]
        p["max_price_raw"] = g["max_price_rs_qtl"]
        p["modal_price"] = p["modal_price_raw"]
        p["min_price"] = p["min_price_raw"]
        p["max_price"] = p["max_price_raw"]
        p["arrivals_tonnes"] = g["arrivals_tonnes"]
        p["was_missing"] = p["modal_price"].isna()
        p["was_corrected"] = False
        for c in ["is_stale", "is_order_violation"]:
            p[c] = g[c].reindex(grid).fillna(False).astype(bool)
        out.append(p.reset_index())
    return pd.concat(out, ignore_index=True).merge(calendar, on="date", how="left")


def apply_corrections(panel: pd.DataFrame, corrections: list[dict[str, str]], log: CleaningLog) -> pd.DataFrame:
    """Set the approved (market, date) quotes to missing, keeping the originals in the `*_raw` columns.

    Each listed point must already be flagged as an outlier; otherwise the pipeline stops, so a
    stale approval list can never silently change a different value. No replacement value is invented:
    modal, min and max price of that day all become missing (a quote with a bad modal is not trusted
    in part) and `was_corrected` is set.
    """
    panel = panel.copy()
    for c in corrections:
        sel = (panel["market_key"] == c["market_key"]) & (panel["date"] == pd.Timestamp(c["date"]))
        if sel.sum() != 1:
            raise ValueError(f"Correction target not found exactly once: {c}")
        if not panel.loc[sel, "is_outlier_flag"].item():
            raise ValueError(f"Correction target is not currently flagged as an outlier: {c}")
        old = panel.loc[sel, "modal_price"].item()
        panel.loc[sel, ["modal_price", "min_price", "max_price"]] = np.nan
        panel.loc[sel, "was_missing"] = True
        panel.loc[sel, "was_corrected"] = True
        log.add("corrections", f"{c['market_key']} {c['date']}: approved by reviewer, modal price {old:.0f} set to missing", 1,
                "set to missing (min/max too); original kept in *_raw", "was_corrected=True; no replacement value")
    return panel


def add_ffill_columns(panel: pd.DataFrame, max_gap: int, log: CleaningLog) -> pd.DataFrame:
    """Add `*_ffill` columns for feature building: gaps of at most `max_gap` days are forward-filled.

    Longer gaps stay missing. The target columns themselves are never filled.
    """
    out = []
    for mk, g in panel.groupby("market_key", sort=False):
        g = g.sort_values("date").copy()
        for c, nm in [("modal_price", "modal_price_ffill"), ("min_price", "min_price_ffill"),
                      ("max_price", "max_price_ffill"), ("arrivals_tonnes", "arrivals_ffill")]:
            g[nm], mask = fill_short_gaps(g[c], max_gap)
            if c == "modal_price":
                g["was_filled"] = mask
        out.append(g)
    panel = pd.concat(out, ignore_index=True)
    for mk, g in panel.groupby("market_key"):
        n_days, n_miss = len(g), int(g["was_missing"].sum())
        log.add("grid", f"{mk}: daily grid from first to last observed date; missing days kept missing",
                n_miss, "missing days in the grid (was_missing=True)",
                f"{n_days} days in span, {n_miss / n_days:.1%} missing; {int(g['was_filled'].sum())} filled for features (gaps <= {max_gap} days)")
    return panel


# --------------------------------------------------------------------------- coverage / ranking
def market_coverage(df: pd.DataFrame, total_days: int, end_date: pd.Timestamp) -> pd.DataFrame:
    """Per-market coverage of the primary-variety series, plus median arrivals.

    `df` holds all rows of the state. Arrivals are taken once per market-day.
    """
    prim = choose_primary_variety(df)
    rows = []
    for mk, g in df.groupby("market_key"):
        pv = prim[mk]
        p = g[g["variety"] == pv].drop_duplicates("date").sort_values("date")
        d = p["date"]
        own_span = (end_date - d.min()).days + 1  # first observed date to the end of the period
        rows.append({
            "market_key": mk, "district": g["district"].iloc[0], "primary_variety": pv,
            "primary_share_of_rows": round((g["variety"] == pv).mean(), 2),
            "first_date": d.min().date(), "last_date": d.max().date(), "observed_days": len(d),
            "longest_gap_days": int(d.diff().dt.days.max() - 1) if len(d) > 1 else 0,
            "missing_vs_4018_days": round(1 - len(d) / total_days, 3), "missing_vs_own_span": round(1 - len(d) / own_span, 3),
            "median_arrivals_t": float(p["arrivals_tonnes"].median()),
        })
    return pd.DataFrame(rows).sort_values("observed_days", ascending=False).reset_index(drop=True)


def rank_markets(cov: pd.DataFrame, min_days: int, max_gap: int, always_show: list[str]) -> pd.DataFrame:
    """Markets with enough days and small enough gaps, ranked by median arrivals.

    Markets in `always_show` are included even if they miss a threshold (marked in `meets_thresholds`).
    """
    ok = (cov["observed_days"] >= min_days) & (cov["longest_gap_days"] <= max_gap)
    t = cov[ok | cov["market_key"].isin(always_show)].copy()
    t["meets_thresholds"] = ok[t.index]
    return t.sort_values("median_arrivals_t", ascending=False).reset_index(drop=True)
