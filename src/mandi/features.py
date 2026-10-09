"""Phase 3: forecast origins, targets, baselines and features on a calendar-day grid.

Definitions (see configs/config.yaml):
* A forecast origin is a calendar day on which a market has an observed quote.
* The target for horizon h is the quote at origin + h calendar days; it is scored only if it exists.
* Every feature at an origin uses data on or before that day only (arrivals: strictly before).
  "Lag k" means "the last observed quote at or before origin - k", never "the value exactly k days ago".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

SEASONAL_OFFSETS = (0, -1, 1, -2, 2, -3, 3)  # nearest first; on a tie the earlier date wins
FIXED_HOLIDAYS = {(1, 26), (5, 1), (8, 15)}  # Republic Day, Maharashtra Day, Independence Day (EDA section 1)


@dataclass
class Arrays:
    """Market x day matrices on a common calendar grid. Missing days are NaN; nothing is filled."""

    days: pd.DatetimeIndex
    markets: list[str]
    short: list[str]
    logp: np.ndarray       # log modal price
    larr: np.ndarray       # log arrivals (tonnes)
    stale: np.ndarray      # is_stale flag (False where no quote)


def short_name(market_key: str) -> str:
    """Short market label used in tables."""
    n = market_key.split("_")[-1]
    return "Sambhajinagar" if n.startswith("Chattrapati") else n.replace(" New Mumbai", "")


def build_arrays(panel: pd.DataFrame, start: str, end: str, markets: list[str] | None = None) -> Arrays:
    """Arrange the Phase 1 panel into market x day arrays between `start` and `end` (inclusive)."""
    days = pd.date_range(start, end, freq="D")
    markets = markets or list(dict.fromkeys(panel["market_key"]))
    logp = np.full((len(markets), len(days)), np.nan)
    larr = np.full_like(logp, np.nan)
    stale = np.zeros(logp.shape, dtype=bool)
    for i, mk in enumerate(markets):
        g = panel[panel["market_key"] == mk].set_index("date").reindex(days)
        price = g["modal_price"].to_numpy(dtype=float)
        logp[i] = np.where(price > 0, np.log(np.where(price > 0, price, 1.0)), np.nan)
        arr = g["arrivals_tonnes"].to_numpy(dtype=float)
        larr[i] = np.where(arr > 0, np.log(np.where(arr > 0, arr, 1.0)), np.nan)
        stale[i] = g["is_stale"].fillna(False).to_numpy(dtype=bool) & ~np.isnan(logp[i])
    return Arrays(days, markets, [short_name(m) for m in markets], logp, larr, stale)


def closure_weekdays(panel: pd.DataFrame, train_end: str, threshold: float = 0.9) -> dict[str, int | None]:
    """Weekday (0=Mon) on which a market is almost always closed, learned from data up to `train_end` only."""
    p = panel[panel["date"] <= train_end]
    out: dict[str, int | None] = {}
    for mk, g in p.groupby("market_key"):
        share = g.groupby(g["date"].dt.dayofweek)["was_missing"].mean()
        out[mk] = int(share.idxmax()) if share.max() > threshold else None
    return out


def last_observed(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each market and day j: value and index of the last non-NaN entry at or before j (NaN / -1 if none)."""
    d = x.shape[1]
    idx = np.where(~np.isnan(x), np.arange(d)[None, :], -1)
    last_idx = np.maximum.accumulate(idx, axis=1)
    val = np.where(last_idx >= 0, np.take_along_axis(x, np.maximum(last_idx, 0), axis=1), np.nan)
    return val, last_idx


def _window_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cumulative sums (value, value^2, count) with a leading zero column, ignoring NaN."""
    ok = ~np.isnan(x)
    z = np.zeros((x.shape[0], 1))
    return (np.concatenate([z, np.cumsum(np.where(ok, x, 0.0), axis=1)], axis=1),
            np.concatenate([z, np.cumsum(np.where(ok, x * x, 0.0), axis=1)], axis=1),
            np.concatenate([z, np.cumsum(ok.astype(float), axis=1)], axis=1))


def _win(cum: np.ndarray, m: int, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Sum over days lo..hi inclusive (lo clipped at 0) from a cumulative array with a leading zero."""
    lo = np.maximum(lo, 0)
    return cum[m, hi + 1] - cum[m, lo]


def build_origin_table(a: Arrays, h: int, closure: dict[str, int | None], params: dict[str, Any]) -> pd.DataFrame:
    """One row per (market, origin) with baselines, features and the realised target for horizon `h`.

    Every row is a forecast origin (an observed quote). `y` is NaN when the target quote does not exist; such
    rows are kept so the caller can decide, but they are never scored. Features use data at or before the
    origin only (arrivals strictly before it).
    """
    n_m, n_d = a.logp.shape
    lastval, lastidx = last_observed(a.logp)
    larr_val, larr_idx = last_observed(a.larr)
    cs, cs2, cc = _window_stats(a.logp)
    # consecutive observed log differences, placed on the later quote's day
    prev_idx_all = np.concatenate([np.full((n_m, 1), -1), lastidx[:, :-1]], axis=1)
    prev_val_all = np.concatenate([np.full((n_m, 1), np.nan), lastval[:, :-1]], axis=1)
    dlog = np.where(~np.isnan(a.logp), a.logp - prev_val_all, np.nan)
    ds, ds2, dc = _window_stats(dlog)
    asum, _, acnt = _window_stats(a.larr)
    stale_days = params["max_stale_days_other_market"]
    seas_len, seas_tol = params["seasonal_period_days"], params["seasonal_tolerance_days"]
    frames = []
    for m in range(n_m):
        I = np.flatnonzero(~np.isnan(a.logp[m]))
        p0 = a.logp[m, I]
        f: dict[str, np.ndarray] = {"origin_idx": I, "p0": p0}
        # lagged log changes from the last quote at or before origin - k
        for k in params["lags_days"]:
            j = I - k
            ok = j >= 0
            lv = np.where(ok, lastval[m, np.maximum(j, 0)], np.nan)
            f[f"chg_{k}"] = p0 - lv
        # previous observed quotes (before the origin)
        prev1 = np.where(I >= 1, lastidx[m, np.maximum(I - 1, 0)], -1)
        prev2 = np.where(prev1 >= 1, lastidx[m, np.maximum(prev1 - 1, 0)], -1)
        f["gap_prev"] = np.where(prev1 >= 0, I - prev1, np.nan)
        v1 = np.where(prev1 >= 0, a.logp[m, np.maximum(prev1, 0)], np.nan)
        v2 = np.where(prev2 >= 0, a.logp[m, np.maximum(prev2, 0)], np.nan)
        last3_log = np.stack([p0, v1, v2])
        f["dev3"] = p0 - last3_log.mean(axis=0)  # NaN unless three quotes exist
        # baseline (b): arithmetic mean of the last 3 observed quotes (rupee scale), then log
        f["base_mean3"] = np.log(np.exp(last3_log).mean(axis=0))
        # 30-day mean and volatility
        w = params["window_days"]
        lo = I - (w - 1)
        n30 = _win(cc, m, lo, I)
        f["dev30"] = np.where(n30 >= 5, p0 - _win(cs, m, lo, I) / np.maximum(n30, 1), np.nan)
        nd = _win(dc, m, lo, I)
        mean_d = _win(ds, m, lo, I) / np.maximum(nd, 1)
        var_d = _win(ds2, m, lo, I) / np.maximum(nd, 1) - mean_d ** 2
        f["vol30"] = np.where(nd >= 4, np.sqrt(np.maximum(var_d, 0)), np.nan)
        f["n_obs_14"] = _win(cc, m, I - 13, I)
        # other markets (their latest quotes at or before the origin day; stale ones are ignored)
        for k in params["cross_market_lags_days"]:
            vals = []
            for o in range(n_m):
                if o == m:
                    continue
                fresh_now = (I - lastidx[o, I] <= stale_days) & (lastidx[o, I] >= 0)
                j = np.maximum(I - k, 0)
                fresh_then = (I - k >= 0) & ((I - k) - lastidx[o, j] <= stale_days) & (lastidx[o, j] >= 0)
                vals.append(np.where(fresh_now & fresh_then, lastval[o, I] - lastval[o, j], np.nan))
            arr = np.stack(vals)
            f[f"xm_chg_{k}"] = np.where((~np.isnan(arr)).sum(axis=0) >= 2, _nanmean0(arr), np.nan)
        lev = []
        for o in range(n_m):
            if o == m:
                continue
            fresh_now = (I - lastidx[o, I] <= stale_days) & (lastidx[o, I] >= 0)
            lev.append(np.where(fresh_now, lastval[o, I], np.nan))
        lev = np.stack(lev)
        f["rel_level"] = np.where((~np.isnan(lev)).sum(axis=0) >= 2, p0 - _nanmean0(lev), np.nan)
        # arrivals: strictly before the origin
        jprev = np.maximum(I - 1, 0)
        f["arr_prev"] = np.where(I >= 1, larr_val[m, jprev], np.nan)
        n_a = _win(acnt, m, I - 14, I - 1)
        f["arr_mean14"] = np.where((n_a >= 3) & (I >= 1), _win(asum, m, I - 14, I - 1) / np.maximum(n_a, 1), np.nan)
        f["arr_dev"] = f["arr_prev"] - f["arr_mean14"]
        tab = pd.DataFrame(f)
        tab["market_idx"] = m
        tab["h"] = h
        # target
        J = I + h
        in_range = J < n_d
        tgt = np.where(in_range, a.logp[m, np.minimum(J, n_d - 1)], np.nan)
        tab["y_log"] = tgt
        tab["y_change"] = tgt - p0
        # baseline (c): nearest observed quote within +-tol days of the target date one period earlier
        ref = J - seas_len
        fc = np.full(len(I), np.nan)
        for off in SEASONAL_OFFSETS:
            if abs(off) > seas_tol:
                continue
            jj = ref + off
            ok = (jj >= 0) & (jj < n_d) & np.isnan(fc)
            v = np.where(ok, a.logp[m, np.clip(jj, 0, n_d - 1)], np.nan)
            fc = np.where(np.isnan(fc) & ~np.isnan(v), v, fc)
        tab["base_seasonal"] = fc
        tab["base_naive"] = p0
        # calendar of the target date
        tdate = a.days[I] + pd.Timedelta(days=h)
        tab["target_month"] = tdate.month.to_numpy()
        tab["target_woy"] = tdate.isocalendar().week.to_numpy().astype(int)
        tab["target_dow"] = tdate.dayofweek.to_numpy()
        tab["origin_dow"] = a.days[I].dayofweek.to_numpy()
        clo = closure.get(a.markets[m])
        hol = np.array([(d.month, d.day) in FIXED_HOLIDAYS for d in tdate])
        tab["target_expected_closed"] = ((tdate.dayofweek.to_numpy() == clo) if clo is not None else False) | hol
        tab["target_expected_closed"] = tab["target_expected_closed"].astype(int)
        tab["stale_origin"] = a.stale[m, I]
        tab["stale_target"] = np.where(in_range, a.stale[m, np.minimum(J, n_d - 1)], False)
        frames.append(tab)
    out = pd.concat(frames, ignore_index=True)
    out["market"] = [a.short[m] for m in out["market_idx"]]
    out["origin"] = a.days[out["origin_idx"].to_numpy()]
    out["target_date"] = out["origin"] + pd.Timedelta(days=h)
    return out


def _nanmean0(arr: np.ndarray) -> np.ndarray:
    """Column-wise mean ignoring NaN (NaN if a column is all NaN)."""
    cnt = (~np.isnan(arr)).sum(axis=0)
    tot = np.where(np.isnan(arr), 0.0, arr).sum(axis=0)
    return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)


FEATURE_COLUMNS = [
    "chg_1", "chg_3", "chg_7", "chg_14", "chg_28", "dev3", "dev30", "vol30", "gap_prev", "n_obs_14",
    "xm_chg_7", "xm_chg_28", "rel_level", "arr_prev", "arr_mean14", "arr_dev",
    "target_month", "target_woy", "target_dow", "origin_dow", "target_expected_closed", "market_idx",
]
