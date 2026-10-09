"""Phase 2: exploratory analysis functions. Each returns tables; plotting lives in eda_plots.py.

Rule: anything that drives a modelling choice is computed on dates <= eda.train_end (2019-12-31).
Functions take a `train` flag or an already-sliced panel so that this is explicit at every call site.
"""
from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import breaks_cusumolsresid
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, adfuller, kpss, levinson_durbin

from mandi.clean import fill_short_gaps

logger = logging.getLogger(__name__)

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def short_name(market_key: str) -> str:
    """Last part of a market_key, e.g. 'Maharashtra_Pune_Pune' -> 'Pune'."""
    n = market_key.split("_")[-1]
    return "Sambhajinagar" if n.startswith("Chattrapati") else n.replace(" New Mumbai", "")


def load_panel(path: Any) -> pd.DataFrame:
    """Read the Phase 1 panel and add a short market name column `m` and weekday."""
    p = pd.read_parquet(path)
    p["m"] = p["market_key"].map(short_name)
    p["weekday"] = p["date"].dt.dayofweek
    return p


# --------------------------------------------------------------------------- weekly series
def weekly_log_series(panel: pd.DataFrame, col: str, min_days: int) -> pd.DataFrame:
    """Weekly (week ending Sunday) mean of log(col) over *observed* days; wide, one column per market.

    A week needs at least `min_days` observed days, otherwise it is NaN. No day is filled.
    """
    cols = {}
    for m, g in panel.groupby("m"):
        s = np.log(g.set_index("date")[col].where(lambda x: x > 0))
        w = s.resample("W-SUN").agg(["mean", "count"])
        w.loc[w["count"] < min_days, "mean"] = np.nan
        cols[m] = w["mean"]
    return pd.DataFrame(cols)


def fill_weekly_gaps(w: pd.DataFrame, max_weeks: int) -> pd.DataFrame:
    """Forward-fill weekly gaps of at most `max_weeks` weeks (EDA tests only); longer gaps stay missing."""
    out = {}
    for c in w.columns:
        first = w[c].first_valid_index()
        s = w[c].loc[first:]
        out[c] = fill_short_gaps(s, max_weeks)[0]
    return pd.DataFrame(out).reindex(w.index)


def longest_segment(s: pd.Series) -> pd.Series:
    """Longest run of consecutive non-missing values."""
    v = s.notna()
    run = (v != v.shift()).cumsum()
    lens = v.groupby(run).sum().where(v.groupby(run).first(), 0)
    best = lens.idxmax()
    return s[run == best]


def closure_weekday(panel: pd.DataFrame, threshold: float = 0.9) -> pd.Series:
    """Weekday on which a market is (almost) always closed: missing share above `threshold`, else NaN."""
    share = panel.groupby(["m", "weekday"])["was_missing"].mean().unstack()
    return share.idxmax(axis=1).where(share.max(axis=1) > threshold)


# --------------------------------------------------------------------------- stationarity, ACF, PACF
def stationarity_table(w: pd.DataFrame) -> pd.DataFrame:
    """ADF and KPSS on weekly log price, in levels and first differences.

    Primary series: the longest contiguous run of weeks (after the 1-week fill). Robustness:
    all valid weeks joined end to end (gaps dropped). ADF null = unit root; KPSS null = stationary.
    """
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message="The test statistic is outside")
    rows = []
    for m in w.columns:
        for name, s in [("longest segment", longest_segment(w[m])), ("joined valid weeks", w[m].dropna())]:
            for tr, x in [("level", s), ("first difference", s.diff().dropna())]:
                if len(x) < 40:
                    continue
                adf = adfuller(x, autolag="AIC")
                kp = kpss(x, regression="c", nlags="auto")
                rows.append({"market": m, "series": name, "transform": tr, "weeks": len(x),
                             "ADF_p": round(adf[1], 4), "KPSS_p": round(kp[1], 4),
                             "reading": _reading(adf[1], kp[1])})
    return pd.DataFrame(rows)


def _reading(adf_p: float, kpss_p: float) -> str:
    """Combine the two tests (5% level). KPSS p-values are clipped to [0.01, 0.1] by statsmodels."""
    adf_stat, kpss_stat = adf_p < 0.05, kpss_p >= 0.05
    if adf_stat and kpss_stat:
        return "stationary (both agree)"
    if not adf_stat and not kpss_stat:
        return "non-stationary (both agree)"
    return "ADF rejects unit root but KPSS rejects stationarity" if adf_stat else "ADF cannot reject unit root, KPSS does not reject stationarity"


def pairwise_acf_pacf(x: pd.Series, nlags: int) -> pd.DataFrame:
    """ACF using only available pairs (statsmodels 'conservative'), and PACF from that ACF via Levinson-Durbin."""
    a = acf(x.values, nlags=nlags, missing="conservative", fft=False)
    pacf = levinson_durbin(a, nlags=nlags, isacov=True)[2]  # isacov=True: `a` is an autocovariance sequence, not a series
    return pd.DataFrame({"lag": np.arange(nlags + 1), "acf": a, "pacf": pacf, "n_valid": int(x.notna().sum())})


def daily_pairwise_acf(panel: pd.DataFrame, nlags: int = 14) -> pd.DataFrame:
    """ACF of daily log price changes using only pairs of *actually observed* days (no filled values)."""
    rows = []
    for m, g in panel.groupby("m"):
        s = np.log(g.set_index("date")["modal_price"])
        d1 = s.diff()  # defined only when both adjacent calendar days are observed
        x = d1.dropna()
        for k in range(1, nlags + 1):
            a, b = d1, d1.shift(k)
            ok = a.notna() & b.notna()
            rows.append({"market": m, "lag_days": k, "acf": float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 30 else np.nan,
                         "pairs": int(ok.sum())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- seasonality
def weekday_effect(panel: pd.DataFrame) -> pd.DataFrame:
    """Mean log price (and log arrivals) by weekday relative to the same week's mean of observed days.

    Weeks run Monday to Sunday; the effect is averaged over weeks with at least 4 observed days.
    Reports mean effect in % and an approximate 95% interval (2 standard errors across weeks).
    """
    rows = []
    for m, g in panel.groupby("m"):
        g = g.dropna(subset=["modal_price"]).copy()
        g["lp"] = np.log(g["modal_price"])
        g["la"] = np.log(g["arrivals_tonnes"].where(g["arrivals_tonnes"] > 0))
        g["week"] = g["date"].dt.to_period("W-SUN")
        g = g[g.groupby("week")["lp"].transform("size") >= 4]
        g["dev_p"] = g["lp"] - g.groupby("week")["lp"].transform("mean")
        g["dev_a"] = g["la"] - g.groupby("week")["la"].transform("mean")
        for wd, h in g.groupby("weekday"):
            if len(h) < 30:  # the market's closure weekday: a handful of exceptional openings, not an effect
                continue
            se_p = h["dev_p"].std() / np.sqrt(len(h))
            rows.append({"market": m, "weekday": WEEKDAYS[wd], "n_days": len(h),
                         "price_effect_pct": round(100 * (np.exp(h["dev_p"].mean()) - 1), 2),
                         "price_ci_lo_pct": round(100 * (np.exp(h["dev_p"].mean() - 2 * se_p) - 1), 2),
                         "price_ci_hi_pct": round(100 * (np.exp(h["dev_p"].mean() + 2 * se_p) - 1), 2),
                         "arrivals_effect_pct": round(100 * (np.exp(h["dev_a"].mean()) - 1), 1)})
    return pd.DataFrame(rows)


def monthly_log_series(panel: pd.DataFrame, min_days: int = 10) -> pd.DataFrame:
    """Monthly mean of log modal price over observed days; months with fewer than `min_days` observed days are NaN."""
    cols = {}
    for m, g in panel.groupby("m"):
        s = np.log(g.set_index("date")["modal_price"])
        r = s.resample("MS").agg(["mean", "count"])
        r.loc[r["count"] < min_days, "mean"] = np.nan
        cols[m] = r["mean"]
    return pd.DataFrame(cols)


def yearly_seasonality(monthly: pd.DataFrame) -> pd.DataFrame:
    """STL (period 12, robust) on monthly log prices: seasonal strength, plus month-profile consistency across years.

    Seasonal strength F_S = max(0, 1 - Var(remainder) / Var(seasonal + remainder)) (Hyndman). Consistency is the
    mean pairwise correlation of the 12-month profiles (each year minus its own mean) between years with >= 10 months.
    """
    rows = []
    for m in monthly.columns:
        s = monthly[m].dropna()
        full = monthly[m].loc[s.index.min(): s.index.max()]
        fill = full.interpolate(limit_area="inside", limit=1)  # at most one missing month bridged, EDA only
        seg = longest_segment(fill)
        if len(seg) < 36:
            rows.append({"market": m, "months_used": len(seg)})
            continue
        res = STL(seg, period=12, robust=True).fit()
        fs = max(0.0, 1 - res.resid.var() / (res.seasonal + res.resid).var())
        prof = full.to_frame("v").assign(y=full.index.year, mo=full.index.month)
        prof["dev"] = prof["v"] - prof.groupby("y")["v"].transform("mean")
        piv = prof.pivot(index="mo", columns="y", values="dev")
        piv = piv.loc[:, piv.notna().sum() >= 10]
        c = piv.corr().values
        cons = float(np.nanmean(c[np.triu_indices_from(c, 1)])) if piv.shape[1] > 1 else np.nan
        avg = piv.mean(axis=1)
        peaks = {int(y): int(piv[y].idxmax()) for y in piv.columns}
        troughs = {int(y): int(piv[y].idxmin()) for y in piv.columns}
        rows.append({"market": m, "months_used": len(seg), "seasonal_strength_FS": round(fs, 3),
                     "peak_month_by_year": ", ".join(f"{y}:{mo}" for y, mo in peaks.items()),
                     "trough_month_by_year": ", ".join(f"{y}:{mo}" for y, mo in troughs.items()),
                     "avg_profile_range_pct": round(100 * (np.exp(avg.max() - avg.min()) - 1), 1),
                     "years_in_consistency": piv.shape[1], "mean_year_to_year_profile_corr": round(cons, 2)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- price vs arrivals, co-movement
def price_arrivals(panel: pd.DataFrame, min_days: int, max_lag: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Weekly correlation of log price with log arrivals (levels and changes), and lagged cross-correlation.

    The lagged table uses weekly *changes* in price against changes in arrivals `k` weeks earlier (k >= 0).
    Arrivals on the same day are not known when forecasting, so only k >= 1 would be usable as a feature.
    """
    wp = weekly_log_series(panel, "modal_price", min_days)
    wa = weekly_log_series(panel, "arrivals_tonnes", min_days)
    s_rows, l_rows = [], []
    for m in wp.columns:
        p, a = wp[m], wa[m]
        ok = p.notna() & a.notna()
        dp, da = p.diff(), a.diff()
        s_rows.append({"market": m, "weeks": int(ok.sum()), "corr_levels": round(p[ok].corr(a[ok]), 3),
                       "corr_changes_same_week": round(dp.corr(da), 3)})
        for k in range(0, max_lag + 1):
            l_rows.append({"market": m, "arrivals_lag_weeks": k, "corr_price_change_vs_lagged_arrivals_change": round(dp.corr(da.shift(k)), 3)})
    return pd.DataFrame(s_rows), pd.DataFrame(l_rows)


def comovement(panel: pd.DataFrame, min_days: int) -> dict[str, pd.DataFrame]:
    """Correlation of weekly log prices (levels and changes) between markets, plus changes cross-correlation at lags -2..2."""
    w = weekly_log_series(panel, "modal_price", min_days)
    out = {"levels": w.corr(min_periods=30).round(3), "changes": w.diff().corr(min_periods=30).round(3)}
    d = w.diff()
    rows = []
    cols = list(w.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            rows.append({"pair": f"{a} vs {b}", **{f"lag {k:+d}": round(d[a].corr(d[b].shift(-k)), 3) for k in range(-2, 3)}})
    out["lead_lag_changes"] = pd.DataFrame(rows)
    return out


# --------------------------------------------------------------------------- volatility, transformation, breaks
def yearly_summary(panel: pd.DataFrame, min_days: int) -> pd.DataFrame:
    """Per market and year: median price, interquartile range, share of days, and std of weekly log changes."""
    w = weekly_log_series(panel, "modal_price", min_days).diff()
    rows = []
    for m, g in panel.groupby("m"):
        for y, h in g.groupby(g["date"].dt.year):
            x = h["modal_price"].dropna()
            if len(x) < 30:
                continue
            ws = w[m][w.index.year == y].dropna()
            rows.append({"market": m, "year": y, "observed_days": len(x), "median_price": round(x.median()),
                         "p10": round(x.quantile(.1)), "p90": round(x.quantile(.9)), "max_price": round(x.max()),
                         "weekly_log_change_std": round(ws.std(), 3)})
    return pd.DataFrame(rows)


def transform_check(panel: pd.DataFrame) -> pd.DataFrame:
    """Does variability grow with the price level? Skewness of price vs log price, and the slope of
    log(rolling std of daily price changes) on log(rolling mean price) over 60 observed-day windows (slope ~1 -> multiplicative -> log)."""
    rows = []
    for m, g in panel.groupby("m"):
        x = g.set_index("date")["modal_price"].dropna()
        d = x.diff().abs()
        roll_mean = x.rolling(60, min_periods=40).mean()
        roll_sd = x.diff().rolling(60, min_periods=40).std()
        ok = (roll_mean > 0) & (roll_sd > 0)
        slope = np.polyfit(np.log(roll_mean[ok]), np.log(roll_sd[ok]), 1)[0]
        rows.append({"market": m, "skew_price": round(x.skew(), 2), "skew_log_price": round(np.log(x).skew(), 2),
                     "slope_log_sd_on_log_mean": round(slope, 2), "windows": int(ok.sum())})
    return pd.DataFrame(rows)


def cusum_ar1(w: pd.DataFrame) -> pd.DataFrame:
    """CUSUM test of parameter stability for an AR(1) fitted to weekly log price (consecutive valid weeks only)."""
    rows = []
    for m in w.columns:
        y = w[m]
        df = pd.DataFrame({"y": y, "x": y.shift(1)}).dropna()
        if len(df) < 60:
            continue
        X = np.column_stack([np.ones(len(df)), df["x"].values])
        beta, *_ = np.linalg.lstsq(X, df["y"].values, rcond=None)
        resid = df["y"].values - X @ beta
        stat, p, _ = breaks_cusumolsresid(resid, ddof=2)
        rows.append({"market": m, "weeks": len(df), "AR1_coef": round(beta[1], 3), "CUSUM_stat": round(float(stat), 3), "CUSUM_p": round(float(p), 4)})
    return pd.DataFrame(rows)


def event_windows(panel: pd.DataFrame, events: list[dict[str, Any]], weeks: int = 4) -> pd.DataFrame:
    """Descriptive only: median price in the `weeks` weeks before vs after each event, per market. No causal claim."""
    rows = []
    for e in events:
        d = pd.Timestamp(e["date"])
        for m, g in panel.groupby("m"):
            s = g.set_index("date")["modal_price"].dropna()
            b = s[(s.index >= d - pd.Timedelta(weeks=weeks)) & (s.index < d)]
            a = s[(s.index >= d) & (s.index < d + pd.Timedelta(weeks=weeks))]
            if len(b) >= 10 and len(a) >= 10:
                rows.append({"event": e["id"], "date": e["date"], "market": m, "median_before": round(b.median()),
                             "median_after": round(a.median()), "change_pct": round(100 * (a.median() / b.median() - 1), 1)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- missingness
def missingness(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Missingness per market: by weekday, month, year; structural (closure weekday) vs unexpected missing days.

    `unexpected` = missing on a weekday other than the market's closure weekday.
    """
    clo = closure_weekday(panel)
    p = panel.copy()
    p["closure_day"] = p["m"].map(clo)
    p["is_closure"] = p["weekday"] == p["closure_day"]
    p["unexpected_missing"] = p["was_missing"] & ~p["is_closure"]
    out = {"closure": clo.map(lambda i: WEEKDAYS[int(i)] if pd.notna(i) else "none").rename("closure_weekday").to_frame()}
    out["by_weekday"] = p.groupby(["m", "weekday"])["was_missing"].mean().unstack().round(3).rename(columns=dict(enumerate(WEEKDAYS)))
    out["by_month"] = p.groupby(["m", p["date"].dt.month])["unexpected_missing"].mean().unstack().round(3)
    out["by_year"] = p.groupby(["m", p["date"].dt.year])["unexpected_missing"].mean().unstack().round(3)
    tot = p.groupby("m").agg(days=("date", "size"), missing=("was_missing", "sum"), closure_days=("is_closure", "sum"),
                             unexpected=("unexpected_missing", "sum"))
    tot["missing_share"] = (tot["missing"] / tot["days"]).round(3)
    tot["share_of_missing_that_is_closure"] = ((tot["missing"] - tot["unexpected"]) / tot["missing"]).round(3)
    tot["unexpected_share_of_open_days"] = (tot["unexpected"] / (tot["days"] - tot["closure_days"])).round(3)
    out["totals"] = tot
    out["_panel"] = p
    return out


def missing_vs_level(p: pd.DataFrame) -> pd.DataFrame:
    """Is unexpected missingness related to the price or arrivals level?

    For every open (non-closure) day, take the local median of observed log price / log arrivals (15 days before and 15 days after,
    the day itself excluded), cut it into terciles within the market, and report the unexpected-missing rate per tercile.
    Also chi-square p-value for independence.
    """
    from scipy.stats import chi2_contingency

    rows = []
    for m, g in p.groupby("m"):
        g = g.sort_values("date").set_index("date")
        for name, col in [("price", "modal_price"), ("arrivals", "arrivals_tonnes")]:
            x = np.log(g[col].where(g[col] > 0))
            # local level from the 15 days before and the 15 days after, never the day itself
            before = x.shift(1).rolling(15, min_periods=5).median()
            after = x[::-1].shift(1).rolling(15, min_periods=5).median()[::-1]
            loc = pd.concat([before, after], axis=1).mean(axis=1)
            open_days = g[~g["is_closure"]].copy()
            open_days["loc"] = loc.reindex(open_days.index)
            open_days = open_days.dropna(subset=["loc"])
            open_days["tercile"] = pd.qcut(open_days["loc"], 3, labels=["low", "mid", "high"])
            rate = open_days.groupby("tercile", observed=True)["unexpected_missing"].mean()
            tab = pd.crosstab(open_days["tercile"], open_days["unexpected_missing"])
            pval = chi2_contingency(tab)[1] if tab.shape == (3, 2) else np.nan
            rows.append({"market": m, "level_of": name, "low": round(rate.get("low", np.nan), 3), "mid": round(rate.get("mid", np.nan), 3),
                         "high": round(rate.get("high", np.nan), 3), "chi2_p": round(float(pval), 4), "open_days": len(open_days)})
    return pd.DataFrame(rows)


def gap_table(p: pd.DataFrame, min_len: int = 14) -> pd.DataFrame:
    """Runs of missing days of at least `min_len` days, per market, with dates."""
    rows = []
    for m, g in p.groupby("m"):
        g = g.sort_values("date")
        v = g["was_missing"].values
        run = np.cumsum(np.r_[True, v[1:] != v[:-1]])
        for r in np.unique(run[v]):
            idx = np.where((run == r) & v)[0]
            if len(idx) >= min_len:
                rows.append({"market": m, "start": g["date"].iloc[idx[0]].date(), "end": g["date"].iloc[idx[-1]].date(), "days": len(idx)})
    return pd.DataFrame(rows)


def cross_market_missing(p: pd.DataFrame, festival_col: str = "is_festival") -> dict[str, pd.DataFrame]:
    """Do markets miss the same days? Pairwise overlap of unexpected-missing days, and days when many markets are missing."""
    w = p.pivot(index="date", columns="m", values="unexpected_missing")
    open_w = p.pivot(index="date", columns="m", values="is_closure")
    rows = []
    cols = list(w.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            both_open = w[a].notna() & w[b].notna() & ~open_w[a].fillna(True) & ~open_w[b].fillna(True)
            x, y = w.loc[both_open, a].astype(bool), w.loc[both_open, b].astype(bool)
            pa, pb = x.mean(), y.mean()
            joint = (x & y).mean()
            rows.append({"pair": f"{a} & {b}", "days_compared": int(both_open.sum()), "P(a missing)": round(pa, 3), "P(b missing)": round(pb, 3),
                         "P(both missing)": round(joint, 4), "expected_if_independent": round(pa * pb, 4),
                         "ratio_obs_to_expected": round(joint / (pa * pb), 1) if pa * pb > 0 else np.nan,
                         "phi_corr": round(float(np.corrcoef(x, y)[0, 1]), 3)})
    n_missing = w.fillna(False).astype(bool).sum(axis=1)
    open_in_span = (w.notna() & ~open_w.fillna(True).astype(bool)).sum(axis=1)
    many = pd.DataFrame({"markets_unexpectedly_missing": n_missing, "markets_open_in_span": open_in_span})
    many = many[many["markets_unexpectedly_missing"] >= 4]
    many["is_festival_flag"] = p.groupby("date")[festival_col].first().reindex(many.index).fillna(False).astype(bool)
    many.index = many.index.date
    # fixed-date public holidays (calendar knowledge, not looked up): do markets close on them?
    ds = pd.Series(w.index, index=w.index)
    frac = w.fillna(False).astype(bool).sum(axis=1) / open_in_span.replace(0, np.nan)
    hol = {"26 Jan": (1, 26), "1 May": (5, 1), "15 Aug": (8, 15), "2 Oct": (10, 2)}
    hrows = []
    for name, (mo, d) in hol.items():
        sel = (ds.dt.month == mo) & (ds.dt.day == d) & (open_in_span >= 3)
        hrows.append({"date": name, "days_observed": int(sel.sum()), "mean_share_of_open_markets_missing": round(float(frac[sel].mean()), 2)})
    base = float(frac[open_in_span >= 3].mean())
    hrows.append({"date": "all other days (baseline)", "days_observed": int((open_in_span >= 3).sum()), "mean_share_of_open_markets_missing": round(base, 2)})
    fest_open = p[~p["is_closure"]]
    isf = fest_open[festival_col].fillna(False).astype(bool)
    fest_tab = pd.DataFrame({"unexpected_missing_rate": {"festival day (file flag)": round(fest_open[isf]["unexpected_missing"].mean(), 3),
                                                        "other open days": round(fest_open[~isf]["unexpected_missing"].mean(), 3)}})
    return {"pairs": pd.DataFrame(rows), "many_markets_missing": many.reset_index(names="date"), "festival": fest_tab,
            "holidays": pd.DataFrame(hrows)}
