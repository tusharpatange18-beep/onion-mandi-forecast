"""Phase 4B: anomaly detection in modal prices.

Detector 1, `stl_residual`: a RETROSPECTIVE LARGE-SWING DETECTOR, NOT A SPIKE DETECTOR (historical labelling: STL uses data after the
date as well, so it cannot be an alert at the time): STL (robust, period 52) on the weekly mean of log modal price; a week is flagged when
|resid - median(resid)| > mad_threshold * mad_scale * MAD(resid). Weeks without enough observed days are bridged by linear
interpolation only to let STL run; such weeks are never flagged and never used for the median and MAD.

Detector 2, `forecast_surprise_h{h}` (usable in real time): the realised quote falls outside the 95% conformal interval of the
headline LightGBM that was issued h days earlier.

Detector 3, `causal_daily` (usable in real time; declared after the first Phase 4 review, run once, not tuned): on OBSERVED quotes
only, baseline = median of the log price of the previous 28 observed quotes; scale = 1.4826 x MAD of the previous 90 observed daily
log changes; flag if |log p_t - baseline| > 4 x scale. Uses only data before the quote being judged.

No causes are claimed anywhere: the output only says when, where, how large, and what arrivals did.

Caveat measured on synthetic noise: with robust=True the STL residuals are heavy-tailed even for Gaussian noise
(kurtosis about 4) and 1.4826 * MAD underestimates their spread, so the nominal "3.5 robust sigma" threshold flags
roughly 5% of CLEAN weeks (1-11% across random series). The false-positive rate reported in the validation is the number to use.
"""
from __future__ import annotations

import logging
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from statsmodels.tsa.seasonal import STL  # noqa: E402

from mandi import evaluate as E  # noqa: E402
from mandi import features as F  # noqa: E402
from mandi import intervals as IV  # noqa: E402
from mandi.config import ROOT, load_config  # noqa: E402

logger = logging.getLogger(__name__)
SYNTHETIC_LABEL = "synthetic test of the detector, not data"


# --------------------------------------------------------------------------- detector 1: STL residual
def weekly_series(panel_m: pd.DataFrame, min_days: int) -> pd.DataFrame:
    """Weekly (week ending Sunday) mean of log modal price and of log arrivals over OBSERVED days; NaN if fewer than `min_days`."""
    g = panel_m.set_index("date")
    lp = np.log(g["modal_price"].where(g["modal_price"] > 0))
    la = np.log(g["arrivals_tonnes"].where(g["arrivals_tonnes"] > 0))
    w = pd.DataFrame({"lp": lp.resample("W-SUN").mean(), "n": lp.resample("W-SUN").count(), "la": la.resample("W-SUN").mean()})
    w.loc[w["n"] < min_days, ["lp", "la"]] = np.nan
    w.index.name = "week_end"
    return w


def robust_flags(resid: pd.Series, observed: pd.Series, threshold: float, mad_scale: float) -> pd.DataFrame:
    """Robust z-score of residuals, with the median and MAD taken over observed points only; flag |score| > threshold."""
    r = resid[observed]
    med = r.median()
    mad = (r - med).abs().median()
    score = (resid - med) / (mad_scale * mad) if mad > 0 else pd.Series(np.nan, index=resid.index)
    flag = observed & (score.abs() > threshold)
    return pd.DataFrame({"resid": resid, "score": score, "flag": flag})


def stl_detect(weekly: pd.DataFrame, cfg: dict[str, Any], robust: bool | None = None) -> pd.DataFrame:
    """STL residual detector on one market's weekly series. Returns a frame indexed by week_end with score, flag, expected (rupees).

    `robust` overrides the configured value; it is used only for the labelled diagnostic comparison in the synthetic validation."""
    ac = cfg["anomalies"]
    s = weekly["lp"]
    first, last = s.first_valid_index(), s.last_valid_index()
    s = s.loc[first:last]
    observed = s.notna()
    filled = s.interpolate(limit_area="inside")               # only so that STL can run; never flagged or used for the MAD
    fit = STL(filled, period=ac["stl"]["period"], robust=ac["stl"]["robust"] if robust is None else robust).fit()
    out = robust_flags(fit.resid, observed, ac["mad_threshold"], ac["mad_scale"])
    out["expected"] = np.exp(fit.trend + fit.seasonal)
    out["price"] = np.exp(s)
    out["observed"] = observed
    return out


# --------------------------------------------------------------------------- detector 3: causal daily
def causal_daily_detect(g: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Causal daily detector on one market's observed quotes (see the module docstring).

    `g` has `date` and `modal_price` (NaN on days without a quote). For the t-th observed quote the baseline is the median of
    the log prices of quotes t-28..t-1 and the scale is 1.4826 x MAD of the 90 observed log changes up to the change into quote
    t-1; nothing from quote t or later is used. Quotes without 90 prior changes are not scored (NaN).
    """
    cd, ms = cfg["anomalies"]["causal_daily"], cfg["anomalies"]["mad_scale"]
    nb, ns, k = cd["baseline_quotes"], cd["scale_changes"], cd["k"]
    q = g[g["modal_price"] > 0][["date", "modal_price"]].reset_index(drop=True)
    lp = np.log(q["modal_price"].to_numpy())
    n = len(lp)
    score, base = np.full(n, np.nan), np.full(n, np.nan)
    if n >= ns + 2:
        c = np.diff(lp)                                              # c[j] = change into quote j + 1
        t = np.arange(ns + 1, n)
        b = np.median(np.lib.stride_tricks.sliding_window_view(lp, nb)[t - nb], axis=1)          # quotes t-nb .. t-1
        w = np.lib.stride_tricks.sliding_window_view(c, ns)[t - ns - 1]                          # changes up to the change into quote t-1
        med = np.median(w, axis=1)
        scale = ms * np.median(np.abs(w - med[:, None]), axis=1)
        ok = scale > 0
        sc = np.full(len(t), np.nan)
        sc[ok] = (lp[t][ok] - b[ok]) / scale[ok]
        score[t], base[t] = sc, b
    flag = np.where(np.isnan(score), False, np.abs(np.nan_to_num(score)) > k)
    return pd.DataFrame({"date": q["date"], "price": np.exp(lp), "expected": np.exp(base), "score": score, "flag": flag,
                         "direction": np.where(score > 0, "up", "down")})


# --------------------------------------------------------------------------- shared helpers
def arrivals_z(daily: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, window_days: int, min_obs: int = 8) -> float:
    """z-score of the mean log arrivals on observed days in [start, end] against the median and MAD of the log arrivals
    on observed days in the `window_days` before `start` (the trailing 30-day median)."""
    la = np.log(daily["arrivals_tonnes"].where(daily["arrivals_tonnes"] > 0))
    cur = la.loc[start:end].dropna()
    ref = la.loc[start - pd.Timedelta(days=window_days): start - pd.Timedelta(days=1)].dropna()
    if cur.empty or len(ref) < min_obs:
        return np.nan
    med = ref.median()
    mad = (ref - med).abs().median()
    if not mad > 0:
        return np.nan
    return float((cur.mean() - med) / (1.4826 * mad))


def label_periods(dates: pd.Series, cfg: dict[str, Any]) -> pd.Series:
    """Event-period label using only the high-confidence periods listed in the config; 'none' outside them."""
    keep = set(cfg["anomalies"]["high_confidence_periods"])
    periods = [p for p in cfg["forecast"]["event_periods"] if p["name"] in keep]
    return E.assign_period(dates, periods).replace("normal", "none")


def week_end(d: pd.Series) -> pd.Series:
    """Sunday that ends the Monday-Sunday week containing each date."""
    d = pd.to_datetime(d)
    return (d + pd.to_timedelta(6 - d.dt.dayofweek, unit="D")).dt.normalize()


# --------------------------------------------------------------------------- detector 2: forecast surprise
def surprise_flags(intervals: pd.DataFrame, level: float, horizons: list[int]) -> pd.DataFrame:
    """Forecasts whose realised quote fell outside the `level` interval issued h days earlier."""
    p = int(level * 100)
    rows = []
    for h in horizons:
        g = intervals[intervals["h"] == h].dropna(subset=[f"lgbm_abs_lo_{p}"])
        half = (g[f"lgbm_abs_hi_{p}"] - g[f"lgbm_abs_lo_{p}"]) / 2
        out = (g["y_log"] < g[f"lgbm_abs_lo_{p}"]) | (g["y_log"] > g[f"lgbm_abs_hi_{p}"])
        f = g[out]
        rows.append(pd.DataFrame({"market": f["market"], "date": f["target_date"], "detector": f"forecast_surprise_h{h}",
                                  "score": ((f["y_log"] - f["f_lgbm"]) / half[out]), "direction": np.where(f["y_log"] > f["f_lgbm"], "up", "down"),
                                  "price": np.exp(f["y_log"]), "expected": np.exp(f["f_lgbm"])}))
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------- real data: the anomaly table
def detect_all(cfg: dict[str, Any], panel: pd.DataFrame, intervals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Both detectors on the real series. Returns (anomalies table, all-observed-weeks table for baselines)."""
    ac = cfg["anomalies"]
    panel = panel.assign(market=panel["market_key"].map(F.short_name))
    flagged, weeks = [], []
    for m, g in panel.groupby("market"):
        w = weekly_series(g, ac["stl"]["min_days_per_week"])
        d = stl_detect(w, cfg)
        weeks.append(d.assign(market=m, la=w["la"].reindex(d.index)))
        f = d[d["flag"]]
        flagged.append(pd.DataFrame({"market": m, "date": f.index, "detector": "stl_residual", "score": f["score"].to_numpy(),
                                     "direction": np.where(f["resid"] > 0, "up", "down"), "price": f["price"].to_numpy(),
                                     "expected": f["expected"].to_numpy()}))
        c = causal_daily_detect(g.sort_values("date"), cfg)
        cf = c[c["flag"]]
        flagged.append(pd.DataFrame({"market": m, "date": cf["date"].to_numpy(), "detector": "causal_daily", "score": cf["score"].to_numpy(),
                                     "direction": cf["direction"].to_numpy(), "price": cf["price"].to_numpy(), "expected": cf["expected"].to_numpy()}))
    sur = surprise_flags(intervals, ac["surprise"]["level"], ac["surprise"]["horizons"])
    tab = pd.concat(flagged + [sur], ignore_index=True)
    # arrivals z-score vs the trailing 30-day median (week for the STL detector, the day for the forecast-surprise detector)
    daily = {m: g.set_index("date").sort_index() for m, g in panel.groupby("market")}
    wnd = ac["arrivals"]["window_days"]
    z = []
    for r in tab.itertuples():
        d = daily[r.market]
        if r.detector == "stl_residual":
            z.append(arrivals_z(d, r.date - pd.Timedelta(days=6), r.date, wnd))
        else:
            z.append(arrivals_z(d, r.date, r.date, wnd))
    tab["arrivals_z"] = z
    tab["arrivals_drop"] = tab["arrivals_z"] <= ac["arrivals"]["drop_z"]
    tab["event_period"] = label_periods(tab["date"], cfg)
    tab["week_end"] = week_end(tab["date"])
    n_by = tab.groupby(["detector", "week_end"])["market"].nunique().rename("n_markets_flagged_that_week")
    tab = tab.merge(n_by, on=["detector", "week_end"], how="left")
    tab["n_other_markets_flagged_same_week"] = tab["n_markets_flagged_that_week"] - 1
    tab["other_markets_flagged_same_week"] = tab["n_other_markets_flagged_same_week"] > 0
    tab = tab.drop(columns=["n_markets_flagged_that_week"]).sort_values(["detector", "market", "date"]).reset_index(drop=True)
    return tab, pd.concat(weeks)


def arrivals_check(tab: pd.DataFrame, weeks: pd.DataFrame, panel: pd.DataFrame, intervals: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Share of flagged spikes that coincide with an arrivals drop, against the share among all comparable weeks / days.

    Descriptive only. A drop means arrivals z <= drop_z (config). `baseline` is the same share over all observed weeks
    (STL detector) or all forecast target days (surprise detectors) so the flagged share can be compared with chance.
    """
    ac = cfg["anomalies"]
    panel = panel.assign(market=panel["market_key"].map(F.short_name))
    daily = {m: g.set_index("date").sort_index() for m, g in panel.groupby("market")}
    wnd, dz = ac["arrivals"]["window_days"], ac["arrivals"]["drop_z"]
    base = {}
    sample = weeks[weeks["observed"]].reset_index()
    base["stl_residual"] = np.nanmean([arrivals_z(daily[r.market], r.week_end - pd.Timedelta(days=6), r.week_end, wnd) <= dz
                                       for r in sample.itertuples()])
    for h in ac["surprise"]["horizons"]:
        g = intervals[(intervals["h"] == h)].dropna(subset=["lgbm_abs_lo_95"])
        sub = g.sample(min(len(g), 1500), random_state=0)
        base[f"forecast_surprise_h{h}"] = np.nanmean([arrivals_z(daily[r.market], r.target_date, r.target_date, wnd) <= dz for r in sub.itertuples()])
    scored = pd.concat([causal_daily_detect(g.sort_values("date"), cfg).assign(market=m).dropna(subset=["score"]) for m, g in panel.groupby("market")])
    sub = scored.sample(min(len(scored), 1500), random_state=0)
    base["causal_daily"] = np.nanmean([arrivals_z(daily[r.market], r.date, r.date, wnd) <= dz for r in sub.itertuples()])
    rows = []
    for (det, direction), g in tab.groupby(["detector", "direction"]):
        has = g["arrivals_z"].notna()
        rows.append({"detector": det, "direction": direction, "n_flags": len(g), "n_with_arrivals_z": int(has.sum()),
                     "share_with_arrivals_drop": float(g.loc[has, "arrivals_drop"].mean()) if has.any() else np.nan,
                     "baseline_share_all_comparable_dates": float(base[det]), "median_arrivals_z": float(g["arrivals_z"].median())})
    return pd.DataFrame(rows)


def counts_table(tab: pd.DataFrame) -> pd.DataFrame:
    """Number of flags per market, year and detector, split into up and down."""
    t = tab.assign(year=tab["date"].dt.year)
    c = t.groupby(["market", "year", "detector", "direction"]).size().unstack("direction", fill_value=0)
    for col in ("up", "down"):
        if col not in c:
            c[col] = 0
    c["n_flags"] = c["up"] + c["down"]
    return c.reset_index().rename(columns={"up": "n_up", "down": "n_down"})[["market", "year", "detector", "n_flags", "n_up", "n_down"]]


# --------------------------------------------------------------------------- validation without labels: synthetic spikes
def place_spikes(candidates: pd.DatetimeIndex, settings: list[dict[str, Any]], min_sep_days: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    """Random start dates for each setting, at least `min_sep_days` apart; returns settings with a `start` date and the spike days."""
    cand = np.array(candidates)
    for _ in range(1000):
        starts = rng.choice(cand, size=len(settings), replace=False)
        ok = all(abs((a - b).astype("timedelta64[D]").astype(int)) >= min_sep_days for i, a in enumerate(starts) for b in starts[:i])
        if ok:
            return [{**s, "start": pd.Timestamp(d), "days": pd.date_range(pd.Timestamp(d), periods=s["duration"])} for s, d in zip(settings, starts)]
    raise RuntimeError("could not place spikes with the requested separation")


def make_settings(sc: dict[str, Any]) -> list[dict[str, Any]]:
    """All (magnitude, direction, duration) combinations."""
    return [{"magnitude": m, "direction": d, "duration": k} for m in sc["magnitudes"] for d in sc["directions"] for k in sc["durations_days"]]


def inject_stl(panel_m: pd.DataFrame, spikes: list[dict[str, Any]]) -> pd.DataFrame:
    """In-memory copy of one market's daily prices with multiplicative spikes. The input is never modified and nothing is saved."""
    out = panel_m[["date", "modal_price", "arrivals_tonnes"]].copy()
    for sp in spikes:
        factor = 1 + sp["magnitude"] if sp["direction"] == "up" else 1 - sp["magnitude"]
        out.loc[out["date"].isin(sp["days"]), "modal_price"] *= factor
    return out


def synthetic_stl(cfg: dict[str, Any], panel: pd.DataFrame, robust: bool | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recall per spike type and the false-positive rate of the STL detector on spiked in-memory copies of the real series."""
    sc, ac = cfg["anomalies"]["synthetic"], cfg["anomalies"]
    lo, hi = map(pd.Timestamp, sc["window"])
    rng = np.random.default_rng(sc["seed"])
    settings = make_settings(sc)
    panel = panel.assign(market=panel["market_key"].map(F.short_name))
    rows, fp_rows = [], []
    for m, g in panel.groupby("market"):
        g = g.sort_values("date")
        base_w = weekly_series(g, ac["stl"]["min_days_per_week"])
        base = stl_detect(base_w, cfg, robust)
        cand = pd.DatetimeIndex(g.loc[g["modal_price"].notna() & (g["date"] >= lo) & (g["date"] <= hi - pd.Timedelta(days=3)), "date"])
        fp_rows.append({"market": m, "run": -1, "false_positive_rate": float(base.loc[base["observed"], "flag"].mean())})
        for r in range(sc["runs"]):
            spikes = place_spikes(cand, settings, sc["min_separation_days"], rng)
            w = weekly_series(inject_stl(g, spikes), ac["stl"]["min_days_per_week"])
            d = stl_detect(w, cfg, robust)
            touched = set()
            for sp in spikes:
                weeks = {pd.Timestamp(x) for x in week_end(pd.Series(sp["days"])).tolist()}
                touched |= {t + pd.Timedelta(days=7 * k) for t in weeks for k in (-1, 0, 1)}
                hit = d.reindex(sorted(weeks))
                want_up = sp["direction"] == "up"
                ok = bool((hit["flag"].fillna(False).astype(bool) & ((hit["resid"] > 0) == want_up)).any())
                rows.append({"market": m, "run": r, "magnitude": sp["magnitude"], "direction": sp["direction"], "duration": sp["duration"], "detected": ok})
            clean = d[d["observed"] & ~d.index.isin(touched)]
            fp_rows.append({"market": m, "run": r, "false_positive_rate": float(clean["flag"].mean())})
    return pd.DataFrame(rows), pd.DataFrame(fp_rows)


def synthetic_surprise(cfg: dict[str, Any], frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recall and false-positive rate of the forecast-surprise detector on spiked in-memory copies of the realised quotes.

    The realised quote is multiplied on the spike days; the forecasts are the ones actually made (forecasts issued from
    inside a spike are not re-made, which is a stated approximation); the conformal intervals are recomputed from the
    modified errors, as a real shock would enter the calibration set.
    """
    sc, ic = cfg["anomalies"]["synthetic"], cfg["intervals"]
    lo, hi = map(pd.Timestamp, sc["window"])
    rng = np.random.default_rng(sc["seed"] + 1)
    settings = make_settings(sc)
    level = cfg["anomalies"]["surprise"]["level"]
    rows, fp_rows = [], []
    for h in cfg["anomalies"]["surprise"]["horizons"]:
        for m, g in frame[frame["h"] == h].groupby("market"):
            g = g.sort_values("origin").reset_index(drop=True)
            fin = g["source"] == "final"
            tgt_final = pd.DatetimeIndex(g.loc[fin & (g["target_date"] >= lo) & (g["target_date"] <= hi - pd.Timedelta(days=3)), "target_date"])
            cand = pd.DatetimeIndex(sorted(set(tgt_final)))

            def flags(y_log: np.ndarray) -> np.ndarray:
                e = np.abs(y_log - g["f_lgbm"].to_numpy())
                q, _ = IV.conformal_halfwidths(g["origin"].to_numpy(), g["target_date"].to_numpy(), e, [level], ic["calibration_window_days"], ic["min_errors"])
                return ((y_log < g["f_lgbm"].to_numpy() - q[level]) | (y_log > g["f_lgbm"].to_numpy() + q[level])) & ~np.isnan(q[level])

            base = flags(g["y_log"].to_numpy())
            fp_rows.append({"detector": f"forecast_surprise_h{h}", "market": m, "run": -1, "false_positive_rate": float(base[fin.to_numpy()].mean())})
            for r in range(sc["runs"]):
                spikes = place_spikes(cand, settings, sc["min_separation_days"], rng)
                y = g["y_log"].to_numpy().copy()
                spike_rows = np.zeros(len(g), dtype=bool)
                for sp in spikes:
                    sel = g["target_date"].isin(sp["days"]).to_numpy()
                    factor = 1 + sp["magnitude"] if sp["direction"] == "up" else 1 - sp["magnitude"]
                    y[sel] = y[sel] + np.log(factor)
                    spike_rows |= sel
                fl = flags(y)
                for sp in spikes:
                    sel = g["target_date"].isin(sp["days"]).to_numpy() & fin.to_numpy()
                    if not sel.any():
                        continue            # no forecast existed for those dates: not testable
                    want_up = sp["direction"] == "up"
                    detected = bool((fl[sel] & ((y[sel] > g["f_lgbm"].to_numpy()[sel]) == want_up)).any())
                    rows.append({"detector": f"forecast_surprise_h{h}", "market": m, "run": r, "magnitude": sp["magnitude"], "direction": sp["direction"],
                                 "duration": sp["duration"], "detected": detected})
                clean = fin.to_numpy() & ~spike_rows
                fp_rows.append({"detector": f"forecast_surprise_h{h}", "market": m, "run": r, "false_positive_rate": float(fl[clean].mean())})
    return pd.DataFrame(rows), pd.DataFrame(fp_rows)


def synthetic_causal(cfg: dict[str, Any], panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recall and false-positive rate of the causal daily detector on spiked in-memory copies of the real series.

    Same seed, candidate days and loop order as `synthetic_stl`, so the spikes are placed on the same dates. A spike is detected
    if any of its days is flagged in the right direction. The false-positive rate is the share of scored non-spike quote days
    in the evaluation window that are flagged (days after a spike are included, so any contamination of the baseline shows up).
    """
    sc = cfg["anomalies"]["synthetic"]
    lo, hi = map(pd.Timestamp, sc["window"])
    rng = np.random.default_rng(sc["seed"])
    settings = make_settings(sc)
    panel = panel.assign(market=panel["market_key"].map(F.short_name))
    rows, fp_rows = [], []
    for m, g in panel.groupby("market"):
        g = g.sort_values("date")
        cand = pd.DatetimeIndex(g.loc[g["modal_price"].notna() & (g["date"] >= lo) & (g["date"] <= hi - pd.Timedelta(days=3)), "date"])
        base = causal_daily_detect(g, cfg)
        inwin = base["date"].between(lo, hi) & base["score"].notna()
        fp_rows.append({"market": m, "run": -1, "false_positive_rate": float(base.loc[inwin, "flag"].mean())})
        for r in range(sc["runs"]):
            spikes = place_spikes(cand, settings, sc["min_separation_days"], rng)
            d = causal_daily_detect(inject_stl(g, spikes), cfg)
            spike_days = set()
            for sp in spikes:
                spike_days |= set(sp["days"])
                sel = d["date"].isin(sp["days"]) & d["score"].notna()
                want_up = sp["direction"] == "up"
                ok = bool((d.loc[sel, "flag"] & ((d.loc[sel, "score"] > 0) == want_up)).any())
                rows.append({"market": m, "run": r, "magnitude": sp["magnitude"], "direction": sp["direction"], "duration": sp["duration"], "detected": ok})
            clean = d["date"].between(lo, hi) & d["score"].notna() & ~d["date"].isin(spike_days)
            fp_rows.append({"market": m, "run": r, "false_positive_rate": float(d.loc[clean, "flag"].mean())})
    return pd.DataFrame(rows), pd.DataFrame(fp_rows)


def synthetic_validation(cfg: dict[str, Any], panel: pd.DataFrame, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recall by spike type and false-positive rates for both detectors (synthetic test, not data)."""
    a, fa = synthetic_stl(cfg, panel)
    a["detector"], fa["detector"] = "stl_residual", "stl_residual"
    # diagnostic only: the same test with non-robust STL, to show what robust=True does to large isolated spikes
    a2, fa2 = synthetic_stl(cfg, panel, robust=False)
    a2["detector"], fa2["detector"] = "stl_residual_NONROBUST_diagnostic", "stl_residual_NONROBUST_diagnostic"
    b, fb = synthetic_surprise(cfg, frame)
    c3, fc3 = synthetic_causal(cfg, panel)
    c3["detector"], fc3["detector"] = "causal_daily", "causal_daily"
    det, fp = pd.concat([a, a2, b, c3], ignore_index=True), pd.concat([fa, fa2, fb, fc3], ignore_index=True)
    recall = det.groupby(["detector", "magnitude", "direction", "duration"]).agg(spikes=("detected", "size"), recall=("detected", "mean")).reset_index()
    recall.insert(0, "label", SYNTHETIC_LABEL)
    fpr = fp.assign(kind=np.where(fp["run"] < 0, "unmodified real series", "series with injected spikes (non-spike weeks/days)")) \
        .groupby(["detector", "kind"]).agg(false_positive_rate=("false_positive_rate", "mean"), n=("false_positive_rate", "size")).reset_index()
    fpr.insert(0, "label", SYNTHETIC_LABEL)
    return recall, fpr


# --------------------------------------------------------------------------- figure
def plot_stl(weeks: pd.DataFrame, tab: pd.DataFrame, out_dir) -> None:
    """Weekly price per market with the STL-flagged weeks marked (historical labelling)."""
    mk = sorted(weeks["market"].unique())
    fig, axes = plt.subplots(len(mk), 1, figsize=(11, 2.2 * len(mk)), sharex=True)
    for ax, m in zip(np.atleast_1d(axes), mk):
        w = weeks[weeks["market"] == m]
        ax.plot(w.index, w["price"], color="#444444", lw=0.8)
        ax.plot(w.index, w["expected"], color="#0072B2", lw=0.8, alpha=0.8)
        f = tab[(tab["market"] == m) & (tab["detector"] == "stl_residual")]
        ax.scatter(f["date"], f["price"], s=18, color=np.where(f["direction"] == "up", "#d7191c", "#2c7bb6"), zorder=3)
        ax.set_yscale("log")
        ax.set_ylabel(m, fontsize=8)
    np.atleast_1d(axes)[0].set_title("Weekly mean price (grey), STL trend+seasonal (blue); flagged weeks: red = above, blue = below. Historical labelling, not real-time alerts.", fontsize=8)
    fig.tight_layout()
    (out_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "anomalies_stl.png", bbox_inches="tight", dpi=130)
    plt.close(fig)


def main() -> None:
    """Write results/anomalies.csv and the related tables, and the synthetic validation (synthetic test, not data)."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config()
    panel = pd.read_parquet(ROOT / cfg["paths"]["panel"])
    intervals = pd.read_parquet(ROOT / "results" / "interval_forecasts.parquet")
    tab, weeks = detect_all(cfg, panel, intervals)
    res = ROOT / "results"
    tab.to_csv(res / "anomalies.csv", index=False)
    counts_table(tab).to_csv(res / "anomaly_counts.csv", index=False)
    arrivals_check(tab, weeks, panel, intervals, cfg).to_csv(res / "anomaly_arrivals_check.csv", index=False)
    plot_stl(weeks, tab, res / "figures")
    frame = IV.build_frame(cfg)
    recall, fpr = synthetic_validation(cfg, panel, frame)
    recall.to_csv(res / "anomaly_synthetic_recall.csv", index=False)
    fpr.to_csv(res / "anomaly_synthetic_fpr.csv", index=False)
    logger.info("flags per detector:\n%s", tab.groupby("detector").size().to_string())


if __name__ == "__main__":
    main()
