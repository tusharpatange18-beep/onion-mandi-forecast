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

matplotlib.use("Agg")  # draw figures to files, no window
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


def _with_short_market_names(panel: pd.DataFrame) -> pd.DataFrame:
    """Copy of the panel with a `market` column holding the short market name."""
    return panel.assign(market=panel["market_key"].map(F.short_name))


def _daily_by_market(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """One date-indexed, date-sorted table per market (the panel must have a `market` column)."""
    daily = {}
    for market, market_rows in panel.groupby("market"):
        daily[market] = market_rows.set_index("date").sort_index()
    return daily


def _spike_factor(spike: dict[str, Any]) -> float:
    """Price multiplier of a spike: 1 + magnitude for "up", 1 - magnitude for "down"."""
    if spike["direction"] == "up":
        return 1 + spike["magnitude"]
    return 1 - spike["magnitude"]


def _up_or_down(values) -> np.ndarray:
    """ "up" where the value is above zero, otherwise "down" (also for NaN)."""
    return np.where(values > 0, "up", "down")


# --------------------------------------------------------------------------- detector 1: STL residual
def weekly_series(panel_m: pd.DataFrame, min_days: int) -> pd.DataFrame:
    """Weekly (week ending Sunday) mean of log modal price and of log arrivals over OBSERVED days; NaN if fewer than `min_days`."""
    daily = panel_m.set_index("date")

    # Log price and log arrivals (zero or negative values become NaN).
    log_price = np.log(daily["modal_price"].where(daily["modal_price"] > 0))
    log_arrivals = np.log(daily["arrivals_tonnes"].where(daily["arrivals_tonnes"] > 0))

    # Weekly mean (weeks end on Sunday) and the number of observed days in each week.
    weekly = pd.DataFrame({
        "lp": log_price.resample("W-SUN").mean(),
        "n": log_price.resample("W-SUN").count(),
        "la": log_arrivals.resample("W-SUN").mean(),
    })

    # Weeks with too few observed days get no value.
    too_few_days = weekly["n"] < min_days
    weekly.loc[too_few_days, ["lp", "la"]] = np.nan
    weekly.index.name = "week_end"
    return weekly


def robust_flags(resid: pd.Series, observed: pd.Series, threshold: float, mad_scale: float) -> pd.DataFrame:
    """Robust z-score of residuals, with the median and MAD taken over observed points only; flag |score| > threshold."""
    # Median and MAD (median absolute deviation) of the observed residuals.
    observed_resid = resid[observed]
    median = observed_resid.median()
    mad = (observed_resid - median).abs().median()

    # Robust z-score (NaN everywhere if the MAD is zero).
    if mad > 0:
        score = (resid - median) / (mad_scale * mad)
    else:
        score = pd.Series(np.nan, index=resid.index)

    # Flag observed points whose score is too large.
    flag = observed & (score.abs() > threshold)
    return pd.DataFrame({"resid": resid, "score": score, "flag": flag})


def stl_detect(weekly: pd.DataFrame, cfg: dict[str, Any], robust: bool | None = None) -> pd.DataFrame:
    """STL residual detector on one market's weekly series. Returns a frame indexed by week_end with score, flag, expected (rupees).

    `robust` overrides the configured value; it is used only for the labelled diagnostic comparison in the synthetic validation."""
    anomaly_cfg = cfg["anomalies"]

    # Cut the series to the span between its first and last value.
    log_price = weekly["lp"]
    first_week = log_price.first_valid_index()
    last_week = log_price.last_valid_index()
    log_price = log_price.loc[first_week:last_week]
    observed = log_price.notna()

    # Fill gaps by straight lines, only so that STL can run; filled weeks are never flagged or used for the MAD.
    filled = log_price.interpolate(limit_area="inside")

    # Split the series into trend + seasonal + residual.
    if robust is None:
        robust = anomaly_cfg["stl"]["robust"]
    fit = STL(filled, period=anomaly_cfg["stl"]["period"], robust=robust).fit()

    # Score the residuals and add the expected and actual prices in rupees.
    out = robust_flags(fit.resid, observed, anomaly_cfg["mad_threshold"], anomaly_cfg["mad_scale"])
    out["expected"] = np.exp(fit.trend + fit.seasonal)
    out["price"] = np.exp(log_price)
    out["observed"] = observed
    return out


# --------------------------------------------------------------------------- detector 3: causal daily
def causal_daily_detect(g: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Causal daily detector on one market's observed quotes (see the module docstring).

    `g` has `date` and `modal_price` (NaN on days without a quote). For the t-th observed quote the baseline is the median of
    the log prices of quotes t-28..t-1 and the scale is 1.4826 x MAD of the 90 observed log changes up to the change into quote
    t-1; nothing from quote t or later is used. Quotes without 90 prior changes are not scored (NaN).
    """
    causal_cfg = cfg["anomalies"]["causal_daily"]
    mad_scale = cfg["anomalies"]["mad_scale"]
    n_baseline = causal_cfg["baseline_quotes"]  # quotes used for the baseline (28)
    n_changes = causal_cfg["scale_changes"]  # changes used for the scale (90)
    k = causal_cfg["k"]  # flag threshold

    # Only days with a real quote.
    quotes = g[g["modal_price"] > 0][["date", "modal_price"]].reset_index(drop=True)
    log_price = np.log(quotes["modal_price"].to_numpy())
    n_quotes = len(log_price)

    score = np.full(n_quotes, np.nan)
    baseline = np.full(n_quotes, np.nan)

    # changes[j] = change into quote j + 1.
    changes = np.diff(log_price)

    # Quote t can be scored once there are enough earlier changes.
    for t in range(n_changes + 1, n_quotes):
        # Baseline: median of the previous n_baseline log prices (quotes t-28 .. t-1).
        baseline[t] = np.median(log_price[t - n_baseline:t])

        # Scale: MAD of the previous n_changes changes (up to the change into quote t-1).
        recent_changes = changes[t - n_changes - 1:t - 1]
        median_change = np.median(recent_changes)
        scale = mad_scale * np.median(np.abs(recent_changes - median_change))

        # Robust score (only if the scale is positive).
        if scale > 0:
            score[t] = (log_price[t] - baseline[t]) / scale

    # Flag scores beyond k (NaN scores are never flagged).
    flag = np.zeros(n_quotes, dtype=bool)
    for t in range(n_quotes):
        if not np.isnan(score[t]) and abs(score[t]) > k:
            flag[t] = True

    return pd.DataFrame({"date": quotes["date"], "price": np.exp(log_price), "expected": np.exp(baseline), "score": score,
                         "flag": flag, "direction": _up_or_down(score)})


# --------------------------------------------------------------------------- shared helpers
def arrivals_z(daily: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, window_days: int, min_obs: int = 8) -> float:
    """z-score of the mean log arrivals on observed days in [start, end] against the median and MAD of the log arrivals
    on observed days in the `window_days` before `start` (the trailing 30-day median)."""
    log_arrivals = np.log(daily["arrivals_tonnes"].where(daily["arrivals_tonnes"] > 0))

    # Arrivals in the period we judge, and in the reference period just before it.
    current = log_arrivals.loc[start:end].dropna()
    reference_start = start - pd.Timedelta(days=window_days)
    reference_end = start - pd.Timedelta(days=1)
    reference = log_arrivals.loc[reference_start:reference_end].dropna()

    # Not enough data for a score.
    if current.empty or len(reference) < min_obs:
        return np.nan

    # Robust z-score: (current mean - reference median) / (1.4826 * MAD).
    median = reference.median()
    mad = (reference - median).abs().median()
    if not mad > 0:
        return np.nan
    return float((current.mean() - median) / (1.4826 * mad))


def label_periods(dates: pd.Series, cfg: dict[str, Any]) -> pd.Series:
    """Event-period label using only the high-confidence periods listed in the config; 'none' outside them."""
    keep = set(cfg["anomalies"]["high_confidence_periods"])
    periods = []
    for period in cfg["forecast"]["event_periods"]:
        if period["name"] in keep:
            periods.append(period)
    return E.assign_period(dates, periods).replace("normal", "none")


def week_end(d: pd.Series) -> pd.Series:
    """Sunday that ends the Monday-Sunday week containing each date."""
    dates = pd.to_datetime(d)
    # dayofweek: Monday = 0 ... Sunday = 6, so add (6 - dayofweek) days to reach Sunday.
    days_to_sunday = pd.to_timedelta(6 - dates.dt.dayofweek, unit="D")
    return (dates + days_to_sunday).dt.normalize()


# --------------------------------------------------------------------------- detector 2: forecast surprise
def surprise_flags(intervals: pd.DataFrame, level: float, horizons: list[int]) -> pd.DataFrame:
    """Forecasts whose realised quote fell outside the `level` interval issued h days earlier."""
    percent = int(level * 100)
    lower_name = f"lgbm_abs_lo_{percent}"
    upper_name = f"lgbm_abs_hi_{percent}"

    parts = []
    for h in horizons:
        # Forecasts of this horizon that have an interval.
        rows_h = intervals[intervals["h"] == h].dropna(subset=[lower_name])
        half_width = (rows_h[upper_name] - rows_h[lower_name]) / 2

        # Realised quote outside the interval.
        outside = (rows_h["y_log"] < rows_h[lower_name]) | (rows_h["y_log"] > rows_h[upper_name])
        flagged = rows_h[outside]

        parts.append(pd.DataFrame({
            "market": flagged["market"],
            "date": flagged["target_date"],
            "detector": f"forecast_surprise_h{h}",
            "score": (flagged["y_log"] - flagged["f_lgbm"]) / half_width[outside],  # error in half-widths
            "direction": np.where(flagged["y_log"] > flagged["f_lgbm"], "up", "down"),
            "price": np.exp(flagged["y_log"]),
            "expected": np.exp(flagged["f_lgbm"]),
        }))
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------- real data: the anomaly table
def detect_all(cfg: dict[str, Any], panel: pd.DataFrame, intervals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Both detectors on the real series. Returns (anomalies table, all-observed-weeks table for baselines)."""
    anomaly_cfg = cfg["anomalies"]
    panel = _with_short_market_names(panel)

    flagged_parts = []
    week_tables = []
    for market, market_rows in panel.groupby("market"):
        # Detector 1: STL on the weekly series.
        weekly = weekly_series(market_rows, anomaly_cfg["stl"]["min_days_per_week"])
        detected = stl_detect(weekly, cfg)
        week_tables.append(detected.assign(market=market, la=weekly["la"].reindex(detected.index)))
        stl_flagged = detected[detected["flag"]]
        flagged_parts.append(pd.DataFrame({
            "market": market,
            "date": stl_flagged.index,
            "detector": "stl_residual",
            "score": stl_flagged["score"].to_numpy(),
            "direction": _up_or_down(stl_flagged["resid"]),
            "price": stl_flagged["price"].to_numpy(),
            "expected": stl_flagged["expected"].to_numpy(),
        }))

        # Detector 3: causal daily.
        causal = causal_daily_detect(market_rows.sort_values("date"), cfg)
        causal_flagged = causal[causal["flag"]]
        flagged_parts.append(pd.DataFrame({
            "market": market,
            "date": causal_flagged["date"].to_numpy(),
            "detector": "causal_daily",
            "score": causal_flagged["score"].to_numpy(),
            "direction": causal_flagged["direction"].to_numpy(),
            "price": causal_flagged["price"].to_numpy(),
            "expected": causal_flagged["expected"].to_numpy(),
        }))

    # Detector 2: forecast surprise.
    surprises = surprise_flags(intervals, anomaly_cfg["surprise"]["level"], anomaly_cfg["surprise"]["horizons"])
    flagged_parts.append(surprises)
    tab = pd.concat(flagged_parts, ignore_index=True)

    # arrivals z-score vs the trailing 30-day median (week for the STL detector, the day for the forecast-surprise detector)
    daily = _daily_by_market(panel)
    window = anomaly_cfg["arrivals"]["window_days"]
    z_scores = []
    for row in tab.itertuples():
        market_daily = daily[row.market]
        if row.detector == "stl_residual":
            # The whole week (Monday to Sunday).
            z_scores.append(arrivals_z(market_daily, row.date - pd.Timedelta(days=6), row.date, window))
        else:
            # Just the day.
            z_scores.append(arrivals_z(market_daily, row.date, row.date, window))
    tab["arrivals_z"] = z_scores
    tab["arrivals_drop"] = tab["arrivals_z"] <= anomaly_cfg["arrivals"]["drop_z"]
    tab["event_period"] = label_periods(tab["date"], cfg)
    tab["week_end"] = week_end(tab["date"])

    # How many markets the same detector flagged in the same week.
    markets_per_week = tab.groupby(["detector", "week_end"])["market"].nunique().rename("n_markets_flagged_that_week")
    tab = tab.merge(markets_per_week, on=["detector", "week_end"], how="left")
    tab["n_other_markets_flagged_same_week"] = tab["n_markets_flagged_that_week"] - 1
    tab["other_markets_flagged_same_week"] = tab["n_other_markets_flagged_same_week"] > 0
    tab = tab.drop(columns=["n_markets_flagged_that_week"])
    tab = tab.sort_values(["detector", "market", "date"]).reset_index(drop=True)
    return tab, pd.concat(week_tables)


def _causal_scored_quotes(panel: pd.DataFrame, cfg: dict[str, Any], add_market: bool) -> pd.DataFrame:
    """All quotes the causal detector could score (score not NaN), for every market."""
    parts = []
    for market, market_rows in panel.groupby("market"):
        detected = causal_daily_detect(market_rows.sort_values("date"), cfg)
        if add_market:
            detected = detected.assign(market=market)
        parts.append(detected.dropna(subset=["score"]))
    return pd.concat(parts)


def _share_with_drop(daily: dict[str, pd.DataFrame], markets, starts, ends, window: int, drop_z: float) -> float:
    """Share of the given (market, start, end) periods whose arrivals z-score is at or below `drop_z`."""
    is_drop = []
    for i in range(len(markets)):
        z = arrivals_z(daily[markets[i]], starts[i], ends[i], window)
        is_drop.append(z <= drop_z)  # NaN <= drop_z is False
    return np.nanmean(is_drop)


def arrivals_check(tab: pd.DataFrame, weeks: pd.DataFrame, panel: pd.DataFrame, intervals: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Share of flagged spikes that coincide with an arrivals drop, against the share among all comparable weeks / days.

    Descriptive only. A drop means arrivals z <= drop_z (config). `baseline` is the same share over all observed weeks
    (STL detector) or all forecast target days (surprise detectors) so the flagged share can be compared with chance.
    """
    anomaly_cfg = cfg["anomalies"]
    panel = _with_short_market_names(panel)
    daily = _daily_by_market(panel)
    window = anomaly_cfg["arrivals"]["window_days"]
    drop_z = anomaly_cfg["arrivals"]["drop_z"]

    baseline = {}

    # STL: all observed weeks.
    sample = weeks[weeks["observed"]].reset_index()
    week_starts = []
    for end_date in sample["week_end"]:
        week_starts.append(end_date - pd.Timedelta(days=6))
    baseline["stl_residual"] = _share_with_drop(daily, sample["market"].tolist(), week_starts, sample["week_end"].tolist(), window, drop_z)

    # Forecast surprise: a random sample of (up to 1500) forecast target days.
    for h in anomaly_cfg["surprise"]["horizons"]:
        rows_h = intervals[(intervals["h"] == h)].dropna(subset=["lgbm_abs_lo_95"])
        sample = rows_h.sample(min(len(rows_h), 1500), random_state=0)
        dates = sample["target_date"].tolist()
        baseline[f"forecast_surprise_h{h}"] = _share_with_drop(daily, sample["market"].tolist(), dates, dates, window, drop_z)

    # Causal daily: a random sample of (up to 1500) scored quotes.
    scored = _causal_scored_quotes(panel, cfg, add_market=True)
    sample = scored.sample(min(len(scored), 1500), random_state=0)
    dates = sample["date"].tolist()
    baseline["causal_daily"] = _share_with_drop(daily, sample["market"].tolist(), dates, dates, window, drop_z)

    # Compare the flagged share with the baseline, per detector and direction.
    rows = []
    for group_key, group in tab.groupby(["detector", "direction"]):
        detector = group_key[0]
        direction = group_key[1]
        has_z = group["arrivals_z"].notna()
        if has_z.any():
            share_with_drop = float(group.loc[has_z, "arrivals_drop"].mean())
        else:
            share_with_drop = np.nan
        rows.append({
            "detector": detector,
            "direction": direction,
            "n_flags": len(group),
            "n_with_arrivals_z": int(has_z.sum()),
            "share_with_arrivals_drop": share_with_drop,
            "baseline_share_all_comparable_dates": float(baseline[detector]),
            "median_arrivals_z": float(group["arrivals_z"].median()),
        })
    return pd.DataFrame(rows)


def counts_table(tab: pd.DataFrame) -> pd.DataFrame:
    """Number of flags per market, year and detector, split into up and down."""
    table = tab.assign(year=tab["date"].dt.year)

    # Count flags, with one column per direction ("up", "down").
    counts = table.groupby(["market", "year", "detector", "direction"]).size().unstack("direction", fill_value=0)

    # Make sure both columns exist even if one direction never occurs.
    for column in ("up", "down"):
        if column not in counts:
            counts[column] = 0
    counts["n_flags"] = counts["up"] + counts["down"]

    counts = counts.reset_index().rename(columns={"up": "n_up", "down": "n_down"})
    return counts[["market", "year", "detector", "n_flags", "n_up", "n_down"]]


def rates_by_period(tab: pd.DataFrame, weeks: pd.DataFrame, panel: pd.DataFrame, intervals: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Flags per 100 comparable dates, by event-period label (high-confidence periods only), for every detector.

    The denominator of each detector is the set of dates it could have judged: observed weeks (STL), scored observed quotes
    inside 2014-2024 (causal daily) and forecasts with an interval, by target date, for each surprise horizon. The unit
    differs by detector (weeks or days), so rates are comparable across periods within a detector, not across detectors.
    """
    anomaly_cfg = cfg["anomalies"]
    panel = _with_short_market_names(panel)

    # Every date each detector could have judged.
    parts = []
    observed_weeks = weeks[weeks["observed"]].reset_index()
    parts.append(pd.DataFrame({"detector": "stl_residual", "date": observed_weeks["week_end"]}))
    scored = _causal_scored_quotes(panel, cfg, add_market=False)
    parts.append(pd.DataFrame({"detector": "causal_daily", "date": scored["date"]}))
    for h in anomaly_cfg["surprise"]["horizons"]:
        rows_h = intervals[(intervals["h"] == h)].dropna(subset=["lgbm_abs_lo_95"])
        parts.append(pd.DataFrame({"detector": f"forecast_surprise_h{h}", "date": rows_h["target_date"]}))
    judged = pd.concat(parts, ignore_index=True)
    judged["event_period"] = label_periods(judged["date"], cfg)

    # Count judged dates and flags per detector and event period.
    judged_count = judged.groupby(["detector", "event_period"]).size().rename("comparable_dates")
    flag_count = tab.groupby(["detector", "event_period"]).size().rename("flags")
    out = pd.concat([judged_count, flag_count], axis=1).fillna(0).reset_index()
    out["flags_per_100_dates"] = 100 * out["flags"] / out["comparable_dates"]
    return out


# --------------------------------------------------------------------------- validation without labels: synthetic spikes
def _far_enough_apart(starts: np.ndarray, min_sep_days: int) -> bool:
    """True if every pair of start dates is at least `min_sep_days` apart."""
    for i in range(len(starts)):
        for j in range(i):
            gap_days = (starts[i] - starts[j]).astype("timedelta64[D]").astype(int)
            if abs(gap_days) < min_sep_days:
                return False
    return True


def place_spikes(candidates: pd.DatetimeIndex, settings: list[dict[str, Any]], min_sep_days: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    """Random start dates for each setting, at least `min_sep_days` apart; returns settings with a `start` date and the spike days."""
    candidate_dates = np.array(candidates)

    # Try up to 1000 random draws until the dates are far enough apart.
    for _ in range(1000):
        starts = rng.choice(candidate_dates, size=len(settings), replace=False)
        if _far_enough_apart(starts, min_sep_days):
            spikes = []
            for i in range(len(settings)):
                start = pd.Timestamp(starts[i])
                spike = dict(settings[i])  # copy, so the settings are not changed
                spike["start"] = start
                spike["days"] = pd.date_range(start, periods=settings[i]["duration"])
                spikes.append(spike)
            return spikes
    raise RuntimeError("could not place spikes with the requested separation")


def make_settings(sc: dict[str, Any]) -> list[dict[str, Any]]:
    """All (magnitude, direction, duration) combinations."""
    settings = []
    for magnitude in sc["magnitudes"]:
        for direction in sc["directions"]:
            for duration in sc["durations_days"]:
                settings.append({"magnitude": magnitude, "direction": direction, "duration": duration})
    return settings


def inject_stl(panel_m: pd.DataFrame, spikes: list[dict[str, Any]]) -> pd.DataFrame:
    """In-memory copy of one market's daily prices with multiplicative spikes. The input is never modified and nothing is saved."""
    out = panel_m[["date", "modal_price", "arrivals_tonnes"]].copy()
    for spike in spikes:
        # Multiply the price on the spike days.
        on_spike_day = out["date"].isin(spike["days"])
        out.loc[on_spike_day, "modal_price"] *= _spike_factor(spike)
    return out


def _candidate_days(market_rows: pd.DataFrame, lo: pd.Timestamp, hi: pd.Timestamp) -> pd.DatetimeIndex:
    """Days with a quote inside the window (ending 3 days early) where a spike may start."""
    has_quote = market_rows["modal_price"].notna()
    in_window = (market_rows["date"] >= lo) & (market_rows["date"] <= hi - pd.Timedelta(days=3))
    return pd.DatetimeIndex(market_rows.loc[has_quote & in_window, "date"])


def synthetic_stl(cfg: dict[str, Any], panel: pd.DataFrame, robust: bool | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recall per spike type and the false-positive rate of the STL detector on spiked in-memory copies of the real series."""
    synthetic_cfg = cfg["anomalies"]["synthetic"]
    anomaly_cfg = cfg["anomalies"]
    lo = pd.Timestamp(synthetic_cfg["window"][0])
    hi = pd.Timestamp(synthetic_cfg["window"][1])
    rng = np.random.default_rng(synthetic_cfg["seed"])
    settings = make_settings(synthetic_cfg)
    panel = _with_short_market_names(panel)
    min_days = anomaly_cfg["stl"]["min_days_per_week"]

    rows = []
    fp_rows = []
    for market, market_rows in panel.groupby("market"):
        market_rows = market_rows.sort_values("date")

        # False-positive rate on the unmodified series (run = -1).
        base = stl_detect(weekly_series(market_rows, min_days), cfg, robust)
        candidates = _candidate_days(market_rows, lo, hi)
        fp_rows.append({"market": market, "run": -1, "false_positive_rate": float(base.loc[base["observed"], "flag"].mean())})

        for run_number in range(synthetic_cfg["runs"]):
            # Add spikes and run the detector again.
            spikes = place_spikes(candidates, settings, synthetic_cfg["min_separation_days"], rng)
            detected = stl_detect(weekly_series(inject_stl(market_rows, spikes), min_days), cfg, robust)

            touched_weeks = set()  # weeks near a spike, left out of the false-positive rate
            for spike in spikes:
                # The week(s) the spike falls in.
                spike_weeks = set()
                for week in week_end(pd.Series(spike["days"])).tolist():
                    spike_weeks.add(pd.Timestamp(week))
                # Those weeks plus one week either side.
                for week in spike_weeks:
                    for shift in (-1, 0, 1):
                        touched_weeks.add(week + pd.Timedelta(days=7 * shift))

                # Detected = a spike week is flagged in the right direction.
                spike_rows = detected.reindex(sorted(spike_weeks))
                want_up = spike["direction"] == "up"
                is_flagged = spike_rows["flag"].fillna(False).astype(bool)
                right_direction = (spike_rows["resid"] > 0) == want_up
                found = bool((is_flagged & right_direction).any())
                rows.append({"market": market, "run": run_number, "magnitude": spike["magnitude"], "direction": spike["direction"],
                             "duration": spike["duration"], "detected": found})

            # False-positive rate on observed weeks away from the spikes.
            clean = detected[detected["observed"] & ~detected.index.isin(touched_weeks)]
            fp_rows.append({"market": market, "run": run_number, "false_positive_rate": float(clean["flag"].mean())})
    return pd.DataFrame(rows), pd.DataFrame(fp_rows)


def _surprise_flags_for(g: pd.DataFrame, y_log: np.ndarray, level: float, window_days: int, min_errors: int) -> np.ndarray:
    """For one market/horizon series: True where the (possibly spiked) quote falls outside the recomputed conformal interval."""
    forecast = g["f_lgbm"].to_numpy()
    abs_error = np.abs(y_log - forecast)
    quantiles, _ = IV.conformal_halfwidths(g["origin"].to_numpy(), g["target_date"].to_numpy(), abs_error, [level], window_days, min_errors)
    half_width = quantiles[level]
    outside = (y_log < forecast - half_width) | (y_log > forecast + half_width)
    has_interval = ~np.isnan(half_width)
    return outside & has_interval


def synthetic_surprise(cfg: dict[str, Any], frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recall and false-positive rate of the forecast-surprise detector on spiked in-memory copies of the realised quotes.

    The realised quote is multiplied on the spike days; the forecasts are the ones actually made (forecasts issued from
    inside a spike are not re-made, which is a stated approximation); the conformal intervals are recomputed from the
    modified errors, as a real shock would enter the calibration set.
    """
    synthetic_cfg = cfg["anomalies"]["synthetic"]
    interval_cfg = cfg["intervals"]
    lo = pd.Timestamp(synthetic_cfg["window"][0])
    hi = pd.Timestamp(synthetic_cfg["window"][1])
    rng = np.random.default_rng(synthetic_cfg["seed"] + 1)
    settings = make_settings(synthetic_cfg)
    level = cfg["anomalies"]["surprise"]["level"]
    window_days = interval_cfg["calibration_window_days"]
    min_errors = interval_cfg["min_errors"]

    rows = []
    fp_rows = []
    for h in cfg["anomalies"]["surprise"]["horizons"]:
        detector = f"forecast_surprise_h{h}"
        for market, g in frame[frame["h"] == h].groupby("market"):
            g = g.sort_values("origin").reset_index(drop=True)
            is_final = (g["source"] == "final").to_numpy()  # rows of the scored window
            forecast = g["f_lgbm"].to_numpy()

            # Candidate spike days: target dates of final-window forecasts inside the window.
            in_window = (g["target_date"] >= lo) & (g["target_date"] <= hi - pd.Timedelta(days=3))
            final_targets = pd.DatetimeIndex(g.loc[is_final & in_window, "target_date"])
            candidates = pd.DatetimeIndex(sorted(set(final_targets)))

            # False-positive rate without spikes (run = -1).
            base = _surprise_flags_for(g, g["y_log"].to_numpy(), level, window_days, min_errors)
            fp_rows.append({"detector": detector, "market": market, "run": -1, "false_positive_rate": float(base[is_final].mean())})

            for run_number in range(synthetic_cfg["runs"]):
                spikes = place_spikes(candidates, settings, synthetic_cfg["min_separation_days"], rng)

                # Add the spikes to the realised log prices.
                y_log = g["y_log"].to_numpy().copy()
                any_spike = np.zeros(len(g), dtype=bool)
                for spike in spikes:
                    on_spike = g["target_date"].isin(spike["days"]).to_numpy()
                    y_log[on_spike] = y_log[on_spike] + np.log(_spike_factor(spike))
                    any_spike = any_spike | on_spike

                flags = _surprise_flags_for(g, y_log, level, window_days, min_errors)

                for spike in spikes:
                    on_spike = g["target_date"].isin(spike["days"]).to_numpy() & is_final
                    if not on_spike.any():
                        continue            # no forecast existed for those dates: not testable
                    want_up = spike["direction"] == "up"
                    right_direction = (y_log[on_spike] > forecast[on_spike]) == want_up
                    found = bool((flags[on_spike] & right_direction).any())
                    rows.append({"detector": detector, "market": market, "run": run_number, "magnitude": spike["magnitude"],
                                 "direction": spike["direction"], "duration": spike["duration"], "detected": found})

                # False-positive rate on final-window rows without a spike.
                clean = is_final & ~any_spike
                fp_rows.append({"detector": detector, "market": market, "run": run_number, "false_positive_rate": float(flags[clean].mean())})
    return pd.DataFrame(rows), pd.DataFrame(fp_rows)


def synthetic_causal(cfg: dict[str, Any], panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recall and false-positive rate of the causal daily detector on spiked in-memory copies of the real series.

    Same seed, candidate days and loop order as `synthetic_stl`, so the spikes are placed on the same dates. A spike is detected
    if any of its days is flagged in the right direction. The false-positive rate is the share of scored non-spike quote days
    in the evaluation window that are flagged (days after a spike are included, so any contamination of the baseline shows up).
    """
    synthetic_cfg = cfg["anomalies"]["synthetic"]
    lo = pd.Timestamp(synthetic_cfg["window"][0])
    hi = pd.Timestamp(synthetic_cfg["window"][1])
    rng = np.random.default_rng(synthetic_cfg["seed"])
    settings = make_settings(synthetic_cfg)
    panel = _with_short_market_names(panel)

    rows = []
    fp_rows = []
    for market, market_rows in panel.groupby("market"):
        market_rows = market_rows.sort_values("date")
        candidates = _candidate_days(market_rows, lo, hi)

        # False-positive rate on the unmodified series (run = -1).
        base = causal_daily_detect(market_rows, cfg)
        scored_in_window = base["date"].between(lo, hi) & base["score"].notna()
        fp_rows.append({"market": market, "run": -1, "false_positive_rate": float(base.loc[scored_in_window, "flag"].mean())})

        for run_number in range(synthetic_cfg["runs"]):
            spikes = place_spikes(candidates, settings, synthetic_cfg["min_separation_days"], rng)
            detected = causal_daily_detect(inject_stl(market_rows, spikes), cfg)

            spike_days = set()
            for spike in spikes:
                for day in spike["days"]:
                    spike_days.add(day)

                # Detected = any scored spike day flagged in the right direction.
                on_spike = detected["date"].isin(spike["days"]) & detected["score"].notna()
                want_up = spike["direction"] == "up"
                right_direction = (detected.loc[on_spike, "score"] > 0) == want_up
                found = bool((detected.loc[on_spike, "flag"] & right_direction).any())
                rows.append({"market": market, "run": run_number, "magnitude": spike["magnitude"], "direction": spike["direction"],
                             "duration": spike["duration"], "detected": found})

            # False-positive rate on scored, non-spike days in the window.
            clean = detected["date"].between(lo, hi) & detected["score"].notna() & ~detected["date"].isin(spike_days)
            fp_rows.append({"market": market, "run": run_number, "false_positive_rate": float(detected.loc[clean, "flag"].mean())})
    return pd.DataFrame(rows), pd.DataFrame(fp_rows)


def synthetic_validation(cfg: dict[str, Any], panel: pd.DataFrame, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recall by spike type and false-positive rates for both detectors (synthetic test, not data)."""
    stl_found, stl_fp = synthetic_stl(cfg, panel)
    stl_found["detector"] = "stl_residual"
    stl_fp["detector"] = "stl_residual"

    # diagnostic only: the same test with non-robust STL, to show what robust=True does to large isolated spikes
    nonrobust_found, nonrobust_fp = synthetic_stl(cfg, panel, robust=False)
    nonrobust_found["detector"] = "stl_residual_NONROBUST_diagnostic"
    nonrobust_fp["detector"] = "stl_residual_NONROBUST_diagnostic"

    surprise_found, surprise_fp = synthetic_surprise(cfg, frame)

    causal_found, causal_fp = synthetic_causal(cfg, panel)
    causal_found["detector"] = "causal_daily"
    causal_fp["detector"] = "causal_daily"

    found = pd.concat([stl_found, nonrobust_found, surprise_found, causal_found], ignore_index=True)
    false_positives = pd.concat([stl_fp, nonrobust_fp, surprise_fp, causal_fp], ignore_index=True)

    # Recall = share of spikes detected, per detector and spike type.
    recall = found.groupby(["detector", "magnitude", "direction", "duration"]).agg(
        spikes=("detected", "size"),
        recall=("detected", "mean"),
    ).reset_index()
    recall.insert(0, "label", SYNTHETIC_LABEL)

    # Average false-positive rate, separately for unmodified series and spiked series.
    false_positives["kind"] = np.where(false_positives["run"] < 0, "unmodified real series",
                                       "series with injected spikes (non-spike weeks/days)")
    fpr = false_positives.groupby(["detector", "kind"]).agg(
        false_positive_rate=("false_positive_rate", "mean"),
        n=("false_positive_rate", "size"),
    ).reset_index()
    fpr.insert(0, "label", SYNTHETIC_LABEL)
    return recall, fpr


# --------------------------------------------------------------------------- figure
def plot_stl(weeks: pd.DataFrame, tab: pd.DataFrame, out_dir) -> None:
    """Weekly price per market with the STL-flagged weeks marked (historical labelling)."""
    markets = sorted(weeks["market"].unique())
    fig, axes = plt.subplots(len(markets), 1, figsize=(11, 2.2 * len(markets)), sharex=True)
    axes = np.atleast_1d(axes)  # always a list of panels, even for one market

    for i in range(len(markets)):
        market = markets[i]
        ax = axes[i]
        market_weeks = weeks[weeks["market"] == market]

        # Weekly price (grey) and the STL trend + seasonal (blue).
        ax.plot(market_weeks.index, market_weeks["price"], color="#444444", lw=0.8)
        ax.plot(market_weeks.index, market_weeks["expected"], color="#0072B2", lw=0.8, alpha=0.8)

        # Flagged weeks: red above, blue below.
        flagged = tab[(tab["market"] == market) & (tab["detector"] == "stl_residual")]
        colours = np.where(flagged["direction"] == "up", "#d7191c", "#2c7bb6")
        ax.scatter(flagged["date"], flagged["price"], s=18, color=colours, zorder=3)

        ax.set_yscale("log")
        ax.set_ylabel(market, fontsize=8)

    axes[0].set_title("Weekly mean price (grey), STL trend+seasonal (blue); flagged weeks: red = above, blue = below. Historical labelling, not real-time alerts.", fontsize=8)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "anomalies_stl.png", bbox_inches="tight", dpi=130)
    plt.close(fig)


def main() -> None:
    """Write results/anomalies.csv and the related tables, and the synthetic validation (synthetic test, not data)."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config()
    panel = pd.read_parquet(ROOT / cfg["paths"]["panel"])
    intervals = pd.read_parquet(ROOT / "results" / "interval_forecasts.parquet")

    # The anomaly table and the related summaries.
    tab, weeks = detect_all(cfg, panel, intervals)
    results_dir = ROOT / "results"
    tab.to_csv(results_dir / "anomalies.csv", index=False)
    counts_table(tab).to_csv(results_dir / "anomaly_counts.csv", index=False)
    arrivals_check(tab, weeks, panel, intervals, cfg).to_csv(results_dir / "anomaly_arrivals_check.csv", index=False)
    rates_by_period(tab, weeks, panel, intervals, cfg).to_csv(results_dir / "anomaly_rates_by_period.csv", index=False)
    plot_stl(weeks, tab, results_dir / "figures")

    # Synthetic validation (spikes added in memory only).
    frame = IV.build_frame(cfg)
    recall, fpr = synthetic_validation(cfg, panel, frame)
    recall.to_csv(results_dir / "anomaly_synthetic_recall.csv", index=False)
    fpr.to_csv(results_dir / "anomaly_synthetic_fpr.csv", index=False)
    logger.info("flags per detector:\n%s", tab.groupby("detector").size().to_string())


if __name__ == "__main__":
    main()
