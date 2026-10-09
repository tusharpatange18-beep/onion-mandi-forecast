"""Phase 4: write docs/interval_report.md and docs/anomaly_report.md from the files in results/.

Everything above the marker line is generated (every number is read from results/); the commentary below it is written by hand and is kept when the file is regenerated.

Usage: python -m mandi.report4
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from mandi.config import ROOT, load_config
from mandi.report3 import _fmt
from mandi.run_phase1 import df_to_md

logger = logging.getLogger(__name__)
R = ROOT / "results"
MARKER = "<!-- COMMENTARY BELOW IS WRITTEN BY HAND; EVERYTHING ABOVE IS GENERATED -->"
METHOD_LABEL = {"lgbm_abs": "LightGBM conformal (primary)", "lgbm_vol": "LightGBM volatility-scaled (sensitivity)", "naive_abs": "naive conformal (comparator)"}


def _csv(name: str) -> pd.DataFrame:
    return pd.read_csv(R / f"{name}.csv")


def _pct(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out:
            out[c] = (100 * out[c]).round(1)
    return out


def _md(df: pd.DataFrame) -> str:
    return df_to_md(_fmt(df))


def interval_report(cfg: dict) -> str:
    """Generated part of docs/interval_report.md."""
    ic = cfg["intervals"]
    cov, mac, by = _csv("interval_coverage"), _csv("interval_coverage_macro"), _csv("interval_breakdowns")
    comp, burn, cal = _csv("interval_comparator"), _csv("interval_burnin"), _csv("interval_calibration")
    for d in (cov, mac, by):
        d["method"] = d["method"].map(METHOD_LABEL)
    d: list[str] = []
    a = d.append
    a("# Prediction intervals for the headline LightGBM (Phase 4A)")
    a("")
    a("## Method (declared before any result was looked at; no switching afterwards)")
    a("")
    a(f"- **Primary: rolling split-conformal, symmetric absolute-error quantile**, on the log scale and exponentiated to rupees, at {', '.join(f'{int(l * 100)}%' for l in ic['levels'])}. "
      f"For a forecast made at origin t (market, horizon) the calibration set is the errors y_log - forecast of earlier forecasts of the same market and horizon whose **target date is strictly before t**, "
      f"within the trailing {ic['calibration_window_days']} days, with at least {ic['min_errors']} errors; the quantile is the k-th smallest absolute error with k = ceil((n + 1) x level) (the finite-sample correction).")
    a(f"- **Sensitivity method:** the same, but each error is divided by a volatility scale known at its own origin (mean absolute log change over the last {ic['vol_window_quotes']} observed quotes) and the quantile is rescaled by the current origin's scale.")
    a("- **Comparator:** the same primary procedure applied to naive's errors, to show whether LightGBM's intervals are narrower at the same coverage.")
    a("- **Out-of-sample forecasts used:** `results/fair_forecasts.parquet` (the 2020-2024 final run; 10,166 forecasts of the frozen headline LightGBM).")
    nseed = bool(ic["seed_with_tuning_forecasts"])
    a(f"- **Burn-in:** {'the first calibration windows were seeded with the saved 2017-2019 tuning-fold forecasts of the same frozen LightGBM (run `tune_20261009_225301`; they are out-of-sample forecasts made with the same monthly expanding-window protocol). Without seeding, every market would have lacked intervals for roughly its first 100 resolved forecasts of 2020.' if nseed else 'no seeding: forecasts with fewer than the minimum calibration errors have no interval.'} "
      f"Forecasts without an interval: **{int(burn['without_interval'].sum())}** of {int(burn['forecasts'].sum()):,}. Smallest calibration set used: {int(burn['min_calibration_errors'].min())} errors.")
    a("- Coverage is the share of realised quotes inside the interval. Intervals are for the quote 7 or 30 calendar days after the origin. Bootstrap intervals are 95% block-bootstrap intervals (30-day blocks of origin days, the same days for all markets, 2000 resamples, seed 42).")
    a("")
    a("## 1. Coverage against nominal (2020-2024)")
    a("")
    m = mac[mac["method"] == METHOD_LABEL["lgbm_abs"]]
    a("Macro (equal-weight) coverage of the primary method, with its block-bootstrap interval:")
    a("")
    a(_md(_pct(m[["h", "level", "macro_coverage", "ci_low", "ci_high", "macro_avg_width_rs", "macro_avg_rel_width", "macro_interval_score_rs"]].assign(level=lambda x: (100 * x["level"]).round(0).astype(int)), ["macro_coverage", "ci_low", "ci_high", "macro_avg_rel_width"]).rename(
        columns={"level": "nominal_%", "macro_coverage": "coverage_%", "ci_low": "ci_low_%", "ci_high": "ci_high_%", "macro_avg_rel_width": "relative_width_%"})))
    a("")
    a("Relative width = (upper - lower) / forecast, averaged. Interval score = width + (2/alpha) x shortfall in rupees (lower is better).")
    a("")
    a("### 1.1 Per market and horizon (primary method)")
    a("")
    p = cov[cov["method"] == METHOD_LABEL["lgbm_abs"]].copy()
    p["coverage [95% CI]"] = [f"{100 * r.coverage:.1f} [{100 * r.ci_low:.1f}, {100 * r.ci_high:.1f}]" for r in p.itertuples()]
    p["nominal_%"] = (100 * p["level"]).round(0).astype(int)
    p["rel_width_%"] = (100 * p["avg_rel_width"]).round(1)
    a(_md(p[["market", "h", "nominal_%", "n_with_interval", "n_without_interval", "coverage [95% CI]", "avg_width_rs", "rel_width_%", "interval_score_rs"]]))
    a("")
    a("### 1.2 Calibration plot and fan charts")
    a("")
    a("![Calibration](../results/figures/interval_calibration.png)")
    a("")
    a("![Fan charts](../results/figures/interval_fan_charts.png)")
    a("")
    a("## 2. LightGBM against the naive comparator at the same procedure")
    a("")
    c = comp[["market", "h", "level", "coverage|lgbm_abs", "coverage|naive_abs", "avg_width_rs|lgbm_abs", "avg_width_rs|naive_abs", "width_ratio_lgbm_over_naive", "interval_score_ratio_lgbm_over_naive"]].copy()
    c["level"] = (100 * c["level"]).round(0).astype(int)
    c = _pct(c, ["coverage|lgbm_abs", "coverage|naive_abs"])
    a(_md(c))
    a("")
    g = comp.groupby(["h", "level"])[["width_ratio_lgbm_over_naive", "interval_score_ratio_lgbm_over_naive"]].mean().reset_index()
    g["level"] = (100 * g["level"]).round(0).astype(int)
    a("Mean over the five markets of the ratio LightGBM / naive (below 1 = LightGBM narrower or better):")
    a("")
    a(_md(g))
    a(f"")
    a(f"LightGBM's interval is narrower than naive's in {int((comp['width_ratio_lgbm_over_naive'] < 1).sum())} of {len(comp)} market-horizon-level cells.")
    a("")
    a("## 3. Coverage by year, event group, gap and volatility (pooled over markets)")
    a("")
    a("Coverage in % of the primary method and of the naive comparator; `n` = forecasts with an interval at that level. Event groups follow the target date (see `docs/model_report.md`); the `mep` group is the Oct-Dec 2023 minimum-export-price period (event E11 verified against a Deccan Herald article dated 28 Oct 2023, ending the day before the E12 ban).")
    a("")
    for gt, title in [("year", "By target year"), ("event_group", "By event group (normal vs ban periods)"), ("gap_bucket", "By days since the previous quote before the origin"), ("vol_tercile", "By volatility tercile of the origin (within market)")]:
        t = by[(by["group_type"] == gt) & by["method"].isin([METHOD_LABEL["lgbm_abs"], METHOD_LABEL["naive_abs"]])].copy()
        t["col"] = [f"{'LGBM' if 'LightGBM' in m else 'naive'} {int(l * 100)}% (h={h})" for m, l, h in zip(t["method"], t["level"], t["h"])]
        pv = t.pivot_table(index="group", columns="col", values="coverage").mul(100).round(1)
        nn = t[(t["method"] == METHOD_LABEL["lgbm_abs"]) & (t["level"] == 0.95)].pivot_table(index="group", columns="h", values="n_with_interval").add_prefix("n_h")
        a(f"### {title}")
        a("")
        a(_md(pd.concat([pv, nn], axis=1).reset_index()))
        a("")
    a("## 4. Sensitivity method: volatility-scaled conformal")
    a("")
    s = mac[mac["method"].isin([METHOD_LABEL["lgbm_abs"], METHOD_LABEL["lgbm_vol"]])].copy()
    s["level"] = (100 * s["level"]).round(0).astype(int)
    a(_md(_pct(s[["h", "method", "level", "macro_coverage", "ci_low", "ci_high", "macro_avg_rel_width", "macro_interval_score_rs"]], ["macro_coverage", "ci_low", "ci_high", "macro_avg_rel_width"])))
    a("")
    t = by[(by["group_type"] == "vol_tercile") & by["method"].isin([METHOD_LABEL["lgbm_abs"], METHOD_LABEL["lgbm_vol"]]) & (by["level"] == 0.8)].copy()
    t["col"] = [("scaled" if "volatility" in m else "primary") + f" (h={h})" for m, h in zip(t["method"], t["h"])]
    a("80% coverage by volatility tercile, primary vs scaled (%):")
    a("")
    a(_md(t.pivot_table(index="group", columns="col", values="coverage").mul(100).round(1).reset_index()))
    a("")
    a("## 5. Forecasts without an interval (burn-in)")
    a("")
    a(_md(burn))
    a("")
    return "\n".join(d)


def anomaly_report(cfg: dict) -> str:
    """Generated part of docs/anomaly_report.md."""
    ac = cfg["anomalies"]
    tab = pd.read_csv(R / "anomalies.csv", parse_dates=["date"])
    counts, chk = _csv("anomaly_counts"), _csv("anomaly_arrivals_check")
    rec, fpr = _csv("anomaly_synthetic_recall"), _csv("anomaly_synthetic_fpr")
    d: list[str] = []
    a = d.append
    a("# Anomaly detection (Phase 4B)")
    a("")
    a("## Detectors (declared before any result was looked at)")
    a("")
    a(f"- **Detector 1, `stl_residual`: a retrospective large-swing detector, not a spike detector.** STL with robust={ac['stl']['robust']} and period {ac['stl']['period']} on the weekly mean of log modal price (observed days only; a week needs at least {ac['stl']['min_days_per_week']} observed days). "
      f"A week is flagged when |residual - median| > {ac['mad_threshold']} x {ac['mad_scale']} x MAD. **It uses data after the date as well, so it is historical labelling, not a real-time alert.** "
      "Unobserved weeks are bridged by linear interpolation only so that STL can run; they are never flagged and are not used for the median or MAD. The date reported is the Sunday that ends the week.")
    a(f"- **Detector 2, `forecast_surprise_h7` and `_h30` (usable in real time).** The realised quote falls outside the {int(ac['surprise']['level'] * 100)}% conformal interval of the headline LightGBM issued h days earlier (section 1 of `docs/interval_report.md`). Available for 2020-2024 only. `score` = (log realised - log forecast) / interval half-width, so |score| > 1 means outside.")
    cd = ac["causal_daily"]
    a(f"- **Detector 3, `causal_daily` (usable in real time; declared after the first Phase 4 review, run once, not tuned).** On observed quotes only: baseline = median of the log price of the previous {cd['baseline_quotes']} observed quotes; "
      f"scale = {ac['mad_scale']} x MAD of the previous {cd['scale_changes']} observed daily log changes; flag if |log p_t - baseline| > {cd['k']:g} x scale. It uses only data before the quote being judged, so it could have raised the alert on the day. "
      "`score` = (log p_t - baseline) / scale. Quotes without 90 prior changes are not scored.")
    a(f"- **Arrivals check.** `arrivals_z` = z-score of the mean log arrivals of the flagged week (or day) against the median and MAD of the log arrivals of the {ac['arrivals']['window_days']} days before it. A \"drop\" means z <= {ac['arrivals']['drop_z']}.")
    a("- **Event-period labels** use only periods whose both end dates are high-confidence in `docs/eda_findings.md`: the 2019-20, 2020-21 and 2023-24 export bans, the COVID lockdown and the Oct-Dec 2023 minimum-export-price period (E11 verified against a Deccan Herald article dated 28 Oct 2023; it ends 2023-12-07, so it does not overlap the ban); `none` means outside those periods.")
    a("- No causal claims are made anywhere: the table says when, where, how large, and what arrivals did.")
    a("- The full table is `results/anomalies.csv` (not committed: it lists individual market-dates); this document shows aggregates and a short excerpt.")
    a("")
    a("## 1. How many flags")
    a("")
    tot = tab.groupby("detector").agg(flags=("score", "size"), up=("direction", lambda s: int((s == "up").sum())), down=("direction", lambda s: int((s == "down").sum()))).reset_index()
    a(_md(tot))
    a("")
    a("Per market and year:")
    a("")
    a(_md(counts.pivot_table(index=["detector", "market"], columns="year", values="n_flags", fill_value=0).reset_index()))
    a("")
    a("Share of flags with another market flagged by the same detector in the same Monday-Sunday week, and share in each event-period label:")
    a("")
    sh = tab.groupby("detector").agg(share_other_market_flagged_same_week_pct=("other_markets_flagged_same_week", lambda s: round(100 * s.mean(), 1))).reset_index()
    ev = tab.groupby(["detector", "event_period"]).size().unstack(0, fill_value=0).reset_index()
    a(_md(sh))
    a("")
    a(_md(ev))
    a("")
    rates = _csv("anomaly_rates_by_period")
    a("Flags per 100 comparable dates by event period (the denominator is the dates each detector could have judged: observed weeks for STL, scored observed quotes for the causal daily detector, forecasts with an interval by target date for the surprise detectors; rates are comparable across periods within a detector, not across detectors):")
    a("")
    a(_md(rates.pivot_table(index="event_period", columns="detector", values="flags_per_100_dates").round(1).reset_index()))
    a("")
    a("## 2. Do flagged spikes coincide with an arrivals drop?")
    a("")
    a("`share_with_arrivals_drop` is among flagged spikes; `baseline` is the same share among all comparable weeks (STL) or forecast target days (surprise detectors), i.e. what chance alone would give. Descriptive only.")
    a("")
    a(_md(_pct(chk, ["share_with_arrivals_drop", "baseline_share_all_comparable_dates"])))
    a("")
    a("## 3. Validation without labels: a synthetic test of the detectors, not data")
    a("")
    a("**Synthetic test of the detector, not data.** Multiplicative spikes (price x (1 +/- magnitude)) were injected at random dates of 2020-2024 into an **in-memory copy** of the cleaned series; nothing was saved to `data/` and the real series is untouched. "
      f"Each run places one spike of each type (magnitude x direction x duration) at least {ac['synthetic']['min_separation_days']} days apart (detectors 1 and 3 are tested on exactly the same spike dates); {ac['synthetic']['runs']} independent placements per market and market-horizon, so each cell below has "
      f"{int(rec['spikes'].min())} or more spikes. Recall = share of injected spikes flagged in the right direction in the affected week(s) (STL) or on the affected target dates (forecast surprise). "
      "For the forecast-surprise detector the forecasts are the ones actually made (forecasts issued from inside a spike are not re-made, an approximation) and the conformal intervals are recomputed from the modified errors.")
    a("")
    a("The detector `stl_residual_NONROBUST_diagnostic` is **not part of the declared method**; it is the same test with `robust=False`, shown only to explain a finding about `robust=True`.")
    a("")
    a("### 3.1 False-positive rate")
    a("")
    a("Share of non-spike weeks (or days) flagged, in the unmodified real series and in the series with injected spikes:")
    a("")
    a(_md(_pct(fpr.drop(columns="label"), ["false_positive_rate"])))
    a("")
    a("### 3.2 Recall by spike size and duration (both directions pooled), %")
    a("")
    for det, gdf in rec.groupby("detector"):
        p = gdf.groupby(["magnitude", "duration"]).apply(lambda x: np.average(x["recall"], weights=x["spikes"]), include_groups=False).unstack().mul(100).round(0)
        p.columns = [f"{int(c)}-day spike" for c in p.columns]
        a(f"**{det}**")
        a("")
        a(_md(p.reset_index()))
        a("")
    a("### 3.3 Recall by direction (durations pooled), %")
    a("")
    for det, gdf in rec.groupby("detector"):
        p = gdf.groupby(["magnitude", "direction"]).apply(lambda x: np.average(x["recall"], weights=x["spikes"]), include_groups=False).unstack().mul(100).round(0)
        a(f"**{det}**")
        a("")
        a(_md(p.reset_index()))
        a("")
    a("## 4. Largest flags (excerpt)")
    a("")
    ex = []
    for det, gdf in tab.groupby("detector"):
        ex.append(gdf.reindex(gdf["score"].abs().sort_values(ascending=False).index).head(8))
    e = pd.concat(ex)[["detector", "market", "date", "direction", "score", "price", "expected", "arrivals_z", "event_period", "other_markets_flagged_same_week"]]
    e["date"] = e["date"].dt.date
    a(_md(e))
    a("")
    a("![STL flags](../results/figures/anomalies_stl.png)")
    a("")
    return "\n".join(d)


def _write(name: str, generated: str) -> None:
    path = ROOT / load_config()["paths"]["docs"] / name
    commentary = ""
    if path.exists() and MARKER in path.read_text(encoding="utf-8"):
        commentary = path.read_text(encoding="utf-8").split(MARKER, 1)[1]
    path.write_text(generated + "\n" + MARKER + commentary, encoding="utf-8")
    logger.info("wrote %s", path)


def main() -> None:
    """Write both Phase 4 reports (commentary below the marker is preserved)."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config()
    _write("interval_report.md", interval_report(cfg))
    _write("anomaly_report.md", anomaly_report(cfg))


if __name__ == "__main__":
    main()
