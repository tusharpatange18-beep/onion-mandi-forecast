"""Streamlit dashboard for the onion price forecasts.

Run:  python -m streamlit run app/dashboard.py        (needs the API: python -m uvicorn api.main:app)

CONSTRAINT: the dataset ends 2024-12-31, so "latest" forecasts are as of the last data date, not live.
Every number shown is read from results/*.csv or from the API.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import dashlib as D  # noqa: E402
from mandi.features import short_name  # noqa: E402

st.set_page_config(page_title="Maharashtra onion price forecasts", layout="wide")
MARKETS = [short_name(m) for m in D.CFG["markets"]]
HORIZONS = [int(h) for h in D.SV["horizons"]]

health, health_err = D.api_get("/health")
data_through = health["data_through"] if health and health.get("data_through") else D.CFG["data"]["end_date"]   # from the API, else from the config
BT = f"{D.CFG['backtest']['first_test_date'][:4]}-{D.CFG['backtest']['last_test_date'][:4]}"

st.title("Maharashtra onion price forecasts")
st.warning(f"The dataset ends {data_through}. \"Latest\" forecasts are as of the last data date, not live: they look ahead from data that stops on {data_through}.")

c1, c2 = st.columns(2)
market = c1.selectbox("Market", MARKETS)
horizon = c2.selectbox("Horizon (days ahead)", HORIZONS)

tab_f, tab_b, tab_a, tab_about = st.tabs(["Forecast", "Backtest", "Anomalies", "About"])

# ---------------------------------------------------------------------------------------------- Forecast
with tab_f:
    latest, err = D.api_get("/forecast", {"market": market, "horizon": horizon})
    if err:
        st.error(err)
    else:
        history = D.price_history(market, D.SV["dashboard"]["history_days"])
        since = history["date"].min() if history is not None and len(history) else None
        backtest = D.backtest_forecasts(market, horizon, since)
        anomalies = D.read_anomalies()
        fpr = D.detector_false_positive_rates()
        dets = sorted(anomalies["detector"].unique()) if anomalies is not None else []
        default = [d for d in dets if d == f"forecast_surprise_h{horizon}"]
        label = lambda d: f"{d} (flags {100 * fpr[d]:.1f}% of ordinary dates)" if d in fpr else d  # noqa: E731
        chosen = st.multiselect("Anomaly layers (default: the forecast-surprise detector for this horizon)", dets, default=default, format_func=label) if dets else []
        flags = None
        if anomalies is not None and chosen and since is not None:
            flags = anomalies[(anomalies["market"] == market) & anomalies["detector"].isin(chosen) & (anomalies["date"] >= since)]
        st.plotly_chart(D.forecast_figure(market, horizon, history, latest, backtest, flags), width="stretch")
        if backtest is None:
            st.info("Out-of-sample backtest forecasts (results/interval_forecasts.parquet) are not available, so only the latest forecast is drawn.")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(f"Forecast for {latest['target_date']} (Rs/quintal)", f"{latest['point']:,.0f}", f"{latest['point'] - latest['naive']:+,.0f} vs naive")
        m2.metric("80% interval", f"{latest['lo80']:,.0f} to {latest['hi80']:,.0f}")
        m3.metric("95% interval", f"{latest['lo95']:,.0f} to {latest['hi95']:,.0f}")
        m4.metric(f"Last observed price ({latest['origin_date']})", f"{latest['last_price']:,.0f}")
        st.subheader("Caveats")
        for c in latest["caveats"]:
            st.markdown(f"- {c}")
        st.caption(f"Model `{latest['model_spec']}`, trained on rows whose target date is before {latest['as_of']}; intervals calibrated on {latest['n_calibration']} resolved out-of-sample errors; "
                   f"generated {latest['generated_at']} (commit {latest['git_commit']}). The blue band is the out-of-sample backtest (forecasts made by models that had not seen the target); "
                   "the latest forecast is the only forward-looking one.")

# ---------------------------------------------------------------------------------------------- Backtest
with tab_b:
    st.subheader(f"Backtest {BT} (headline model: LightGBM)")
    note = D.tie_note(horizon)
    if note:
        st.info(note)
    summ, macro, boot, holm = D.read_table("summary_table"), D.read_table("summary_macro"), D.read_table("bootstrap_ci"), D.read_table("holm_dm")
    if macro is None:
        st.error("results/summary_macro.csv not found: run the backtest first (see docs/model_report.md).")
    else:
        st.markdown("**Improvement in rupee MAE over the strongest baseline, macro average over the five markets (%)**")
        st.dataframe(macro[["h", "headline_impr_vs_strongest_pct", "headline_impr_vs_naive_pct"] + [c for c in macro.columns if c.startswith("impr_") and c.endswith("_vs_strongest_pct")]].round(2), hide_index=True)
        if boot is not None:
            st.markdown(f"**Block-bootstrap interval of the macro improvement ({int(boot['block_days'].iloc[0])}-day blocks, {int(boot['resamples'].iloc[0])} resamples)**")
            st.dataframe(boot[["h", "point_estimate_pct", "ci95_low_strongest", "ci95_high_strongest", "share_resamples_above_0_strongest", "point_estimate_vs_naive_pct", "ci95_low_vs_naive", "ci95_high_vs_naive"]].round(2), hide_index=True)
        if summ is not None:
            st.markdown(f"**Per market, {horizon}-day horizon**")
            s = summ[summ["h"] == horizon][["market", "strongest_baseline", "baseline_MAE", "naive_MAE", "headline_MAE", "headline_impr_vs_strongest_pct", "headline_impr_vs_naive_pct", "n"]]
            st.dataframe(s.round(2), hide_index=True)
        if holm is not None:
            st.markdown("**Diebold-Mariano p-values of the headline model against the strongest baseline, with the Holm adjustment**")
            st.dataframe(holm[holm["h"] == horizon][["market", "DM", "p_value", "p_holm", "significant_raw_5pct", "significant_holm_5pct"]].round(4), hide_index=True)
        for name, title, cols in [("per_year", "By target year", None), ("interval_coverage_macro", "Prediction-interval coverage (macro; nominal = level)", None)]:
            t = D.read_table(name)
            if t is not None:
                st.markdown(f"**{title}**")
                if name == "per_year":
                    t = t[t["h"] == horizon].groupby("year")[["impr_vs_strongest_pct", "impr_vs_naive_pct"]].mean().round(2).reset_index()
                else:
                    t = t[(t["method"] == "lgbm_abs") & (t["h"] == horizon)]
                    t = t[[c for c in ["level", "macro_coverage", "ci_low", "ci_high", "macro_avg_rel_width"] if c in t.columns]].round(3)
                st.dataframe(t, hide_index=True)
    st.caption("All tables are read from results/*.csv. Methods and caveats: docs/model_report.md and docs/interval_report.md.")

# ---------------------------------------------------------------------------------------------- Anomalies
with tab_a:
    st.subheader("Anomaly flags")
    st.markdown("Three detectors, tested on synthetic spikes injected into an in-memory copy of the series (a test of the detectors, not data). "
                "No flag is evidence of an error or of a cause; flags and arrivals are only compared descriptively.")
    rec, fpr = D.read_table("anomaly_synthetic_recall"), D.read_table("anomaly_synthetic_fpr")
    desc = {
        "stl_residual": "Retrospective large-swing detector, not a spike detector. STL on weekly mean prices; uses later data, so it cannot alert in real time.",
        "forecast_surprise_h7": "Real time. The realised quote falls outside the high-coverage prediction interval issued h days earlier (the short horizon).",
        "forecast_surprise_h30": "Real time. The realised quote falls outside the high-coverage prediction interval issued h days earlier (the long horizon).",
        "causal_daily": "Real time. Price vs the median of the previous quotes, scaled by the recent daily changes; flags large deviations.",
    }
    if rec is not None and fpr is not None:
        fp = fpr[fpr["kind"].str.startswith("unmodified")][["detector", "false_positive_rate"]].rename(columns={"false_positive_rate": "false positives, ordinary dates (%)"})
        fp["false positives, ordinary dates (%)"] = (100 * fp["false positives, ordinary dates (%)"]).round(1)
        r = rec.groupby(["detector", "magnitude"]).apply(lambda x: (100 * (x["recall"] * x["spikes"]).sum() / x["spikes"].sum()), include_groups=False).unstack().round(0)
        r.columns = [f"recall, {int(c * 100)}% spike (%)" for c in r.columns]
        tbl = fp.merge(r.reset_index(), on="detector", how="left")
        tbl.insert(1, "what it is", tbl["detector"].map(desc).fillna("diagnostic only (non-robust STL), not part of the declared method"))
        st.dataframe(tbl, hide_index=True)
    st.markdown("**Limitations.** The retrospective STL detector and the causal daily detector flag a large share of ordinary dates (see the table above); the forecast-surprise detector has a false-positive rate near its nominal level but low recall for small spikes, "
                "especially at 30 days. There is no ground truth, so recall on real anomalies is unknown. Details: docs/anomaly_report.md.")
    flags = D.read_anomalies()
    if flags is None:
        st.info("results/anomalies.csv is not available (it is generated locally and not committed): run `python -m mandi.anomalies`.")
    else:
        fpr_a = D.detector_false_positive_rates()
        opts = sorted(flags["detector"].unique())
        chosen = st.multiselect("Detectors (default: forecast surprise for this horizon; the others are noisy comparators)", opts,
                                default=[d for d in opts if d == f"forecast_surprise_h{horizon}"], key="adet",
                                format_func=lambda d: f"{d} (flags {100 * fpr_a[d]:.1f}% of ordinary dates)" if d in fpr_a else d)
        sel = flags[(flags["market"] == market) & flags["detector"].isin(chosen)].sort_values("date", ascending=False)
        st.caption(f"{len(sel)} flags for {market} (showing the latest 500)")
        st.dataframe(sel.head(500)[["date", "detector", "direction", "score", "price", "expected", "arrivals_z", "event_period", "other_markets_flagged_same_week"]].round({"score": 2, "price": 1, "expected": 1, "arrivals_z": 2}), hide_index=True)
        counts = D.read_table("anomaly_counts")
        if counts is not None:
            st.markdown("**Flags per market and year**")
            st.dataframe(counts[(counts["market"] == market) & counts["detector"].isin(chosen)].pivot_table(index="detector", columns="year", values="n_flags", fill_value=0).rename(columns=str).reset_index(), hide_index=True)

# ---------------------------------------------------------------------------------------------- About
with tab_about:
    st.subheader("About")
    st.markdown("""
**Data.** Daily modal onion prices and arrivals for five Maharashtra mandis (Solapur, Pune, Vashi New Mumbai, Kolhapur, Chattrapati Sambhajinagar) from the Mendeley Data set
*Preprocessed onion 2014-2024 prices, weather and events* (Shelar, Hajare, Mangarule; DOI 10.17632/ds9jmxp9zy.1; licence CC BY 4.0). Prices and arrivals come from **Agmarknet** (Directorate of Marketing and Inspection);
the weather columns in that file are from **IMD** gridded data, but this project uses no weather feature in version 1.

**Lasalgaon is not in this dataset** (nor Niphad, Pimpalgaon, Saykheda, Yeola or Chandwad). These are five Maharashtra mandis, not the main price-setting onion market.

**Data as of.** The dataset ends {data_through}. "Latest" forecasts are as of the last data date, not live.

**Limitations.** At 7 days the model is statistically close to naive; 30-day intervals are wide and under-cover in shock periods (export-ban periods); the five markets move together, so they are not five independent tests;
the strongest baseline was chosen on the test window; quotes are reported modal prices with a reporting lag and many days are missing (closure weekdays, holidays and gaps). Full discussion: docs/model_report.md, docs/interval_report.md, docs/anomaly_report.md.
""".replace("{data_through}", str(data_through)))
    if health:
        st.caption(f"API {health['api_version']}: forecast file generated {health.get('forecast_generated_at')}, as of {health.get('as_of')}, data through {health.get('data_through')} (age {health.get('age_days')} days).")
    elif health_err:
        st.caption(health_err)
