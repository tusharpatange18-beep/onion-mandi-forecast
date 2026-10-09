"""Helpers for the Streamlit dashboard (kept separate from the page so they can be tested).

CONSTRAINT: the dataset ends 2024-12-31, so "latest" forecasts are as of the last data date, not live.
Every number the dashboard shows is read from results/*.csv or the API; nothing is typed here.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import plotly.graph_objects as go

from mandi.config import ROOT, load_config

CFG = load_config()
SV = CFG["serving"]
RESULTS = Path(os.environ.get("MANDI_RESULTS") or ROOT / SV["results_dir"])
PANEL = Path(os.environ.get("MANDI_PANEL") or ROOT / SV["panel_file"])
API_URL = os.environ.get("MANDI_API_URL") or SV["dashboard"]["api_url"]
TIMEOUT = float(SV["dashboard"]["request_timeout_seconds"])
BLUE, RED, GREY, ORANGE = "#0072B2", "#D55E00", "#888888", "#E69F00"


# --------------------------------------------------------------------------- API
def _http_get(path: str, params: dict[str, Any] | None = None) -> httpx.Response:
    """The one place that touches the network (replaced by a test client in the tests)."""
    return httpx.get(API_URL.rstrip("/") + path, params=params, timeout=TIMEOUT)


def api_get(path: str, params: dict[str, Any] | None = None) -> tuple[Any, str | None]:
    """GET `path` from the API. Returns (json, None) or (None, a readable error message); never raises."""
    try:
        r = _http_get(path, params)
    except httpx.HTTPError as exc:
        return None, f"The forecast API at {API_URL} is not reachable ({type(exc).__name__}). Start it with `python -m uvicorn api.main:app` or `docker compose up`."
    if r.status_code != 200:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        return None, f"The API answered {r.status_code}: {detail}"
    return r.json(), None


# --------------------------------------------------------------------------- results tables and data
def read_table(name: str) -> pd.DataFrame | None:
    """results/<name>.csv, or None if the file does not exist."""
    p = RESULTS / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


def read_anomalies() -> pd.DataFrame | None:
    p = RESULTS / "anomalies.csv"
    return pd.read_csv(p, parse_dates=["date"]) if p.exists() else None


def price_history(market: str, days: int) -> pd.DataFrame | None:
    """Observed modal prices of one market for the last `days` calendar days of the dataset (from the processed panel)."""
    if not PANEL.exists():
        return None
    from mandi.features import short_name

    p = pd.read_parquet(PANEL, columns=["market_key", "date", "modal_price"])
    p = p[p["market_key"].map(short_name) == market].dropna(subset=["modal_price"])
    if p.empty:
        return None
    return p[p["date"] > p["date"].max() - pd.Timedelta(days=days)][["date", "modal_price"]]


# --------------------------------------------------------------------------- figures
def forecast_figure(market: str, horizon: int, history: pd.DataFrame | None, latest: dict | None, recent: list[dict] | None,
                    flags: pd.DataFrame | None) -> go.Figure:
    """Price history, the latest forecast with 80% and 95% bands, the naive forecast, recent forecasts and anomaly markers."""
    fig = go.Figure()
    if history is not None and len(history):
        fig.add_trace(go.Scatter(x=history["date"], y=history["modal_price"], mode="lines+markers", name="observed modal price", line=dict(color="black", width=1), marker=dict(size=3)))
    if recent:
        r = pd.DataFrame(recent).dropna(subset=["lo80", "hi80"])
        r["target_date"] = pd.to_datetime(r["target_date"])
        for flag, name, color in [(True, "recent forecasts, in sample (fitted on the target)", GREY), (False, "recent forecasts, out of sample", BLUE)]:
            s = r[r["in_sample"] == flag].sort_values("target_date")
            if len(s):
                fig.add_trace(go.Scatter(x=s["target_date"], y=s["point"], mode="markers", name=name, marker=dict(size=4, color=color, opacity=0.6),
                                         error_y=dict(type="data", symmetric=False, array=(s["hi80"] - s["point"]), arrayminus=(s["point"] - s["lo80"]), thickness=0.6, width=0, color=color)))
    if latest:
        t = pd.Timestamp(latest["target_date"])
        o = pd.Timestamp(latest["origin_date"])
        for lvl, lo, hi, w in [("95%", "lo95", "hi95", 2), ("80%", "lo80", "hi80", 6)]:
            fig.add_trace(go.Scatter(x=[t, t], y=[latest[lo], latest[hi]], mode="lines", name=f"{lvl} interval", line=dict(color=ORANGE if lvl == "80%" else "#f5c27a", width=w + 2)))
        fig.add_trace(go.Scatter(x=[o, t], y=[latest["last_price"], latest["point"]], mode="lines+markers", name="LightGBM forecast", line=dict(color=BLUE, dash="dot"), marker=dict(size=[0, 11], color=BLUE)))
        fig.add_trace(go.Scatter(x=[t], y=[latest["naive"]], mode="markers", name="naive forecast (last price)", marker=dict(size=10, symbol="diamond-open", color=RED)))
    if flags is not None and len(flags):
        for d, grp in flags.groupby("detector"):
            fig.add_trace(go.Scatter(x=grp["date"], y=grp["price"], mode="markers", name=f"anomaly flag: {d}", marker=dict(size=9, symbol="x", line=dict(width=1)), text=[f"score {s:.1f}, {dr}" for s, dr in zip(grp["score"], grp["direction"])]))
    fig.update_layout(height=470, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="Rs per quintal", legend=dict(orientation="h", y=-0.2), title=f"{market}: {horizon}-day-ahead forecast")
    return fig


# --------------------------------------------------------------------------- text built from tables
def tie_note(horizon: int) -> str | None:
    """The 7-day 'statistical tie with naive' note, assembled from results/bootstrap_ci.csv and results/robustness.csv."""
    b, rob = read_table("bootstrap_ci"), read_table("robustness")
    if horizon != 7 or b is None:
        return None
    x = b[b["h"] == 7].iloc[0]
    lo, hi = x["ci95_low_strongest"], x["ci95_high_strongest"]
    s = (f"At 7 days LightGBM improved on the strongest baseline by {x['point_estimate_pct']:+.1f}% (95% block-bootstrap interval {lo:+.1f}% to {hi:+.1f}%)"
         + (", which includes zero: statistically a tie with naive" if lo <= 0 <= hi else ", which excludes zero"))
    if rob is not None:
        r = rob[(rob["model"] == "lgbm") & (rob["h"] == 7)]
        if len(r):
            s += (f". It beat the strongest baseline on {r['win_rate_vs_strongest_pct'].mean():.1f}% of forecasts and its median absolute error changed by "
                  f"{r['median_AE_impr_vs_strongest_pct'].mean():+.1f}% (average over markets); the mean gain comes from a minority of large moves.")
    return s
