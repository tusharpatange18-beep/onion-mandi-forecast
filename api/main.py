"""FastAPI service for the onion price forecasts.

CONSTRAINT: the dataset ends 2024-12-31, so "latest" forecasts are as of the last data date, not live.

Run:  python -m uvicorn api.main:app --port 8000      (plain uvicorn, no [standard] extras)

Endpoints: GET /health, GET /markets, GET /forecast?market=&horizon=, GET /forecast/history?market=&horizon=
Forecasts are read from results/forecasts.parquet (written by `python -m mandi.forecast`) and reloaded when the file's mtime changes.
Caveat texts are assembled from the tables in results/, so no number is typed here.
"""
from __future__ import annotations

import os
import threading
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from mandi.config import ROOT, load_config
from mandi.features import short_name


# --------------------------------------------------------------------------- response models (Pydantic v2)
class HealthResponse(BaseModel):
    status: str = Field(description="'ok' if the forecast file exists, otherwise 'degraded'")
    forecast_file_exists: bool
    data_through: date | None = Field(None, description="last date of the dataset (it ends 2024-12-31)")
    as_of: date | None = Field(None, description="the as-of date of the forecasts")
    forecast_generated_at: datetime | None = None
    age_days: float | None = Field(None, description="days since the forecasts were generated")
    api_version: str
    note: str


class MarketInfo(BaseModel):
    market: str
    horizons: list[int]
    latest_origin_date: date | None = None


class MarketsResponse(BaseModel):
    markets: list[MarketInfo]


class ForecastResponse(BaseModel):
    market: str
    horizon: int
    origin_date: date
    target_date: date
    last_price: float = Field(description="the latest observed modal price at the origin, Rs per quintal")
    point: float
    lo80: float
    hi80: float
    lo95: float
    hi95: float
    naive: float = Field(description="the naive forecast: the last observed price")
    n_calibration: int
    model_spec: str
    trained_through: date
    generated_at: datetime
    git_commit: str
    as_of: date
    data_through: date
    in_sample: bool
    gap_before_origin_days: float | None = None
    caveats: list[str]


class HistoryRow(BaseModel):
    origin_date: date
    target_date: date
    last_price: float
    point: float
    lo80: float | None
    hi80: float | None
    lo95: float | None
    hi95: float | None
    naive: float
    actual_price: float | None
    in_sample: bool


class HistoryResponse(BaseModel):
    market: str
    horizon: int
    as_of: date
    note: str
    rows: list[HistoryRow]


class ErrorResponse(BaseModel):
    detail: str


# --------------------------------------------------------------------------- data access
class _MtimeCache:
    """Read a file with `loader` and reload it only when its modification time changes."""

    def __init__(self, path: Path, loader):
        self.path, self.loader = path, loader
        self._mtime: float | None = None
        self._value: Any = None
        self._lock = threading.Lock()

    def get(self) -> Any:
        try:
            mtime = self.path.stat().st_mtime
        except FileNotFoundError:
            with self._lock:
                self._mtime, self._value = None, None
            return None
        with self._lock:
            if mtime != self._mtime:
                self._value, self._mtime = self.loader(self.path), mtime
            return self._value


def _load_forecasts(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def _read_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


class Caveats:
    """Builds caveat texts from the result tables (read fresh when they change)."""

    def __init__(self, results_dir: Path, long_gap_days: int):
        self.dir, self.long_gap = results_dir, long_gap_days
        self.boot = _MtimeCache(results_dir / "bootstrap_ci.csv", _read_csv_loader)
        self.cov = _MtimeCache(results_dir / "interval_coverage_macro.csv", _read_csv_loader)
        self.brk = _MtimeCache(results_dir / "interval_breakdowns.csv", _read_csv_loader)
        self.err = _MtimeCache(results_dir / "error_analysis.csv", _read_csv_loader)

    def for_row(self, r: pd.Series) -> list[str]:
        h = int(r["horizon"])
        out = [f"The dataset ends {r['data_through'].date()}; this forecast is as of {r['as_of'].date()}, the last data date, not live."]
        if h == 7:
            b = self.boot.get()
            if b is not None and (b["h"] == 7).any():
                x = b[b["h"] == 7].iloc[0]
                lo, hi = x["ci95_low_strongest"], x["ci95_high_strongest"]
                tie = "which includes zero, so at 7 days the model is statistically a tie with naive" if lo <= 0 <= hi else "which excludes zero"
                out.append(f"At 7 days LightGBM improved on the strongest baseline by {x['point_estimate_pct']:+.1f}% in the 2020-2024 backtest "
                           f"(95% bootstrap interval {lo:+.1f}% to {hi:+.1f}%, {tie}); treat the point forecast as close to the last price.")
            else:
                out.append("At 7 days the point forecast is statistically close to the naive forecast (the last price); see the backtest report.")
        else:
            c, k = self.cov.get(), self.brk.get()
            if c is not None:
                x = c[(c["h"] == 30) & (c["method"] == "lgbm_abs") & (c["level"] == 0.95)]
                if len(x):
                    out.append(f"At 30 days the 95% interval is wide: on average {100 * x.iloc[0]['macro_avg_rel_width']:.0f}% of the forecast in the backtest.")
            if k is not None:
                y = k[(k["group_type"] == "event_group") & (k["group"] == "export_ban") & (k["h"] == 30) & (k["level"] == 0.8) & (k["method"] == "lgbm_abs")]
                if len(y):
                    out.append(f"Intervals under-cover in shock periods: in the 2020-2024 export-ban periods the 80% interval covered {100 * y.iloc[0]['coverage']:.0f}% of 30-day-ahead quotes.")
        gap = r.get("gap_before_origin_days")
        if gap is not None and pd.notna(gap) and gap >= self.long_gap:
            e = self.err.get()
            extra = ""
            if e is not None:
                z = e[(e["split_by"] == "gap_before_origin") & (e["h"] == h) & (e["segment"] == "7+ days")]
                if len(z):
                    extra = f" In the backtest the model was worse than the baseline after such gaps ({z.iloc[0]['log_impr_vs_strongest_pct']:+.0f}% on the log scale, {int(z.iloc[0]['n'])} forecasts)."
            out.append(f"The quote before the latest one was {int(gap)} days earlier.{extra}")
        stale = (r["data_through"] - r["origin_date"]).days
        if stale >= self.long_gap:
            out.append(f"This market's latest quote is {stale} days older than the end of the dataset.")
        if bool(r["in_sample"]):
            out.append("The target date has passed in the dataset and the model was fitted on that row: this is not an out-of-sample test.")
        return out


def _read_csv_loader(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


# --------------------------------------------------------------------------- app
def create_app(forecasts_path: Path | None = None, results_dir: Path | None = None, cfg: dict[str, Any] | None = None) -> FastAPI:
    """Build the application. Paths default to the configuration (env MANDI_FORECASTS / MANDI_RESULTS override)."""
    cfg = cfg or load_config()
    sv = cfg["serving"]
    fpath = Path(forecasts_path or os.environ.get("MANDI_FORECASTS") or ROOT / sv["forecasts_file"])
    rdir = Path(results_dir or os.environ.get("MANDI_RESULTS") or ROOT / sv["results_dir"])
    markets = [short_name(m) for m in cfg["markets"]]
    horizons = [int(h) for h in sv["horizons"]]
    version = sv["api"]["version"]
    store = _MtimeCache(fpath, _load_forecasts)
    caveats = Caveats(rdir, int(sv["api"]["long_gap_days"]))
    by_lower = {m.lower(): m for m in markets}
    app = FastAPI(title="Maharashtra onion price forecasts", version=version,
                  description="Forecasts of the modal onion price in five Maharashtra mandis. The dataset ends 2024-12-31, so 'latest' means as of the last data date, not live.")

    def need_data() -> pd.DataFrame:
        df = store.get()
        if df is None:
            raise HTTPException(status_code=503, detail=f"forecast file not found ({fpath.name}); run `python -m mandi.forecast`")
        return df

    def valid(market: str, horizon: int) -> tuple[str, int]:
        m = by_lower.get(market.strip().lower())
        if m is None:
            raise HTTPException(status_code=422, detail=f"unknown market '{market}'; allowed: {', '.join(markets)}")
        if horizon not in horizons:
            raise HTTPException(status_code=422, detail=f"horizon must be one of {horizons}")
        return m, horizon

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        df = store.get()
        if df is None:
            return HealthResponse(status="degraded", forecast_file_exists=False, api_version=version,
                                  note="forecast file missing; the dataset ends 2024-12-31 and forecasts are as of the last data date, not live")
        first = df.iloc[0]
        gen = pd.Timestamp(first["generated_at"])
        return HealthResponse(status="ok", forecast_file_exists=True, data_through=first["data_through"].date(), as_of=first["as_of"].date(),
                              forecast_generated_at=gen.to_pydatetime(), age_days=round((pd.Timestamp.now() - gen).total_seconds() / 86400, 3),
                              api_version=version, note="the dataset ends 2024-12-31; forecasts are as of the last data date, not live")

    @app.get("/markets", response_model=MarketsResponse)
    def list_markets() -> MarketsResponse:
        df = store.get()
        out = []
        for m in markets:
            latest = None
            if df is not None:
                g = df[(df["market"] == m) & df["is_latest"]]
                latest = g["origin_date"].max().date() if len(g) else None
            out.append(MarketInfo(market=m, horizons=horizons, latest_origin_date=latest))
        return MarketsResponse(markets=out)

    @app.get("/forecast", response_model=ForecastResponse, responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})
    def forecast(market: str = Query(..., description="one of the configured markets, case-insensitive"),
                 horizon: int = Query(..., description="7 or 30 (days ahead)")) -> ForecastResponse:
        m, h = valid(market, horizon)
        df = need_data()
        g = df[(df["market"] == m) & (df["horizon"] == h) & df["is_latest"]]
        if g.empty:
            raise HTTPException(status_code=404, detail=f"no forecast for {m} at horizon {h}")
        r = g.sort_values("origin_date").iloc[-1]
        return ForecastResponse(
            market=m, horizon=h, origin_date=r["origin_date"].date(), target_date=r["target_date"].date(), last_price=r["last_price"], point=r["point"],
            lo80=r["lo80"], hi80=r["hi80"], lo95=r["lo95"], hi95=r["hi95"], naive=r["naive"], n_calibration=int(r["n_calibration"]), model_spec=r["model_spec"],
            trained_through=r["trained_through"].date(), generated_at=pd.Timestamp(r["generated_at"]).to_pydatetime(), git_commit=r["git_commit"],
            as_of=r["as_of"].date(), data_through=r["data_through"].date(), in_sample=bool(r["in_sample"]),
            gap_before_origin_days=None if pd.isna(r["gap_before_origin_days"]) else float(r["gap_before_origin_days"]), caveats=caveats.for_row(r))

    @app.get("/forecast/history", response_model=HistoryResponse, responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})
    def forecast_history(market: str = Query(...), horizon: int = Query(...)) -> HistoryResponse:
        m, h = valid(market, horizon)
        df = need_data()
        g = df[(df["market"] == m) & (df["horizon"] == h)].sort_values("origin_date")
        if g.empty:
            raise HTTPException(status_code=404, detail=f"no forecast history for {m} at horizon {h}")
        nan = lambda v: None if pd.isna(v) else float(v)  # noqa: E731
        rows = [HistoryRow(origin_date=r.origin_date.date(), target_date=r.target_date.date(), last_price=r.last_price, point=r.point, lo80=nan(r.lo80), hi80=nan(r.hi80),
                           lo95=nan(r.lo95), hi95=nan(r.hi95), naive=r.naive, actual_price=nan(r.actual_price), in_sample=bool(r.in_sample)) for r in g.itertuples()]
        return HistoryResponse(market=m, horizon=h, as_of=g.iloc[0]["as_of"].date(), rows=rows,
                               note="Rows whose target has passed in the dataset were fitted on that row (in_sample = true): they show the model's recent track, not out-of-sample accuracy.")

    return app


app = create_app()
