# Serving (Phase 5)

**Constraint: the dataset ends 2024-12-31, so "latest" forecasts are as of the last data date, not live.** The latest forecasts look ahead
from the last quote; nothing in this repository fetches new prices.

## Order of work

1. Build the cleaned panel and the backtest (`docs/model_report.md`). To reproduce the interval calibration you need the **tuning run first**:
   the seeded conformal calibration uses the saved forecasts of `results/runs/tune_*` (not in git).
2. Generate the serving forecasts: `python -m mandi.forecast [--as-of YYYY-MM-DD]` (with `PYTHONPATH=src`). It trains the frozen LightGBM specs
   from `configs/selected_models.yaml` on rows whose **target date is strictly before the as-of date** (default: the day after the last quote),
   forecasts from each market's latest origin and its last 60 origins, adds 80% and 95% conformal intervals calibrated on resolved out-of-sample
   errors of the previous 365 days, writes `results/forecasts.parquet` (not in git) and adds a row to `results/run_log.csv`.
3. Start the API: `python -m uvicorn api.main:app --port 8000` (plain uvicorn; set `PYTHONPATH=src;.` on Windows or `src:.` elsewhere).
4. Start the dashboard: `python -m streamlit run app/dashboard.py` (it calls the API; set `MANDI_API_URL` if the API is elsewhere).
5. Or with Docker: `docker compose up --build` (see the status below).

## API

| Endpoint | Meaning |
|---|---|
| `GET /health` | status, whether the forecast file exists, data-through date, forecast age in days, API version |
| `GET /markets` | the five configured markets with their latest origin |
| `GET /forecast?market=&horizon=` | the latest forecast: all fields of `results/forecasts.parquet` plus a `caveats` list. Market is case-insensitive; horizon is 7 or 30 |
| `GET /forecast/history?market=&horizon=` | the last 60 forecasts (the dashboard's recent track) |

Status codes: 200, 404 (no forecast for a valid market and horizon), 422 (invalid input), 503 (forecast file missing). The file is reloaded when its
modification time changes. Caveat texts are assembled from `results/*.csv` at request time, so the numbers in them always match the result files.

## What the forecasts are, and are not

- The forecast for a recent origin whose target date has already passed in the dataset was fitted on that very row (`in_sample = true`): it shows the model's
  recent track, not out-of-sample accuracy. Only forecasts whose target date lies after the as-of date are genuinely out of sample.
- At 7 days the model is statistically close to naive; 30-day intervals are wide and under-cover in shock periods (`docs/model_report.md`, `docs/interval_report.md`).

## Verification status

See the final report of the phase. In short: the forecast script, API and dashboard were run and tested on this machine; **the Docker image and docker-compose were not built or started
because Docker is not installed here** (`docker compose config` could not be run either); the files were checked statically and the pinned `requirements-serving.txt` was installed into a fresh
Python 3.12 venv and every module the image runs was imported there.
