# Project brief: Forecasting and anomaly detection for Indian mandi onion prices

You are helping a B.Tech student build a portfolio-grade data science project. The student must be able to explain every decision in an interview, so prefer simple, well-justified choices over clever ones, and explain what you did and why after each phase.

## Goal

Forecast the daily modal price of onions in 3 to 5 large Maharashtra mandis at 7-day and 30-day horizons, with prediction intervals, and flag anomalous price movements. Serve the forecasts through a small FastAPI service and a Streamlit dashboard.

The headline result is the percentage improvement in MAE of the best model over the **strongest baseline per market and horizon** (best of naive, mean of the last 3 observed quotes, and 365-day seasonal-naive) under honest rolling-origin backtesting. If no model beats the baselines at some horizon, report that plainly. Never invent, round up or cherry-pick numbers.

## Data

- Source: "Preprocessed onion 2014-2024 prices, weather and events" (Shelar, Hajare, Mangarule; Mendeley Data, DOI 10.17632/ds9jmxp9zy.1, v1, published 2025-12-04, licence CC BY 4.0). It replaces the CEDA download. Prices and arrivals come from Agmarknet; weather (rain, tmax, tmin) from IMD gridded data matched to the nearest grid cell; festivals and events from curated calendars. The student downloads the file manually to `data/raw/preprocessed_onion_2014_2024_prices_weather_events.parquet`. Do not scrape.
- Fields: date; state, district, market, `market_key`; variety, grade (all FAQ in Maharashtra); arrivals (tonnes); min, max, modal price (Rs/quintal); weather; calendar and festival flags; and many pre-computed lag, rolling and climate features. The page documents no columns, so every derived column must be treated as unverified (see `docs/data_audit.md`).
- Known structure: the file is not sorted; there can be several rows per (market, date), one per variety, with the same arrivals value repeated on each. The dataset's own preprocessing and imputation are not described. Rolling and lag features are computed over rows, not calendar days.
- Rule: use only raw columns (date, ids, prices, arrivals, weather observations, calendar and festival flags) as the base. Recompute lags and rolling features ourselves. Use a pre-computed feature only after it has been proven past-only.
- If `data/raw/` is empty or the file does not match these fields, stop and ask. Never generate fake or sample data to continue.
- Raw data and any derived data stay out of git (`.gitignore`); only code is committed. CC BY 4.0 would allow redistribution, but the file is 113 MB, so it stays out anyway.

## Decisions (confirmed by the student)

- **Markets:** five Maharashtra mandis: Solapur, Pune, Vashi New Mumbai, Kolhapur, Chattrapati Sambhajinagar (Jalgaon was replaced by Vashi after the Phase 1 review: its median arrivals are 50 t/day against 1,084 for Vashi; Vashi starts 2016-05-19). Nasik is excluded (214-day gap). Karad was considered but its median arrivals are only 15 t/day. Lasalgaon, Niphad, Pimpalgaon, Saykheda, Yeola and Chandwad are not in this dataset (case-insensitive search of market, district, market_key); the README must say so and present these as "five Maharashtra mandis".
- **Variety:** per market, the single most frequent variety is the primary series. Only its rows are used. If it has no row on a day, that day is missing; never substitute another variety. Arrivals are deduplicated to one value per market-day (the file repeats it on every variety row). Document the chosen variety per market.
- **Features:** use only raw columns plus calendar and festival columns. Ignore the file's lag, rolling, climate-anomaly, cum_rain, 5m and price_spike columns. Recompute all lags and rolling features on a calendar-day grid, so lag_7 means 7 days ago.
- **Weather:** version 1 has no weather features. Later, test lagged rain and temperature as an ablation and report whether they help.
- **Missing days:** build a daily grid per market and keep missing days missing. Fill short gaps (config limit) only to build features, never for the target. Evaluate only on target dates with an actual observed quote. Report the share of missing days per market.
- **Stale values:** count runs of 5 or more identical (min, max, modal) quotes in the chosen markets and propose a rule before changing anything.
- **Outliers:** handle price extremes per market with a robust rule (rolling median and MAD). Log every change. Only the chosen markets matter.
- **Backtest:** train on 2014-2019 and test forward across 2020-2024 (rolling-origin, expanding window), so the test period includes COVID and the export-ban periods.
- **Corrections:** six approved outlier prices are set to missing, never replaced: Sambhajinagar 2015-04-30, 2019-12-03, 2024-10-29; Kolhapur 2015-06-22, 2015-06-23, 2023-08-04. Originals are kept in `modal_price_raw` (also `min_price_raw`, `max_price_raw`), `was_corrected=True`. All other flagged points stay unchanged and flagged. The 0.3 outlier threshold was chosen after seeing 223 flags with the first rule and is not tuned further.
- **Evaluation:** only target dates with an actual observed quote are scored. Missing share is reported both against 4,018 days and against each market's own span.
- **Modelling evidence rule:** any EDA finding that drives a modelling choice (seasonal period, transform, differencing) must come from 2014-2019 only. 2020-2024 may be plotted for description.
- **Notebook execution:** `requirements-dev.txt` pins jupyter, ipykernel and nbconvert, but executing notebooks is blocked here (Application Control blocks `Scripts\jupyter-nbconvert.exe`). `notebooks/02_eda.ipynb` therefore has embedded figures and no executed outputs; do not work around the policy.
- **Pre-declared main result (Phase 3):** the main result is the configuration frozen in `configs/selected_models.yaml` (chosen on the 2017-2019 tuning folds only). The "likely error set to missing" and `is_stale` runs are sensitivity checks, reported side by side whatever they show, and never used to choose between models.
- **Phase 3 tuning history:** tuning run 1 had a silent NaN ETS fit that removed Kolhapur from the selection; it was fixed (explicit start values, non-finite guard, coverage guard) and tuning re-run. `prophet_anchored` was added before freezing at the student's request. No 2020-2024 result had been looked at.
- **Git policy:** the 12 EDA figures stay tracked; row-level data stays out of git; small aggregate tables (metrics, summary tables) under `results/` may be committed.
- **Sensitivity checks (Phase 3):** Vashi 2016-10-20 stays unchanged and flagged. Report (a) all "likely error" flagged points set to missing and (b) with and without `is_stale` dates.
- **Event E11 (28 Oct 2023, MEP USD 800/t to 31 Dec 2023)** is high-confidence: verified by the student against a Deccan Herald article dated 28 Oct 2023. The `mep_oct_dec_2023` period ends 2023-12-07 (the day before the E12 ban) and is one of the five high-confidence periods used for anomaly labels.
- **Phase 4 decisions:** the STL detector is kept exactly as reported and relabelled a "retrospective large-swing detector, not a spike detector"; detector 3 (`causal_daily`: 28-quote median baseline, 1.4826 x MAD of the previous 90 daily log changes, threshold 4) was declared after the first Phase 4 review and run once, untuned. No adaptive conformal variant was added.
- **Serving constraint (Phase 5):** the dataset ends 2024-12-31, so "latest" forecasts are as of the last data date, not live.
- **Environment:** Python 3.12.10 in a venv at `%USERPROFILE%\.venvs\fada`, outside OneDrive and Temp. Python 3.14 was abandoned: Windows Application Control blocked `statsmodels\tsa\statespace\_tools.cp314-win_amd64.pyd`, which breaks ETS and SARIMAX. A venv under Temp was also blocked. Smoke tests (a real fit of Prophet, LightGBM, ETS, SARIMAX, STL) pass on 3.12. Run tests with `.\scripts.ps1 test`.

## Repository layout

```
data/raw/  data/interim/  data/processed/
src/mandi/  (ingest.py, clean.py, features.py, models.py, backtest.py, anomalies.py, intervals.py)
api/main.py   app/dashboard.py
notebooks/    tests/    configs/config.yaml
results/      docs/     README.md   pyproject.toml or requirements.txt
```

All paths, markets, horizons, seeds and model settings live in `configs/config.yaml`. Pin dependency versions. Set random seeds everywhere.

## Phases (finish one, then stop for review)

### Phase 0: Setup
Create the layout, environment, `.gitignore`, config, and a `Makefile` or simple scripts for each stage. Done when `pytest` runs (even with a trivial test).

### Phase 1: Load and clean
- Load raw files, keep onion only, filter to Maharashtra, and choose 3 to 5 markets with the longest and most complete history. Show the coverage table and justify the choice.
- Handle: duplicate rows, inconsistent market or commodity names, non-positive or absurd prices, min > modal > max violations, unit changes, and obvious typos (use robust rules such as rolling median and MAD, and log every row changed or dropped).
- Build a daily panel per market. Mandis close on Sundays and holidays, so missing days are real. Add a `was_missing` flag. For modelling, forward-fill only short gaps (config limit, default 3 days) and never fill across longer gaps.
- Save `data/processed/panel.parquet` and write `docs/cleaning_log.md` listing every rule and how many rows it affected.
- Tests: no duplicate (market, date), dates sorted, prices positive, no fill beyond the gap limit.

### Phase 2: EDA
Notebook plus saved figures: trend, yearly and weekly seasonality, price vs arrivals, co-movement between markets, ADF and KPSS tests, ACF and PACF, structural breaks (for example export bans or policy changes). Write `docs/eda_findings.md` with the findings that will influence modelling.

### Phase 3: Baselines and models
**Forecast task (defined in `configs/config.yaml`).** A forecast origin is a calendar day on which the market has an observed quote. The target is the quote at origin + h calendar days (h = 7 and 30), scored only if that quote exists. Models forecast the log change from the origin's quote to the target.

**Baselines** (the benchmark to beat; the headline uses the best of the three per market and horizon):
- (a) naive: the last observed quote (the origin's quote).
- (b) the mean of the last 3 observed quotes (daily changes have negative autocorrelation, so a smoothed anchor may do better).
- (c) seasonal-naive, 365-day period: the nearest observed quote within 3 days of the target date one year earlier. A 7-day seasonal-naive is effectively naive and is not used.

**Models.** Statistical: ETS and SARIMA (statsmodels state-space versions, which handle missing days without filling them) and Prophet. ETS local level and ARIMA(0,1,1) are mathematically the same model and count as one result. Prophet is reported in two forms: `prophet` (off the shelf, ignores recent prices) and `prophet_anchored` (declared before freezing: forecast = last observed log price + Prophet's predicted log price at the target date minus its predicted log price at the origin date, from the model fitted at the latest refit). Both are tuned on the same 2017-2019 folds and both stay in the report. ML: LightGBM, one model per horizon, target = log change from the origin's quote.

**Features** (all computed on a calendar-day grid from data up to the origin; "lag k" means the last observed quote at or before origin - k): lagged log changes; deviation of the origin's quote from the mean of the last 3 observed quotes and from the 30-day mean; recent volatility; days since the previous observed quote; other markets' lagged prices and their average recent change; arrivals only lagged (strictly before the origin); calendar (month, week of year) and an expected-closure flag for the target date. No weather in version 1. A test proves no leakage (shifting future values must not change features at the origin).

**Backtest.** Rolling origin, expanding window, never a random split. Hyperparameters and features are chosen only with folds inside 2014-2019 (tuning folds: origins in 2017, 2018, 2019, each trained on everything before the fold). The final evaluation uses an expanding window with origins from 2020-01-01 to 2024-12-31, with models retrained at a fixed interval (set in the config, justified in `docs/model_report.md`). Training rows only use origins whose target was already known at the retrain date. Nothing is changed after seeing 2020-2024 results, and every run is logged in `results/`.

**Fairness.** Every model and baseline is scored on exactly the same set of (market, origin, horizon) triples. A triple that any model cannot forecast is dropped for all, and the number dropped is reported.

**Metrics** per market and horizon on the rupee scale: MAE, RMSE, MASE (scaled by the in-sample naive error), sMAPE; plus MAE on the log scale. A Diebold-Mariano test (Harvey correction, HAC with h-1 lags) of the best model against the strongest baseline per market and horizon; forecasts overlap, so the test is approximate. The "best model" is the one that did best on the 2017-2019 tuning folds, fixed before the test period was run; the model that did best on the test period is reported separately and labelled optimistic.

**Sensitivity.** Report (a) every "likely error" flagged point set to missing, and (b) with and without `is_stale` dates.

**Outputs.** `results/backtest_metrics.csv`, a summary table of percentage improvement over the strongest baseline, runtime per model, and `docs/model_report.md` with what worked, what did not, error analysis by event period (export-ban vs normal periods) and by gap length since the last observation. Report real numbers, including where no model beats the baselines. Do not tune until a model wins.

### Phase 4: Intervals and anomalies
- Prediction intervals via LightGBM quantile regression or split conformal prediction. Report empirical coverage versus nominal (for example 80% and 95%) per horizon.
- Anomaly detection: STL decomposition and model residuals with a robust threshold. Output a table of flagged dates and check them against arrivals. Do not claim causes you cannot support.

### Phase 5: Serving
**Constraint, stated everywhere: the dataset ends 2024-12-31, so "latest" forecasts are as of the last data date, not live.**
- `python -m mandi.forecast [--as-of DATE]` trains the frozen LightGBM specs on rows whose target date is strictly before the as-of date (default: the day after the last quote), forecasts from each market's latest origin and its last 60 origins, adds 80% and 95% conformal intervals (calibrated on resolved out-of-sample errors of the previous 365 days; the Phase 4 errors for 2024), and writes `results/forecasts.parquet` (not in git) plus a row in `results/run_log.csv`. A regression test links serving to the backtest (the 2024-06-01 refit reproduces `results/fair_forecasts.parquet`).
- FastAPI (`api/main.py`, Pydantic v2, plain uvicorn): `GET /health`, `/markets`, `/forecast?market=&horizon=` (market case-insensitive; horizon 7 or 30; 404, 422, 503; a `caveats` list assembled from `results/`), and `/forecast/history`. The file is reloaded when its mtime changes.
- Streamlit dashboard (`app/dashboard.py`): market and horizon selectors; tabs Forecast, Backtest, Anomalies, About. Every number is read from `results/` or the API.
- Dockerfile and docker-compose.yml (one image, two services with healthchecks, `./results` and `./data/processed` mounted read-only; data stays out of git and out of the image). Not built or started here (Docker is not installed); see `docs/serving.md`.
- Pinned dependencies: `requirements.txt` (full) and `requirements-serving.txt` (the image).

### Phase 6: README and report
README includes: problem, data source and its limits (including reporting lag), architecture, the results table generated from `results/` (never typed by hand), error analysis, limitations, how to reproduce, and what you would do next. **How to reproduce: the tuning run must be run first** (`python -m mandi.run_phase3 tune`, then commit `configs/selected_models.yaml`), because the seeded interval calibration (`docs/interval_report.md`) and the anomaly validation use the saved forecasts of that tuning run; reproduction therefore goes: Phase 1 clean, tune, final, intervals, anomalies, forecast. **Next steps to list: adaptive conformal inference** to repair the interval under-coverage in shock periods (decided at the Phase 4 review; not implemented), and a daily anomaly detector with a threshold calibrated to a target false-positive rate (the three detectors in `docs/anomaly_report.md` flag 5-20% of ordinary dates or miss most short spikes). The headline is the percentage improvement in MAE over the strongest baseline per market and horizon (naive, mean of the last 3 observed quotes, 365-day seasonal-naive), with the Diebold-Mariano result beside it, plus the five-market caveat (Lasalgaon is not in this dataset).

## Working rules

- Work one phase at a time. At the end of each phase, summarise what was done, show key numbers from files, list assumptions and open questions, then wait.
- Small, meaningful commits with clear messages.
- Use type hints, docstrings on public functions, and logging instead of prints.
- Do not add dependencies or features beyond this brief without asking.
- If something is ambiguous or the data contradicts an assumption here, ask rather than guess.