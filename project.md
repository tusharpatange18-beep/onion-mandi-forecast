# Project brief: Forecasting and anomaly detection for Indian mandi onion prices

You are helping a B.Tech student build a portfolio-grade data science project. The student must be able to explain every decision in an interview, so prefer simple, well-justified choices over clever ones, and explain what you did and why after each phase.

## Goal

Forecast the daily modal price of onions in 3 to 5 large Maharashtra mandis at 7-day and 30-day horizons, with prediction intervals, and flag anomalous price movements. Serve the forecasts through a small FastAPI service and a Streamlit dashboard.

The headline result is the percentage improvement of the best model over a seasonal-naive baseline under honest rolling-origin backtesting. If no model beats the baseline at some horizon, report that plainly. Never invent, round up or cherry-pick numbers.

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
- Baselines: naive and seasonal-naive. These are the benchmark to beat.
- Statistical: ETS and SARIMA (statsmodels), Prophet.
- ML: LightGBM with lag features, rolling mean and std, calendar features, and arrivals.
- Feature rule: every feature at forecast origin `t` uses only data up to `t`. Arrivals are only usable as lagged values unless forecasted. Write a test that proves no leakage (for example, shifting future values must not change features at `t`).
- Backtesting: rolling-origin with an expanding window. Never use a random split. Never tune on the final test period; tune only on earlier folds.
- Metrics: MAE, RMSE, MASE and sMAPE, per market and per horizon. Save to `results/backtest_metrics.csv`, plus a summary table showing percentage improvement over seasonal-naive.
- Write `docs/model_report.md` covering what worked, what did not, and where models fail (festivals, shocks, long gaps).

### Phase 4: Intervals and anomalies
- Prediction intervals via LightGBM quantile regression or split conformal prediction. Report empirical coverage versus nominal (for example 80% and 95%) per horizon.
- Anomaly detection: STL decomposition and model residuals with a robust threshold. Output a table of flagged dates and check them against arrivals. Do not claim causes you cannot support.

### Phase 5: Serving
- A scheduled-style script generates forecasts and writes them to `results/forecasts.parquet`.
- FastAPI reads the latest forecasts: `GET /forecast?market=&horizon=` returns point forecast and interval, plus `/health`. Pydantic models, input validation, and tests.
- Streamlit dashboard: history, forecast with interval, anomaly markers, and the backtest results table.
- Dockerfile and docker-compose for the API and dashboard.

### Phase 6: README and report
README includes: problem, data source and its limits (including reporting lag), architecture, the results table generated from `results/` (never typed by hand), error analysis, limitations, how to reproduce, and what you would do next.

## Working rules

- Work one phase at a time. At the end of each phase, summarise what was done, show key numbers from files, list assumptions and open questions, then wait.
- Small, meaningful commits with clear messages.
- Use type hints, docstrings on public functions, and logging instead of prints.
- Do not add dependencies or features beyond this brief without asking.
- If something is ambiguous or the data contradicts an assumption here, ask rather than guess.