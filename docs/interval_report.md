# Prediction intervals for the headline LightGBM (Phase 4A)

## Method (declared before any result was looked at; no switching afterwards)

- **Primary: rolling split-conformal, symmetric absolute-error quantile**, on the log scale and exponentiated to rupees, at 80%, 95%. For a forecast made at origin t (market, horizon) the calibration set is the errors y_log - forecast of earlier forecasts of the same market and horizon whose **target date is strictly before t**, within the trailing 365 days, with at least 100 errors; the quantile is the k-th smallest absolute error with k = ceil((n + 1) x level) (the finite-sample correction).
- **Sensitivity method:** the same, but each error is divided by a volatility scale known at its own origin (mean absolute log change over the last 30 observed quotes) and the quantile is rescaled by the current origin's scale.
- **Comparator:** the same primary procedure applied to naive's errors, to show whether LightGBM's intervals are narrower at the same coverage.
- **Out-of-sample forecasts used:** `results/fair_forecasts.parquet` (the 2020-2024 final run; 10,166 forecasts of the frozen headline LightGBM).
- **Burn-in:** the first calibration windows were seeded with the saved 2017-2019 tuning-fold forecasts of the same frozen LightGBM (run `tune_20261009_225301`; they are out-of-sample forecasts made with the same monthly expanding-window protocol). Without seeding, every market would have lacked intervals for roughly its first 100 resolved forecasts of 2020. Forecasts without an interval: **0** of 10,166. Smallest calibration set used: 102 errors.
- Coverage is the share of realised quotes inside the interval. Intervals are for the quote 7 or 30 calendar days after the origin. Bootstrap intervals are 95% block-bootstrap intervals (30-day blocks of origin days, the same days for all markets, 2000 resamples, seed 42).

## 1. Coverage against nominal (2020-2024)

Macro (equal-weight) coverage of the primary method, with its block-bootstrap interval:

| h | nominal_% | coverage_% | ci_low_% | ci_high_% | macro_avg_width_rs | relative_width_% | macro_interval_score_rs |
|---|---|---|---|---|---|---|---|
| 7 | 80 | 81.5 | 78.2 | 85.1 | 716.07 | 45.0 | 1092.07 |
| 7 | 95 | 95.2 | 93.4 | 96.7 | 1300.44 | 80.3 | 1670.04 |
| 30 | 80 | 82.1 | 77.3 | 86.9 | 1496.97 | 92.1 | 2009.77 |
| 30 | 95 | 95.1 | 92.7 | 97.2 | 2419.74 | 151.6 | 2890.58 |


Relative width = (upper - lower) / forecast, averaged. Interval score = width + (2/alpha) x shortfall in rupees (lower is better).

### 1.1 Per market and horizon (primary method)

| market | h | nominal_% | n_with_interval | n_without_interval | coverage [95% CI] | avg_width_rs | rel_width_% | interval_score_rs |
|---|---|---|---|---|---|---|---|---|
| Kolhapur | 7 | 80 | 1149 | 0 | 81.5 [77.2, 85.9] | 637.77 | 37.9 | 916.02 |
| Kolhapur | 7 | 95 | 1149 | 0 | 95.6 [93.3, 97.4] | 1198.17 | 69.2 | 1399.89 |
| Kolhapur | 30 | 80 | 933 | 0 | 83.2 [77.8, 88.4] | 1353.87 | 78.0 | 1804.16 |
| Kolhapur | 30 | 95 | 933 | 0 | 95.3 [92.2, 97.8] | 2078.7 | 120.7 | 2406.19 |
| Pune | 7 | 80 | 1211 | 0 | 83.3 [79.6, 87.2] | 731.37 | 44.0 | 1086.03 |
| Pune | 7 | 95 | 1211 | 0 | 96.0 [93.9, 97.9] | 1300.28 | 76.3 | 1703.84 |
| Pune | 30 | 80 | 983 | 0 | 82.6 [76.6, 87.9] | 1470.31 | 88.5 | 1950.77 |
| Pune | 30 | 95 | 983 | 0 | 96.5 [93.9, 98.6] | 2389.36 | 144.8 | 2736.89 |
| Sambhajinagar | 7 | 80 | 1185 | 0 | 81.0 [77.3, 85.0] | 721.3 | 57.0 | 1098.34 |
| Sambhajinagar | 7 | 95 | 1185 | 0 | 94.4 [92.2, 96.4] | 1228.0 | 96.6 | 1620.51 |
| Sambhajinagar | 30 | 80 | 956 | 0 | 81.0 [75.8, 86.3] | 1377.72 | 107.4 | 1836.97 |
| Sambhajinagar | 30 | 95 | 956 | 0 | 95.2 [92.2, 97.5] | 2265.37 | 179.0 | 2632.91 |
| Solapur | 7 | 80 | 956 | 0 | 80.2 [75.5, 84.6] | 667.54 | 46.0 | 1086.42 |
| Solapur | 7 | 95 | 956 | 0 | 94.6 [91.6, 97.0] | 1294.3 | 87.6 | 1719.55 |
| Solapur | 30 | 80 | 773 | 0 | 80.3 [73.3, 86.9] | 1399.59 | 92.8 | 1953.03 |
| Solapur | 30 | 95 | 773 | 0 | 93.9 [89.6, 97.3] | 2446.96 | 168.3 | 3195.06 |
| Vashi | 7 | 80 | 1113 | 0 | 81.7 [77.5, 86.7] | 822.34 | 40.4 | 1273.53 |
| Vashi | 7 | 95 | 1113 | 0 | 95.4 [93.1, 97.6] | 1481.45 | 72.0 | 1906.42 |
| Vashi | 30 | 80 | 907 | 0 | 83.5 [77.4, 89.5] | 1883.35 | 93.8 | 2503.94 |
| Vashi | 30 | 95 | 907 | 0 | 94.7 [91.1, 97.9] | 2918.33 | 145.4 | 3481.87 |


### 1.2 Calibration plot and fan charts

![Calibration](../results/figures/interval_calibration.png)

![Fan charts](../results/figures/interval_fan_charts.png)

## 2. LightGBM against the naive comparator at the same procedure

| market | h | level | coverage|lgbm_abs | coverage|naive_abs | avg_width_rs|lgbm_abs | avg_width_rs|naive_abs | width_ratio_lgbm_over_naive | interval_score_ratio_lgbm_over_naive |
|---|---|---|---|---|---|---|---|---|
| Kolhapur | 7 | 80 | 81.5 | 82.7 | 637.77 | 692.81 | 0.92 | 0.94 |
| Kolhapur | 7 | 95 | 95.6 | 95.5 | 1198.17 | 1246.95 | 0.96 | 0.94 |
| Kolhapur | 30 | 80 | 83.2 | 81.8 | 1353.87 | 1515.6 | 0.89 | 0.87 |
| Kolhapur | 30 | 95 | 95.3 | 94.7 | 2078.7 | 2416.89 | 0.86 | 0.85 |
| Pune | 7 | 80 | 83.3 | 82.7 | 731.37 | 779.11 | 0.94 | 0.98 |
| Pune | 7 | 95 | 96.0 | 96.0 | 1300.28 | 1373.95 | 0.95 | 0.98 |
| Pune | 30 | 80 | 82.6 | 83.1 | 1470.31 | 1651.07 | 0.89 | 0.88 |
| Pune | 30 | 95 | 96.5 | 95.4 | 2389.36 | 2757.06 | 0.87 | 0.82 |
| Sambhajinagar | 7 | 80 | 81.0 | 80.8 | 721.3 | 794.09 | 0.91 | 0.9 |
| Sambhajinagar | 7 | 95 | 94.4 | 95.1 | 1228.0 | 1386.89 | 0.89 | 0.89 |
| Sambhajinagar | 30 | 80 | 81.0 | 80.0 | 1377.72 | 1520.23 | 0.91 | 0.9 |
| Sambhajinagar | 30 | 95 | 95.2 | 94.7 | 2265.37 | 2490.93 | 0.91 | 0.9 |
| Solapur | 7 | 80 | 80.2 | 80.6 | 667.54 | 723.29 | 0.92 | 0.95 |
| Solapur | 7 | 95 | 94.6 | 94.5 | 1294.3 | 1352.49 | 0.96 | 0.91 |
| Solapur | 30 | 80 | 80.3 | 79.3 | 1399.59 | 1680.88 | 0.83 | 0.88 |
| Solapur | 30 | 95 | 93.9 | 94.8 | 2446.96 | 2879.77 | 0.85 | 0.95 |
| Vashi | 7 | 80 | 81.7 | 81.1 | 822.34 | 848.52 | 0.97 | 0.96 |
| Vashi | 7 | 95 | 95.4 | 95.1 | 1481.45 | 1554.51 | 0.95 | 0.98 |
| Vashi | 30 | 80 | 83.5 | 81.0 | 1883.35 | 2112.01 | 0.89 | 0.86 |
| Vashi | 30 | 95 | 94.7 | 95.0 | 2918.33 | 3380.92 | 0.86 | 0.87 |


Mean over the five markets of the ratio LightGBM / naive (below 1 = LightGBM narrower or better):

| h | level | width_ratio_lgbm_over_naive | interval_score_ratio_lgbm_over_naive |
|---|---|---|---|
| 7 | 80 | 0.93 | 0.95 |
| 7 | 95 | 0.94 | 0.94 |
| 30 | 80 | 0.88 | 0.88 |
| 30 | 95 | 0.87 | 0.88 |


LightGBM's interval is narrower than naive's in 20 of 20 market-horizon-level cells.

## 3. Coverage by year, event group, gap and volatility (pooled over markets)

Coverage in % of the primary method and of the naive comparator; `n` = forecasts with an interval at that level. Event groups follow the target date (see `docs/model_report.md`); the `mep` group is the Oct-Dec 2023 minimum-export-price period (event E11 verified against a Deccan Herald article dated 28 Oct 2023, ending the day before the E12 ban).

### By target year

| group | LGBM 80% (h=30) | LGBM 80% (h=7) | LGBM 95% (h=30) | LGBM 95% (h=7) | naive 80% (h=30) | naive 80% (h=7) | naive 95% (h=30) | naive 95% (h=7) | n_h7 | n_h30 |
|---|---|---|---|---|---|---|---|---|---|---|
| 2020 | 78.2 | 83.0 | 93.1 | 97.1 | 72.7 | 82.4 | 92.4 | 96.3 | 922.0 | 634.0 |
| 2021 | 87.0 | 82.8 | 96.1 | 95.1 | 88.8 | 84.7 | 95.2 | 95.8 | 1109.0 | 931.0 |
| 2022 | 84.3 | 80.8 | 97.5 | 96.0 | 78.5 | 80.6 | 97.1 | 96.4 | 1124.0 | 952.0 |
| 2023 | 77.2 | 80.4 | 93.6 | 92.2 | 76.4 | 78.6 | 93.1 | 92.0 | 1250.0 | 1039.0 |
| 2024 | 83.2 | 81.5 | 95.2 | 96.2 | 86.7 | 82.3 | 96.3 | 96.2 | 1209.0 | 996.0 |


### By event group (normal vs ban periods)

| group | LGBM 80% (h=30) | LGBM 80% (h=7) | LGBM 95% (h=30) | LGBM 95% (h=7) | naive 80% (h=30) | naive 80% (h=7) | naive 95% (h=30) | naive 95% (h=7) | n_h7 | n_h30 |
|---|---|---|---|---|---|---|---|---|---|---|
| covid | 89.5 | 66.7 | 100.0 | 88.1 | 92.1 | 66.7 | 100.0 | 97.6 | 42.0 | 38.0 |
| export_ban | 72.3 | 76.8 | 91.8 | 94.5 | 70.5 | 77.3 | 90.9 | 93.9 | 1030.0 | 792.0 |
| mep | 53.8 | 71.5 | 75.5 | 88.6 | 50.9 | 70.7 | 69.8 | 87.8 | 123.0 | 106.0 |
| normal | 85.1 | 83.1 | 96.5 | 95.6 | 84.2 | 83.1 | 96.5 | 95.7 | 4419.0 | 3616.0 |


### By days since the previous quote before the origin

| group | LGBM 80% (h=30) | LGBM 80% (h=7) | LGBM 95% (h=30) | LGBM 95% (h=7) | naive 80% (h=30) | naive 80% (h=7) | naive 95% (h=30) | naive 95% (h=7) | n_h7 | n_h30 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 day (consecutive) | 82.6 | 81.3 | 95.3 | 95.1 | 81.6 | 81.7 | 95.0 | 95.2 | 4183.0 | 3260.0 |
| 2 days | 82.3 | 82.3 | 95.4 | 95.0 | 80.7 | 80.5 | 95.4 | 94.9 | 1147.0 | 1028.0 |
| 3-6 days | 74.5 | 83.8 | 92.6 | 96.9 | 75.7 | 84.9 | 91.8 | 96.9 | 259.0 | 243.0 |
| 7+ days | 95.2 | 76.0 | 100.0 | 96.0 | 95.2 | 84.0 | 100.0 | 100.0 | 25.0 | 21.0 |


### By volatility tercile of the origin (within market)

| group | LGBM 80% (h=30) | LGBM 80% (h=7) | LGBM 95% (h=30) | LGBM 95% (h=7) | naive 80% (h=30) | naive 80% (h=7) | naive 95% (h=30) | naive 95% (h=7) | n_h7 | n_h30 |
|---|---|---|---|---|---|---|---|---|---|---|
| high | 81.6 | 79.6 | 95.5 | 95.6 | 80.4 | 79.5 | 95.9 | 95.2 | 1889.0 | 1497.0 |
| low | 83.0 | 83.8 | 95.0 | 95.0 | 80.8 | 83.2 | 94.8 | 95.0 | 1868.0 | 1527.0 |
| mid | 81.9 | 81.4 | 95.1 | 95.1 | 82.1 | 82.2 | 94.2 | 95.6 | 1857.0 | 1528.0 |


## 4. Sensitivity method: volatility-scaled conformal

| h | method | level | macro_coverage | ci_low | ci_high | macro_avg_rel_width | macro_interval_score_rs |
|---|---|---|---|---|---|---|---|
| 7 | LightGBM conformal (primary) | 80 | 81.5 | 78.2 | 85.1 | 45.0 | 1092.07 |
| 7 | LightGBM conformal (primary) | 95 | 95.2 | 93.4 | 96.7 | 80.3 | 1670.04 |
| 7 | LightGBM volatility-scaled (sensitivity) | 80 | 80.0 | 76.7 | 83.4 | 45.6 | 1126.65 |
| 7 | LightGBM volatility-scaled (sensitivity) | 95 | 94.3 | 92.2 | 96.1 | 88.9 | 1883.59 |
| 30 | LightGBM conformal (primary) | 80 | 82.1 | 77.3 | 86.9 | 92.1 | 2009.77 |
| 30 | LightGBM conformal (primary) | 95 | 95.1 | 92.7 | 97.2 | 151.6 | 2890.58 |
| 30 | LightGBM volatility-scaled (sensitivity) | 80 | 79.9 | 75.0 | 84.4 | 94.3 | 2168.11 |
| 30 | LightGBM volatility-scaled (sensitivity) | 95 | 93.9 | 90.9 | 96.3 | 205.8 | 3946.81 |


80% coverage by volatility tercile, primary vs scaled (%):

| group | primary (h=30) | primary (h=7) | scaled (h=30) | scaled (h=7) |
|---|---|---|---|---|
| high | 81.6 | 79.6 | 90.8 | 87.7 |
| low | 83.0 | 83.8 | 67.5 | 71.4 |
| mid | 81.9 | 81.4 | 81.8 | 81.0 |


## 5. Forecasts without an interval (burn-in)

| market | h | forecasts | without_interval | min_calibration_errors | median_calibration_errors | seeded_with_tuning_forecasts |
|---|---|---|---|---|---|---|
| Kolhapur | 7 | 1149 | 0 | 163 | 239.0 | True |
| Kolhapur | 30 | 933 | 0 | 122 | 199.0 | True |
| Pune | 7 | 1211 | 0 | 195 | 248.0 | True |
| Pune | 30 | 983 | 0 | 147 | 209.0 | True |
| Sambhajinagar | 7 | 1185 | 0 | 185 | 238.0 | True |
| Sambhajinagar | 30 | 956 | 0 | 146 | 196.0 | True |
| Solapur | 7 | 956 | 0 | 139 | 195.5 | True |
| Solapur | 30 | 773 | 0 | 102 | 162.0 | True |
| Vashi | 7 | 1113 | 0 | 181 | 220.0 | True |
| Vashi | 30 | 907 | 0 | 149 | 185.0 | True |


<!-- COMMENTARY BELOW IS WRITTEN BY HAND; EVERYTHING ABOVE IS GENERATED -->


## 6. What the intervals show (hand-written; every number is read from the tables above)

### 6.1 Overall, the intervals are well calibrated

- Pooled over the five markets the 80% intervals cover **81.5%** of realised quotes at 7 days (95% bootstrap interval 78.2-85.1) and **82.1%** at 30 days (77.3-86.9); the 95% intervals cover **95.2%** (93.4-96.7) and **95.1%** (92.7-97.2). Every market's bootstrap interval contains the nominal level, at both horizons and both levels (section 1.1).
- The calibration curve (section 1.2) stays within about three points of the diagonal from 50% to 95%; the primary method sits slightly above it in the middle levels.
- No forecast was left without an interval: the first calibration windows were **seeded with the 2017-2019 tuning-fold forecasts** (saved in the tuning run, not in git). Without that seeding, roughly the first 100 resolved forecasts of 2020 in each market would have had none. The smallest calibration set actually used had 102 errors.

### 6.2 LightGBM's intervals are narrower than naive's at the same coverage, but still wide

- In all 20 market-horizon-level cells LightGBM's interval is narrower than naive's. Averaged over markets, the width ratio LightGBM / naive is 0.93 (80%) and 0.94 (95%) at 7 days and 0.88 and 0.87 at 30 days; the interval-score ratios are 0.95, 0.94, 0.88 and 0.88. This is consistent in direction with LightGBM's small accuracy gain.
- **The intervals are wide.** At 7 days the 80% interval averages Rs 716 wide (45% of the forecast) and the 95% interval Rs 1,300 (80%). At 30 days the 80% interval averages Rs 1,497 (92% of the forecast) and the 95% interval Rs 2,420 (152%): a 95% interval for a 30-day-ahead price is typically wider than the forecast itself. For comparison, LightGBM's mean absolute error is about Rs 219 at 7 days and Rs 424 at 30 days. A relative width of 1.5 corresponds to an interval running from about half to about twice the forecast, so the 30-day 95% intervals say little beyond that.

### 6.3 Coverage falls short in shocks

Conformal prediction guarantees coverage on average, not within every kind of period, and the breakdowns show the gap plainly (section 3):

- **Export-ban periods** (the 2019-20, 2020-21 and 2023-24 bans, whose dates are high-confidence in `docs/eda_findings.md`): the 80% interval covers **76.8%** at 7 days and **72.3%** at 30 days, and the 95% interval **94.5%** and **91.8%**.
- **Oct-Dec 2023 minimum-export-price period** (event E11, 28 October 2023, verified against a Deccan Herald article of that date; the period ends the day before the E12 ban): 71.5% and **53.8%** for the 80% interval, 88.6% and **75.5%** for the 95% interval (n = 123 and 106). This is the worst case in the results.
- **COVID lockdown:** 66.7% at 7 days for the 80% interval, but only 42 forecasts, so too few to read.
- **Normal periods** are slightly **over**-covered (83.1% and 85.1% for 80%; 95.6% and 96.5% for 95%). The average coverage is right because normal periods over-cover and shocks under-cover.
- By year, the weakest are 2023 (95% interval: 92.2% at 7 days, 93.6% at 30 days) and 2020 (30-day 80% interval: 78.2%); for 2023 the 80% interval covers 77.2% at 30 days.
- In the fan charts (Pune and Solapur, 30 days) the quotes outside the 95% band cluster at the **start** of the 2020 and 2023-24 ban episodes. Because the calibration window is the trailing 365 days, the interval is still narrow when a shock begins and widens only after the large errors have resolved. I looked at two markets and three periods only.
- After a gap of 3-6 days before the origin, 30-day coverage is 74.5% (80%) and 92.6% (95%); the 7+ day bucket has only 25 and 21 forecasts.
- Coverage is roughly the same across volatility terciles for the primary method (80%: 79.6-83.8% at 7 days, 81.6-83.0% at 30 days), so the volatility scale does not identify the periods where it fails.

### 6.4 The volatility-scaled sensitivity method does not help

It has the same overall coverage (80.0% and 79.9% at 80%; 94.3% and 93.9% at 95%) but wider 95% intervals (relative width 89% against 80% at 7 days; 206% against 152% at 30 days) and worse interval scores at both levels and horizons (for example 3,947 against 2,891 at 30 days and 95%). It over-corrects by volatility: low-volatility origins are under-covered (80% interval: 71.4% at 7 days and 67.5% at 30 days) and high-volatility origins over-covered (87.7% and 90.8%). It improves the 30-day export-ban coverage (72.3% to 77.8%) but worsens the Oct-Dec 2023 period (53.8% to 43.4%). The primary method was declared in advance and stays the primary method.

### 6.5 Limitations

- Coverage is marginal, not conditional: it is right on average and wrong in shocks (6.3).
- The 365-day trailing window reacts to a regime change only after the errors resolve, which for a 30-day horizon is 30 days later. **No adaptive variant was added** (decided at the Phase 4 review); see the next steps.
- Intervals are symmetric on the log scale; that is a choice, not something tested against asymmetric alternatives.
- Seeding used 2017-2019 forecasts that exist only in the local tuning run; reproducing this needs that run (`results/runs/` is not in git).
- The block bootstrap resamples origin days, keeping the same days for all markets; the intervals for individual markets are wide because each has about 1,000 forecasts that are serially correlated.
- Five markets that move together are not five independent tests (see `docs/model_report.md`, section 11).

### 6.6 Next steps

- **Adaptive conformal inference (ACI)**: update the working miscoverage level online after each resolved forecast, so that intervals widen quickly after a run of misses. This is the standard remedy for the shock under-coverage in 6.3 (export-ban periods, and especially the Oct-Dec 2023 MEP period). It would be a new, separately declared method; the primary method in this report is unchanged.
- Re-test at the same breakdowns (event group, gap, volatility) and report whether conditional coverage in shocks improves, and at what cost in width.
- A reproduction note: the seeded calibration needs the saved forecasts of the tuning run (`results/runs/tune_20261009_225301`), so the tuning run must be reproduced first.
