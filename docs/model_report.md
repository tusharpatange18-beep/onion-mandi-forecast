# Model report (Phase 3)

## Rules declared before the final run

- **Primary metric:** MAE in rupees per quintal. MAE on the log scale is secondary. RMSE, MASE (scaled by the in-sample naive error of 2014-2019) and sMAPE are in section 4.
- **Strongest baseline** for each market and horizon = whichever of (a) naive (last observed quote), (b) the mean of the last 3 observed quotes and (c) the 365-day seasonal-naive has the lowest rupee MAE on the final window 2020-2024. This is a conservative, hindsight choice: it makes the benchmark as hard as possible. Improvement over plain naive is reported next to it.
- **Headline model:** at each horizon, the family with the best pooled score on the 2017-2019 tuning folds, using the frozen configurations. That is **LightGBM at both horizons** (7 days: `lgbm|huber_nl16_mc60_n150`; 30 days: `lgbm|huber_nl16_mc20_n150`). The per-market tuning-selected model and the model that happens to be best on the test window are secondary rows (section 3); the latter is optimistic because it is chosen after seeing the results. Every family is shown on every market and horizon.
- **Task:** a forecast origin is a day with an observed quote; the target is the quote 7 or 30 calendar days later and is scored only if it exists. Every model is scored on the same 10,166 (market, origin, horizon) triples; 254 of 10,420 were dropped for all models because baseline (c) has no quote within 3 days of the target date one year earlier.
- **Protocol:** models are refit on the first day of every month using only rows whose target date is before that day (embargo; tested). Hyperparameters were chosen on 2017-2019 folds only. The final run was made once, on a clean tree. The sensitivity runs (section 7) and the ablation (section 9) are never used to choose between models.
- **Provenance:** final run `final_20261010_003002`, code commit `5fcfe05`, working tree dirty = `False`, selection file commit `a440e84`, configuration hash `11bf5ccf03`.
- ETS local level and ARIMA(0,1,1) are mathematically the same model, so the `ets` and `sarima` columns count as one result (their numbers agree).

## 1. Headline result

LightGBM improves rupee MAE over the strongest baseline by **2.7% at 7 days and 7.7% at 30 days** (equal-weight average over the five markets). Against plain naive the figures are 3.0% and 7.9%. It beats the strongest baseline in 4 of 5 markets at 7 days and 5 of 5 at 30 days. Diebold-Mariano tests against the strongest baseline (absolute error, Harvey correction, h-1 lags) reject equality at the 5% level in 1 of 5 markets at 7 days and 1 of 5 at 30 days; the forecasts overlap, so these p-values are approximate, and with ten cells some significance is expected by chance.

| market | h | n | strongest_baseline | baseline_MAE | naive_MAE | headline_MAE | impr_vs_strongest_% | impr_vs_naive_% |
|---|---|---|---|---|---|---|---|---|
| Kolhapur | 7 | 1149 | base_naive | 188.69 | 188.69 | 184.86 | 2.03 | 2.03 |
| Kolhapur | 30 | 933 | base_naive | 396.78 | 396.78 | 366.74 | 7.57 | 7.57 |
| Pune | 7 | 1211 | base_naive | 221.15 | 221.15 | 215.8 | 2.42 | 2.42 |
| Pune | 30 | 983 | base_naive | 467.14 | 467.14 | 418.61 | 10.39 | 10.39 |
| Sambhajinagar | 7 | 1185 | base_mean3 | 246.01 | 250.04 | 225.98 | 8.14 | 9.62 |
| Sambhajinagar | 30 | 956 | base_mean3 | 414.17 | 417.58 | 383.15 | 7.49 | 8.25 |
| Solapur | 7 | 956 | base_naive | 214.07 | 214.07 | 208.64 | 2.53 | 2.53 |
| Solapur | 30 | 773 | base_naive | 429.3 | 429.3 | 403.52 | 6.01 | 6.01 |
| Vashi | 7 | 1113 | base_naive | 256.24 | 256.24 | 260.27 | -1.57 | -1.57 |
| Vashi | 30 | 907 | base_naive | 592.92 | 592.92 | 549.86 | 7.26 | 7.26 |


Macro average over the five markets (equal weight), improvement in rupee MAE in %:

| h | headline_impr_vs_strongest_pct | headline_impr_vs_naive_pct |
|---|---|---|
| 7 | 2.7 | 3.0 |
| 30 | 7.7 | 7.9 |


Diebold-Mariano test of the headline model against the strongest baseline (positive DM = LightGBM better; `mean_loss_diff` in rupees):

| market | h | strongest_baseline | DM | p_value | mean_loss_diff | n |
|---|---|---|---|---|---|---|
| Kolhapur | 7 | base_naive | 0.9 | 0.3672 | 3.83 | 1149 |
| Kolhapur | 30 | base_naive | 1.28 | 0.2016 | 30.04 | 933 |
| Pune | 7 | base_naive | 1.38 | 0.1676 | 5.35 | 1211 |
| Pune | 30 | base_naive | 2.06 | 0.04 | 48.53 | 983 |
| Sambhajinagar | 7 | base_mean3 | 4.06 | 0.0001 | 20.03 | 1185 |
| Sambhajinagar | 30 | base_mean3 | 1.69 | 0.0922 | 31.02 | 956 |
| Solapur | 7 | base_naive | 1.13 | 0.2583 | 5.43 | 956 |
| Solapur | 30 | base_naive | 0.92 | 0.3578 | 25.78 | 773 |
| Vashi | 7 | base_naive | -0.66 | 0.5122 | -4.03 | 1113 |
| Vashi | 30 | base_naive | 1.76 | 0.0788 | 43.06 | 907 |


### 1.1 Uncertainty of the macro improvement: block bootstrap (analysis only)

Origin days are resampled in blocks of 30 days, with the same days used for all five markets (so cross-market correlation, serial correlation and the overlap of the horizons stay inside each block); 2000 resamples, seed 42. The strongest baseline of each market is held at its full-window identity; the `reselected` columns re-choose it in every resample (more conservative). Intervals are 95% percentile intervals for the macro improvement in rupee MAE, in %.

| h | point_estimate_pct | ci95_low_strongest | ci95_high_strongest | share_resamples_above_0_strongest | ci95_low_strongest_reselected | ci95_high_strongest_reselected | point_estimate_vs_naive_pct | ci95_low_vs_naive | ci95_high_vs_naive |
|---|---|---|---|---|---|---|---|---|---|
| 7 | 2.7 | -0.07 | 5.03 | 0.97 | -0.16 | 5.0 | 3.0 | 0.5 | 5.25 |
| 30 | 7.7 | -1.84 | 14.05 | 0.95 | -1.94 | 14.05 | 7.9 | -1.47 | 14.14 |


### 1.2 Multiple testing: Holm adjustment of the ten Diebold-Mariano p-values (analysis only)

Ten tests (5 markets x 2 horizons) are made for the headline model. The Bonferroni threshold at 5% is 0.0050. Raw p < 0.05 in 2 cells; after Holm adjustment 1 remain significant at 5% (Bonferroni: 1).

| market | h | strongest_baseline | DM | p_value | p_holm | significant_raw_5pct | significant_holm_5pct | significant_bonferroni |
|---|---|---|---|---|---|---|---|---|
| Sambhajinagar | 7 | base_mean3 | 4.06 | 0.0001 | 0.0005 | True | True | True |
| Pune | 30 | base_naive | 2.06 | 0.04 | 0.3604 | True | False | False |
| Vashi | 30 | base_naive | 1.76 | 0.0788 | 0.6304 | False | False | False |
| Sambhajinagar | 30 | base_mean3 | 1.69 | 0.0922 | 0.6454 | False | False | False |
| Pune | 7 | base_naive | 1.38 | 0.1676 | 1.0 | False | False | False |
| Kolhapur | 30 | base_naive | 1.28 | 0.2016 | 1.0 | False | False | False |
| Solapur | 7 | base_naive | 1.13 | 0.2583 | 1.0 | False | False | False |
| Solapur | 30 | base_naive | 0.92 | 0.3578 | 1.0 | False | False | False |
| Kolhapur | 7 | base_naive | 0.9 | 0.3672 | 1.0 | False | False | False |
| Vashi | 7 | base_naive | -0.66 | 0.5122 | 1.0 | False | False | False |


## 2. Every family on every market and horizon

Improvement in rupee MAE over the strongest baseline, in % (positive = better). `prophet` is off the shelf; `prophet_anchored` is the declared variant.

| market | h | impr_ets_vs_strongest_pct | impr_sarima_vs_strongest_pct | impr_prophet_vs_strongest_pct | impr_prophet_anchored_vs_strongest_pct | impr_lgbm_vs_strongest_pct |
|---|---|---|---|---|---|---|
| Kolhapur | 7 | -1.7 | -1.7 | -191.4 | -4.7 | 2.0 |
| Kolhapur | 30 | -0.8 | -0.8 | -36.7 | 2.7 | 7.6 |
| Pune | 7 | 0.8 | 0.8 | -159.8 | -0.2 | 2.4 |
| Pune | 30 | -0.4 | -0.4 | -18.9 | 7.1 | 10.4 |
| Sambhajinagar | 7 | 1.5 | 1.5 | -117.7 | 0.9 | 8.1 |
| Sambhajinagar | 30 | 0.8 | 0.8 | -43.4 | 3.2 | 7.5 |
| Solapur | 7 | -0.5 | -0.5 | -162.8 | -0.3 | 2.5 |
| Solapur | 30 | -0.1 | -0.1 | -43.1 | 7.2 | 6.0 |
| Vashi | 7 | -1.5 | -1.5 | -206.4 | -4.7 | -1.6 |
| Vashi | 30 | -0.4 | -0.4 | -41.7 | -5.1 | 7.3 |


Macro average:

| h | impr_ets_vs_strongest_pct | impr_sarima_vs_strongest_pct | impr_prophet_vs_strongest_pct | impr_prophet_anchored_vs_strongest_pct | impr_lgbm_vs_strongest_pct |
|---|---|---|---|---|---|
| 7 | -0.3 | -0.3 | -167.6 | -1.8 | 2.7 |
| 30 | -0.2 | -0.2 | -36.8 | 3.0 | 7.7 |


Same, against plain naive:

| h | impr_ets_vs_naive_pct | impr_sarima_vs_naive_pct | impr_prophet_vs_naive_pct | impr_prophet_anchored_vs_naive_pct | impr_lgbm_vs_naive_pct |
|---|---|---|---|---|---|
| 7 | 0.1 | 0.1 | -166.9 | -1.5 | 3.0 |
| 30 | -0.0 | -0.0 | -36.5 | 3.2 | 7.9 |


Diebold-Mariano p-values of every family against the strongest baseline (no model selection involved):

| market | h | ets | sarima | prophet | prophet_anchored | lgbm |
|---|---|---|---|---|---|---|
| Kolhapur | 7 | 0.0 | 0.0 | 0.0 | 0.07 | 0.37 |
| Kolhapur | 30 | 0.01 | 0.01 | 0.16 | 0.7 | 0.2 |
| Pune | 7 | 0.37 | 0.36 | 0.0 | 0.95 | 0.17 |
| Pune | 30 | 0.37 | 0.37 | 0.39 | 0.35 | 0.04 |
| Sambhajinagar | 7 | 0.04 | 0.04 | 0.0 | 0.69 | 0.0 |
| Sambhajinagar | 30 | 0.21 | 0.21 | 0.18 | 0.63 | 0.09 |
| Solapur | 7 | 0.13 | 0.13 | 0.0 | 0.91 | 0.26 |
| Solapur | 30 | 0.55 | 0.55 | 0.09 | 0.17 | 0.36 |
| Vashi | 7 | 0.01 | 0.01 | 0.0 | 0.22 | 0.51 |
| Vashi | 30 | 0.12 | 0.12 | 0.06 | 0.56 | 0.08 |


## 3. Secondary rows: per-cell tuning-selected and test-best

The per-cell model is the family with the lowest rupee MAE on the 2017-2019 folds in that market and horizon (fixed before the final run). The test-best model is the family with the lowest MAE on 2020-2024 in that cell; it is chosen after seeing the results, so its improvement is optimistic and is shown only for transparency.

| market | h | per_cell_selected_model | per_cell_impr_vs_strongest_pct | test_best_model_(optimistic) | test_best_impr_vs_strongest_pct | headline_impr_vs_strongest_pct |
|---|---|---|---|---|---|---|
| Kolhapur | 7 | lgbm | 2.0 | lgbm | 2.0 | 2.0 |
| Kolhapur | 30 | lgbm | 7.6 | lgbm | 7.6 | 7.6 |
| Pune | 7 | ets | 0.8 | lgbm | 2.4 | 2.4 |
| Pune | 30 | lgbm | 10.4 | lgbm | 10.4 | 10.4 |
| Sambhajinagar | 7 | lgbm | 8.1 | lgbm | 8.1 | 8.1 |
| Sambhajinagar | 30 | lgbm | 7.5 | lgbm | 7.5 | 7.5 |
| Solapur | 7 | lgbm | 2.5 | lgbm | 2.5 | 2.5 |
| Solapur | 30 | lgbm | 6.0 | prophet_anchored | 7.2 | 6.0 |
| Vashi | 7 | lgbm | -1.6 | ets | -1.5 | -1.6 |
| Vashi | 30 | lgbm | 7.3 | lgbm | 7.3 | 7.3 |


Macro average of these three rows:

| h | headline_impr_vs_strongest_pct | per_cell_impr_vs_strongest_pct | test_best_impr_vs_strongest_pct |
|---|---|---|---|
| 7 | 2.7 | 2.4 | 2.7 |
| 30 | 7.7 | 7.7 | 8.0 |


## 4. All metrics per market, horizon and model

Rupee scale except `MAE_log`. MASE is scaled by the in-sample naive error of 2014-2019 for that market and horizon.

Macro average over markets:

| h | model | MAE | RMSE | MASE | sMAPE | MAE_log |
|---|---|---|---|---|---|---|
| 7 | base_mean3 | 234.38 | 366.77 | 1.19 | 14.04 | 0.1416 |
| 7 | base_naive | 226.04 | 357.1 | 1.14 | 13.67 | 0.1379 |
| 7 | base_seasonal | 721.71 | 990.63 | 3.64 | 45.11 | 0.4768 |
| 7 | ets | 225.73 | 355.42 | 1.14 | 13.59 | 0.137 |
| 7 | lgbm | 219.11 | 339.1 | 1.11 | 13.17 | 0.1327 |
| 7 | prophet | 601.51 | 887.38 | 3.03 | 33.84 | 0.3516 |
| 7 | prophet_anchored | 229.18 | 352.91 | 1.16 | 13.82 | 0.1393 |
| 7 | sarima | 225.73 | 355.42 | 1.14 | 13.59 | 0.137 |
| 30 | base_mean3 | 465.89 | 672.49 | 1.22 | 27.74 | 0.2845 |
| 30 | base_naive | 460.74 | 668.61 | 1.21 | 27.46 | 0.2815 |
| 30 | base_seasonal | 695.95 | 964.4 | 1.83 | 43.09 | 0.4515 |
| 30 | ets | 460.9 | 666.52 | 1.21 | 27.48 | 0.2817 |
| 30 | lgbm | 424.38 | 619.31 | 1.12 | 24.74 | 0.2526 |
| 30 | prophet | 629.3 | 963.7 | 1.66 | 34.46 | 0.3617 |
| 30 | prophet_anchored | 448.55 | 651.1 | 1.17 | 26.2 | 0.2676 |
| 30 | sarima | 460.9 | 666.52 | 1.21 | 27.48 | 0.2817 |


Per market:

| market | h | model | n | MAE | RMSE | MASE | sMAPE | MAE_log |
|---|---|---|---|---|---|---|---|---|
| Kolhapur | 7 | base_naive | 1149 | 188.69 | 289.32 | 0.99 | 10.85 | 0.1091 |
| Kolhapur | 7 | base_mean3 | 1149 | 200.23 | 298.23 | 1.05 | 11.51 | 0.1156 |
| Kolhapur | 7 | base_seasonal | 1149 | 699.78 | 921.87 | 3.67 | 41.55 | 0.4349 |
| Kolhapur | 7 | ets | 1149 | 191.91 | 288.84 | 1.01 | 11.03 | 0.1109 |
| Kolhapur | 7 | sarima | 1149 | 191.9 | 288.84 | 1.01 | 11.03 | 0.1109 |
| Kolhapur | 7 | prophet | 1149 | 549.89 | 761.16 | 2.88 | 29.77 | 0.3055 |
| Kolhapur | 7 | prophet_anchored | 1149 | 197.64 | 286.04 | 1.04 | 11.51 | 0.1156 |
| Kolhapur | 7 | lgbm | 1149 | 184.86 | 271.05 | 0.97 | 10.74 | 0.1079 |
| Kolhapur | 30 | base_naive | 933 | 396.78 | 588.75 | 1.07 | 22.14 | 0.2252 |
| Kolhapur | 30 | base_mean3 | 933 | 405.09 | 594.58 | 1.1 | 22.68 | 0.2307 |
| Kolhapur | 30 | base_seasonal | 933 | 677.71 | 898.64 | 1.83 | 40.12 | 0.4168 |
| Kolhapur | 30 | ets | 933 | 399.77 | 589.54 | 1.08 | 22.34 | 0.2272 |
| Kolhapur | 30 | sarima | 933 | 399.76 | 589.53 | 1.08 | 22.34 | 0.2272 |
| Kolhapur | 30 | prophet | 933 | 542.28 | 809.72 | 1.47 | 28.65 | 0.2968 |
| Kolhapur | 30 | prophet_anchored | 933 | 386.14 | 543.97 | 1.04 | 21.53 | 0.2183 |
| Kolhapur | 30 | lgbm | 933 | 366.74 | 536.47 | 0.99 | 20.16 | 0.2044 |
| Pune | 7 | base_naive | 1211 | 221.15 | 354.47 | 1.02 | 12.5 | 0.1258 |
| Pune | 7 | base_mean3 | 1211 | 224.82 | 362.05 | 1.04 | 12.72 | 0.1281 |
| Pune | 7 | base_seasonal | 1211 | 708.04 | 980.58 | 3.27 | 42.82 | 0.4518 |
| Pune | 7 | ets | 1211 | 219.3 | 352.46 | 1.01 | 12.42 | 0.1249 |
| Pune | 7 | sarima | 1211 | 219.3 | 352.46 | 1.01 | 12.42 | 0.1249 |
| Pune | 7 | prophet | 1211 | 574.52 | 883.69 | 2.65 | 30.84 | 0.3191 |
| Pune | 7 | prophet_anchored | 1211 | 221.51 | 348.57 | 1.02 | 12.59 | 0.1267 |
| Pune | 7 | lgbm | 1211 | 215.8 | 345.88 | 1.0 | 12.23 | 0.123 |
| Pune | 30 | base_naive | 983 | 467.14 | 664.55 | 1.01 | 26.5 | 0.2707 |
| Pune | 30 | base_mean3 | 983 | 472.13 | 670.38 | 1.03 | 26.8 | 0.2738 |
| Pune | 30 | base_seasonal | 983 | 693.21 | 961.45 | 1.51 | 41.3 | 0.4324 |
| Pune | 30 | ets | 983 | 468.89 | 665.58 | 1.02 | 26.63 | 0.272 |
| Pune | 30 | sarima | 983 | 468.89 | 665.57 | 1.02 | 26.63 | 0.272 |
| Pune | 30 | prophet | 983 | 555.5 | 875.46 | 1.21 | 30.58 | 0.3209 |
| Pune | 30 | prophet_anchored | 983 | 433.99 | 615.36 | 0.94 | 24.9 | 0.2535 |
| Pune | 30 | lgbm | 983 | 418.61 | 602.5 | 0.91 | 23.63 | 0.2404 |
| Sambhajinagar | 7 | base_naive | 1185 | 250.04 | 393.14 | 1.25 | 19.06 | 0.1929 |
| Sambhajinagar | 7 | base_mean3 | 1185 | 246.01 | 388.31 | 1.23 | 18.51 | 0.1873 |
| Sambhajinagar | 7 | base_seasonal | 1185 | 650.79 | 886.39 | 3.24 | 51.35 | 0.5477 |
| Sambhajinagar | 7 | ets | 1185 | 242.35 | 382.24 | 1.21 | 18.33 | 0.1855 |
| Sambhajinagar | 7 | sarima | 1185 | 242.35 | 382.24 | 1.21 | 18.33 | 0.1855 |
| Sambhajinagar | 7 | prophet | 1185 | 535.46 | 758.09 | 2.67 | 39.48 | 0.4137 |
| Sambhajinagar | 7 | prophet_anchored | 1185 | 243.77 | 385.52 | 1.21 | 18.58 | 0.188 |
| Sambhajinagar | 7 | lgbm | 1185 | 225.98 | 356.11 | 1.13 | 17.15 | 0.1733 |
| Sambhajinagar | 30 | base_naive | 956 | 417.58 | 609.84 | 1.2 | 32.53 | 0.3352 |
| Sambhajinagar | 30 | base_mean3 | 956 | 414.17 | 602.94 | 1.19 | 32.22 | 0.332 |
| Sambhajinagar | 30 | base_seasonal | 956 | 627.16 | 845.68 | 1.8 | 49.57 | 0.524 |
| Sambhajinagar | 30 | ets | 956 | 410.7 | 595.43 | 1.18 | 32.15 | 0.3311 |
| Sambhajinagar | 30 | sarima | 956 | 410.7 | 595.44 | 1.18 | 32.15 | 0.3311 |
| Sambhajinagar | 30 | prophet | 956 | 594.07 | 916.01 | 1.71 | 40.87 | 0.4381 |
| Sambhajinagar | 30 | prophet_anchored | 956 | 400.74 | 581.15 | 1.15 | 30.54 | 0.3138 |
| Sambhajinagar | 30 | lgbm | 956 | 383.15 | 559.91 | 1.1 | 29.05 | 0.2978 |
| Solapur | 7 | base_naive | 956 | 214.07 | 343.68 | 1.38 | 14.11 | 0.1425 |
| Solapur | 7 | base_mean3 | 956 | 225.7 | 354.51 | 1.46 | 14.88 | 0.1502 |
| Solapur | 7 | base_seasonal | 956 | 674.35 | 920.67 | 4.35 | 47.31 | 0.5014 |
| Solapur | 7 | ets | 956 | 215.03 | 343.9 | 1.39 | 14.18 | 0.1433 |
| Solapur | 7 | sarima | 956 | 215.04 | 343.9 | 1.39 | 14.18 | 0.1433 |
| Solapur | 7 | prophet | 956 | 562.51 | 815.89 | 3.63 | 35.99 | 0.3774 |
| Solapur | 7 | prophet_anchored | 956 | 214.61 | 336.57 | 1.38 | 14.16 | 0.1429 |
| Solapur | 7 | lgbm | 956 | 208.64 | 325.79 | 1.35 | 13.8 | 0.1391 |
| Solapur | 30 | base_naive | 773 | 429.3 | 621.88 | 1.43 | 28.87 | 0.2976 |
| Solapur | 30 | base_mean3 | 773 | 435.71 | 625.12 | 1.45 | 29.34 | 0.3026 |
| Solapur | 30 | base_seasonal | 773 | 638.23 | 899.4 | 2.13 | 43.78 | 0.4605 |
| Solapur | 30 | ets | 773 | 429.82 | 621.08 | 1.43 | 28.93 | 0.2982 |
| Solapur | 30 | sarima | 773 | 429.83 | 621.08 | 1.43 | 28.93 | 0.2982 |
| Solapur | 30 | prophet | 773 | 614.24 | 933.77 | 2.05 | 36.04 | 0.3768 |
| Solapur | 30 | prophet_anchored | 773 | 398.57 | 597.48 | 1.33 | 26.15 | 0.2679 |
| Solapur | 30 | lgbm | 773 | 403.52 | 605.45 | 1.35 | 25.76 | 0.2641 |
| Vashi | 7 | base_naive | 1113 | 256.24 | 404.91 | 1.08 | 11.84 | 0.1191 |
| Vashi | 7 | base_mean3 | 1113 | 275.14 | 430.74 | 1.16 | 12.6 | 0.1268 |
| Vashi | 7 | base_seasonal | 1113 | 875.61 | 1243.64 | 3.69 | 42.54 | 0.4479 |
| Vashi | 7 | ets | 1113 | 260.04 | 409.64 | 1.1 | 11.98 | 0.1205 |
| Vashi | 7 | sarima | 1113 | 260.05 | 409.65 | 1.1 | 11.98 | 0.1205 |
| Vashi | 7 | prophet | 1113 | 785.17 | 1218.06 | 3.31 | 33.1 | 0.3423 |
| Vashi | 7 | prophet_anchored | 1113 | 268.37 | 407.84 | 1.13 | 12.28 | 0.1234 |
| Vashi | 7 | lgbm | 1113 | 260.27 | 396.68 | 1.1 | 11.94 | 0.12 |
| Vashi | 30 | base_naive | 907 | 592.92 | 858.05 | 1.33 | 27.24 | 0.2788 |
| Vashi | 30 | base_mean3 | 907 | 602.32 | 869.41 | 1.35 | 27.67 | 0.2833 |
| Vashi | 30 | base_seasonal | 907 | 843.47 | 1216.81 | 1.89 | 40.68 | 0.4239 |
| Vashi | 30 | ets | 907 | 595.3 | 860.96 | 1.33 | 27.35 | 0.2799 |
| Vashi | 30 | sarima | 907 | 595.3 | 860.96 | 1.33 | 27.35 | 0.2799 |
| Vashi | 30 | prophet | 907 | 840.4 | 1283.55 | 1.88 | 36.19 | 0.3758 |
| Vashi | 30 | prophet_anchored | 907 | 623.3 | 917.55 | 1.4 | 27.88 | 0.2847 |
| Vashi | 30 | lgbm | 907 | 549.86 | 792.23 | 1.23 | 25.12 | 0.256 |


## 5. Robustness: per year, win rate, median error

### 5.1 By target year (headline model vs the strongest baseline, macro average over markets, %)

| year | impr_h7_% | impr_h30_% | markets_positive_h7_of_5 | markets_positive_h30_of_5 |
|---|---|---|---|---|
| 2020 | 6.36 | 16.15 | 5 | 5 |
| 2021 | -2.51 | -10.73 | 3 | 0 |
| 2022 | 2.45 | 11.58 | 3 | 5 |
| 2023 | 5.22 | 18.01 | 5 | 5 |
| 2024 | 1.18 | 4.5 | 3 | 4 |


Years in which the macro improvement is negative: 7 days [2021]; 30 days [2021].

Per market and year:

| market | h | year | n | strongest_baseline_MAE | naive_MAE | headline_MAE | impr_vs_strongest_pct | impr_vs_naive_pct |
|---|---|---|---|---|---|---|---|---|
| Kolhapur | 7 | 2020 | 171 | 319.3 | 319.3 | 290.1 | 9.1 | 9.1 |
| Kolhapur | 7 | 2021 | 222 | 200.0 | 200.0 | 198.22 | 0.9 | 0.9 |
| Kolhapur | 7 | 2022 | 233 | 146.35 | 146.35 | 148.05 | -1.2 | -1.2 |
| Kolhapur | 7 | 2023 | 263 | 158.56 | 158.56 | 158.32 | 0.2 | 0.2 |
| Kolhapur | 7 | 2024 | 260 | 161.54 | 161.54 | 164.06 | -1.6 | -1.6 |
| Kolhapur | 30 | 2020 | 113 | 791.15 | 791.15 | 696.99 | 11.9 | 11.9 |
| Kolhapur | 30 | 2021 | 190 | 358.42 | 358.42 | 395.94 | -10.5 | -10.5 |
| Kolhapur | 30 | 2022 | 198 | 267.17 | 267.17 | 223.97 | 16.2 | 16.2 |
| Kolhapur | 30 | 2023 | 218 | 356.42 | 356.42 | 289.65 | 18.7 | 18.7 |
| Kolhapur | 30 | 2024 | 214 | 383.64 | 383.64 | 377.07 | 1.7 | 1.7 |
| Pune | 7 | 2020 | 206 | 306.31 | 306.31 | 286.18 | 6.6 | 6.6 |
| Pune | 7 | 2021 | 230 | 219.0 | 219.0 | 231.26 | -5.6 | -5.6 |
| Pune | 7 | 2022 | 254 | 141.54 | 141.54 | 135.53 | 4.2 | 4.2 |
| Pune | 7 | 2023 | 262 | 198.05 | 198.05 | 186.89 | 5.6 | 5.6 |
| Pune | 7 | 2024 | 259 | 256.76 | 256.76 | 254.06 | 1.1 | 1.1 |
| Pune | 30 | 2020 | 139 | 686.94 | 686.94 | 565.98 | 17.6 | 17.6 |
| Pune | 30 | 2021 | 195 | 445.28 | 445.28 | 522.3 | -17.3 | -17.3 |
| Pune | 30 | 2022 | 221 | 280.84 | 280.84 | 230.43 | 18.0 | 18.0 |
| Pune | 30 | 2023 | 216 | 402.38 | 402.38 | 317.3 | 21.1 | 21.1 |
| Pune | 30 | 2024 | 212 | 603.3 | 603.3 | 526.01 | 12.8 | 12.8 |
| Sambhajinagar | 7 | 2020 | 208 | 293.8 | 291.97 | 261.65 | 10.9 | 10.4 |
| Sambhajinagar | 7 | 2021 | 236 | 199.79 | 214.79 | 188.16 | 5.8 | 12.4 |
| Sambhajinagar | 7 | 2022 | 228 | 179.32 | 183.18 | 161.41 | 10.0 | 11.9 |
| Sambhajinagar | 7 | 2023 | 262 | 251.91 | 245.86 | 226.63 | 10.0 | 7.8 |
| Sambhajinagar | 7 | 2024 | 251 | 304.27 | 313.56 | 289.95 | 4.7 | 7.5 |
| Sambhajinagar | 30 | 2020 | 150 | 656.0 | 634.8 | 516.44 | 21.3 | 18.6 |
| Sambhajinagar | 30 | 2021 | 193 | 325.24 | 336.27 | 346.77 | -6.6 | -3.1 |
| Sambhajinagar | 30 | 2022 | 188 | 275.48 | 278.24 | 261.18 | 5.2 | 6.1 |
| Sambhajinagar | 30 | 2023 | 216 | 369.65 | 383.62 | 336.25 | 9.0 | 12.3 |
| Sambhajinagar | 30 | 2024 | 209 | 493.49 | 497.2 | 479.27 | 2.9 | 3.6 |
| Solapur | 7 | 2020 | 146 | 198.29 | 198.29 | 193.56 | 2.4 | 2.4 |
| Solapur | 7 | 2021 | 186 | 240.59 | 240.59 | 240.16 | 0.2 | 0.2 |
| Solapur | 7 | 2022 | 184 | 135.87 | 135.87 | 141.93 | -4.5 | -4.5 |
| Solapur | 7 | 2023 | 215 | 177.21 | 177.21 | 163.27 | 7.9 | 7.9 |
| Solapur | 7 | 2024 | 225 | 301.56 | 301.56 | 290.29 | 3.7 | 3.7 |
| Solapur | 30 | 2020 | 94 | 507.45 | 507.45 | 419.26 | 17.4 | 17.4 |
| Solapur | 30 | 2021 | 155 | 430.97 | 430.97 | 467.64 | -8.5 | -8.5 |
| Solapur | 30 | 2022 | 160 | 268.75 | 268.75 | 232.64 | 13.4 | 13.4 |
| Solapur | 30 | 2023 | 182 | 429.12 | 429.12 | 340.81 | 20.6 | 20.6 |
| Solapur | 30 | 2024 | 182 | 528.85 | 528.85 | 553.73 | -4.7 | -4.7 |
| Vashi | 7 | 2020 | 191 | 389.01 | 389.01 | 378.24 | 2.8 | 2.8 |
| Vashi | 7 | 2021 | 235 | 249.36 | 249.36 | 283.85 | -13.8 | -13.8 |
| Vashi | 7 | 2022 | 225 | 193.78 | 193.78 | 186.77 | 3.6 | 3.6 |
| Vashi | 7 | 2023 | 248 | 199.4 | 199.4 | 194.55 | 2.4 | 2.4 |
| Vashi | 7 | 2024 | 214 | 276.87 | 276.87 | 282.56 | -2.1 | -2.1 |
| Vashi | 30 | 2020 | 138 | 991.3 | 991.3 | 866.68 | 12.6 | 12.6 |
| Vashi | 30 | 2021 | 198 | 568.18 | 568.18 | 629.34 | -10.8 | -10.8 |
| Vashi | 30 | 2022 | 185 | 427.43 | 427.43 | 405.46 | 5.1 | 5.1 |
| Vashi | 30 | 2023 | 207 | 420.53 | 420.53 | 334.05 | 20.6 | 20.6 |
| Vashi | 30 | 2024 | 179 | 683.52 | 683.52 | 616.5 | 9.8 | 9.8 |


### 5.3 The 2021 loss year, described (analysis only; no cause is claimed)

Improvement of the headline model over the strongest baseline in 2021 by target month and market, in % (negative = the model loses). Nothing is changed in the model; this only shows where the losses are.

**h = 7 days**

| month | Kolhapur | Pune | Sambhajinagar | Solapur | Vashi | all markets (mean) |
|---|---|---|---|---|---|---|
| 1 | 18.92 | -5.12 | 21.66 | 33.73 | -69.61 | -0.08 |
| 2 | -26.79 | -6.28 | -5.12 | -17.57 | -49.05 | -20.96 |
| 3 | 2.52 | 0.83 | 9.0 | -5.43 | 5.51 | 2.49 |
| 4 | -14.0 | 9.77 | 7.0 | -52.45 | 49.6 | -0.02 |
| 5 | 9.94 | 19.61 | 8.81 | nan | 5.05 | 10.85 |
| 6 | 20.39 | -0.62 | 32.44 | 18.42 | -6.74 | 12.78 |
| 7 | -19.19 | 10.21 | -9.91 | -80.61 | -15.49 | -23.0 |
| 8 | -9.76 | -27.57 | -23.23 | -45.77 | -50.98 | -31.46 |
| 9 | 7.56 | 0.0 | -1.78 | -3.25 | 2.64 | 1.03 |
| 10 | 21.07 | -10.31 | 6.52 | 4.17 | 5.7 | 5.43 |
| 11 | 11.1 | 0.57 | -1.97 | 13.6 | -3.24 | 4.01 |
| 12 | -12.41 | -29.46 | 8.47 | 4.14 | -43.45 | -14.54 |


**h = 30 days**

| month | Kolhapur | Pune | Sambhajinagar | Solapur | Vashi | all markets (mean) |
|---|---|---|---|---|---|---|
| 1 | -34.81 | -59.73 | -55.52 | -44.11 | -47.97 | -48.43 |
| 2 | -125.52 | -69.79 | -140.62 | -85.29 | -82.89 | -100.82 |
| 3 | 12.75 | 13.57 | 19.79 | 14.06 | 26.36 | 17.31 |
| 4 | 26.69 | 15.03 | 28.85 | 4.84 | 19.3 | 18.94 |
| 5 | 11.36 | 15.94 | 30.04 | nan | -15.33 | 10.5 |
| 6 | 62.98 | 50.84 | 37.63 | 67.89 | 34.31 | 50.73 |
| 7 | 48.21 | -70.32 | -6.89 | -8.1 | -102.84 | -27.99 |
| 8 | -145.88 | -125.41 | -98.07 | -409.45 | -49.76 | -165.72 |
| 9 | -50.23 | -30.34 | -16.99 | -12.36 | -5.54 | -23.09 |
| 10 | 11.17 | 10.16 | 12.66 | 14.08 | 9.11 | 11.44 |
| 11 | -89.98 | -106.31 | -39.03 | -112.03 | -48.77 | -79.22 |
| 12 | -10.31 | -24.82 | 27.26 | -6.77 | -18.96 | -6.72 |


Excess absolute error of the headline model over the strongest baseline (rupees per forecast; positive = worse), by month, and each month's share of the year's total excess:

| h | month | excess_rupees_per_forecast | share_of_year_excess_% |
|---|---|---|---|
| 7 | 1 | -14.29 | -10.75 |
| 7 | 2 | 79.09 | 91.35 |
| 7 | 3 | -11.16 | -11.59 |
| 7 | 4 | -3.91 | -1.88 |
| 7 | 5 | -9.36 | -6.92 |
| 7 | 6 | -15.37 | -24.53 |
| 7 | 7 | 7.85 | 11.41 |
| 7 | 8 | 23.9 | 34.11 |
| 7 | 9 | -1.46 | -2.25 |
| 7 | 10 | -21.19 | -23.09 |
| 7 | 11 | -8.67 | -12.37 |
| 7 | 12 | 33.49 | 56.51 |
| 30 | 1 | 214.42 | 35.5 |
| 30 | 2 | 471.65 | 69.53 |
| 30 | 3 | -232.27 | -36.35 |
| 30 | 4 | -158.15 | -11.84 |
| 30 | 5 | -7.14 | -0.73 |
| 30 | 6 | -188.32 | -37.58 |
| 30 | 7 | 28.32 | 6.29 |
| 30 | 8 | 127.04 | 26.79 |
| 30 | 9 | 29.93 | 6.52 |
| 30 | 10 | -90.69 | -15.43 |
| 30 | 11 | 242.19 | 48.88 |
| 30 | 12 | 34.63 | 8.4 |


2021 compared with the other years, by market (movement = mean absolute log change from the origin quote to the target; signed error = forecast minus actual on the log scale):

| h | market | period | n | impr_vs_strongest_pct | mean_abs_log_move | mean_signed_log_error_headline | mean_signed_log_error_strongest_baseline |
|---|---|---|---|---|---|---|---|
| 7 | Kolhapur | other years | 927 | 2.3 | 0.11 | -0.0 | 0.0 |
| 7 | Kolhapur | 2021 | 222 | 0.9 | 0.11 | 0.01 | 0.01 |
| 7 | Pune | other years | 981 | 4.3 | 0.12 | 0.0 | 0.0 |
| 7 | Pune | 2021 | 230 | -5.6 | 0.13 | 0.01 | 0.01 |
| 7 | Sambhajinagar | other years | 949 | 8.6 | 0.2 | 0.02 | 0.0 |
| 7 | Sambhajinagar | 2021 | 236 | 5.8 | 0.17 | 0.02 | 0.0 |
| 7 | Solapur | other years | 770 | 3.2 | 0.14 | -0.01 | -0.01 |
| 7 | Solapur | 2021 | 186 | 0.2 | 0.16 | 0.02 | 0.02 |
| 7 | Vashi | other years | 878 | 1.6 | 0.12 | 0.0 | 0.01 |
| 7 | Vashi | 2021 | 235 | -13.8 | 0.11 | -0.02 | -0.0 |
| 30 | Kolhapur | other years | 743 | 11.6 | 0.23 | -0.01 | -0.01 |
| 30 | Kolhapur | 2021 | 190 | -10.5 | 0.2 | 0.02 | 0.03 |
| 30 | Pune | other years | 788 | 16.8 | 0.27 | -0.0 | -0.01 |
| 30 | Pune | 2021 | 195 | -17.3 | 0.28 | 0.02 | 0.01 |
| 30 | Sambhajinagar | other years | 763 | 10.1 | 0.35 | 0.02 | -0.01 |
| 30 | Sambhajinagar | 2021 | 193 | -6.6 | 0.28 | -0.02 | -0.02 |
| 30 | Solapur | other years | 618 | 9.7 | 0.3 | -0.0 | -0.03 |
| 30 | Solapur | 2021 | 155 | -8.5 | 0.3 | 0.02 | 0.0 |
| 30 | Vashi | other years | 709 | 12.0 | 0.28 | 0.0 | 0.0 |
| 30 | Vashi | 2021 | 198 | -10.8 | 0.27 | -0.04 | -0.02 |


### 5.2 Win rate and median error (macro average over markets)

Win rate = share of triples in which the model's absolute rupee error is smaller than the comparator's (ties count half). A model can win on the mean and lose on the median when its gain comes from a minority of large moves.

| h | model | win_rate_vs_strongest_pct | win_rate_vs_naive_pct | median_AE | median_AE_impr_vs_strongest_pct |
|---|---|---|---|---|---|
| 7 | base_mean3 | 46.1 | 46.3 | 137.47 | -11.1 |
| 7 | base_naive | 49.8 | 50.0 | 130.0 | -1.8 |
| 7 | base_seasonal | 17.3 | 17.5 | 550.0 | -346.1 |
| 7 | ets | 45.9 | 46.1 | 126.93 | -1.5 |
| 7 | lgbm | 49.6 | 50.1 | 127.54 | -2.4 |
| 7 | prophet | 22.8 | 22.9 | 371.4 | -199.5 |
| 7 | prophet_anchored | 46.1 | 46.7 | 137.88 | -11.3 |
| 7 | sarima | 45.9 | 46.1 | 126.92 | -1.5 |
| 30 | base_mean3 | 47.0 | 47.0 | 285.0 | -2.1 |
| 30 | base_naive | 50.1 | 50.0 | 280.0 | 0.6 |
| 30 | base_seasonal | 31.3 | 31.4 | 520.0 | -90.1 |
| 30 | ets | 46.2 | 45.9 | 289.39 | -3.2 |
| 30 | lgbm | 60.9 | 61.6 | 259.53 | 6.9 |
| 30 | prophet | 41.2 | 41.3 | 375.36 | -34.6 |
| 30 | prophet_anchored | 57.1 | 57.4 | 282.4 | -1.4 |
| 30 | sarima | 46.2 | 45.9 | 289.38 | -3.2 |


LightGBM: win rate against the strongest baseline 49.6% at 7 days and 60.9% at 30 days; change in median absolute error -2.4% and +6.9% (positive = smaller error).

Per market:

| market | h | model | n | median_AE | median_AE_strongest_baseline | win_rate_vs_strongest_pct | win_rate_vs_naive_pct |
|---|---|---|---|---|---|---|---|
| Kolhapur | 7 | lgbm | 1149 | 114.64 | 100.0 | 43.9 | 43.9 |
| Kolhapur | 7 | base_naive | 1149 | 100.0 | 100.0 | 50.0 | 50.0 |
| Kolhapur | 30 | lgbm | 933 | 216.73 | 200.0 | 59.7 | 59.7 |
| Kolhapur | 30 | base_naive | 933 | 200.0 | 200.0 | 50.0 | 50.0 |
| Pune | 7 | lgbm | 1211 | 124.17 | 150.0 | 51.4 | 51.4 |
| Pune | 7 | base_naive | 1211 | 150.0 | 150.0 | 50.0 | 50.0 |
| Pune | 30 | lgbm | 983 | 252.78 | 300.0 | 62.6 | 62.6 |
| Pune | 30 | base_naive | 983 | 300.0 | 300.0 | 50.0 | 50.0 |
| Sambhajinagar | 7 | lgbm | 1185 | 134.05 | 137.33 | 57.0 | 59.5 |
| Sambhajinagar | 7 | base_naive | 1185 | 150.0 | 137.33 | 48.8 | 50.0 |
| Sambhajinagar | 30 | lgbm | 956 | 238.76 | 258.33 | 59.3 | 62.7 |
| Sambhajinagar | 30 | base_naive | 956 | 250.0 | 258.33 | 50.5 | 50.0 |
| Solapur | 7 | lgbm | 956 | 121.03 | 100.0 | 47.9 | 47.9 |
| Solapur | 7 | base_naive | 956 | 100.0 | 100.0 | 50.0 | 50.0 |
| Solapur | 30 | lgbm | 773 | 254.08 | 300.0 | 62.6 | 62.6 |
| Solapur | 30 | base_naive | 773 | 300.0 | 300.0 | 50.0 | 50.0 |
| Vashi | 7 | lgbm | 1113 | 143.81 | 150.0 | 48.0 | 48.0 |
| Vashi | 7 | base_naive | 1113 | 150.0 | 150.0 | 50.0 | 50.0 |
| Vashi | 30 | lgbm | 907 | 335.3 | 350.0 | 60.3 | 60.3 |
| Vashi | 30 | base_naive | 907 | 350.0 | 350.0 | 50.0 | 50.0 |


### 5.4 Where does the mean gain come from? (analysis only)

Saving on each forecast = |strongest-baseline error| - |headline error| in rupees. The table shows how much of the total saving comes from the 10% of forecasts with the largest baseline errors (the large moves), the share of forecasts on which the model is better, and the typical (median) saving. A mean gain that comes from the large moves with a zero median saving means the model is tied on ordinary days.

| market | h | n | mean_saving_rs | median_saving_rs | share_of_forecasts_model_better_pct | share_of_total_saving_from_top_10pct_largest_baseline_errors_pct | mean_saving_rs_excluding_top | mean_saving_rs_top |
|---|---|---|---|---|---|---|---|---|
| Kolhapur | 7 | 1149.0 | 3.83 | -8.32 | 43.9 | 213.6 | -4.83 | 81.72 |
| Kolhapur | 30 | 933.0 | 30.04 | 47.59 | 59.7 | 74.5 | 8.52 | 222.13 |
| Pune | 7 | 1211.0 | 5.35 | 3.07 | 51.4 | 99.2 | 0.05 | 52.67 |
| Pune | 30 | 983.0 | 48.53 | 61.7 | 62.6 | 55.7 | 23.92 | 268.26 |
| Sambhajinagar | 7 | 1185.0 | 20.03 | 11.52 | 57.0 | 68.6 | 7.0 | 136.74 |
| Sambhajinagar | 30 | 956.0 | 31.02 | 35.02 | 59.3 | 62.8 | 12.81 | 194.14 |
| Solapur | 7 | 956.0 | 5.43 | -2.95 | 47.9 | 117.4 | -1.05 | 63.45 |
| Solapur | 30 | 773.0 | 25.78 | 58.95 | 62.6 | 46.6 | 15.3 | 119.12 |
| Vashi | 7 | 1113.0 | -4.03 | -3.45 | 48.0 | nan | -11.3 | 60.94 |
| Vashi | 30 | 907.0 | 43.06 | 54.2 | 60.3 | 73.9 | 12.5 | 317.05 |
| ALL (mean of markets) | 7 | 1122.8 | 6.12 | -0.02 | 49.6 | 124.7 | -2.03 | 79.1 |
| ALL (mean of markets) | 30 | 910.4 | 35.69 | 51.49 | 60.9 | 62.7 | 14.61 | 224.14 |


## 6. Error analysis

Segments are defined by the **target date** (the date the price is realised) for event periods, and by the days since the previous observed quote before the origin for gaps. Pooled on the log scale (equal weight per forecast); rupee MAE is shown too. Event periods follow the sourced dates in `docs/eda_findings.md` (export bans E5-E6, E9-E10, E12-E13; COVID lockdown E7-E8; MEP-only E11-E12).

### 6.1 Export-ban periods vs normal periods

| h | segment | n | logMAE_strongest_baseline | logMAE_naive | logMAE_headline | rupeeMAE_strongest_baseline | rupeeMAE_headline | log_impr_vs_strongest_pct | log_impr_vs_naive_pct |
|---|---|---|---|---|---|---|---|---|---|
| 7 | covid | 42 | 0.2129 | 0.2109 | 0.2188 | 259.33 | 255.24 | -2.8 | -3.8 |
| 7 | export_ban | 1030 | 0.1685 | 0.1684 | 0.1602 | 331.04 | 308.64 | 4.9 | 4.8 |
| 7 | mep | 123 | 0.1765 | 0.1742 | 0.1629 | 552.44 | 498.55 | 7.7 | 6.5 |
| 7 | normal | 4419 | 0.1277 | 0.1293 | 0.1246 | 191.48 | 190.25 | 2.4 | 3.6 |
| 30 | covid | 38 | 0.2453 | 0.2398 | 0.2282 | 289.91 | 269.47 | 7.0 | 4.8 |
| 30 | export_ban | 792 | 0.3898 | 0.3865 | 0.3319 | 780.14 | 639.97 | 14.9 | 14.1 |
| 30 | mep | 106 | 0.4094 | 0.4142 | 0.3313 | 1063.21 | 907.72 | 19.1 | 20.0 |
| 30 | normal | 3616 | 0.2531 | 0.2545 | 0.2327 | 374.28 | 364.29 | 8.0 | 8.6 |


### 6.2 Each event period

| h | segment | n | logMAE_strongest_baseline | logMAE_naive | logMAE_headline | rupeeMAE_strongest_baseline | rupeeMAE_headline | log_impr_vs_strongest_pct | log_impr_vs_naive_pct |
|---|---|---|---|---|---|---|---|---|---|
| 7 | covid_lockdown_2020 | 42 | 0.2129 | 0.2109 | 0.2188 | 259.33 | 255.24 | -2.8 | -3.8 |
| 7 | export_ban_2019_20 | 234 | 0.1761 | 0.172 | 0.1564 | 386.4 | 337.3 | 11.2 | 9.1 |
| 7 | export_ban_2020_21 | 336 | 0.1793 | 0.1825 | 0.1696 | 428.8 | 404.68 | 5.4 | 7.1 |
| 7 | export_ban_2023_24 | 460 | 0.1568 | 0.1563 | 0.1553 | 231.47 | 223.92 | 1.0 | 0.6 |
| 7 | mep_oct_dec_2023 | 123 | 0.1765 | 0.1742 | 0.1629 | 552.44 | 498.55 | 7.7 | 6.5 |
| 7 | normal | 4419 | 0.1277 | 0.1293 | 0.1246 | 191.48 | 190.25 | 2.4 | 3.6 |
| 30 | covid_lockdown_2020 | 38 | 0.2453 | 0.2398 | 0.2282 | 289.91 | 269.47 | 7.0 | 4.8 |
| 30 | export_ban_2019_20 | 133 | 0.438 | 0.4289 | 0.307 | 943.28 | 598.88 | 29.9 | 28.4 |
| 30 | export_ban_2020_21 | 260 | 0.4666 | 0.4622 | 0.4078 | 1060.58 | 955.62 | 12.6 | 11.8 |
| 30 | export_ban_2023_24 | 399 | 0.3237 | 0.323 | 0.2907 | 543.01 | 447.98 | 10.2 | 10.0 |
| 30 | mep_oct_dec_2023 | 106 | 0.4094 | 0.4142 | 0.3313 | 1063.21 | 907.72 | 19.1 | 20.0 |
| 30 | normal | 3616 | 0.2531 | 0.2545 | 0.2327 | 374.28 | 364.29 | 8.0 | 8.6 |


### 6.3 By gap since the previous observed quote

| h | segment | n | logMAE_strongest_baseline | logMAE_naive | logMAE_headline | rupeeMAE_strongest_baseline | rupeeMAE_headline | log_impr_vs_strongest_pct | log_impr_vs_naive_pct |
|---|---|---|---|---|---|---|---|---|---|
| 7 | 1 day (consecutive) | 4183 | 0.1371 | 0.1381 | 0.133 | 225.25 | 219.62 | 3.0 | 3.7 |
| 7 | 2 days | 1147 | 0.138 | 0.1397 | 0.1321 | 233.09 | 222.26 | 4.3 | 5.4 |
| 7 | 3-6 days | 259 | 0.1277 | 0.1308 | 0.1254 | 200.14 | 197.01 | 1.8 | 4.1 |
| 7 | 7+ days | 25 | 0.1326 | 0.1283 | 0.1722 | 183.0 | 241.06 | -29.9 | -34.3 |
| 30 | 1 day (consecutive) | 3260 | 0.2785 | 0.2791 | 0.251 | 455.65 | 420.02 | 9.9 | 10.1 |
| 30 | 2 days | 1028 | 0.2816 | 0.2815 | 0.2499 | 471.12 | 429.71 | 11.2 | 11.2 |
| 30 | 3-6 days | 243 | 0.3047 | 0.3088 | 0.2766 | 479.89 | 449.74 | 9.2 | 10.4 |
| 30 | 7+ days | 21 | 0.2424 | 0.2492 | 0.2757 | 410.95 | 490.59 | -13.8 | -10.7 |


Small segments (the COVID lockdown, and gaps of 7+ days) have few forecasts; read them as indications only.

## 7. Sensitivity checks (side by side, never used to choose between models)

### 7.1 Sensitivity (a): every 'likely error' flagged point set to missing

Run `final_error_20261010_004022` (commit `5fcfe05`, dirty `False`), same frozen models refit on the modified panel. The review table lists 8 'likely error' points; the ones not already set to missing were: ['Vashi New Mumbai 2016-10-20', 'Pune 2023-05-12'].

| market | h | n_main | strongest_baseline_main | headline_impr_vs_strongest_pct_main | n_error_set_missing | strongest_baseline_error_set_missing | headline_impr_vs_strongest_pct_error_set_missing |
|---|---|---|---|---|---|---|---|
| Kolhapur | 7 | 1149 | base_naive | 2.03 | 1149 | base_naive | 2.3 |
| Kolhapur | 30 | 933 | base_naive | 7.57 | 933 | base_naive | 6.94 |
| Pune | 7 | 1211 | base_naive | 2.42 | 1209 | base_naive | 3.05 |
| Pune | 30 | 983 | base_naive | 10.39 | 981 | base_naive | 10.14 |
| Sambhajinagar | 7 | 1185 | base_mean3 | 8.14 | 1185 | base_mean3 | 7.91 |
| Sambhajinagar | 30 | 956 | base_mean3 | 7.49 | 956 | base_mean3 | 7.66 |
| Solapur | 7 | 956 | base_naive | 2.53 | 956 | base_naive | 2.46 |
| Solapur | 30 | 773 | base_naive | 6.01 | 773 | base_naive | 5.3 |
| Vashi | 7 | 1113 | base_naive | -1.57 | 1113 | base_naive | -1.5 |
| Vashi | 30 | 907 | base_naive | 7.26 | 907 | base_naive | 6.72 |


Macro average, improvement over the strongest baseline in %:

| h | headline_impr_vs_strongest_pct_main | headline_impr_vs_strongest_pct_error_set_missing |
|---|---|---|
| 7 | 2.71 | 2.84 |
| 30 | 7.74 | 7.35 |


### 7.2 Sensitivity (b): with and without `is_stale` dates

Triples whose origin or target date lies in a run of 5 or more identical (min, max, modal) quotes are excluded in the second scoring set; the strongest baseline is recomputed on each set.

| market | h | headline_impr_vs_strongest_pct|all dates | headline_impr_vs_strongest_pct|excluding is_stale dates | n|all dates | n|excluding is_stale dates |
|---|---|---|---|---|---|
| Kolhapur | 7 | 2.03 | 2.83 | 1149 | 1091 |
| Kolhapur | 30 | 7.57 | 7.11 | 933 | 881 |
| Pune | 7 | 2.42 | 2.31 | 1211 | 1107 |
| Pune | 30 | 10.39 | 9.49 | 983 | 892 |
| Sambhajinagar | 7 | 8.14 | 8.14 | 1185 | 1185 |
| Sambhajinagar | 30 | 7.49 | 7.49 | 956 | 956 |
| Solapur | 7 | 2.53 | 2.55 | 956 | 942 |
| Solapur | 30 | 6.01 | 5.86 | 773 | 756 |
| Vashi | 7 | -1.57 | -1.28 | 1113 | 1072 |
| Vashi | 30 | 7.26 | 7.52 | 907 | 878 |


Macro average improvement over the strongest baseline, in %:

| h | all dates | excluding is_stale dates |
|---|---|---|
| 7 | 2.71 | 2.91 |
| 30 | 7.74 | 7.49 |


## 8. Runtime and dropped triples

Wall-clock for the final run: 611.3 s on 12 worker processes (so per-fit seconds include contention; LightGBM used 4 threads per fit).

| model | cpu_seconds | fits | not_converged | cpu_seconds_per_fit |
|---|---|---|---|---|
| ets|local_level | 188.4 | 300 | 0 | 0.63 |
| lgbm|huber_nl16_mc20_n150 | 1083.8 | 120 | 0 | 9.03 |
| lgbm|huber_nl16_mc60_n150 | 1077.4 | 120 | 0 | 8.98 |
| prophet_anchored|cps0.05 | 1293.1 | 299 | 0 | 4.32 |
| prophet|cps0.05 | 1299.1 | 299 | 0 | 4.34 |
| prophet|cps0.3 | 1450.9 | 299 | 0 | 4.85 |
| sarima|011 | 425.7 | 300 | 2 | 1.42 |


Triples dropped for all models (a triple any model cannot forecast is dropped for everyone):

| column | cannot_forecast | total_triples | dropped_for_all_models_total |
|---|---|---|---|
| base_naive | 0 | 10420 | 254 |
| base_mean3 | 0 | 10420 | 254 |
| base_seasonal | 254 | 10420 | 254 |
| ets|local_level | 0 | 10420 | 254 |
| lgbm|huber_nl16_mc20_n150 | 0 | 10420 | 254 |
| lgbm|huber_nl16_mc60_n150 | 0 | 10420 | 254 |
| prophet_anchored|cps0.05 | 0 | 10420 | 254 |
| prophet|cps0.05 | 0 | 10420 | 254 |
| prophet|cps0.3 | 0 | 10420 | 254 |
| sarima|011 | 0 | 10420 | 254 |


## 9. Post-hoc: feature-group ablation of the frozen LightGBM (not used for any choice)

**This section is post-hoc and was not used for any choice.** The frozen LightGBM was refit with one feature group removed at a time, with the same hyperparameters, the same monthly refits and the same triples as the final run. Groups: calendar (target month, week of year, weekday, origin weekday, expected-closure flag); other-market (other markets' 7- and 28-day changes and the origin's price relative to them); mean-reversion (deviation of the last quote from its 3-quote mean and from its 30-day mean); days-since-last-quote (gap before the origin). Run `ablation_20261010_004236` (commit `5fcfe05`, dirty `False`).

Macro average over markets. `change_vs_full_%` is the change in rupee MAE when the group is dropped (positive = the group was helping); the last column is the remaining improvement over the strongest baseline:

| h | variant | MAE | change_vs_full_MAE | change_vs_full_pct | impr_vs_strongest_pct |
|---|---|---|---|---|---|
| 7 | drop_calendar | 222.63 | 3.52 | 1.7 | 1.0 |
| 7 | drop_days_since_last_quote | 219.15 | 0.04 | 0.0 | 2.7 |
| 7 | drop_mean_reversion | 218.76 | -0.35 | -0.2 | 2.9 |
| 7 | drop_other_market | 220.66 | 1.55 | 0.7 | 2.1 |
| 7 | full | 219.11 | 0.0 | 0.0 | 2.7 |
| 30 | drop_calendar | 452.7 | 28.32 | 6.6 | 1.7 |
| 30 | drop_days_since_last_quote | 425.03 | 0.66 | 0.2 | 7.6 |
| 30 | drop_mean_reversion | 425.19 | 0.82 | 0.2 | 7.6 |
| 30 | drop_other_market | 428.69 | 4.32 | 1.0 | 6.8 |
| 30 | full | 424.38 | 0.0 | 0.0 | 7.7 |


Per market:

| market | h | variant | n | MAE | change_vs_full_pct | impr_vs_strongest_pct |
|---|---|---|---|---|---|---|
| Kolhapur | 7 | full | 1149 | 184.86 | 0.0 | 2.0 |
| Kolhapur | 7 | drop_calendar | 1149 | 192.6 | 4.2 | -2.1 |
| Kolhapur | 7 | drop_other_market | 1149 | 185.02 | 0.1 | 1.9 |
| Kolhapur | 7 | drop_mean_reversion | 1149 | 185.3 | 0.2 | 1.8 |
| Kolhapur | 7 | drop_days_since_last_quote | 1149 | 184.97 | 0.1 | 2.0 |
| Kolhapur | 30 | full | 933 | 366.74 | 0.0 | 7.6 |
| Kolhapur | 30 | drop_calendar | 933 | 402.42 | 9.7 | -1.4 |
| Kolhapur | 30 | drop_other_market | 933 | 370.06 | 0.9 | 6.7 |
| Kolhapur | 30 | drop_mean_reversion | 933 | 367.71 | 0.3 | 7.3 |
| Kolhapur | 30 | drop_days_since_last_quote | 933 | 367.39 | 0.2 | 7.4 |
| Pune | 7 | full | 1211 | 215.8 | 0.0 | 2.4 |
| Pune | 7 | drop_calendar | 1211 | 220.03 | 2.0 | 0.5 |
| Pune | 7 | drop_other_market | 1211 | 214.14 | -0.8 | 3.2 |
| Pune | 7 | drop_mean_reversion | 1211 | 215.06 | -0.3 | 2.8 |
| Pune | 7 | drop_days_since_last_quote | 1211 | 215.09 | -0.3 | 2.7 |
| Pune | 30 | full | 983 | 418.61 | 0.0 | 10.4 |
| Pune | 30 | drop_calendar | 983 | 458.1 | 9.4 | 1.9 |
| Pune | 30 | drop_other_market | 983 | 418.0 | -0.1 | 10.5 |
| Pune | 30 | drop_mean_reversion | 983 | 419.71 | 0.3 | 10.2 |
| Pune | 30 | drop_days_since_last_quote | 983 | 418.88 | 0.1 | 10.3 |
| Sambhajinagar | 7 | full | 1185 | 225.98 | 0.0 | 8.1 |
| Sambhajinagar | 7 | drop_calendar | 1185 | 229.37 | 1.5 | 6.8 |
| Sambhajinagar | 7 | drop_other_market | 1185 | 233.25 | 3.2 | 5.2 |
| Sambhajinagar | 7 | drop_mean_reversion | 1185 | 226.38 | 0.2 | 8.0 |
| Sambhajinagar | 7 | drop_days_since_last_quote | 1185 | 226.46 | 0.2 | 7.9 |
| Sambhajinagar | 30 | full | 956 | 383.15 | 0.0 | 7.5 |
| Sambhajinagar | 30 | drop_calendar | 956 | 396.01 | 3.4 | 4.4 |
| Sambhajinagar | 30 | drop_other_market | 956 | 395.12 | 3.1 | 4.6 |
| Sambhajinagar | 30 | drop_mean_reversion | 956 | 383.44 | 0.1 | 7.4 |
| Sambhajinagar | 30 | drop_days_since_last_quote | 956 | 383.76 | 0.2 | 7.3 |
| Solapur | 7 | full | 956 | 208.64 | 0.0 | 2.5 |
| Solapur | 7 | drop_calendar | 956 | 211.1 | 1.2 | 1.4 |
| Solapur | 7 | drop_other_market | 956 | 209.3 | 0.3 | 2.2 |
| Solapur | 7 | drop_mean_reversion | 956 | 207.42 | -0.6 | 3.1 |
| Solapur | 7 | drop_days_since_last_quote | 956 | 209.4 | 0.4 | 2.2 |
| Solapur | 30 | full | 773 | 403.52 | 0.0 | 6.0 |
| Solapur | 30 | drop_calendar | 773 | 412.14 | 2.1 | 4.0 |
| Solapur | 30 | drop_other_market | 773 | 402.5 | -0.3 | 6.2 |
| Solapur | 30 | drop_mean_reversion | 773 | 403.78 | 0.1 | 5.9 |
| Solapur | 30 | drop_days_since_last_quote | 773 | 404.68 | 0.3 | 5.7 |
| Vashi | 7 | full | 1113 | 260.27 | 0.0 | -1.6 |
| Vashi | 7 | drop_calendar | 1113 | 260.05 | -0.1 | -1.5 |
| Vashi | 7 | drop_other_market | 1113 | 261.57 | 0.5 | -2.1 |
| Vashi | 7 | drop_mean_reversion | 1113 | 259.64 | -0.2 | -1.3 |
| Vashi | 7 | drop_days_since_last_quote | 1113 | 259.83 | -0.2 | -1.4 |
| Vashi | 30 | full | 907 | 549.86 | 0.0 | 7.3 |
| Vashi | 30 | drop_calendar | 907 | 594.83 | 8.2 | -0.3 |
| Vashi | 30 | drop_other_market | 907 | 557.78 | 1.4 | 5.9 |
| Vashi | 30 | drop_mean_reversion | 907 | 551.32 | 0.3 | 7.0 |
| Vashi | 30 | drop_days_since_last_quote | 907 | 550.45 | 0.1 | 7.2 |


<!-- COMMENTARY BELOW IS WRITTEN BY HAND; EVERYTHING ABOVE IS GENERATED -->


## 10. What the results say (hand-written; every number below is read from the tables above)

### 10.1 At 7 days LightGBM is effectively tied with naive; its mean gain comes from a few large moves

- Macro improvement over the strongest baseline: **+2.7%**, 95% block-bootstrap interval **[-0.1%, +5.0%]**, which includes zero (section 1.1). It beats the strongest baseline in 4 of 5 markets and loses in Vashi (-1.6%).
- It is better than the strongest baseline on **49.6%** of forecasts (50.1% against plain naive): a coin flip. The **median** absolute error is Rs 127.5, the same as the strongest baseline (Rs 127.5) and about 2% below naive (Rs 130.0); the median saving per forecast is Rs -0.02, i.e. zero (sections 5.2 and 5.4).
- The mean gain (Rs 6.1 per forecast) comes entirely from the 10% of forecasts with the largest baseline errors: there the model saves Rs 79 on average, while on the other 90% it saves Rs -2. On ordinary days it is not better than naive; it helps when prices move a lot.
- Only 1 of 10 Diebold-Mariano tests survives a Holm adjustment, and it is a 7-day cell (Sambhajinagar, where the strongest baseline is the mean of the last 3 quotes). Pune at 30 days is significant on its raw p-value (0.040) but not after adjustment (0.36).
- Honest summary: **at 7 days the evidence for a gain over naive is weak.** Against plain naive the bootstrap interval is [+0.5%, +5.2%], so a small positive mean effect is plausible, but it is not a gain on a typical day.

### 10.2 At 30 days the gain is larger and broader, but its interval still includes zero

- Macro improvement over the strongest baseline: **+7.7%**, 95% block-bootstrap interval **[-1.8%, +14.1%]** (95% of the resamples are above zero). Positive in 5 of 5 markets (+6.0% to +10.4%).
- Better than the strongest baseline on **60.9%** of forecasts (61.6% against naive); the median absolute error is lower (Rs 259.5 against Rs 281.7, averaged over markets; 6.9% lower as the average of the per-market percentages), and the median saving is Rs 51 per forecast. The gain is not only from large moves: the largest 10% of baseline errors give 63% of the total saving and the other 90% of forecasts still save Rs 15 each.
- It is not statistically conclusive: no 30-day cell survives the Holm adjustment, and the interval reaches below zero because the year-to-year spread is large (next point).

### 10.3 2021 is a loss year at both horizons

- Macro improvement by target year (7 days / 30 days): 2020 +6.4% / +16.2%, **2021 -2.5% / -10.7%**, 2022 +2.5% / +11.6%, 2023 +5.2% / +18.0%, 2024 +1.2% / +4.5%.
- At 30 days LightGBM loses to the strongest baseline in **all five markets** in 2021 (-6.6% to -17.3%), against +9.7% to +16.8% in the other years. At 7 days it loses in Pune (-5.6%) and Vashi (-13.8%) and is slightly positive elsewhere.
- By month (section 5.3), the 2021 excess error is concentrated, not spread out. At 30 days, February accounts for 70% of the year's net excess absolute error, November 49%, January 35% and August 27%, offset by June (-38%), March (-36%) and October (-15%). At 7 days: February 91%, December 57%, August 34%.
- What the data does not show: the realised price movement in 2021 is similar to other years (mean absolute log change 0.20-0.30 at 30 days against 0.23-0.35 elsewhere), and the forecast errors have no large persistent bias (mean signed log error within +-0.05). **I do not know why 2021 is a loss year and have not changed the model.**

### 10.4 Seasonal-naive is a weak baseline here

The 365-day seasonal-naive has a macro MAE of Rs 721.7 against Rs 226.0 for naive at 7 days (3.2 times) and Rs 696.0 against Rs 460.7 at 30 days (1.5 times), because price levels differ by a factor of 2-3 between years. It is never the strongest baseline in any market. The headline is therefore against the strongest baseline (naive in 8 of 10 cells, mean-of-last-3 in the other 2), and against plain naive, not against seasonal-naive.

### 10.5 The other families

- **ETS (= ARIMA(0,1,1)) and SARIMA:** ties with naive (-0.3% at 7 days, -0.2% at 30 days). This is what the near-random-walk finding in the EDA predicts.
- **Prophet off the shelf:** far worse (-168% at 7 days, -37% at 30 days): it fits a curve and ignores recent prices.
- **Prophet anchored (declared variant):** -1.8% at 7 days and **+3.0% at 30 days** (better than the strongest baseline in 4 of 5 markets at 30 days). On the 2017-2019 tuning folds it lost at 30 days (-10.8%), so the sign flipped on the test window. It was not used for any choice and the headline is unchanged.
- **Per-cell tuning-selected and test-best rows** are close to the headline (macro +2.4% / +7.7% and +2.7% / +8.0%); the test-best row is optimistic by construction.

### 10.6 Events (only dates marked high-confidence in `docs/eda_findings.md`)

Log-scale improvement of the headline over the strongest baseline, by the period of the target date (section 6). Periods whose both end dates are high-confidence: the 2019 export ban (E5-E6; only the part inside the test window, 1 January to 14 March 2020), the COVID lockdown (E7-E8), the 2020 export ban (E9-E10) and the 2023-24 export ban (E12-E13).

| Period | 7 days | 30 days |
|---|---|---|
| Normal periods | +2.4% (n = 4,419) | +8.0% (n = 3,616) |
| Export ban 2019-20, test-window part (E5-E6) | +11.2% (n = 234) | +29.9% (n = 133) |
| COVID lockdown (E7-E8) | -2.8% (n = 42) | +7.0% (n = 38) |
| Export ban 2020-21 (E9-E10) | +5.4% (n = 336) | +12.6% (n = 260) |
| Export ban 2023-24 (E12-E13) | +1.0% (n = 460) | +10.2% (n = 399) |

The gain is larger in the three export-ban periods than in normal periods (pooled: +4.9% against +2.4% at 7 days; +14.9% against +8.0% at 30 days). This is descriptive: ban periods also have the largest price moves, there are only three independent episodes, and nothing here shows that a ban caused anything. The COVID lockdown has too few forecasts to read. The **Oct-Dec 2023 minimum-export-price period** (section 6.2: +7.7% at 7 days, +19.1% at 30 days) starts at event E11, whose date is only medium-confidence in the EDA table (the cited article may not carry it), so **that row is unverified**.

By gap before the origin (section 6.3): the gain holds when the previous quote is 1, 2 or 3-6 days earlier, but at 7 or more days it is negative (-29.9% at 7 days, n = 25; -13.8% at 30 days, n = 21): too few cases to be sure, but a sign that the model is least reliable after a long reporting gap.

### 10.7 Sensitivity checks (reported side by side; not used to choose anything)

- **(a) "Likely error" points set to missing:** only two points were not already missing (Vashi New Mumbai 2016-10-20, in the training period, and Pune 2023-05-12, in the test period). Macro improvement 2.71% -> 2.84% at 7 days and 7.74% -> 7.35% at 30 days.
- **(b) Without `is_stale` dates** (0-9% of the forecasts of each market removed): 2.71% -> 2.91% at 7 days and 7.74% -> 7.49% at 30 days.
- Neither check moves a macro figure by more than 0.4 percentage points, and the sign of every market's result is unchanged at both horizons.

### 10.8 Post-hoc: where the gain comes from (section 9; not used for any choice)

Dropping a feature group and refitting with the same hyperparameters, change in macro rupee MAE (positive = the group was helping):

| Group dropped | 7 days | 30 days |
|---|---|---|
| Calendar (month, week of year, weekday, expected-closure flag) | +1.7% (improvement falls from 2.7% to 1.0%) | **+6.6%** (improvement falls from 7.7% to 1.7%) |
| Other markets (their recent changes and relative level) | +0.7% | +1.0% |
| Mean-reversion (distance from 3-quote and 30-day mean) | -0.2% | +0.2% |
| Days since the previous quote | 0.0% | +0.2% |

Most of the 30-day gain disappears without the calendar features, and the other-market features add a little. The mean-reversion and gap features add nothing measurable. The calendar features let the model learn recurring seasonal drift from earlier years, but the EDA found the yearly pattern irregular and only 6-10 years of history exist, so this may not persist; it is a post-hoc pattern in one test window, not a tested mechanism.

### 10.9 Where the models fail

- After long gaps (7+ days) and in 2021 (above).
- Quote noise: daily quotes have a noise part that reverses the next day (EDA section 5), so no model can forecast a single day's quote closely; MAE is about Rs 220 at 7 days and Rs 425 at 30 days for LightGBM, against typical prices of Rs 500-3,000.
- Any model that ignores the latest quote (off-the-shelf Prophet, 365-day seasonal-naive) is far worse than naive.

## 11. Limitations

- **Lasalgaon is absent from the dataset** (and Niphad, Pimpalgaon, Saykheda, Yeola, Chandwad): these are five Maharashtra mandis, not the main price-setting onion market. Results may not carry over to it.
- **Vashi has a shorter history** (from May 2016), so its models train on fewer years; it is the only market where LightGBM loses at 7 days.
- **The markets are strongly correlated** (weekly changes correlate at 0.57-0.72), so the five markets are not five independent tests; the macro average and the ten Diebold-Mariano tests overstate the independent evidence. The bootstrap resamples the same days across markets to account for this.
- **Horizons overlap**, so consecutive forecast errors are correlated. The Diebold-Mariano test uses h-1 lags counted in origins, not calendar days, so it is approximate; the block bootstrap (30-day blocks) is a second, more direct check.
- **The strongest baseline is chosen on the test window** (a conservative, hindsight choice that makes the benchmark harder). The bootstrap interval also holds that choice fixed; re-selecting the baseline in every resample changes the intervals only slightly (7 days [-0.2%, +5.0%]; 30 days [-1.9%, +14.1%]).
- **One test period, one realisation:** 2020-2024 contains several price surges and three export-ban episodes; it is a single path and the year-to-year spread is large (section 5.1).
- **Prophet is disadvantaged by the monthly refit design** (it does not update between refits), which is a property of the setup, not necessarily of the model; the anchored variant partly addresses this.
- **Event dates are partly unverified:** see `docs/eda_findings.md`; only the high-confidence ones are named above.
- **Data limits:** quotes are reported modal prices with a reporting lag, one variety per market, a quarter to a third of days missing, and a few quotes were set to missing by a reviewed rule (section 6 of the cleaning log). The derived Mendeley dataset's own preprocessing is undocumented.
- **No weather features** in version 1 (planned as a later ablation).
