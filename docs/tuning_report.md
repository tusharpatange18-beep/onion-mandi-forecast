# Tuning report (2017-2019 folds only)

Run `tune_20261009_225301`, code commit `27f3c55` (working tree dirty at the end of the run: True). Window 2017-01-01 to 2019-12-31; models refit on the first day of every month using only rows whose target date is before that day. Nothing from 2020-2024 was used. **The selection below is a proposal: `configs/selected_models.yaml` has not been committed or frozen.**

## 1. Triples kept and dropped, by market and reason

A triple is a (market, origin, horizon) with an observed target. It is kept only if every candidate and every baseline can forecast it.

| market | h | triples | no_365d_quote_for_baseline_c | a_model_could_not_forecast | dropped_total | kept | kept_pct |
|---|---|---|---|---|---|---|---|
| Kolhapur | 7 | 793 | 0 | 0 | 0 | 793 | 100.0 |
| Kolhapur | 30 | 664 | 0 | 0 | 0 | 664 | 100.0 |
| Pune | 7 | 833 | 2 | 0 | 2 | 831 | 99.8 |
| Pune | 30 | 702 | 2 | 0 | 2 | 700 | 99.7 |
| Sambhajinagar | 7 | 830 | 0 | 0 | 0 | 830 | 100.0 |
| Sambhajinagar | 30 | 701 | 0 | 0 | 0 | 701 | 100.0 |
| Solapur | 7 | 675 | 2 | 0 | 2 | 673 | 99.7 |
| Solapur | 30 | 579 | 2 | 0 | 2 | 577 | 99.7 |
| Vashi | 7 | 745 | 82 | 0 | 82 | 663 | 89.0 |
| Vashi | 30 | 635 | 60 | 0 | 60 | 575 | 90.6 |
| ALL | both | 7157 | 150 | 0 | 150 | 7007 | 97.9 |


`no_365d_quote_for_baseline_c`: no observed quote within 3 days of the target date one year earlier (a baseline-(c) limitation, mostly Vashi, which starts in 2016-05). `a_model_could_not_forecast` is a guard: the run aborts if any model misses more than 2% of a market's triples, so a failing candidate can no longer remove a market silently.

## 2. Fits, non-converged fits and cost per model

| candidate | fits | not_converged | not_converged_pct | cpu_seconds | cpu_seconds_per_fit |
|---|---|---|---|---|---|
| ets|local_level | 180 | 0 | 0.0 | 22.1 | 0.12 |
| ets|damped_trend | 180 | 35 | 19.4 | 46.9 | 0.26 |
| sarima|011 | 180 | 2 | 1.1 | 37.3 | 0.21 |
| sarima|111 | 180 | 2 | 1.1 | 73.9 | 0.41 |
| sarima|111_s7 | 180 | 0 | 0.0 | 179.9 | 1.0 |
| prophet|cps0.05 | 180 | 0 | 0.0 | 178.3 | 0.99 |
| prophet|cps0.3 | 180 | 0 | 0.0 | 292.3 | 1.62 |
| prophet_anchored|cps0.05 | 180 | 0 | 0.0 | 201.0 | 1.12 |
| prophet_anchored|cps0.3 | 180 | 0 | 0.0 | 273.2 | 1.52 |
| lgbm|l1_nl8_mc20_n150 | 72 | 0 | 0.0 | 50.1 | 0.7 |
| lgbm|l1_nl8_mc20_n400 | 72 | 0 | 0.0 | 136.7 | 1.9 |
| lgbm|l1_nl8_mc60_n150 | 72 | 0 | 0.0 | 64.3 | 0.89 |
| lgbm|l1_nl8_mc60_n400 | 72 | 0 | 0.0 | 130.4 | 1.81 |
| lgbm|l1_nl16_mc20_n150 | 72 | 0 | 0.0 | 92.0 | 1.28 |
| lgbm|l1_nl16_mc20_n400 | 72 | 0 | 0.0 | 194.8 | 2.71 |
| lgbm|l1_nl16_mc60_n150 | 72 | 0 | 0.0 | 80.6 | 1.12 |
| lgbm|l1_nl16_mc60_n400 | 72 | 0 | 0.0 | 175.0 | 2.43 |
| lgbm|huber_nl8_mc20_n150 | 72 | 0 | 0.0 | 42.0 | 0.58 |
| lgbm|huber_nl8_mc20_n400 | 72 | 0 | 0.0 | 95.1 | 1.32 |
| lgbm|huber_nl8_mc60_n150 | 72 | 0 | 0.0 | 38.5 | 0.53 |
| lgbm|huber_nl8_mc60_n400 | 72 | 0 | 0.0 | 90.7 | 1.26 |
| lgbm|huber_nl16_mc20_n150 | 72 | 0 | 0.0 | 66.5 | 0.92 |
| lgbm|huber_nl16_mc20_n400 | 72 | 0 | 0.0 | 124.2 | 1.73 |
| lgbm|huber_nl16_mc60_n150 | 72 | 0 | 0.0 | 48.8 | 0.68 |
| lgbm|huber_nl16_mc60_n400 | 72 | 0 | 0.0 | 84.9 | 1.18 |


Wall-clock for the whole run: 263.0 s. Non-converged means the optimiser hit its iteration limit; the fitted parameters are used as they are. LightGBM and Prophet have no convergence flag in this setup (shown as 0).

## 3. All candidates and baselines, by market and horizon

Scores on the shared triples (section 1). Lower is better. `macro_mean` is the equal-weight mean over the five markets. ETS local level and ARIMA(0,1,1) are mathematically the same model, so `ets|local_level` and `sarima|011` count as one result (identical scores below confirm it). `prophet` is off the shelf; `prophet_anchored` is the declared variant.

### h = 7 days: log-scale MAE

| candidate | Solapur | Pune | Vashi | Kolhapur | Sambhajinagar | macro_mean |
|---|---|---|---|---|---|---|
| base_naive | 0.1274 | 0.1565 | 0.1253 | 0.1245 | 0.1857 | 0.1439 |
| base_mean3 | 0.1307 | 0.1572 | 0.1279 | 0.1261 | 0.1817 | 0.1447 |
| base_seasonal | 0.689 | 0.675 | 0.8435 | 0.7405 | 0.8297 | 0.7555 |
| ets|local_level | 0.1274 | 0.154 | 0.1254 | 0.1244 | 0.1784 | 0.1419 |
| ets|damped_trend | 0.1274 | 0.1541 | 0.1258 | 0.1244 | 0.1784 | 0.142 |
| sarima|011 | 0.1274 | 0.154 | 0.1254 | 0.1244 | 0.1784 | 0.1419 |
| sarima|111 | 0.1269 | 0.1553 | 0.1261 | 0.1239 | 0.1782 | 0.1421 |
| sarima|111_s7 | 0.1269 | 0.1554 | 0.1269 | 0.1242 | 0.1781 | 0.1423 |
| prophet|cps0.05 | 0.3376 | 0.3463 | 0.3289 | 0.3081 | 0.4462 | 0.3534 |
| prophet|cps0.3 | 0.2807 | 0.3528 | 0.2767 | 0.2772 | 0.3549 | 0.3084 |
| prophet_anchored|cps0.05 | 0.1379 | 0.1751 | 0.1439 | 0.1351 | 0.1948 | 0.1574 |
| prophet_anchored|cps0.3 | 0.1395 | 0.1791 | 0.1443 | 0.1372 | 0.1948 | 0.159 |
| lgbm|l1_nl8_mc20_n150 | 0.1266 | 0.1541 | 0.123 | 0.1234 | 0.1746 | 0.1403 |
| lgbm|l1_nl8_mc20_n400 | 0.1265 | 0.1549 | 0.1237 | 0.1243 | 0.1718 | 0.1402 |
| lgbm|l1_nl8_mc60_n150 | 0.1259 | 0.1542 | 0.1225 | 0.1231 | 0.1761 | 0.1404 |
| lgbm|l1_nl8_mc60_n400 | 0.1257 | 0.1555 | 0.1236 | 0.1242 | 0.1735 | 0.1405 |
| lgbm|l1_nl16_mc20_n150 | 0.1269 | 0.155 | 0.1224 | 0.1235 | 0.1725 | 0.1401 |
| lgbm|l1_nl16_mc20_n400 | 0.1273 | 0.1561 | 0.1239 | 0.1251 | 0.1705 | 0.1406 |
| lgbm|l1_nl16_mc60_n150 | 0.1263 | 0.1555 | 0.1224 | 0.1237 | 0.1733 | 0.1402 |
| lgbm|l1_nl16_mc60_n400 | 0.1271 | 0.1578 | 0.1234 | 0.1255 | 0.1722 | 0.1412 |
| lgbm|huber_nl8_mc20_n150 | 0.1267 | 0.1539 | 0.1218 | 0.1228 | 0.1749 | 0.14 |
| lgbm|huber_nl8_mc20_n400 | 0.128 | 0.1553 | 0.1229 | 0.1234 | 0.1714 | 0.1402 |
| lgbm|huber_nl8_mc60_n150 | 0.1259 | 0.154 | 0.1216 | 0.123 | 0.1748 | 0.1399 |
| lgbm|huber_nl8_mc60_n400 | 0.1274 | 0.1556 | 0.1234 | 0.1242 | 0.1723 | 0.1406 |
| lgbm|huber_nl16_mc20_n150 | 0.127 | 0.1551 | 0.1219 | 0.1228 | 0.1728 | 0.1399 |
| lgbm|huber_nl16_mc20_n400 | 0.1299 | 0.1586 | 0.1247 | 0.1255 | 0.1711 | 0.142 |
| lgbm|huber_nl16_mc60_n150 | 0.1262 | 0.1548 | 0.1214 | 0.1222 | 0.1733 | 0.1396 |
| lgbm|huber_nl16_mc60_n400 | 0.129 | 0.1585 | 0.1238 | 0.126 | 0.1724 | 0.1419 |


### h = 7 days: MAE in rupees per quintal

| candidate | Solapur | Pune | Vashi | Kolhapur | Sambhajinagar | macro_mean |
|---|---|---|---|---|---|---|
| base_naive | 166.2 | 236.1 | 308.3 | 217.3 | 222.5 | 230.1 |
| base_mean3 | 164.7 | 232.1 | 306.7 | 215.7 | 214.3 | 226.7 |
| base_seasonal | 728.3 | 876.9 | 1471.4 | 1023.9 | 821.1 | 984.3 |
| ets|local_level | 163.9 | 227.9 | 302.8 | 215.6 | 210.9 | 224.2 |
| ets|damped_trend | 163.9 | 227.9 | 299.2 | 215.6 | 211.0 | 223.5 |
| sarima|011 | 163.9 | 227.9 | 302.8 | 215.6 | 210.9 | 224.2 |
| sarima|111 | 161.4 | 228.2 | 305.3 | 213.3 | 209.7 | 223.6 |
| sarima|111_s7 | 161.5 | 228.3 | 309.0 | 214.0 | 209.4 | 224.5 |
| prophet|cps0.05 | 384.2 | 460.1 | 624.3 | 454.7 | 484.9 | 481.6 |
| prophet|cps0.3 | 317.1 | 531.1 | 695.0 | 418.3 | 396.4 | 471.6 |
| prophet_anchored|cps0.05 | 176.3 | 254.5 | 340.8 | 232.4 | 235.1 | 247.8 |
| prophet_anchored|cps0.3 | 174.4 | 264.4 | 349.6 | 234.9 | 235.4 | 251.7 |
| lgbm|l1_nl8_mc20_n150 | 160.0 | 233.5 | 293.6 | 210.6 | 207.2 | 221.0 |
| lgbm|l1_nl8_mc20_n400 | 157.7 | 236.4 | 293.0 | 212.1 | 202.1 | 220.3 |
| lgbm|l1_nl8_mc60_n150 | 159.7 | 233.8 | 292.2 | 210.3 | 209.4 | 221.1 |
| lgbm|l1_nl8_mc60_n400 | 158.5 | 237.4 | 293.1 | 211.7 | 205.1 | 221.2 |
| lgbm|l1_nl16_mc20_n150 | 160.6 | 236.8 | 292.8 | 211.2 | 206.0 | 221.5 |
| lgbm|l1_nl16_mc20_n400 | 160.7 | 241.0 | 296.3 | 214.8 | 202.8 | 223.1 |
| lgbm|l1_nl16_mc60_n150 | 161.1 | 235.5 | 293.9 | 211.3 | 207.0 | 221.8 |
| lgbm|l1_nl16_mc60_n400 | 161.7 | 240.6 | 293.3 | 214.4 | 204.6 | 222.9 |
| lgbm|huber_nl8_mc20_n150 | 161.8 | 234.7 | 292.2 | 212.5 | 208.3 | 221.9 |
| lgbm|huber_nl8_mc20_n400 | 162.4 | 239.2 | 294.4 | 213.3 | 202.6 | 222.3 |
| lgbm|huber_nl8_mc60_n150 | 161.4 | 234.2 | 291.4 | 212.6 | 208.6 | 221.6 |
| lgbm|huber_nl8_mc60_n400 | 163.4 | 239.9 | 296.4 | 214.9 | 204.7 | 223.8 |
| lgbm|huber_nl16_mc20_n150 | 162.8 | 238.4 | 292.2 | 212.7 | 205.6 | 222.3 |
| lgbm|huber_nl16_mc20_n400 | 166.1 | 246.3 | 297.3 | 218.3 | 202.2 | 226.0 |
| lgbm|huber_nl16_mc60_n150 | 161.8 | 235.7 | 290.8 | 210.6 | 206.6 | 221.1 |
| lgbm|huber_nl16_mc60_n400 | 165.7 | 245.9 | 297.2 | 217.8 | 204.3 | 226.2 |


### h = 30 days: log-scale MAE

| candidate | Solapur | Pune | Vashi | Kolhapur | Sambhajinagar | macro_mean |
|---|---|---|---|---|---|---|
| base_naive | 0.2638 | 0.3288 | 0.3013 | 0.2695 | 0.3355 | 0.2998 |
| base_mean3 | 0.2699 | 0.3352 | 0.3077 | 0.2744 | 0.3385 | 0.3051 |
| base_seasonal | 0.7256 | 0.6971 | 0.8548 | 0.7688 | 0.8559 | 0.7805 |
| ets|local_level | 0.2646 | 0.3325 | 0.3032 | 0.271 | 0.3371 | 0.3017 |
| ets|damped_trend | 0.2646 | 0.333 | 0.3068 | 0.271 | 0.3371 | 0.3025 |
| sarima|011 | 0.2646 | 0.3325 | 0.3032 | 0.271 | 0.3371 | 0.3017 |
| sarima|111 | 0.2657 | 0.334 | 0.3035 | 0.2714 | 0.3381 | 0.3025 |
| sarima|111_s7 | 0.2656 | 0.3341 | 0.3029 | 0.2715 | 0.3382 | 0.3024 |
| prophet|cps0.05 | 0.4496 | 0.453 | 0.4743 | 0.4136 | 0.5826 | 0.4746 |
| prophet|cps0.3 | 0.4169 | 0.5115 | 0.5746 | 0.39 | 0.498 | 0.4782 |
| prophet_anchored|cps0.05 | 0.297 | 0.3348 | 0.3592 | 0.293 | 0.3792 | 0.3326 |
| prophet_anchored|cps0.3 | 0.3058 | 0.3895 | 0.449 | 0.3018 | 0.3788 | 0.365 |
| lgbm|l1_nl8_mc20_n150 | 0.254 | 0.3068 | 0.2814 | 0.2652 | 0.3214 | 0.2858 |
| lgbm|l1_nl8_mc20_n400 | 0.2558 | 0.3079 | 0.2871 | 0.2716 | 0.3191 | 0.2883 |
| lgbm|l1_nl8_mc60_n150 | 0.2544 | 0.3087 | 0.2812 | 0.265 | 0.3212 | 0.2861 |
| lgbm|l1_nl8_mc60_n400 | 0.2563 | 0.3101 | 0.2854 | 0.272 | 0.3227 | 0.2893 |
| lgbm|l1_nl16_mc20_n150 | 0.2513 | 0.3082 | 0.2779 | 0.2659 | 0.3187 | 0.2844 |
| lgbm|l1_nl16_mc20_n400 | 0.2517 | 0.3081 | 0.2819 | 0.271 | 0.3201 | 0.2866 |
| lgbm|l1_nl16_mc60_n150 | 0.2511 | 0.3088 | 0.2811 | 0.2683 | 0.3216 | 0.2862 |
| lgbm|l1_nl16_mc60_n400 | 0.2511 | 0.3117 | 0.2828 | 0.2745 | 0.3226 | 0.2885 |
| lgbm|huber_nl8_mc20_n150 | 0.2528 | 0.3054 | 0.28 | 0.2592 | 0.3206 | 0.2836 |
| lgbm|huber_nl8_mc20_n400 | 0.2537 | 0.3049 | 0.2838 | 0.264 | 0.3196 | 0.2852 |
| lgbm|huber_nl8_mc60_n150 | 0.2518 | 0.3036 | 0.2777 | 0.2579 | 0.3193 | 0.2821 |
| lgbm|huber_nl8_mc60_n400 | 0.2556 | 0.3052 | 0.2824 | 0.264 | 0.3196 | 0.2854 |
| lgbm|huber_nl16_mc20_n150 | 0.2535 | 0.3013 | 0.2741 | 0.2592 | 0.3172 | 0.2811 |
| lgbm|huber_nl16_mc20_n400 | 0.2571 | 0.3033 | 0.2832 | 0.2674 | 0.3212 | 0.2864 |
| lgbm|huber_nl16_mc60_n150 | 0.2545 | 0.3025 | 0.2762 | 0.2611 | 0.3178 | 0.2824 |
| lgbm|huber_nl16_mc60_n400 | 0.2563 | 0.3045 | 0.2821 | 0.2707 | 0.3183 | 0.2864 |


### h = 30 days: MAE in rupees per quintal

| candidate | Solapur | Pune | Vashi | Kolhapur | Sambhajinagar | macro_mean |
|---|---|---|---|---|---|---|
| base_naive | 330.0 | 489.2 | 667.6 | 422.1 | 397.4 | 461.3 |
| base_mean3 | 336.2 | 495.2 | 680.5 | 428.4 | 399.4 | 467.9 |
| base_seasonal | 771.3 | 906.0 | 1468.7 | 1056.2 | 854.7 | 1011.4 |
| ets|local_level | 329.6 | 490.4 | 670.6 | 423.7 | 397.0 | 462.3 |
| ets|damped_trend | 329.6 | 491.1 | 676.5 | 423.7 | 397.0 | 463.6 |
| sarima|011 | 329.6 | 490.4 | 670.6 | 423.7 | 397.0 | 462.3 |
| sarima|111 | 330.5 | 491.7 | 671.6 | 423.4 | 397.8 | 463.0 |
| sarima|111_s7 | 330.4 | 491.8 | 671.0 | 423.9 | 397.9 | 463.0 |
| prophet|cps0.05 | 502.8 | 561.1 | 902.9 | 602.5 | 628.3 | 639.5 |
| prophet|cps0.3 | 446.8 | 714.6 | 1172.8 | 557.5 | 511.6 | 680.7 |
| prophet_anchored|cps0.05 | 369.2 | 495.9 | 774.4 | 468.0 | 451.8 | 511.9 |
| prophet_anchored|cps0.3 | 356.8 | 586.2 | 898.0 | 480.7 | 444.9 | 553.3 |
| lgbm|l1_nl8_mc20_n150 | 313.5 | 464.5 | 621.6 | 414.4 | 376.6 | 438.1 |
| lgbm|l1_nl8_mc20_n400 | 310.9 | 464.7 | 636.3 | 427.1 | 368.0 | 441.4 |
| lgbm|l1_nl8_mc60_n150 | 315.3 | 470.6 | 622.1 | 416.0 | 377.2 | 440.3 |
| lgbm|l1_nl8_mc60_n400 | 312.4 | 471.0 | 631.1 | 429.5 | 373.8 | 443.6 |
| lgbm|l1_nl16_mc20_n150 | 304.4 | 464.7 | 616.5 | 411.7 | 370.8 | 433.6 |
| lgbm|l1_nl16_mc20_n400 | 303.4 | 462.5 | 626.6 | 422.7 | 367.5 | 436.5 |
| lgbm|l1_nl16_mc60_n150 | 307.4 | 467.2 | 623.6 | 421.3 | 377.1 | 439.3 |
| lgbm|l1_nl16_mc60_n400 | 303.6 | 469.4 | 624.8 | 433.7 | 376.3 | 441.6 |
| lgbm|huber_nl8_mc20_n150 | 317.1 | 463.4 | 619.0 | 407.4 | 379.2 | 437.2 |
| lgbm|huber_nl8_mc20_n400 | 314.0 | 465.2 | 625.6 | 414.5 | 373.1 | 438.5 |
| lgbm|huber_nl8_mc60_n150 | 315.7 | 460.6 | 615.9 | 404.8 | 377.8 | 435.0 |
| lgbm|huber_nl8_mc60_n400 | 315.4 | 466.5 | 625.3 | 415.3 | 374.8 | 439.5 |
| lgbm|huber_nl16_mc20_n150 | 312.3 | 456.9 | 607.1 | 404.2 | 374.0 | 430.9 |
| lgbm|huber_nl16_mc20_n400 | 314.1 | 458.5 | 625.1 | 419.0 | 373.2 | 438.0 |
| lgbm|huber_nl16_mc60_n150 | 315.7 | 459.8 | 615.1 | 410.8 | 376.7 | 435.6 |
| lgbm|huber_nl16_mc60_n400 | 311.3 | 460.6 | 626.3 | 427.4 | 372.6 | 439.6 |


### Log-scale MAE by fold year (macro mean over markets), h = 7 and 30

| h | col | logMAE_2017 | logMAE_2018 | logMAE_2019 | logMAE |
|---|---|---|---|---|---|
| 7 | base_mean3 | 0.1481 | 0.1359 | 0.1514 | 0.1447 |
| 7 | base_naive | 0.1438 | 0.1362 | 0.1529 | 0.1439 |
| 7 | base_seasonal | 0.7091 | 0.6773 | 0.8895 | 0.7555 |
| 7 | ets|damped_trend | 0.1444 | 0.1337 | 0.1493 | 0.142 |
| 7 | ets|local_level | 0.1444 | 0.1335 | 0.1492 | 0.1419 |
| 7 | lgbm|huber_nl16_mc20_n150 | 0.1438 | 0.1294 | 0.1479 | 0.1399 |
| 7 | lgbm|huber_nl16_mc20_n400 | 0.1458 | 0.1327 | 0.1487 | 0.142 |
| 7 | lgbm|huber_nl16_mc60_n150 | 0.1428 | 0.1294 | 0.148 | 0.1396 |
| 7 | lgbm|huber_nl16_mc60_n400 | 0.1445 | 0.1325 | 0.15 | 0.1419 |
| 7 | lgbm|huber_nl8_mc20_n150 | 0.1445 | 0.1298 | 0.1474 | 0.14 |
| 7 | lgbm|huber_nl8_mc20_n400 | 0.1446 | 0.1307 | 0.1467 | 0.1402 |
| 7 | lgbm|huber_nl8_mc60_n150 | 0.1443 | 0.1291 | 0.1477 | 0.1399 |
| 7 | lgbm|huber_nl8_mc60_n400 | 0.1445 | 0.13 | 0.1486 | 0.1406 |
| 7 | lgbm|l1_nl16_mc20_n150 | 0.1439 | 0.1305 | 0.1471 | 0.1401 |
| 7 | lgbm|l1_nl16_mc20_n400 | 0.1445 | 0.1314 | 0.1471 | 0.1406 |
| 7 | lgbm|l1_nl16_mc60_n150 | 0.1456 | 0.1292 | 0.1475 | 0.1402 |
| 7 | lgbm|l1_nl16_mc60_n400 | 0.146 | 0.1308 | 0.1482 | 0.1412 |
| 7 | lgbm|l1_nl8_mc20_n150 | 0.1458 | 0.1304 | 0.1464 | 0.1403 |
| 7 | lgbm|l1_nl8_mc20_n400 | 0.1456 | 0.1311 | 0.1454 | 0.1402 |
| 7 | lgbm|l1_nl8_mc60_n150 | 0.1459 | 0.1297 | 0.1471 | 0.1404 |
| 7 | lgbm|l1_nl8_mc60_n400 | 0.1461 | 0.1298 | 0.1471 | 0.1405 |
| 7 | prophet_anchored|cps0.05 | 0.1567 | 0.151 | 0.1653 | 0.1574 |
| 7 | prophet_anchored|cps0.3 | 0.1612 | 0.1508 | 0.1663 | 0.159 |
| 7 | prophet|cps0.05 | 0.3355 | 0.3555 | 0.3713 | 0.3534 |
| 7 | prophet|cps0.3 | 0.2987 | 0.2784 | 0.3506 | 0.3084 |
| 7 | sarima|011 | 0.1444 | 0.1335 | 0.1492 | 0.1419 |
| 7 | sarima|111 | 0.1459 | 0.1335 | 0.1484 | 0.1421 |
| 7 | sarima|111_s7 | 0.1463 | 0.1336 | 0.1486 | 0.1423 |
| 30 | base_mean3 | 0.2999 | 0.2962 | 0.3196 | 0.3051 |
| 30 | base_naive | 0.2999 | 0.2873 | 0.3126 | 0.2998 |
| 30 | base_seasonal | 0.7563 | 0.6932 | 0.9029 | 0.7805 |
| 30 | ets|damped_trend | 0.2995 | 0.2915 | 0.3168 | 0.3025 |
| 30 | ets|local_level | 0.2987 | 0.2909 | 0.3158 | 0.3017 |
| 30 | lgbm|huber_nl16_mc20_n150 | 0.2917 | 0.2415 | 0.3135 | 0.2811 |
| 30 | lgbm|huber_nl16_mc20_n400 | 0.2939 | 0.2478 | 0.3212 | 0.2864 |
| 30 | lgbm|huber_nl16_mc60_n150 | 0.2905 | 0.2448 | 0.3152 | 0.2824 |
| 30 | lgbm|huber_nl16_mc60_n400 | 0.2904 | 0.2522 | 0.3196 | 0.2864 |
| 30 | lgbm|huber_nl8_mc20_n150 | 0.2961 | 0.2467 | 0.3111 | 0.2836 |
| 30 | lgbm|huber_nl8_mc20_n400 | 0.2977 | 0.2467 | 0.3151 | 0.2852 |
| 30 | lgbm|huber_nl8_mc60_n150 | 0.2917 | 0.2458 | 0.3115 | 0.2821 |
| 30 | lgbm|huber_nl8_mc60_n400 | 0.2956 | 0.2485 | 0.3153 | 0.2854 |
| 30 | lgbm|l1_nl16_mc20_n150 | 0.2908 | 0.2528 | 0.3124 | 0.2844 |
| 30 | lgbm|l1_nl16_mc20_n400 | 0.2888 | 0.2545 | 0.3191 | 0.2866 |
| 30 | lgbm|l1_nl16_mc60_n150 | 0.2894 | 0.257 | 0.3148 | 0.2862 |
| 30 | lgbm|l1_nl16_mc60_n400 | 0.2869 | 0.2622 | 0.319 | 0.2885 |
| 30 | lgbm|l1_nl8_mc20_n150 | 0.2962 | 0.2536 | 0.3107 | 0.2858 |
| 30 | lgbm|l1_nl8_mc20_n400 | 0.296 | 0.2557 | 0.3167 | 0.2883 |
| 30 | lgbm|l1_nl8_mc60_n150 | 0.297 | 0.2525 | 0.3121 | 0.2861 |
| 30 | lgbm|l1_nl8_mc60_n400 | 0.2971 | 0.2592 | 0.3148 | 0.2893 |
| 30 | prophet_anchored|cps0.05 | 0.3195 | 0.3131 | 0.3663 | 0.3326 |
| 30 | prophet_anchored|cps0.3 | 0.4234 | 0.3108 | 0.3813 | 0.365 |
| 30 | prophet|cps0.05 | 0.4746 | 0.4771 | 0.4734 | 0.4746 |
| 30 | prophet|cps0.3 | 0.5431 | 0.4319 | 0.4857 | 0.4782 |
| 30 | sarima|011 | 0.2987 | 0.2909 | 0.3158 | 0.3017 |
| 30 | sarima|111 | 0.2992 | 0.2923 | 0.3166 | 0.3025 |
| 30 | sarima|111_s7 | 0.299 | 0.2922 | 0.3165 | 0.3024 |


## 4. Proposed selection (lowest pooled tuning log-MAE per family and horizon)

| h | ets | lgbm | prophet | prophet_anchored | sarima |
|---|---|---|---|---|---|
| 7 | ets|local_level | lgbm|huber_nl16_mc60_n150 | prophet|cps0.3 | prophet_anchored|cps0.05 | sarima|011 |
| 30 | ets|local_level | lgbm|huber_nl16_mc20_n150 | prophet|cps0.05 | prophet_anchored|cps0.05 | sarima|011 |


Model chosen per market and horizon on the tuning folds (lowest rupee MAE among the family choices above):

| market | 7 | 30 |
|---|---|---|
| Kolhapur | lgbm | lgbm |
| Pune | ets | lgbm |
| Sambhajinagar | lgbm | lgbm |
| Solapur | lgbm | lgbm |
| Vashi | lgbm | lgbm |


## 5. Frozen candidate of each family vs the strongest baseline on the tuning folds

Improvement in rupee MAE = 100 x (strongest baseline MAE - model MAE) / strongest baseline MAE. Positive = the model is better. The strongest baseline is chosen per market and horizon among naive, mean of the last 3 quotes and 365-day seasonal-naive.

### Per market

| market | h | n | strongest_baseline | baseline_MAE_rs | ets | sarima | prophet | prophet_anchored | lgbm |
|---|---|---|---|---|---|---|---|---|---|
| Kolhapur | 7 | 793 | base_mean3 | 215.68 | 0.03 | 0.02 | -93.95 | -7.73 | 2.37 |
| Kolhapur | 30 | 664 | base_naive | 422.14 | -0.36 | -0.36 | -42.72 | -10.87 | 4.25 |
| Pune | 7 | 831 | base_mean3 | 232.09 | 1.82 | 1.81 | -128.81 | -9.67 | -1.55 |
| Pune | 30 | 700 | base_naive | 489.21 | -0.25 | -0.25 | -14.7 | -1.36 | 6.6 |
| Sambhajinagar | 7 | 830 | base_mean3 | 214.29 | 1.56 | 1.56 | -85.0 | -9.71 | 3.61 |
| Sambhajinagar | 30 | 701 | base_naive | 397.43 | 0.11 | 0.11 | -58.09 | -13.69 | 5.89 |
| Solapur | 7 | 673 | base_mean3 | 164.75 | 0.49 | 0.49 | -92.45 | -7.03 | 1.81 |
| Solapur | 30 | 577 | base_naive | 329.96 | 0.11 | 0.11 | -52.39 | -11.9 | 5.36 |
| Vashi | 7 | 663 | base_mean3 | 306.74 | 1.29 | 1.3 | -126.59 | -11.1 | 5.19 |
| Vashi | 30 | 575 | base_naive | 667.57 | -0.45 | -0.45 | -35.25 | -16.0 | 9.06 |


### Macro average over markets (rupee MAE)

| h | ets | sarima | prophet | prophet_anchored | lgbm |
|---|---|---|---|---|---|
| 7 | 1.04 | 1.04 | -105.36 | -9.05 | 2.29 |
| 30 | -0.17 | -0.17 | -40.63 | -10.76 | 6.23 |


### Pooled log-scale MAE improvement (equal weight per market)

| h | strongest_baseline_(pooled_log) | ets | sarima | prophet | prophet_anchored | lgbm |
|---|---|---|---|---|---|---|
| 7 | base_naive | 1.37 | 1.37 | -114.36 | -9.36 | 3.0 |
| 30 | base_naive | -0.63 | -0.63 | -58.32 | -10.95 | 6.24 |


### Does the best candidate of each model type beat the strongest baseline on the tuning folds?

| h | family | markets_where_it_beats_strongest_baseline | macro_improvement_pct |
|---|---|---|---|
| 7 | ets | 5 of 5 | 1.04 |
| 7 | sarima | 5 of 5 | 1.04 |
| 7 | prophet | 0 of 5 | -105.36 |
| 7 | prophet_anchored | 0 of 5 | -9.05 |
| 7 | lgbm | 4 of 5 | 2.29 |
| 30 | ets | 2 of 5 | -0.17 |
| 30 | sarima | 2 of 5 | -0.17 |
| 30 | prophet | 0 of 5 | -40.63 |
| 30 | prophet_anchored | 0 of 5 | -10.76 |
| 30 | lgbm | 5 of 5 | 6.23 |


## 6. Notes

- **Tuning history.** Run 1 had a silent NaN in the damped-trend ETS fit for Kolhapur (a 71-day gap right after its first quote made the library's start value NaN), so Kolhapur was dropped from the selection without any error. This was fixed with explicit start values, a non-finite-parameter check and a coverage guard, and tuning was re-run (run 2). `prophet_anchored` was then added at the student's request before anything was frozen, and tuning was re-run once more (this run). No 2020-2024 result was looked at in any of the runs.
- **Off-the-shelf Prophet** forecasts from a curve fitted at the last monthly refit; it does not use prices after the refit, which is why it is worse than naive even on refit day. `prophet_anchored` keeps only its predicted movement and anchors it to the latest quote.
- **ETS local level and ARIMA(0,1,1) are the same model**; they are listed twice only because both were declared candidates, and they count as one result.
- The 365-day seasonal baseline is much worse than naive because price levels differ by a factor of 2-3 between years; an independent recomputation of 40 random triples from the panel matched exactly.