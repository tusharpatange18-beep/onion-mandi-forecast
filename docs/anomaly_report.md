# Anomaly detection (Phase 4B)

## Detectors (declared before any result was looked at)

- **Detector 1, `stl_residual`: a retrospective large-swing detector, not a spike detector.** STL with robust=True and period 52 on the weekly mean of log modal price (observed days only; a week needs at least 3 observed days). A week is flagged when |residual - median| > 3.5 x 1.4826 x MAD. **It uses data after the date as well, so it is historical labelling, not a real-time alert.** Unobserved weeks are bridged by linear interpolation only so that STL can run; they are never flagged and are not used for the median or MAD. The date reported is the Sunday that ends the week.
- **Detector 2, `forecast_surprise_h7` and `_h30` (usable in real time).** The realised quote falls outside the 95% conformal interval of the headline LightGBM issued h days earlier (section 1 of `docs/interval_report.md`). Available for 2020-2024 only. `score` = (log realised - log forecast) / interval half-width, so |score| > 1 means outside.
- **Detector 3, `causal_daily` (usable in real time; declared after the first Phase 4 review, run once, not tuned).** On observed quotes only: baseline = median of the log price of the previous 28 observed quotes; scale = 1.4826 x MAD of the previous 90 observed daily log changes; flag if |log p_t - baseline| > 4 x scale. It uses only data before the quote being judged, so it could have raised the alert on the day. `score` = (log p_t - baseline) / scale. Quotes without 90 prior changes are not scored.
- **Arrivals check.** `arrivals_z` = z-score of the mean log arrivals of the flagged week (or day) against the median and MAD of the log arrivals of the 30 days before it. A "drop" means z <= -1.0.
- **Event-period labels** use only periods whose both end dates are high-confidence in `docs/eda_findings.md`: the 2019-20, 2020-21 and 2023-24 export bans, the COVID lockdown and the Oct-Dec 2023 minimum-export-price period (E11 verified against a Deccan Herald article dated 28 Oct 2023; it ends 2023-12-07, so it does not overlap the ban); `none` means outside those periods.
- No causal claims are made anywhere: the table says when, where, how large, and what arrivals did.
- The full table is `results/anomalies.csv` (not committed: it lists individual market-dates); this document shows aggregates and a short excerpt.

## 1. How many flags

| detector | flags | up | down |
|---|---|---|---|
| causal_daily | 1930 | 918 | 1012 |
| forecast_surprise_h30 | 219 | 94 | 125 |
| forecast_surprise_h7 | 269 | 131 | 138 |
| stl_residual | 460 | 233 | 227 |


Per market and year:

| detector | market | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| causal_daily | Kolhapur | 0.0 | 67.0 | 12.0 | 25.0 | 50.0 | 24.0 | 48.0 | 32.0 | 5.0 | 23.0 | 22.0 |
| causal_daily | Pune | 0.0 | 25.0 | 17.0 | 25.0 | 9.0 | 49.0 | 35.0 | 43.0 | 52.0 | 46.0 | 72.0 |
| causal_daily | Sambhajinagar | 1.0 | 26.0 | 4.0 | 23.0 | 24.0 | 12.0 | 40.0 | 38.0 | 13.0 | 41.0 | 19.0 |
| causal_daily | Solapur | 0.0 | 75.0 | 19.0 | 2.0 | 40.0 | 49.0 | 63.0 | 43.0 | 27.0 | 35.0 | 78.0 |
| causal_daily | Vashi | 0.0 | 0.0 | 1.0 | 40.0 | 81.0 | 61.0 | 91.0 | 78.0 | 86.0 | 61.0 | 78.0 |
| forecast_surprise_h30 | Kolhapur | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 11.0 | 6.0 | 4.0 | 15.0 | 8.0 |
| forecast_surprise_h30 | Pune | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 3.0 | 8.0 | 4.0 | 10.0 | 9.0 |
| forecast_surprise_h30 | Sambhajinagar | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 7.0 | 5.0 | 12.0 | 14.0 | 8.0 |
| forecast_surprise_h30 | Solapur | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 8.0 | 12.0 | 0.0 | 15.0 | 12.0 |
| forecast_surprise_h30 | Vashi | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 15.0 | 5.0 | 4.0 | 13.0 | 11.0 |
| forecast_surprise_h7 | Kolhapur | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 4.0 | 11.0 | 6.0 | 22.0 | 8.0 |
| forecast_surprise_h7 | Pune | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 6.0 | 4.0 | 13.0 | 17.0 | 9.0 |
| forecast_surprise_h7 | Sambhajinagar | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 9.0 | 9.0 | 13.0 | 24.0 | 11.0 |
| forecast_surprise_h7 | Solapur | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 3.0 | 20.0 | 0.0 | 19.0 | 10.0 |
| forecast_surprise_h7 | Vashi | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 5.0 | 10.0 | 13.0 | 15.0 | 8.0 |
| stl_residual | Kolhapur | 0.0 | 14.0 | 10.0 | 14.0 | 23.0 | 19.0 | 11.0 | 9.0 | 4.0 | 5.0 | 0.0 |
| stl_residual | Pune | 0.0 | 0.0 | 18.0 | 18.0 | 15.0 | 20.0 | 14.0 | 4.0 | 6.0 | 2.0 | 0.0 |
| stl_residual | Sambhajinagar | 0.0 | 12.0 | 6.0 | 14.0 | 17.0 | 21.0 | 12.0 | 3.0 | 3.0 | 8.0 | 1.0 |
| stl_residual | Solapur | 0.0 | 10.0 | 1.0 | 7.0 | 12.0 | 16.0 | 4.0 | 4.0 | 3.0 | 7.0 | 0.0 |
| stl_residual | Vashi | 0.0 | 0.0 | 0.0 | 21.0 | 14.0 | 25.0 | 16.0 | 6.0 | 4.0 | 6.0 | 1.0 |


Share of flags with another market flagged by the same detector in the same Monday-Sunday week, and share in each event-period label:

| detector | share_other_market_flagged_same_week_pct |
|---|---|
| causal_daily | 89.2 |
| forecast_surprise_h30 | 78.1 |
| forecast_surprise_h7 | 71.0 |
| stl_residual | 87.6 |


| event_period | causal_daily | forecast_surprise_h30 | forecast_surprise_h7 | stl_residual |
|---|---|---|---|---|
| covid_lockdown_2020 | 32 | 0 | 5 | 5 |
| export_ban_2019_20 | 133 | 5 | 3 | 20 |
| export_ban_2020_21 | 101 | 21 | 10 | 6 |
| export_ban_2023_24 | 133 | 39 | 44 | 5 |
| mep_oct_dec_2023 | 45 | 26 | 14 | 20 |
| none | 1486 | 128 | 193 | 404 |


Flags per 100 comparable dates by event period (the denominator is the dates each detector could have judged: observed weeks for STL, scored observed quotes for the causal daily detector, forecasts with an interval by target date for the surprise detectors; rates are comparable across periods within a detector, not across detectors):

| event_period | causal_daily | forecast_surprise_h30 | forecast_surprise_h7 | stl_residual |
|---|---|---|---|---|
| covid_lockdown_2020 | 37.2 | 0.0 | 11.9 | 27.8 |
| export_ban_2019_20 | 20.0 | 3.8 | 1.3 | 16.7 |
| export_ban_2020_21 | 26.7 | 8.1 | 3.0 | 8.6 |
| export_ban_2023_24 | 25.1 | 9.8 | 9.6 | 5.2 |
| mep_oct_dec_2023 | 31.9 | 24.5 | 11.4 | 66.7 |
| none | 16.6 | 3.5 | 4.4 | 18.9 |


## 2. Do flagged spikes coincide with an arrivals drop?

`share_with_arrivals_drop` is among flagged spikes; `baseline` is the same share among all comparable weeks (STL) or forecast target days (surprise detectors), i.e. what chance alone would give. Descriptive only.

| detector | direction | n_flags | n_with_arrivals_z | share_with_arrivals_drop | baseline_share_all_comparable_dates | median_arrivals_z |
|---|---|---|---|---|---|---|
| causal_daily | down | 1012 | 976 | 14.8 | 19.4 | 0.26 |
| causal_daily | up | 918 | 918 | 19.9 | 19.4 | 0.03 |
| forecast_surprise_h30 | down | 125 | 125 | 19.2 | 17.9 | 0.15 |
| forecast_surprise_h30 | up | 94 | 94 | 14.9 | 17.9 | -0.01 |
| forecast_surprise_h7 | down | 138 | 137 | 13.9 | 20.4 | 0.62 |
| forecast_surprise_h7 | up | 131 | 131 | 23.7 | 20.4 | -0.03 |
| stl_residual | down | 227 | 223 | 12.6 | 15.6 | 0.03 |
| stl_residual | up | 233 | 233 | 15.5 | 15.6 | -0.1 |


## 3. Validation without labels: a synthetic test of the detectors, not data

**Synthetic test of the detector, not data.** Multiplicative spikes (price x (1 +/- magnitude)) were injected at random dates of 2020-2024 into an **in-memory copy** of the cleaned series; nothing was saved to `data/` and the real series is untouched. Each run places one spike of each type (magnitude x direction x duration) at least 45 days apart (detectors 1 and 3 are tested on exactly the same spike dates); 20 independent placements per market and market-horizon, so each cell below has 100 or more spikes. Recall = share of injected spikes flagged in the right direction in the affected week(s) (STL) or on the affected target dates (forecast surprise). For the forecast-surprise detector the forecasts are the ones actually made (forecasts issued from inside a spike are not re-made, an approximation) and the conformal intervals are recomputed from the modified errors.

The detector `stl_residual_NONROBUST_diagnostic` is **not part of the declared method**; it is the same test with `robust=False`, shown only to explain a finding about `robust=True`.

### 3.1 False-positive rate

Share of non-spike weeks (or days) flagged, in the unmodified real series and in the series with injected spikes:

| detector | kind | false_positive_rate | n |
|---|---|---|---|
| causal_daily | series with injected spikes (non-spike weeks/days) | 19.0 | 100 |
| causal_daily | unmodified real series | 19.6 | 5 |
| forecast_surprise_h30 | series with injected spikes (non-spike weeks/days) | 4.6 | 100 |
| forecast_surprise_h30 | unmodified real series | 4.9 | 5 |
| forecast_surprise_h7 | series with injected spikes (non-spike weeks/days) | 4.0 | 100 |
| forecast_surprise_h7 | unmodified real series | 4.8 | 5 |
| stl_residual | series with injected spikes (non-spike weeks/days) | 19.1 | 100 |
| stl_residual | unmodified real series | 18.7 | 5 |
| stl_residual_NONROBUST_diagnostic | series with injected spikes (non-spike weeks/days) | 0.6 | 100 |
| stl_residual_NONROBUST_diagnostic | unmodified real series | 0.5 | 5 |


### 3.2 Recall by spike size and duration (both directions pooled), %

**causal_daily**

| magnitude | 1-day spike | 3-day spike |
|---|---|---|
| 0.25 | 30.0 | 38.0 |
| 0.4 | 54.0 | 58.0 |
| 0.6 | 73.0 | 76.0 |


**forecast_surprise_h30**

| magnitude | 1-day spike | 3-day spike |
|---|---|---|
| 0.25 | 8.0 | 12.0 |
| 0.4 | 24.0 | 30.0 |
| 0.6 | 53.0 | 56.0 |


**forecast_surprise_h7**

| magnitude | 1-day spike | 3-day spike |
|---|---|---|
| 0.25 | 19.0 | 28.0 |
| 0.4 | 55.0 | 58.0 |
| 0.6 | 80.0 | 88.0 |


**stl_residual**

| magnitude | 1-day spike | 3-day spike |
|---|---|---|
| 0.25 | 5.0 | 6.0 |
| 0.4 | 6.0 | 10.0 |
| 0.6 | 12.0 | 22.0 |


**stl_residual_NONROBUST_diagnostic**

| magnitude | 1-day spike | 3-day spike |
|---|---|---|
| 0.25 | 0.0 | 0.0 |
| 0.4 | 0.0 | 0.0 |
| 0.6 | 0.0 | 3.0 |


### 3.3 Recall by direction (durations pooled), %

**causal_daily**

| magnitude | down | up |
|---|---|---|
| 0.25 | 36.0 | 31.0 |
| 0.4 | 66.0 | 46.0 |
| 0.6 | 90.0 | 59.0 |


**forecast_surprise_h30**

| magnitude | down | up |
|---|---|---|
| 0.25 | 10.0 | 10.0 |
| 0.4 | 33.0 | 20.0 |
| 0.6 | 78.0 | 32.0 |


**forecast_surprise_h7**

| magnitude | down | up |
|---|---|---|
| 0.25 | 27.0 | 20.0 |
| 0.4 | 76.0 | 38.0 |
| 0.6 | 99.0 | 69.0 |


**stl_residual**

| magnitude | down | up |
|---|---|---|
| 0.25 | 3.0 | 8.0 |
| 0.4 | 8.0 | 8.0 |
| 0.6 | 22.0 | 12.0 |


**stl_residual_NONROBUST_diagnostic**

| magnitude | down | up |
|---|---|---|
| 0.25 | 0.0 | 0.0 |
| 0.4 | 0.0 | 0.0 |
| 0.6 | 2.0 | 0.0 |


## 4. Largest flags (excerpt)

| detector | market | date | direction | score | price | expected | arrivals_z | event_period | other_markets_flagged_same_week |
|---|---|---|---|---|---|---|---|---|---|
| causal_daily | Solapur | 2024-06-08 | up | 24.0919 | 2600.0 | 1350.0 | -0.52 | none | True |
| causal_daily | Vashi | 2024-12-16 | down | -22.9349 | 1650.0 | 3924.28 | 4.71 | none | True |
| causal_daily | Solapur | 2024-06-06 | up | 22.5369 | 1800.0 | 1324.76 | -1.67 | none | True |
| causal_daily | Pune | 2017-08-03 | up | 21.0156 | 1800.0 | 700.0 | -0.75 | none | True |
| causal_daily | Kolhapur | 2024-02-21 | up | 20.6042 | 1600.0 | 1000.0 | -2.66 | export_ban_2023_24 | True |
| causal_daily | Vashi | 2024-12-20 | down | -20.3378 | 1500.0 | 3250.0 | 2.6 | none | True |
| causal_daily | Kolhapur | 2019-08-24 | up | 18.8539 | 2300.0 | 1300.0 | 2.52 | none | True |
| causal_daily | Vashi | 2024-12-21 | down | -18.6402 | 1600.0 | 3250.0 | 1.79 | none | True |
| forecast_surprise_h30 | Kolhapur | 2023-08-25 | up | 2.1047 | 2400.0 | 1040.28 | 0.19 | none | False |
| forecast_surprise_h30 | Solapur | 2021-03-24 | down | -2.0582 | 600.0 | 2927.45 | -3.02 | none | True |
| forecast_surprise_h30 | Solapur | 2021-03-22 | down | -2.0019 | 600.0 | 2803.37 | 0.28 | none | True |
| forecast_surprise_h30 | Solapur | 2021-03-25 | down | -1.9806 | 600.0 | 2757.7 | -2.43 | none | True |
| forecast_surprise_h30 | Solapur | 2021-03-26 | down | -1.946 | 600.0 | 2685.11 | -1.92 | none | True |
| forecast_surprise_h30 | Sambhajinagar | 2022-03-28 | down | -1.9393 | 500.0 | 1896.83 | 0.38 | none | False |
| forecast_surprise_h30 | Solapur | 2024-10-26 | down | -1.9328 | 1800.0 | 5126.83 | 2.44 | none | False |
| forecast_surprise_h30 | Solapur | 2024-10-28 | down | -1.7476 | 1800.0 | 4637.6 | 4.17 | none | True |
| forecast_surprise_h7 | Solapur | 2023-05-26 | up | 2.6303 | 800.0 | 367.24 | -0.04 | none | False |
| forecast_surprise_h7 | Sambhajinagar | 2023-04-23 | down | -2.4763 | 143.0 | 436.34 | 1.37 | none | False |
| forecast_surprise_h7 | Solapur | 2023-05-25 | up | 2.4732 | 800.0 | 396.8 | -0.28 | none | False |
| forecast_surprise_h7 | Sambhajinagar | 2023-10-29 | up | 2.4455 | 6000.0 | 1993.9 | 0.9 | mep_oct_dec_2023 | True |
| forecast_surprise_h7 | Solapur | 2023-05-27 | up | 2.3182 | 900.0 | 437.54 | 0.42 | none | False |
| forecast_surprise_h7 | Pune | 2024-12-22 | down | -2.2475 | 2100.0 | 4332.35 | 1.77 | none | True |
| forecast_surprise_h7 | Pune | 2024-12-17 | down | -2.2338 | 2200.0 | 4518.66 | 1.76 | none | True |
| forecast_surprise_h7 | Vashi | 2023-12-19 | down | -2.1106 | 1900.0 | 3249.16 | 0.1 | export_ban_2023_24 | True |
| stl_residual | Vashi | 2019-12-29 | up | 15.9903 | 7240.31 | 1193.64 | -0.76 | export_ban_2019_20 | True |
| stl_residual | Pune | 2019-01-13 | down | -13.9274 | 582.04 | 2839.94 | 1.7 | none | True |
| stl_residual | Kolhapur | 2015-09-13 | up | 12.7845 | 4749.38 | 1088.51 | -2.07 | none | True |
| stl_residual | Kolhapur | 2018-12-30 | down | -12.7193 | 579.32 | 2532.74 | -0.3 | none | True |
| stl_residual | Kolhapur | 2018-12-09 | down | -12.7043 | 772.94 | 3373.39 | -0.3 | none | True |
| stl_residual | Pune | 2019-12-22 | up | 12.5933 | 5596.1 | 1372.22 | 1.46 | export_ban_2019_20 | True |
| stl_residual | Kolhapur | 2015-09-20 | up | 12.3566 | 4302.39 | 1036.07 | -0.3 | none | True |
| stl_residual | Pune | 2016-10-02 | down | -12.2173 | 600.0 | 2414.11 | 0.08 | none | True |


![STL flags](../results/figures/anomalies_stl.png)

<!-- COMMENTARY BELOW IS WRITTEN BY HAND; EVERYTHING ABOVE IS GENERATED -->


## 5. What the results say (hand-written; every number below is read from the tables above)

### 5.1 The three detectors side by side

| | Detector 1: STL residual | Detector 2: forecast surprise | Detector 3: causal daily |
|---|---|---|---|
| What it is | retrospective large-swing detector, **not a spike detector** | real time, interval-based | real time, rule-based |
| Real-time usable | no (uses later data) | yes (2020-2024 only) | yes |
| False-positive rate | **18.7%** of ordinary weeks | **4.0-4.9%** (nominal 5%) | **19.6%** of ordinary days |
| Recall, 25% spike | 5-6% | 19-28% (7 d); 8-12% (30 d) | 30-38% |
| Recall, 40% spike | 6-10% | 55-58% (7 d); 24-30% (30 d) | 54-58% |
| Recall, 60% spike | 12-22% | 80-88% (7 d); 53-56% (30 d) | 73-76% |

(Recall ranges are the 1-day and 3-day spikes; sections 3.1-3.2. Detectors 1 and 3 were tested on exactly the same spike dates; detector 2 on its own placements, in the forecast window.)

**Only the forecast-surprise detector combines a usable false-positive rate with useful recall, and only for large spikes at 7 days.** The other two flag about one ordinary week or day in five.

### 5.2 Detector 1 (STL residual): a retrospective large-swing detector, not a spike detector

- It flags **18.7% of ordinary weeks** in the real series (19.1% in the series with injected spikes): 460 flags in about 2,500 market-weeks. Its recall for 1-3 day spikes of 25-60% is only **5-22%**, little better than flagging weeks at random.
- Two things explain this, both measured here: (1) weekly averaging dilutes a short spike (a 1-day +60% quote moves the weekly mean of the log price by about 8%); (2) with `robust = True` STL gives an isolated large outlier a robustness weight of zero, yet the seasonal component then absorbs much of it: in a test case a 90% drop lasting 3 days left a residual of about -0.01 against -0.71 for non-robust STL. The robust fit also makes the residuals heavy-tailed, so `1.4826 x MAD` underestimates their spread and the nominal "3.5 sigma" flags about 5% of weeks even for pure Gaussian noise (and 19% on the real series). I did not look inside the statsmodels code for the cause.
- The non-robust diagnostic has the expected false-positive rate (0.5%) but almost no recall for these spike sizes (0-3%). Neither variant detects 1-3 day spikes at weekly resolution.
- What it picks out is runs of weeks in the big price swings (the late-2015, 2017-18 and 2019 peaks, the 2019 and mid-2020 troughs), where the price differs from the smooth STL fit by a factor of 4-6 (for example Vashi, week ending 2019-12-29: Rs 7,240 against an expected Rs 1,194). It is best used as **historical labelling of large swings**.
- STL flags per year are 36-101 in 2015-2019, 20-57 in 2020-2023 and 2 in 2024. I did not test why; a trend smoother follows the data near the end of a series, which is a possible reason.

### 5.3 Detector 2 (forecast surprise): behaves as designed

- Its false-positive rate matches the nominal 5%. Recall rises steeply with spike size. At 7 days: 19-28% for a 25% spike, 55-58% for 40%, 80-88% for 60%. At 30 days it is much lower (8-12%, 24-30%, 53-56%) because the 30-day intervals are wide (`docs/interval_report.md`).
- Down-spikes are detected far more often than up-spikes of the same percentage (7 days, 60%: 99% against 69%): a -60% move is larger on the log scale than a +60% move (log 0.4 = -0.92 against log 1.6 = +0.47), and the intervals are symmetric on the log scale.
- A 3-day spike is detected only slightly more often than a 1-day spike, because each day is judged on its own against an interval issued before the spike began.

### 5.4 Detector 3 (causal daily): declared after the first review, run once, not tuned

- It flagged **1,930 quotes** (918 up, 1,012 down), **19.6%** of the scored ordinary days (19.0% in the series with injected spikes). That is not a rare-event detector.
- Its recall is clearly higher than detector 1's on the same spikes: 30% (1-day) and 38% (3-day) for a 25% spike, 54% and 58% for 40%, 73% and 76% for 60%. By direction (durations pooled): 36% down and 31% up at 25%, 66% and 46% at 40%, 90% and 59% at 60%.
- Compared with detector 2 at 7 days, it is better at 25% (30-38% against 19-28%), about equal at 40% (54-58% against 55-58%) and worse at 60% (73-76% against 80-88%), with about **four times the false-positive rate** (19.6% against 4-5%). A recall gain at that price in false alarms is not a good trade, and the 7-day surprise detector needs a trained model but no tuning.
- Why the false-positive rate is high is not tested here. The rule compares a price with the median of the last 28 quotes but scales by the typical one-day change; in a trending series the first is far larger than the second. This reading is consistent with the data (flag rates are higher in sustained-move periods and flags cluster across markets, below) but it was not tested separately.
- Arrivals: 19.9% of its up-spikes coincide with an arrivals drop, against 19.4% of ordinary days (down-spikes: 14.8%): no excess.

### 5.5 Do flagged spikes coincide with an arrivals drop? No more than chance

The share of flagged **up**-spikes with an arrivals drop (z <= -1) is 15.5% for detector 1 (15.6% among all weeks), 23.7% for the 7-day surprise detector (20.4% among all days), 14.9% for the 30-day detector (17.9%) and 19.9% for detector 3 (19.4%). The flagged shares are close to the chance shares, with 94-918 flags per group; the largest difference (+3.3 points at 7 days) is within what chance would give. For down-spikes the shares are 12.6%, 13.9%, 19.2% and 14.8% against 15.6%, 20.4%, 17.9% and 19.4%. **There is no evidence that flagged spikes coincide with arrivals drops more often than ordinary days do.** This says nothing about causes.

### 5.6 Flags cluster across markets and in event periods

- 71-89% of flags have another market flagged by the same detector in the same week. If the other four markets were flagged independently at each detector's own rate, one would expect about 56-58% for detectors 1 and 3 and about 18% for the surprise detectors (my own back-of-envelope calculation), so the flags are clustered in time across markets. That fits market-wide moves rather than single-market data errors, but it does not prove it.
- Flags per 100 comparable dates by event period (all five periods are high-confidence; the first column is the label `none`, i.e. outside them; section 1):

| Period | Detector 1 (per 100 weeks) | Detector 2, 7 d | Detector 2, 30 d | Detector 3 (per 100 days) |
|---|---|---|---|---|
| none | 18.9 | 4.4 | 3.5 | 16.6 |
| Export ban 2019-20 (E5-E6) | 16.7 | 1.3 | 3.8 | 20.0 |
| COVID lockdown (E7-E8) | 27.8 | 11.9 | 0.0 | 37.2 |
| Export ban 2020-21 (E9-E10) | 8.6 | 3.0 | 8.1 | 26.7 |
| Oct-Dec 2023 MEP period (E11-E12) | 66.7 | 11.4 | 24.5 | 31.9 |
| Export ban 2023-24 (E12-E13) | 5.2 | 9.6 | 9.8 | 25.1 |

  Detector 3 flags more in all five event periods than outside them (20.0-37.2 against 16.6 per 100 days); detector 2 does so in three of five periods at 7 days (COVID, the MEP period, the 2023-24 ban) and in four of five at 30 days (all but COVID). The clearest case is the Oct-Dec 2023 MEP period (30-day surprise detector: 24.5 per 100 against 3.5 outside; detector 3: 31.9 against 16.6), followed by the 2023-24 ban. The 2019-20 ban figures cover only January to mid-March 2020 for detectors 1-2 (the test window), whereas detector 3 covers the whole period from 29 September 2019. The COVID lockdown has few forecasts (18 weeks, 38-86 days). Detector 1's period rates are not monotone with event size (66.7 in the MEP window, 5.2 in the 2023-24 ban); I do not interpret them. None of this shows that an event caused any flag.
- Consecutive days are often flagged together (for example Solapur, 22-26 March 2021, a realised Rs 600 against forecasts near Rs 2,800), so the number of distinct episodes is smaller than the number of flags.
- Counts per market and year are in section 1. Some are zero (Solapur 2022 for both surprise detectors); after a large-error year the trailing calibration window is wide for the following year, so few quotes fall outside it.

### 5.7 Limitations

- **Detector 1 is retrospective and at weekly resolution**: it cannot raise an alert at the time and cannot see a 1-3 day spike; as specified it flags about one ordinary week in five. I report it as a large-swing labeller, not as a spike detector.
- **Detector 3 flags about one ordinary day in five.** It was declared after the first review and run once, with no tuning; any change of its rule or threshold would be a new, separately declared method.
- **Detector 2 inherits the intervals**: its power at 30 days is low (section 3.2) and it misses up-spikes much more often than down-spikes of equal percentage size. It exists only for 2020-2024.
- **The synthetic test is a test, not data**: spikes are multiplicative, placed at random dates and independent of the series; real shocks are persistent and clustered. For the forecast-surprise test the forecasts issued from inside a spike are not re-made. The false-positive rate is measured over the whole series for detector 1 and over 2020-2024 for detectors 2 and 3.
- **No ground truth exists**, so recall on real anomalies is unknown; "flagged" does not mean "an error" or "an event".
- **Arrivals** are reported by the same source and can themselves be missing or imputed; the z-score uses a trailing median over observed days only and is undefined when fewer than 8 reference days exist.
- **Event labels** use the five high-confidence periods (E11 verified against a Deccan Herald article dated 28 Oct 2023). Events E1-E4 are medium or low confidence in `docs/eda_findings.md` and are not used here.
- The thresholds (3.5, 95%, 4) were declared in advance and were not tuned on these results.
