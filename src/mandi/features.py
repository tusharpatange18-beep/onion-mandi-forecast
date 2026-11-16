"""Phase 3: forecast origins, targets, baselines and features on a calendar-day grid.

Definitions (see configs/config.yaml):
* A forecast origin is a calendar day on which a market has an observed quote.
* The target for horizon h is the quote at origin + h calendar days; it is scored only if it exists.
* Every feature at an origin uses data on or before that day only (arrivals: strictly before).
  "Lag k" means "the last observed quote at or before origin - k", never "the value exactly k days ago".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

SEASONAL_OFFSETS = (0, -1, 1, -2, 2, -3, 3)  # nearest first; on a tie the earlier date wins
FIXED_HOLIDAYS = {(1, 26), (5, 1), (8, 15)}  # Republic Day, Maharashtra Day, Independence Day (EDA section 1)
NAN = float("nan")  # short name for "missing value"


@dataclass
class Arrays:
    """Market x day matrices on a common calendar grid. Missing days are NaN; nothing is filled."""

    days: pd.DatetimeIndex
    markets: list[str]
    short: list[str]
    logp: np.ndarray       # log modal price
    larr: np.ndarray       # log arrivals (tonnes)
    stale: np.ndarray      # is_stale flag (False where no quote)


def short_name(market_key: str) -> str:
    """Short market label used in tables."""
    # The market name is the last part of the key, e.g. "Maharashtra_Pune" -> "Pune".
    parts = market_key.split("_")
    name = parts[-1]

    # "Chattrapati Sambhajinagar" is too long for tables.
    if name.startswith("Chattrapati"):
        return "Sambhajinagar"

    # "Vashi New Mumbai" -> "Vashi".
    return name.replace(" New Mumbai", "")


def _log_of_positive(values: np.ndarray) -> np.ndarray:
    """Natural log of each value; zero, negative or missing values become NaN."""
    result = np.full(len(values), NAN)  # start with all missing
    is_positive = values > 0  # only positive numbers have a log
    result[is_positive] = np.log(values[is_positive])
    return result


def build_arrays(panel: pd.DataFrame, start: str, end: str, markets: list[str] | None = None) -> Arrays:
    """Arrange the Phase 1 panel into market x day arrays between `start` and `end` (inclusive)."""
    # Every calendar day between start and end.
    days = pd.date_range(start, end, freq="D")

    # If no market list is given, use the markets in the order they first appear in the panel.
    if not markets:
        markets = []
        for market_key in panel["market_key"]:
            if market_key not in markets:
                markets.append(market_key)

    # Empty matrices: one row per market, one column per day.
    n_markets = len(markets)
    n_days = len(days)
    logp = np.full((n_markets, n_days), NAN)
    larr = np.full((n_markets, n_days), NAN)
    stale = np.zeros((n_markets, n_days), dtype=bool)

    for row, market_key in enumerate(markets):
        # This market's rows, placed on the full calendar (days without data become NaN).
        market_rows = panel[panel["market_key"] == market_key]
        market_rows = market_rows.set_index("date").reindex(days)

        # Log of the modal price.
        price = market_rows["modal_price"].to_numpy(dtype=float)
        logp[row] = _log_of_positive(price)

        # Log of the arrivals.
        arrivals = market_rows["arrivals_tonnes"].to_numpy(dtype=float)
        larr[row] = _log_of_positive(arrivals)

        # Stale flag: only True on days that actually have a price.
        stale_flag = market_rows["is_stale"].fillna(False).to_numpy(dtype=bool)
        has_price = ~np.isnan(logp[row])
        stale[row] = stale_flag & has_price

    # Short names for tables.
    short_names = []
    for market_key in markets:
        short_names.append(short_name(market_key))

    return Arrays(days, markets, short_names, logp, larr, stale)


def closure_weekdays(panel: pd.DataFrame, train_end: str, threshold: float = 0.9) -> dict[str, int | None]:
    """Weekday (0=Mon) on which a market is almost always closed, learned from data up to `train_end` only."""
    # Only use the training period.
    train_rows = panel[panel["date"] <= train_end]

    closed_day = {}
    for market_key, market_rows in train_rows.groupby("market_key"):
        # Share of missing days for each weekday (0 = Monday ... 6 = Sunday).
        weekday = market_rows["date"].dt.dayofweek
        missing_share = market_rows.groupby(weekday)["was_missing"].mean()

        # If one weekday is missing more than `threshold` of the time, the market is closed that day.
        if missing_share.max() > threshold:
            closed_day[market_key] = int(missing_share.idxmax())
        else:
            closed_day[market_key] = None
    return closed_day


def last_observed(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each market and day j: value and index of the last non-NaN entry at or before j (NaN / -1 if none)."""
    n_markets = x.shape[0]
    n_days = x.shape[1]
    last_value = np.full((n_markets, n_days), NAN)
    last_index = np.full((n_markets, n_days), -1, dtype=int)

    for m in range(n_markets):
        # Walk forward through the days, remembering the latest observed value.
        current_value = NAN
        current_index = -1
        for day in range(n_days):
            if not np.isnan(x[m, day]):
                current_value = x[m, day]
                current_index = day
            last_value[m, day] = current_value
            last_index[m, day] = current_index
    return last_value, last_index


def _cumulative_sums(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Running totals of value, value squared and count (NaN skipped), with a 0 in front.

    With these, the sum over days lo..hi is just total[hi + 1] - total[lo], which makes window sums fast.
    """
    n_markets = x.shape[0]
    n_days = x.shape[1]
    sum_values = np.zeros((n_markets, n_days + 1))
    sum_squares = np.zeros((n_markets, n_days + 1))
    count = np.zeros((n_markets, n_days + 1))

    for m in range(n_markets):
        running_sum = 0.0
        running_squares = 0.0
        running_count = 0.0
        for day in range(n_days):
            value = x[m, day]
            # Missing days add nothing.
            if not np.isnan(value):
                running_sum = running_sum + value
                running_squares = running_squares + value * value
                running_count = running_count + 1.0
            # Position day + 1, because position 0 is the leading zero.
            sum_values[m, day + 1] = running_sum
            sum_squares[m, day + 1] = running_squares
            count[m, day + 1] = running_count
    return sum_values, sum_squares, count


def _window_sum(cumulative: np.ndarray, m: int, lo: int, hi: int) -> float:
    """Sum over days lo..hi (inclusive) for market m, using a running-total array. lo below 0 counts as 0."""
    if lo < 0:
        lo = 0
    return cumulative[m, hi + 1] - cumulative[m, lo]


def _mean_ignoring_missing(values: list[float]) -> float:
    """Average of the values, skipping NaN. NaN if every value is missing."""
    total = 0.0
    count = 0
    for value in values:
        if not np.isnan(value):
            total = total + value
            count = count + 1
    if count == 0:
        return NAN
    return total / count


def _count_present(values: list[float]) -> int:
    """How many values are not NaN."""
    count = 0
    for value in values:
        if not np.isnan(value):
            count = count + 1
    return count


def build_origin_table(a: Arrays, h: int, closure: dict[str, int | None], params: dict[str, Any]) -> pd.DataFrame:
    """One row per (market, origin) with baselines, features and the realised target for horizon `h`.

    Every row is a forecast origin (an observed quote). `y` is NaN when the target quote does not exist; such
    rows are kept so the caller can decide, but they are never scored. Features use data at or before the
    origin only (arrivals strictly before it).
    """
    n_markets = a.logp.shape[0]
    n_days = a.logp.shape[1]

    # For every day: the latest observed price / arrivals at or before that day, and on which day it was seen.
    last_price, last_price_day = last_observed(a.logp)
    last_arrivals, _ = last_observed(a.larr)

    # Running totals of the log price (used for the 30-day mean and the 14-day quote count).
    price_sum, _, price_count = _cumulative_sums(a.logp)

    # Day-to-day change between consecutive observed quotes, placed on the day of the later quote.
    price_change = np.full((n_markets, n_days), NAN)
    for m in range(n_markets):
        for day in range(1, n_days):
            if not np.isnan(a.logp[m, day]):
                price_change[m, day] = a.logp[m, day] - last_price[m, day - 1]
    change_sum, change_squares, change_count = _cumulative_sums(price_change)

    # Running totals of the log arrivals.
    arrivals_sum, _, arrivals_count = _cumulative_sums(a.larr)

    # Settings from the config file.
    lag_list = params["lags_days"]
    cross_lag_list = params["cross_market_lags_days"]
    max_stale_days = params["max_stale_days_other_market"]
    window = params["window_days"]
    seasonal_period = params["seasonal_period_days"]
    seasonal_tolerance = params["seasonal_tolerance_days"]

    market_tables = []
    for m in range(n_markets):
        # Forecast origins = days on which this market has a price.
        origin_days = np.flatnonzero(~np.isnan(a.logp[m]))

        # One list per column; each origin adds one value to every list.
        columns: dict[str, list] = {"origin_idx": [], "p0": []}
        for k in lag_list:
            columns[f"chg_{k}"] = []
        for name in ["gap_prev", "dev3", "base_mean3", "dev30", "vol30", "n_obs_14"]:
            columns[name] = []
        for k in cross_lag_list:
            columns[f"xm_chg_{k}"] = []
        for name in ["rel_level", "arr_prev", "arr_mean14", "arr_dev", "y_log", "y_change", "base_seasonal", "stale_target"]:
            columns[name] = []

        for origin in origin_days:
            origin = int(origin)
            price_now = a.logp[m, origin]  # log price on the origin day
            columns["origin_idx"].append(origin)
            columns["p0"].append(price_now)

            # --- Lagged changes: price now minus the last price at or before (origin - k).
            for k in lag_list:
                lag_day = origin - k
                if lag_day >= 0:
                    old_price = last_price[m, lag_day]
                else:
                    old_price = NAN
                columns[f"chg_{k}"].append(price_now - old_price)

            # --- The two quotes before this one.
            if origin >= 1:
                prev1_day = last_price_day[m, origin - 1]
            else:
                prev1_day = -1
            if prev1_day >= 1:
                prev2_day = last_price_day[m, prev1_day - 1]
            else:
                prev2_day = -1

            # Days since the previous quote.
            if prev1_day >= 0:
                columns["gap_prev"].append(float(origin - prev1_day))
            else:
                columns["gap_prev"].append(NAN)

            # Prices of the previous two quotes (NaN if they do not exist).
            if prev1_day >= 0:
                prev1_price = a.logp[m, prev1_day]
            else:
                prev1_price = NAN
            if prev2_day >= 0:
                prev2_price = a.logp[m, prev2_day]
            else:
                prev2_price = NAN

            # dev3: price now minus the mean of the last three log prices (NaN unless three quotes exist).
            mean_last3_log = (price_now + prev1_price + prev2_price) / 3
            columns["dev3"].append(price_now - mean_last3_log)

            # Baseline (b): average of the last 3 prices in rupees, then back to log.
            mean_last3_rupees = (np.exp(price_now) + np.exp(prev1_price) + np.exp(prev2_price)) / 3
            columns["base_mean3"].append(np.log(mean_last3_rupees))

            # --- 30-day window ending on the origin day.
            window_start = origin - (window - 1)

            # dev30: price now minus the 30-day mean (needs at least 5 quotes).
            n_quotes_30 = _window_sum(price_count, m, window_start, origin)
            if n_quotes_30 >= 5:
                mean_30 = _window_sum(price_sum, m, window_start, origin) / max(n_quotes_30, 1)
                columns["dev30"].append(price_now - mean_30)
            else:
                columns["dev30"].append(NAN)

            # vol30: standard deviation of the quote-to-quote changes (needs at least 4 changes).
            n_changes = _window_sum(change_count, m, window_start, origin)
            mean_change = _window_sum(change_sum, m, window_start, origin) / max(n_changes, 1)
            mean_square = _window_sum(change_squares, m, window_start, origin) / max(n_changes, 1)
            variance = mean_square - mean_change ** 2
            if n_changes >= 4:
                columns["vol30"].append(np.sqrt(max(variance, 0)))
            else:
                columns["vol30"].append(NAN)

            # Number of quotes in the last 14 days.
            columns["n_obs_14"].append(_window_sum(price_count, m, origin - 13, origin))

            # --- Other markets: average change over k days (ignore stale quotes).
            for k in cross_lag_list:
                other_changes = []
                for other in range(n_markets):
                    if other == m:
                        continue  # skip our own market
                    # Is the other market's latest quote recent enough today?
                    seen_day_now = last_price_day[other, origin]
                    fresh_now = seen_day_now >= 0 and origin - seen_day_now <= max_stale_days
                    # Was its quote recent enough k days ago?
                    then_day = origin - k
                    if then_day >= 0:
                        seen_day_then = last_price_day[other, then_day]
                        fresh_then = seen_day_then >= 0 and then_day - seen_day_then <= max_stale_days
                    else:
                        fresh_then = False
                    # Only use the change if both quotes are fresh.
                    if fresh_now and fresh_then:
                        other_changes.append(last_price[other, origin] - last_price[other, then_day])
                    else:
                        other_changes.append(NAN)
                # Need at least two markets with a value.
                if _count_present(other_changes) >= 2:
                    columns[f"xm_chg_{k}"].append(_mean_ignoring_missing(other_changes))
                else:
                    columns[f"xm_chg_{k}"].append(NAN)

            # rel_level: our price minus the average of the other markets' fresh prices.
            other_levels = []
            for other in range(n_markets):
                if other == m:
                    continue
                seen_day_now = last_price_day[other, origin]
                fresh_now = seen_day_now >= 0 and origin - seen_day_now <= max_stale_days
                if fresh_now:
                    other_levels.append(last_price[other, origin])
                else:
                    other_levels.append(NAN)
            if _count_present(other_levels) >= 2:
                columns["rel_level"].append(price_now - _mean_ignoring_missing(other_levels))
            else:
                columns["rel_level"].append(NAN)

            # --- Arrivals: only use days strictly before the origin.
            if origin >= 1:
                arrivals_prev = last_arrivals[m, origin - 1]
            else:
                arrivals_prev = NAN
            columns["arr_prev"].append(arrivals_prev)

            # Mean arrivals over the 14 days before the origin (needs at least 3 days).
            n_arrival_days = _window_sum(arrivals_count, m, origin - 14, origin - 1)
            if n_arrival_days >= 3 and origin >= 1:
                arrivals_mean14 = _window_sum(arrivals_sum, m, origin - 14, origin - 1) / max(n_arrival_days, 1)
            else:
                arrivals_mean14 = NAN
            columns["arr_mean14"].append(arrivals_mean14)
            columns["arr_dev"].append(arrivals_prev - arrivals_mean14)

            # --- Target: the price h days after the origin (NaN if past the end or not observed).
            target_day = origin + h
            target_in_range = target_day < n_days
            if target_in_range:
                target_price = a.logp[m, target_day]
                columns["stale_target"].append(bool(a.stale[m, target_day]))
            else:
                target_price = NAN
                columns["stale_target"].append(False)
            columns["y_log"].append(target_price)
            columns["y_change"].append(target_price - price_now)

            # --- Baseline (c): the nearest observed price around the same date one year earlier.
            seasonal_price = NAN
            reference_day = target_day - seasonal_period
            for offset in SEASONAL_OFFSETS:
                if abs(offset) > seasonal_tolerance:
                    continue  # too far from the reference date
                check_day = reference_day + offset
                if check_day < 0 or check_day >= n_days:
                    continue  # outside the data
                if not np.isnan(a.logp[m, check_day]):
                    seasonal_price = a.logp[m, check_day]
                    break  # the first (nearest) observed price wins
            columns["base_seasonal"].append(seasonal_price)

        # --- Turn the lists into a table (same column order as before).
        table = pd.DataFrame()
        table["origin_idx"] = np.array(columns["origin_idx"], dtype=np.int64)
        table["p0"] = np.array(columns["p0"], dtype=float)
        for k in lag_list:
            table[f"chg_{k}"] = np.array(columns[f"chg_{k}"], dtype=float)
        for name in ["gap_prev", "dev3", "base_mean3", "dev30", "vol30", "n_obs_14"]:
            table[name] = np.array(columns[name], dtype=float)
        for k in cross_lag_list:
            table[f"xm_chg_{k}"] = np.array(columns[f"xm_chg_{k}"], dtype=float)
        for name in ["rel_level", "arr_prev", "arr_mean14", "arr_dev"]:
            table[name] = np.array(columns[name], dtype=float)
        table["market_idx"] = m
        table["h"] = h
        table["y_log"] = np.array(columns["y_log"], dtype=float)
        table["y_change"] = np.array(columns["y_change"], dtype=float)
        table["base_seasonal"] = np.array(columns["base_seasonal"], dtype=float)
        table["base_naive"] = table["p0"]  # baseline (a): tomorrow equals today

        # --- Calendar features of the target date and the origin date.
        origin_dates = a.days[origin_days]
        target_dates = origin_dates + pd.Timedelta(days=h)
        table["target_month"] = target_dates.month.to_numpy()
        table["target_woy"] = target_dates.isocalendar().week.to_numpy().astype(int)
        table["target_dow"] = target_dates.dayofweek.to_numpy()
        table["origin_dow"] = origin_dates.dayofweek.to_numpy()

        # Is the market expected to be closed on the target day (weekly closing day or a fixed holiday)?
        closed_weekday = closure.get(a.markets[m])
        expected_closed = []
        for target_date in target_dates:
            is_holiday = (target_date.month, target_date.day) in FIXED_HOLIDAYS
            is_closing_day = closed_weekday is not None and target_date.dayofweek == closed_weekday
            if is_holiday or is_closing_day:
                expected_closed.append(1)
            else:
                expected_closed.append(0)
        table["target_expected_closed"] = np.array(expected_closed, dtype=int)

        # Stale flags of the origin and target quotes.
        table["stale_origin"] = a.stale[m, origin_days]
        table["stale_target"] = np.array(columns["stale_target"], dtype=bool)
        market_tables.append(table)

    # Stack all markets into one table.
    result = pd.concat(market_tables, ignore_index=True)

    # Readable market name and real dates.
    market_names = []
    for market_index in result["market_idx"]:
        market_names.append(a.short[market_index])
    result["market"] = market_names
    result["origin"] = a.days[result["origin_idx"].to_numpy()]
    result["target_date"] = result["origin"] + pd.Timedelta(days=h)
    return result


FEATURE_COLUMNS = [
    "chg_1", "chg_3", "chg_7", "chg_14", "chg_28", "dev3", "dev30", "vol30", "gap_prev", "n_obs_14",
    "xm_chg_7", "xm_chg_28", "rel_level", "arr_prev", "arr_mean14", "arr_dev",
    "target_month", "target_woy", "target_dow", "origin_dow", "target_expected_closed", "market_idx",
]
