"""Forecast accuracy metrics and the Diebold-Mariano test.

All point forecasts are made on the log scale; rupee forecasts are exp(log forecast). Metrics are computed
per market and horizon on the rupee scale (MAE, RMSE, MASE, sMAPE) plus MAE on the log scale.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def point_metrics(y_log: np.ndarray, f_log: np.ndarray, mase_scale: float) -> dict[str, float]:
    """MAE, RMSE, MASE, sMAPE (rupee scale) and MAE on the log scale for one set of forecasts."""
    # Turn the log prices back into rupee prices.
    actual_price = np.exp(y_log)
    forecast_price = np.exp(f_log)

    # Error = actual minus forecast (positive means we forecast too low).
    error = actual_price - forecast_price
    absolute_error = np.abs(error)

    # MAE: the average size of the error in rupees.
    mae = np.mean(absolute_error)

    # RMSE: square the errors, average them, then take the square root (punishes big errors more).
    squared_error = error ** 2
    rmse = np.sqrt(np.mean(squared_error))

    # MASE: MAE divided by the error of the simple "last price" forecast (below 1 means we beat it).
    mase = mae / mase_scale

    # sMAPE: percentage error that treats over- and under-forecasts the same way.
    size_of_both = np.abs(actual_price) + np.abs(forecast_price)
    smape = 100 * np.mean(2 * absolute_error / size_of_both)

    # MAE on the log scale (roughly the average percentage error).
    mae_log = np.mean(np.abs(y_log - f_log))

    # Put every number in a dictionary with plain Python types.
    result = {
        "n": int(len(actual_price)),
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MASE": float(mase),
        "sMAPE": float(smape),
        "MAE_log": float(mae_log),
    }
    return result


def naive_scale(y_log_target: np.ndarray, y_log_origin: np.ndarray) -> float:
    """In-sample naive error: mean absolute rupee error of 'last observed quote' over the training triples."""
    # Rupee price on the target day and on the day the forecast was made.
    target_price = np.exp(y_log_target)
    origin_price = np.exp(y_log_origin)

    # The naive forecast says "the price stays the same", so its error is the price change.
    naive_error = np.abs(target_price - origin_price)

    # Average size of that error.
    return float(np.mean(naive_error))


def diebold_mariano(loss_base: np.ndarray, loss_model: np.ndarray, h: int) -> dict[str, float]:
    """Diebold-Mariano test with the Harvey-Leybourne-Newbold small-sample correction.

    `loss_*` are per-forecast losses (here absolute rupee error) ordered by forecast origin. The loss
    differential d = loss_base - loss_model is positive when the model is better. The long-run variance uses
    autocovariances up to lag h - 1 (the forecasts overlap, so errors are serially correlated); the lags are counted in
    origins, which are consecutive observed days, so they only approximate h calendar days. The corrected statistic is
    compared with a Student t distribution with n - 1 degrees of freedom (two-sided).
    """
    # Convert both inputs to float arrays.
    base_loss = np.asarray(loss_base, dtype=float)
    model_loss = np.asarray(loss_model, dtype=float)

    # Loss difference for each forecast: positive when the model did better than the baseline.
    loss_diff = base_loss - model_loss
    n = len(loss_diff)

    # With fewer than 20 forecasts the test is not reliable, so return "no result".
    if n < 20:
        if n > 0:
            mean_diff_small = float(np.mean(loss_diff))
        else:
            mean_diff_small = np.nan
        return {"DM": np.nan, "p_value": np.nan, "mean_loss_diff": mean_diff_small, "n": n}

    # Average loss difference.
    mean_diff = loss_diff.mean()

    # Centre the differences around zero.
    centred = loss_diff - mean_diff

    # Variance of the differences (lag 0).
    variance = np.dot(centred, centred) / n

    # Add the covariance with earlier values (lags 1 .. h-1), because overlapping forecasts are correlated.
    for lag in range(1, h):
        later_values = centred[lag:]
        earlier_values = centred[:-lag]
        variance = variance + 2 * np.dot(later_values, earlier_values) / n

    # A variance of zero or less means the test cannot be computed.
    if variance <= 0:
        return {"DM": np.nan, "p_value": np.nan, "mean_loss_diff": float(mean_diff), "n": n}

    # The raw Diebold-Mariano statistic.
    dm_stat = mean_diff / np.sqrt(variance / n)

    # Harvey-Leybourne-Newbold correction for small samples.
    correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_corrected = dm_stat * correction

    # Two-sided p-value from the Student t distribution.
    p_value = 2 * (1 - stats.t.cdf(abs(dm_corrected), df=n - 1))

    return {"DM": float(dm_corrected), "p_value": float(p_value), "mean_loss_diff": float(mean_diff), "n": n}


def metrics_table(tab: pd.DataFrame, models: list[str], scales: dict[tuple[str, int], float]) -> pd.DataFrame:
    """Metrics per market, horizon and model. `tab` has y_log, market, h and one log-forecast column per model."""
    rows = []

    # Go through each (market, horizon) group of forecasts.
    for group_key, group in tab.groupby(["market", "h"]):
        market = group_key[0]
        horizon = group_key[1]

        # Actual log prices for this group.
        actual_log = group["y_log"].to_numpy()

        # Compute the metrics for every model.
        for model_name in models:
            forecast_log = group[model_name].to_numpy()
            scale = scales[(market, horizon)]
            model_metrics = point_metrics(actual_log, forecast_log, scale)

            # Build one row: who/what first, then the metric values.
            row = {"market": market, "h": horizon, "model": model_name}
            for metric_name in model_metrics:
                row[metric_name] = model_metrics[metric_name]
            rows.append(row)

    return pd.DataFrame(rows)
