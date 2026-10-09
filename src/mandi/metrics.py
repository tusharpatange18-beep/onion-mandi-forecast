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
    y, f = np.exp(y_log), np.exp(f_log)
    e = y - f
    return {
        "n": int(len(y)),
        "MAE": float(np.mean(np.abs(e))),
        "RMSE": float(np.sqrt(np.mean(e ** 2))),
        "MASE": float(np.mean(np.abs(e)) / mase_scale),
        "sMAPE": float(100 * np.mean(2 * np.abs(e) / (np.abs(y) + np.abs(f)))),
        "MAE_log": float(np.mean(np.abs(y_log - f_log))),
    }


def naive_scale(y_log_target: np.ndarray, y_log_origin: np.ndarray) -> float:
    """In-sample naive error: mean absolute rupee error of 'last observed quote' over the training triples."""
    return float(np.mean(np.abs(np.exp(y_log_target) - np.exp(y_log_origin))))


def diebold_mariano(loss_base: np.ndarray, loss_model: np.ndarray, h: int) -> dict[str, float]:
    """Diebold-Mariano test with the Harvey-Leybourne-Newbold small-sample correction.

    `loss_*` are per-forecast losses (here absolute rupee error) ordered by forecast origin. The loss
    differential d = loss_base - loss_model is positive when the model is better. The long-run variance uses
    autocovariances up to lag h - 1 (the forecasts overlap, so errors are serially correlated); the lags are counted in
    origins, which are consecutive observed days, so they only approximate h calendar days. The corrected statistic is
    compared with a Student t distribution with n - 1 degrees of freedom (two-sided).
    """
    d = np.asarray(loss_base, dtype=float) - np.asarray(loss_model, dtype=float)
    n = len(d)
    if n < 20:
        return {"DM": np.nan, "p_value": np.nan, "mean_loss_diff": float(np.mean(d)) if n else np.nan, "n": n}
    dbar = d.mean()
    dc = d - dbar
    var = np.dot(dc, dc) / n
    for k in range(1, h):
        var += 2 * np.dot(dc[k:], dc[:-k]) / n
    if var <= 0:
        return {"DM": np.nan, "p_value": np.nan, "mean_loss_diff": float(dbar), "n": n}
    dm = dbar / np.sqrt(var / n)
    corr = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_c = dm * corr
    p = 2 * (1 - stats.t.cdf(abs(dm_c), df=n - 1))
    return {"DM": float(dm_c), "p_value": float(p), "mean_loss_diff": float(dbar), "n": n}


def metrics_table(tab: pd.DataFrame, models: list[str], scales: dict[tuple[str, int], float]) -> pd.DataFrame:
    """Metrics per market, horizon and model. `tab` has y_log, market, h and one log-forecast column per model."""
    rows = []
    for (m, h), g in tab.groupby(["market", "h"]):
        for mod in models:
            r = point_metrics(g["y_log"].to_numpy(), g[mod].to_numpy(), scales[(m, h)])
            rows.append({"market": m, "h": h, "model": mod, **r})
    return pd.DataFrame(rows)
