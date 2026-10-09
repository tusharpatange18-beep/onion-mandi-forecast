"""Phase 3: statistical and ML forecasting models.

ETS and SARIMA use statsmodels *state-space* implementations because they accept missing days (NaN) without
filling them. Parameters are estimated on data before the refit date; the model is then run (filtered) with those
fixed parameters over the following days, and the h-step forecast at every origin is read from the filtered state,
which uses data up to that origin only.
"""
from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- state-space models
def _build(kind: str, spec: dict[str, Any], endog: pd.Series):
    """Create the (unfitted) state-space model for `kind` in {'ets', 'sarima'}."""
    if kind == "ets":
        from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing

        return ExponentialSmoothing(endog, trend=spec["trend"], damped_trend=spec["damped"], initialization_method="estimated")
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    return SARIMAX(endog, order=tuple(spec["order"]), seasonal_order=tuple(spec["seasonal_order"]),
                   enforce_stationarity=False, enforce_invertibility=False)


def h_step_forecasts(res: Any, horizons: list[int]) -> dict[int, np.ndarray]:
    """h-step-ahead mean forecast made at every time point, from the filtered state a_{t|t}.

    For a time-invariant model without intercepts the forecast of y_{t+h} is Z T^h a_{t|t}.
    """
    fr = res.filter_results
    T, Z = fr.transition[:, :, 0], fr.design[:, :, 0]
    if np.abs(fr.state_intercept).max() > 0 or np.abs(fr.obs_intercept).max() > 0:
        raise ValueError("h_step_forecasts assumes no intercepts")
    a = fr.filtered_state
    return {h: (Z @ np.linalg.matrix_power(T, h) @ a).ravel() for h in horizons}


def _start_params(kind: str, spec: dict[str, Any], y_train: pd.Series) -> np.ndarray | None:
    """Explicit start values for ETS. The library's own heuristic returns NaN for the initial trend when one of the first
    two observations is missing (Kolhapur has a 71-day gap right after its first quote), which silently ruins the fit."""
    if kind != "ets":
        return None
    names = _build(kind, spec, y_train).param_names
    defaults = {"smoothing_level": 0.3, "smoothing_trend": 0.05, "damping_trend": 0.95,
                "initial_level": float(y_train.dropna().iloc[0]), "initial_trend": 0.0}
    return np.array([defaults[n] for n in names])


def fit_state_space(kind: str, spec: dict[str, Any], y_train: pd.Series, start_params: np.ndarray | None = None,
                    maxiter: int = 100, return_info: bool = False):
    """Estimate parameters on `y_train` (log price on a daily index, NaN on missing days).

    Raises if any estimated parameter is not finite, so a broken fit can never pass silently. With
    `return_info=True` returns (params, converged).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = _build(kind, spec, y_train)
        sp = start_params if start_params is not None and np.all(np.isfinite(start_params)) else _start_params(kind, spec, y_train)
        res = mod.fit(start_params=sp, disp=False, maxiter=maxiter)
    params = np.asarray(res.params)
    if not np.all(np.isfinite(params)):
        raise ValueError(f"non-finite parameters from {kind} {spec}: {params}")
    converged = bool(res.mle_retvals.get("converged", True)) if getattr(res, "mle_retvals", None) else True
    return (params, converged) if return_info else params


def state_space_forecasts(kind: str, spec: dict[str, Any], y_run: pd.Series, params: np.ndarray, horizons: list[int]) -> dict[int, np.ndarray]:
    """Run the model with fixed `params` over `y_run` and return h-step forecasts made at every day of `y_run`."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = _build(kind, spec, y_run)
        res = mod.filter(params)
    return h_step_forecasts(res, horizons)


# --------------------------------------------------------------------------- Prophet
def prophet_forecast(y_train: pd.Series, target_dates: pd.DatetimeIndex, spec: dict[str, Any], seed: int = 0) -> np.ndarray:
    """Fit Prophet on observed log prices (days with no quote are simply absent) and predict `target_dates`.

    Weekly seasonality is off (EDA: weekday price effects are within +-1.8%); yearly seasonality is on.
    """
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    logging.getLogger("prophet").setLevel(logging.ERROR)
    from prophet import Prophet

    df = y_train.dropna().rename("y").rename_axis("ds").reset_index()
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
                changepoint_prior_scale=spec["changepoint_prior_scale"], uncertainty_samples=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(df)
        pred = m.predict(pd.DataFrame({"ds": pd.DatetimeIndex(target_dates)}))
    return pred["yhat"].to_numpy()


# --------------------------------------------------------------------------- LightGBM
def lgbm_fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str], params: dict[str, Any], seed: int) -> np.ndarray:
    """Train on (features -> log change) and predict the log change for `test`. Missing feature values stay NaN."""
    import lightgbm as lgb

    p = {
        "objective": params["objective"], "learning_rate": params["learning_rate"], "num_leaves": params["num_leaves"],
        "min_data_in_leaf": params["min_child_samples"], "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
        "lambda_l2": 1.0, "seed": seed, "deterministic": True, "force_row_wise": True, "num_threads": 4, "verbose": -1,
    }
    if params["objective"] == "huber":
        p["alpha"] = params.get("huber_alpha", 0.15)
    ds = lgb.Dataset(train[features], label=train["y_change"], categorical_feature=["market_idx"], free_raw_data=False)
    booster = lgb.train(p, ds, num_boost_round=params["n_estimators"])
    return booster.predict(test[features])
