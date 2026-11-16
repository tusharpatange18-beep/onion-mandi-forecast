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
        # Exponential smoothing (ETS) with an optional (damped) trend.
        from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing

        return ExponentialSmoothing(endog, trend=spec["trend"], damped_trend=spec["damped"], initialization_method="estimated")

    # Otherwise build a SARIMA model with the given (p, d, q) and seasonal orders.
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    order = tuple(spec["order"])
    seasonal_order = tuple(spec["seasonal_order"])
    return SARIMAX(endog, order=order, seasonal_order=seasonal_order,
                   enforce_stationarity=False, enforce_invertibility=False)


def h_step_forecasts(res: Any, horizons: list[int]) -> dict[int, np.ndarray]:
    """h-step-ahead mean forecast made at every time point, from the filtered state a_{t|t}.

    For a time-invariant model without intercepts the forecast of y_{t+h} is Z T^h a_{t|t}.
    """
    filter_results = res.filter_results

    # T moves the state one day forward; Z turns the state into a price.
    transition = filter_results.transition[:, :, 0]
    design = filter_results.design[:, :, 0]

    # The formula below only works when the model has no intercept terms.
    has_state_intercept = np.abs(filter_results.state_intercept).max() > 0
    has_obs_intercept = np.abs(filter_results.obs_intercept).max() > 0
    if has_state_intercept or has_obs_intercept:
        raise ValueError("h_step_forecasts assumes no intercepts")

    # The model's best estimate of the state on each day (using data up to that day only).
    filtered_state = filter_results.filtered_state

    forecasts = {}
    for h in horizons:
        # Move the state h days forward (T to the power h), then read off the price (Z).
        transition_h = np.linalg.matrix_power(transition, h)
        forecast = design @ transition_h @ filtered_state
        forecasts[h] = forecast.ravel()  # flatten to a 1-D array (one value per day)
    return forecasts


def _start_params(kind: str, spec: dict[str, Any], y_train: pd.Series) -> np.ndarray | None:
    """Explicit start values for ETS. The library's own heuristic returns NaN for the initial trend when one of the first
    two observations is missing (Kolhapur has a 71-day gap right after its first quote), which silently ruins the fit."""
    # SARIMA uses the library's own start values.
    if kind != "ets":
        return None

    # The names of the parameters this ETS model needs, in the model's order.
    param_names = _build(kind, spec, y_train).param_names

    # A sensible start value for every possible ETS parameter.
    first_price = float(y_train.dropna().iloc[0])
    defaults = {
        "smoothing_level": 0.3,
        "smoothing_trend": 0.05,
        "damping_trend": 0.95,
        "initial_level": first_price,
        "initial_trend": 0.0,
    }

    # Pick the start values in the order the model expects.
    start_values = []
    for name in param_names:
        start_values.append(defaults[name])
    return np.array(start_values)


def fit_state_space(kind: str, spec: dict[str, Any], y_train: pd.Series, start_params: np.ndarray | None = None,
                    maxiter: int = 100, return_info: bool = False):
    """Estimate parameters on `y_train` (log price on a daily index, NaN on missing days).

    Raises if any estimated parameter is not finite, so a broken fit can never pass silently. With
    `return_info=True` returns (params, converged).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # statsmodels prints many harmless warnings
        model = _build(kind, spec, y_train)

        # Use the given start values if they are all real numbers; otherwise use our defaults.
        if start_params is not None and np.all(np.isfinite(start_params)):
            first_guess = start_params
        else:
            first_guess = _start_params(kind, spec, y_train)

        # Estimate the parameters.
        result = model.fit(start_params=first_guess, disp=False, maxiter=maxiter)

    # Stop if any parameter is NaN or infinite.
    params = np.asarray(result.params)
    if not np.all(np.isfinite(params)):
        raise ValueError(f"non-finite parameters from {kind} {spec}: {params}")

    # Did the optimiser report that it converged? (Assume yes if it gives no information.)
    converged = True
    optimiser_info = getattr(result, "mle_retvals", None)
    if optimiser_info:
        converged = bool(optimiser_info.get("converged", True))

    if return_info:
        return params, converged
    return params


def state_space_forecasts(kind: str, spec: dict[str, Any], y_run: pd.Series, params: np.ndarray, horizons: list[int]) -> dict[int, np.ndarray]:
    """Run the model with fixed `params` over `y_run` and return h-step forecasts made at every day of `y_run`."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = _build(kind, spec, y_run)
        # Run the model with the already-estimated parameters (no new fitting).
        result = model.filter(params)
    return h_step_forecasts(result, horizons)


# --------------------------------------------------------------------------- Prophet
def prophet_forecast(y_train: pd.Series, target_dates: pd.DatetimeIndex, spec: dict[str, Any], seed: int = 0) -> np.ndarray:
    """Fit Prophet on observed log prices (days with no quote are simply absent) and predict `target_dates`.

    Weekly seasonality is off (EDA: weekday price effects are within +-1.8%); yearly seasonality is on.
    """
    # Hide Prophet's progress messages.
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    logging.getLogger("prophet").setLevel(logging.ERROR)
    from prophet import Prophet

    # Prophet wants a table with columns "ds" (date) and "y" (value); drop days without a price.
    observed = y_train.dropna()
    train_table = pd.DataFrame({"ds": observed.index, "y": observed.to_numpy()})

    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
                    changepoint_prior_scale=spec["changepoint_prior_scale"], uncertainty_samples=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(train_table)
        # Predict the requested dates.
        future_table = pd.DataFrame({"ds": pd.DatetimeIndex(target_dates)})
        prediction = model.predict(future_table)
    return prediction["yhat"].to_numpy()


# --------------------------------------------------------------------------- LightGBM
def lgbm_fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str], params: dict[str, Any], seed: int) -> np.ndarray:
    """Train on (features -> log change) and predict the log change for `test`. Missing feature values stay NaN."""
    import lightgbm as lgb

    # LightGBM settings (the first four come from the config; the rest are fixed).
    lgbm_params = {
        "objective": params["objective"],
        "learning_rate": params["learning_rate"],
        "num_leaves": params["num_leaves"],
        "min_data_in_leaf": params["min_child_samples"],
        "feature_fraction": 0.8,   # each tree sees 80% of the features
        "bagging_fraction": 0.8,   # each tree sees 80% of the rows
        "bagging_freq": 1,
        "lambda_l2": 1.0,          # L2 regularisation
        "seed": seed,
        "deterministic": True,     # same result every run
        "force_row_wise": True,
        "num_threads": 4,
        "verbose": -1,             # no log output
    }
    # The Huber loss needs its alpha parameter.
    if params["objective"] == "huber":
        lgbm_params["alpha"] = params.get("huber_alpha", 0.15)

    # Training data: features -> change in log price. The market number is a category, not a quantity.
    train_data = lgb.Dataset(train[features], label=train["y_change"], categorical_feature=["market_idx"], free_raw_data=False)

    # Train the trees and predict the test rows.
    booster = lgb.train(lgbm_params, train_data, num_boost_round=params["n_estimators"])
    return booster.predict(test[features])
