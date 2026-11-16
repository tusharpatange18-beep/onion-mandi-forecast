"""A small, reusable backtesting interface: ``Model`` (fit / predict), ``Backtester.run()`` and ``BacktestResult``.

The rules that make the evaluation honest live here, in one place:

* **Rolling origin, expanding window.** Models are refit on the first day of every month; the refit applies to the
  origins of that month (``refit_schedule`` / ``blocks``).
* **Embargo on labels.** A training row is used only if its *target date* is before the refit date, so no unseen
  target leaks into training.
* **Same rows for every model.** ``BacktestResult.align`` keeps only the (market, origin) rows that every result can
  forecast, so models are compared on identical forecasts.

Typical use::

    bt = Backtester(table_h30, window=("2020-01-01", "2024-12-31"))
    res = bt.run(LGBMModel(params, seed=42))
    res.point_metrics(), res.tail_risk(), res.trading(threshold=0.03)

A ``Model`` works on an *origin table* (``mandi.features.build_origin_table``): one row per (market, origin) with
features, the last log price ``p0`` and the realised log target ``y_log``. ``predict`` returns the forecast of the log
price at the target date. Statistical per-market models (ETS, SARIMA, Prophet) keep their own runners in
``mandi.backtest``; their saved forecasts can still be evaluated with ``BacktestResult.from_columns``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from mandi import models as M
from mandi import risk, trading

KEYS = ["market", "origin", "h"]  # these three columns identify one forecast
RESULT_COLUMNS = ["market", "market_idx", "origin", "h", "target_date", "y_log", "p0", "f_log"]


# --------------------------------------------------------------------------- refit schedule
def refit_schedule(start: str, end: str) -> list[pd.Timestamp]:
    """First day of every month from `start` to `end` (the retrain dates)."""
    # "MS" = month start.
    month_starts = pd.date_range(start, end, freq="MS")
    return list(month_starts)


def refit_blocks(refits: list[pd.Timestamp], end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """(refit date, last day of the block) pairs; origins in the block use the model fitted at the refit date."""
    blocks = []
    for i in range(len(refits)):
        refit_date = refits[i]
        if i + 1 < len(refits):
            # A block ends the day before the next refit.
            block_end = refits[i + 1] - pd.Timedelta(days=1)
        else:
            # The last block ends at the end of the window.
            block_end = end
        blocks.append((refit_date, block_end))
    return blocks


# --------------------------------------------------------------------------- models
class Model(ABC):
    """Anything that can be fitted on past origin rows and asked for log-price forecasts of later ones."""

    name: str = "model"

    @abstractmethod
    def fit(self, train: pd.DataFrame) -> "Model":
        """Learn from `train` (rows whose targets are all known before the refit date)."""

    @abstractmethod
    def predict(self, test: pd.DataFrame) -> np.ndarray:
        """Forecast of the log price at each row's target date (same length and order as `test`)."""


class ColumnModel(Model):
    """A baseline that needs no fitting: its forecast is already a column of the origin table.

    ``ColumnModel("base_naive")`` is "tomorrow equals today"; ``base_mean3`` and ``base_seasonal`` are the other two baselines.
    """

    def __init__(self, column: str, name: str | None = None):
        self.column = column  # the column that holds the forecast
        # Use the column name as the model name unless another name is given.
        if name:
            self.name = name
        else:
            self.name = column

    def fit(self, train: pd.DataFrame) -> "ColumnModel":
        # Nothing to learn.
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        # The forecast is simply the column.
        return test[self.column].to_numpy(dtype=float)


class LGBMModel(Model):
    """Pooled LightGBM on (features -> log change from the last price). Forecast = p0 + predicted change."""

    def __init__(self, params: dict[str, Any], features: Sequence[str], seed: int = 42, name: str = "lgbm"):
        self.params = dict(params)  # LightGBM settings
        self.features = list(features)  # feature column names
        self.seed = seed  # random seed for reproducible results
        self.name = name
        self._train: pd.DataFrame | None = None  # training rows, stored by fit()

    def fit(self, train: pd.DataFrame) -> "LGBMModel":
        # Training happens inside predict(), so here we only remember the training rows.
        self._train = train
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        if self._train is None:
            raise AssertionError("call fit() before predict()")
        # looked up on the module at call time on purpose: tests replace models.lgbm_fit_predict to inspect the training rows
        predicted_change = M.lgbm_fit_predict(self._train, test, self.features, self.params, self.seed)
        # Forecast = today's log price + predicted change.
        return test["p0"].to_numpy() + predicted_change


# --------------------------------------------------------------------------- results
@dataclass
class BacktestResult:
    """Forecasts of one model over one window: columns market, origin, h, target_date, y_log, p0, f_log (one row per forecast)."""

    name: str
    frame: pd.DataFrame
    n_fits: int = 0

    @classmethod
    def from_columns(cls, tab: pd.DataFrame, column: str, name: str | None = None) -> "BacktestResult":
        """Wrap saved forecasts: `tab` has market, origin, h, target_date, y_log, base_naive (= p0) and a log-forecast `column`."""
        frame = tab[["market", "origin", "h", "target_date", "y_log"]].copy()
        frame["p0"] = tab["base_naive"].to_numpy()  # naive forecast = last price
        frame["f_log"] = tab[column].to_numpy()  # the model's forecast

        # Drop rows without a forecast.
        frame = frame.dropna(subset=["f_log"]).reset_index(drop=True)

        if not name:
            name = column
        return cls(name, frame)

    @staticmethod
    def align(results: Sequence["BacktestResult"]) -> list["BacktestResult"]:
        """Keep only the rows every result can forecast, so models are scored on exactly the same forecasts."""
        # Find the (market, origin, h) keys that appear in every result.
        common_keys = None
        for result in results:
            keys = pd.MultiIndex.from_frame(result.frame[KEYS])
            if common_keys is None:
                common_keys = keys
            else:
                common_keys = common_keys.intersection(keys)

        # Keep only those rows in each result.
        aligned = []
        for result in results:
            keys = pd.MultiIndex.from_frame(result.frame[KEYS])
            keep = keys.isin(common_keys)
            kept_frame = result.frame[keep].reset_index(drop=True)
            aligned.append(BacktestResult(result.name, kept_frame, result.n_fits))
        return aligned

    # rupee-scale views ---------------------------------------------------
    def errors(self) -> pd.Series:
        """Signed rupee error actual - forecast (positive = the forecast was too low)."""
        actual_price = np.exp(self.frame["y_log"].to_numpy())
        forecast_price = np.exp(self.frame["f_log"].to_numpy())
        return pd.Series(actual_price - forecast_price, index=self.frame.index)

    def point_metrics(self) -> dict[str, float]:
        error = self.errors().to_numpy()
        mae = np.mean(np.abs(error))  # average error size
        rmse = np.sqrt(np.mean(error ** 2))  # root mean squared error
        return {"n": int(len(error)), "MAE": float(mae), "RMSE": float(rmse)}

    def tail_risk(self, q: float = 0.05) -> dict[str, float]:
        """Worst-5% error, maximum error and CVaR-style (mean of the worst 5%) absolute error; see ``mandi.risk``."""
        error = self.errors().to_numpy()

        # Error as a percentage of the actual price.
        actual_price = np.exp(self.frame["y_log"].to_numpy())
        error_pct = 100 * error / actual_price

        # Tail summary in rupees, and in percent.
        result = risk.tail_risk(error, q)
        tail_pct = risk.tail_risk(error_pct, q)

        # Add the percentage versions of four numbers with a "_pct" suffix.
        for key in ["MAE", "p95_abs_error", "cvar5_abs_error", "max_abs_error"]:
            result[key + "_pct"] = tail_pct[key]
        return result

    def trading(self, threshold: float = 0.03, cost_bps: float = 0.0, entry_lag: bool = False) -> trading.TradingResult:
        """Illustrative long/flat strategy driven by this model's forecasts; see ``mandi.trading``."""
        return trading.backtest_long_flat(self.frame, threshold=threshold, cost_bps=cost_bps, entry_lag=entry_lag)


# --------------------------------------------------------------------------- backtester
class Backtester:
    """Rolling-origin, expanding-window backtest of a ``Model`` on an origin table of ONE horizon.

    Parameters
    ----------
    table: origin table (all rows, labelled or not) for a single horizon.
    window: (first, last) origin date to score, inclusive.
    h: the horizon in days (read from the table's `h` column when omitted).
    min_train_rows: skip a refit block when fewer labelled training rows exist (as in the original LightGBM runner).
    """

    def __init__(self, table: pd.DataFrame, window: tuple[str, str], h: int | None = None, min_train_rows: int = 500,
                 blocks: list[tuple[pd.Timestamp, pd.Timestamp]] | None = None):
        # Which horizons does the table contain?
        if "h" in table:
            horizons_in_table = table["h"].unique()
        else:
            horizons_in_table = []

        if h is None:
            # No horizon given: the table must hold exactly one.
            if len(horizons_in_table) != 1:
                raise ValueError(f"a Backtester works on one horizon at a time, got {sorted(horizons_in_table)}; pass h=")
            h = int(horizons_in_table[0])
        elif len(horizons_in_table) > 0 and set(horizons_in_table) != {h}:
            # A horizon was given, but the table holds other horizons too.
            raise ValueError(f"table holds horizons {sorted(horizons_in_table)}, not just h={h}")

        self.table = table
        self.h = int(h)
        self.min_train_rows = min_train_rows
        self.window = (pd.Timestamp(window[0]), pd.Timestamp(window[1]))
        self._blocks = blocks

    def blocks(self) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        """(refit date, last origin of the block) pairs; given explicitly or derived from the window."""
        if self._blocks is not None:
            return self._blocks
        # One block per month inside the window.
        first_day = str(self.window[0].date())
        last_day = str(self.window[1].date())
        refit_dates = refit_schedule(first_day, last_day)
        return refit_blocks(refit_dates, self.window[1])

    def run(self, model: Model) -> BacktestResult:
        table = self.table
        has_target = table["y_log"].notna()  # rows whose target price is known
        labelled = table[has_target]

        parts = []
        n_fits = 0
        for refit_date, block_end in self.blocks():
            # Embargo: train only on rows whose target date is before the refit date.
            train = labelled[labelled["target_date"] < refit_date]

            # Test rows: origins inside this block that have a target.
            in_block = (table["origin"] >= refit_date) & (table["origin"] <= block_end)
            test = table[in_block & has_target]

            # Skip the block if there is too little training data or nothing to test.
            if len(train) < self.min_train_rows or len(test) == 0:
                continue

            # Fit the model and forecast the test rows.
            forecast = model.fit(train).predict(test)
            n_fits = n_fits + 1

            # Market name (or the market number as text if there is no name column).
            if "market" in test:
                market_names = test["market"].to_numpy()
            else:
                market_names = test["market_idx"].astype(str).to_numpy()

            part = pd.DataFrame({
                "market": market_names,
                "market_idx": test["market_idx"].to_numpy(),
                "origin": test["origin"].to_numpy(),
                "h": self.h,
                "target_date": test["target_date"].to_numpy(),
                "y_log": test["y_log"].to_numpy(),
                "p0": test["p0"].to_numpy(),
                "f_log": forecast,
            })
            parts.append(part)

        # Join the blocks (or make an empty table if no block was run).
        if parts:
            frame = pd.concat(parts, ignore_index=True)
        else:
            frame = pd.DataFrame(columns=RESULT_COLUMNS)

        # a model that cannot forecast a row (NaN) simply has no forecast there; `compare` then drops that row for every model
        frame = frame.dropna(subset=["f_log"]).reset_index(drop=True)
        return BacktestResult(model.name, frame, n_fits)

    def compare(self, models: Sequence[Model]) -> list[BacktestResult]:
        """Run several models and score them on exactly the same forecasts (the fairness filter)."""
        results = []
        for model in models:
            results.append(self.run(model))
        return BacktestResult.align(results)
