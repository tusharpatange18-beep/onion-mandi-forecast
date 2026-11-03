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

KEYS = ["market", "origin", "h"]


# --------------------------------------------------------------------------- refit schedule
def refit_schedule(start: str, end: str) -> list[pd.Timestamp]:
    """First day of every month from `start` to `end` (the retrain dates)."""
    return list(pd.date_range(start, end, freq="MS"))


def refit_blocks(refits: list[pd.Timestamp], end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """(refit date, last day of the block) pairs; origins in the block use the model fitted at the refit date."""
    ends = [r - pd.Timedelta(days=1) for r in refits[1:]] + [end]
    return list(zip(refits, ends))


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
        self.column = column
        self.name = name or column

    def fit(self, train: pd.DataFrame) -> "ColumnModel":
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        return test[self.column].to_numpy(dtype=float)


class LGBMModel(Model):
    """Pooled LightGBM on (features -> log change from the last price). Forecast = p0 + predicted change."""

    def __init__(self, params: dict[str, Any], features: Sequence[str], seed: int = 42, name: str = "lgbm"):
        self.params, self.features, self.seed, self.name = dict(params), list(features), seed, name
        self._train: pd.DataFrame | None = None

    def fit(self, train: pd.DataFrame) -> "LGBMModel":
        self._train = train
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        assert self._train is not None, "call fit() before predict()"
        # looked up on the module at call time on purpose: tests replace models.lgbm_fit_predict to inspect the training rows
        change = M.lgbm_fit_predict(self._train, test, self.features, self.params, self.seed)
        return test["p0"].to_numpy() + change


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
        f = tab[["market", "origin", "h", "target_date", "y_log"]].copy()
        f["p0"] = tab["base_naive"].to_numpy()
        f["f_log"] = tab[column].to_numpy()
        return cls(name or column, f.dropna(subset=["f_log"]).reset_index(drop=True))

    @staticmethod
    def align(results: Sequence["BacktestResult"]) -> list["BacktestResult"]:
        """Keep only the rows every result can forecast, so models are scored on exactly the same forecasts."""
        idx = None
        for r in results:
            k = pd.MultiIndex.from_frame(r.frame[KEYS])
            idx = k if idx is None else idx.intersection(k)
        return [BacktestResult(r.name, r.frame[pd.MultiIndex.from_frame(r.frame[KEYS]).isin(idx)].reset_index(drop=True), r.n_fits) for r in results]

    # rupee-scale views ---------------------------------------------------
    def errors(self) -> pd.Series:
        """Signed rupee error actual - forecast (positive = the forecast was too low)."""
        return pd.Series(np.exp(self.frame["y_log"].to_numpy()) - np.exp(self.frame["f_log"].to_numpy()), index=self.frame.index)

    def point_metrics(self) -> dict[str, float]:
        e = self.errors().to_numpy()
        return {"n": int(len(e)), "MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e ** 2)))}

    def tail_risk(self, q: float = 0.05) -> dict[str, float]:
        """Worst-5% error, maximum error and CVaR-style (mean of the worst 5%) absolute error; see ``mandi.risk``."""
        e = self.errors().to_numpy()
        rel = 100 * e / np.exp(self.frame["y_log"].to_numpy())  # error as % of the realised price
        pct = risk.tail_risk(rel, q)
        return {**risk.tail_risk(e, q), **{k + "_pct": pct[k] for k in ("MAE", "p95_abs_error", "cvar5_abs_error", "max_abs_error")}}

    def trading(self, threshold: float = 0.03, cost_bps: float = 0.0) -> trading.TradingResult:
        """Illustrative long/flat strategy driven by this model's forecasts; see ``mandi.trading``."""
        return trading.backtest_long_flat(self.frame, threshold=threshold, cost_bps=cost_bps)


# --------------------------------------------------------------------------- backtester
class Backtester:
    """Rolling-origin, expanding-window backtest of a ``Model`` on an origin table of ONE horizon.

    Parameters
    ----------
    table: origin table (all rows, labelled or not) for a single horizon.
    window: (first, last) origin date to score, inclusive.
    h: the horizon in days (read from the table\u0027s `h` column when omitted).
    min_train_rows: skip a refit block when fewer labelled training rows exist (as in the original LightGBM runner).
    """

    def __init__(self, table: pd.DataFrame, window: tuple[str, str], h: int | None = None, min_train_rows: int = 500,
                 blocks: list[tuple[pd.Timestamp, pd.Timestamp]] | None = None):
        hs = table["h"].unique() if "h" in table else []
        if h is None:
            if len(hs) != 1:
                raise ValueError(f"a Backtester works on one horizon at a time, got {sorted(hs)}; pass h=")
            h = int(hs[0])
        elif len(hs) and set(hs) != {h}:
            raise ValueError(f"table holds horizons {sorted(hs)}, not just h={h}")
        self.table, self.h, self.min_train_rows = table, int(h), min_train_rows
        self.window = (pd.Timestamp(window[0]), pd.Timestamp(window[1]))
        self._blocks = blocks

    def blocks(self) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        """(refit date, last origin of the block) pairs; given explicitly or derived from the window."""
        if self._blocks is not None:
            return self._blocks
        return refit_blocks(refit_schedule(str(self.window[0].date()), str(self.window[1].date())), self.window[1])

    def run(self, model: Model) -> BacktestResult:
        labelled = self.table[self.table["y_log"].notna()]
        parts, n_fits = [], 0
        for refit, block_end in self.blocks():
            train = labelled[labelled["target_date"] < refit]  # embargo: every training target is known before the refit date
            test = self.table[(self.table["origin"] >= refit) & (self.table["origin"] <= block_end) & self.table["y_log"].notna()]
            if len(train) < self.min_train_rows or len(test) == 0:
                continue
            f = model.fit(train).predict(test)
            n_fits += 1
            parts.append(pd.DataFrame({"market": (test["market"] if "market" in test else test["market_idx"].astype(str)).to_numpy(), "market_idx": test["market_idx"].to_numpy(),
                                       "origin": test["origin"].to_numpy(), "h": self.h, "target_date": test["target_date"].to_numpy(),
                                       "y_log": test["y_log"].to_numpy(), "p0": test["p0"].to_numpy(), "f_log": f}))
        frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["market", "market_idx", "origin", "h", "target_date", "y_log", "p0", "f_log"])
        # a model that cannot forecast a row (NaN) simply has no forecast there; `compare` then drops that row for every model
        return BacktestResult(model.name, frame.dropna(subset=["f_log"]).reset_index(drop=True), n_fits)

    def compare(self, models: Sequence[Model]) -> list[BacktestResult]:
        """Run several models and score them on exactly the same forecasts (the fairness filter)."""
        return BacktestResult.align([self.run(m) for m in models])
