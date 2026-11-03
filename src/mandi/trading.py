"""Illustrative long/flat trading evaluation of the price forecasts.

**Mandi prices are not directly tradable.** There is no instrument that pays the Solapur modal price in 30 days, and
the quotes are reported prices, not fills. This module only shows how a forecast can be turned into a rule and scored
with the metrics a trading desk would ask for (P&L, drawdown, Sharpe, win rate, tail loss). Treat the numbers as an
illustration of the method, not as a strategy.

The rule (fixed in advance, not tuned on the test years):

* at a forecast origin, go **long** if the forecast beats today's price by more than `threshold` (default 3%), else stay flat;
* hold for the forecast horizon `h` (exit at the target date) and take the realised price change as the trade return;
* **no overlapping trades:** a market can hold one position at a time, so a new entry needs ``origin >= previous exit``;
* optional round-trip cost in basis points, charged on every trade (default 0);
* the capital is split equally across markets and every trade uses a fixed notional (P&L is added up, never compounded).

Sharpe uses the daily change of the equal-weight equity curve (calendar days, zero risk-free rate, sqrt(365) scaling); drawdown is
the largest peak-to-trough fall of that curve in percent of the starting capital.
The tail measure is the mean return of the worst 5% of trades (CVaR-style) and the single worst trade.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from mandi import risk


@dataclass
class TradingResult:
    """Trades (one row each), the equal-weight equity curve and the settings that produced them."""

    trades: pd.DataFrame            # market, entry, exit, ret
    equity: pd.Series               # equal-weight equity, daily, starts at 1.0
    n_markets: int
    params: dict = field(default_factory=dict)

    def summary(self) -> dict[str, float]:
        r = self.trades["ret"].to_numpy()
        n = len(r)
        eq = self.equity
        daily = eq.diff().dropna()                       # daily P&L in units of starting capital
        dd = (eq - eq.cummax()).min() if len(eq) else np.nan   # largest peak-to-trough fall, in units of starting capital
        sharpe = float(daily.mean() / daily.std(ddof=1) * np.sqrt(365)) if len(daily) > 1 and daily.std(ddof=1) > 0 else np.nan
        span_days = max((eq.index[-1] - eq.index[0]).days, 1) if len(eq) > 1 else 1
        held = float((self.trades["exit"] - self.trades["entry"]).dt.days.sum()) if n else 0.0
        return {
            "trades": n,
            "exposure_pct": 100 * held / (span_days * max(self.n_markets, 1)),
            "total_return_pct": 100 * (float(eq.iloc[-1]) - 1) if len(eq) else 0.0,
            "mean_trade_return_pct": 100 * float(r.mean()) if n else np.nan,
            "win_rate_pct": 100 * float((r > 0).mean()) if n else np.nan,
            "sharpe": sharpe,
            "max_drawdown_pct": 100 * float(dd) if len(eq) else np.nan,
            "cvar5_trade_return_pct": -100 * risk.cvar(-r, 0.05) if n else np.nan,   # mean return of the worst 5% of trades
            "worst_trade_pct": 100 * float(r.min()) if n else np.nan,
        }


    def by_year(self) -> pd.DataFrame:
        """P&L (in % of starting capital, fixed notional), trades and win rate by exit year: shows whether the result depends on a few years."""
        t = self.trades.assign(year=self.trades["exit"].dt.year)
        g = t.groupby("year")["ret"]
        return pd.DataFrame({"trades": g.size(), "pnl_pct_of_capital": 100 * g.sum() / max(self.n_markets, 1), "win_rate_pct": 100 * g.apply(lambda x: (x > 0).mean())}).reset_index()


def trade_schedule(frame: pd.DataFrame, enter: np.ndarray, cost_bps: float = 0.0) -> pd.DataFrame:
    """Walk each market forward in time and take every flagged origin that is not inside an open trade."""
    f = frame.assign(enter=np.asarray(enter, dtype=bool)).sort_values(["market", "origin"])
    ret = np.exp(f["y_log"].to_numpy() - f["p0"].to_numpy()) - 1 - cost_bps / 1e4
    f = f.assign(ret=ret)
    rows = []
    for m, g in f.groupby("market", sort=False):
        busy_until = pd.Timestamp.min
        for o, t, e, r in zip(g["origin"], g["target_date"], g["enter"], g["ret"]):
            if e and o >= busy_until:
                rows.append((m, o, t, r))
                busy_until = t
    return pd.DataFrame(rows, columns=["market", "entry", "exit", "ret"])


def equity_curve(trades: pd.DataFrame, markets: list[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Equal-weight equity at FIXED notional: each trade risks one unit of its market's capital slice and the P&L is added, not compounded.

    Capital starts at 1.0, split equally over the markets; a trade's return is booked on its exit date. Compounding hundreds of
    trades would produce meaningless figures (the quotes are not fillable), so profits are never reinvested.
    """
    days = pd.date_range(start, end, freq="D")
    pnl = trades.groupby("exit")["ret"].sum().reindex(days, fill_value=0.0) / max(len(markets), 1)
    return 1.0 + pnl.cumsum()


def past_return(frame: pd.DataFrame, lookback_days: int, tolerance_days: int = 3) -> np.ndarray:
    """Return of each market's price over the `lookback_days` before every origin (NaN when no quote exists near that earlier date).

    Uses only quotes at or before the origin, so it is a legitimate real-time signal. Row order follows `frame`.
    """
    out = np.full(len(frame), np.nan)
    pos = np.arange(len(frame))
    for _, g in frame.assign(_pos=pos).groupby("market"):
        hist = g[["origin", "p0"]].sort_values("origin").rename(columns={"origin": "t", "p0": "p_then"})
        q = g[["origin", "_pos"]].assign(t=g["origin"] - pd.Timedelta(days=lookback_days)).sort_values("t")
        m = pd.merge_asof(q, hist, on="t", direction="backward", tolerance=pd.Timedelta(days=tolerance_days))
        out[m["_pos"].to_numpy()] = np.exp(g.set_index("_pos").loc[m["_pos"], "p0"].to_numpy() - m["p_then"].to_numpy()) - 1
    return out


def backtest_long_flat(frame: pd.DataFrame, threshold: float = 0.03, cost_bps: float = 0.0, always_long: bool = False,
                       momentum_days: int | None = None) -> TradingResult:
    """Score the long/flat rule on a forecast frame (columns market, origin, target_date, y_log, p0, f_log).

    `always_long=True` ignores the forecast and enters whenever allowed: the buy-and-hold-style benchmark with the same trade schedule.
    `momentum_days=k` replaces the forecast by "the price rose more than `threshold` over the last k days": the no-model benchmark.
    """
    f = frame.dropna(subset=["f_log", "y_log", "p0"]).reset_index(drop=True)
    expected = np.exp(f["f_log"].to_numpy() - f["p0"].to_numpy()) - 1
    if momentum_days is not None:
        expected = np.nan_to_num(past_return(f, momentum_days), nan=-1.0)
    enter = np.ones(len(f), dtype=bool) if always_long else expected > threshold
    trades = trade_schedule(f, enter, cost_bps)
    markets = sorted(f["market"].unique())
    start, end = f["origin"].min(), f["target_date"].max()
    eq = equity_curve(trades, markets, start, end)
    return TradingResult(trades, eq, len(markets), {"threshold": threshold, "cost_bps": cost_bps, "always_long": always_long})
