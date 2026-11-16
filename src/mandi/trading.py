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

import math
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
        returns = self.trades["ret"].to_numpy()  # return of each trade
        n_trades = len(returns)
        equity = self.equity
        n_markets = max(self.n_markets, 1)  # avoid dividing by zero

        # Daily profit or loss, in units of the starting capital.
        daily_pnl = equity.diff().dropna()

        # Sharpe ratio: average daily P&L divided by its standard deviation, scaled to a year.
        sharpe = np.nan
        if len(daily_pnl) > 1:
            daily_std = daily_pnl.std(ddof=1)
            if daily_std > 0:
                sharpe = float(daily_pnl.mean() / daily_std * np.sqrt(365))

        # Number of calendar days the equity curve covers (at least 1).
        span_days = 1
        if len(equity) > 1:
            span_days = (equity.index[-1] - equity.index[0]).days
            if span_days < 1:
                span_days = 1

        # Total number of days we were holding a position.
        days_held = 0.0
        if n_trades > 0:
            holding_days = (self.trades["exit"] - self.trades["entry"]).dt.days
            days_held = float(holding_days.sum())

        # Values that depend on having an equity curve.
        if len(equity) > 0:
            running_peak = equity.cummax()  # highest equity so far
            drawdown = (equity - running_peak).min()  # largest fall from a peak
            total_return_pct = 100 * (float(equity.iloc[-1]) - 1)
            max_drawdown_pct = 100 * float(drawdown)
        else:
            total_return_pct = 0.0
            max_drawdown_pct = np.nan

        # Values that depend on having at least one trade.
        if n_trades > 0:
            mean_trade_return_pct = 100 * float(returns.mean())
            win_rate_pct = 100 * float((returns > 0).mean())  # share of trades that made money
            losses = -returns  # a loss is a negative return
            cvar5_trade_return_pct = -100 * risk.cvar(losses, 0.05)  # mean return of the worst 5% of trades
            worst_trade_pct = 100 * float(returns.min())
        else:
            mean_trade_return_pct = np.nan
            win_rate_pct = np.nan
            cvar5_trade_return_pct = np.nan
            worst_trade_pct = np.nan

        return {
            "trades": n_trades,
            "exposure_pct": 100 * days_held / (span_days * n_markets),  # share of the time capital was in a trade
            "total_return_pct": total_return_pct,
            "mean_trade_return_pct": mean_trade_return_pct,
            "win_rate_pct": win_rate_pct,
            "sharpe": sharpe,
            "max_drawdown_pct": max_drawdown_pct,
            "cvar5_trade_return_pct": cvar5_trade_return_pct,
            "worst_trade_pct": worst_trade_pct,
        }

    def by_year(self) -> pd.DataFrame:
        """P&L (in % of starting capital, fixed notional), trades and win rate by exit year: shows whether the result depends on a few years."""
        trades = self.trades.copy()
        trades["year"] = trades["exit"].dt.year  # year in which the trade closed
        trades["is_win"] = trades["ret"] > 0  # True when the trade made money

        by_year = trades.groupby("year")
        n_trades = by_year["ret"].size()  # number of trades per year
        total_return = by_year["ret"].sum()  # sum of trade returns per year
        win_rate = by_year["is_win"].mean()  # share of winning trades per year

        table = pd.DataFrame({
            "trades": n_trades,
            "pnl_pct_of_capital": 100 * total_return / max(self.n_markets, 1),
            "win_rate_pct": 100 * win_rate,
        })
        return table.reset_index()  # make "year" a normal column


def trade_schedule(frame: pd.DataFrame, enter: np.ndarray, cost_bps: float = 0.0) -> pd.DataFrame:
    """Walk each market forward in time and take every flagged origin that is not inside an open trade."""
    # Add the entry signal, then sort each market by date.
    table = frame.copy()
    table["enter"] = np.asarray(enter, dtype=bool)
    table = table.sort_values(["market", "origin"])

    # Return of a trade = price change from origin to target, minus the cost (bps = 1/10000).
    price_ratio = np.exp(table["y_log"].to_numpy() - table["p0"].to_numpy())
    table["ret"] = price_ratio - 1 - cost_bps / 1e4

    rows = []
    for market, market_rows in table.groupby("market", sort=False):
        # The market is "busy" until the current trade exits. At the start it is free.
        busy_until = pd.Timestamp.min

        origins = market_rows["origin"].tolist()
        target_dates = market_rows["target_date"].tolist()
        enter_flags = market_rows["enter"].tolist()
        trade_returns = market_rows["ret"].tolist()

        for i in range(len(origins)):
            # Enter only if the signal says so and no trade is open.
            if enter_flags[i] and origins[i] >= busy_until:
                rows.append((market, origins[i], target_dates[i], trade_returns[i]))
                busy_until = target_dates[i]  # busy until this trade exits
    return pd.DataFrame(rows, columns=["market", "entry", "exit", "ret"])


def equity_curve(trades: pd.DataFrame, markets: list[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Equal-weight equity at FIXED notional: each trade risks one unit of its market's capital slice and the P&L is added, not compounded.

    Capital starts at 1.0, split equally over the markets; a trade's return is booked on its exit date. Compounding hundreds of
    trades would produce meaningless figures (the quotes are not fillable), so profits are never reinvested.
    """
    days = pd.date_range(start, end, freq="D")
    n_markets = max(len(markets), 1)

    # Sum of trade returns booked on each exit day (0 on days with no exit).
    pnl_per_exit_day = trades.groupby("exit")["ret"].sum()
    pnl_per_day = pnl_per_exit_day.reindex(days, fill_value=0.0)

    # Each market gets 1/n of the capital.
    pnl_per_day = pnl_per_day / n_markets

    # Equity = 1.0 plus all P&L so far.
    return 1.0 + pnl_per_day.cumsum()


def past_return(frame: pd.DataFrame, lookback_days: int, tolerance_days: int = 3) -> np.ndarray:
    """Return of each market's price over the `lookback_days` before every origin (NaN when no quote exists near that earlier date).

    Uses only quotes at or before the origin, so it is a legitimate real-time signal. Row order follows `frame`.
    """
    result = np.full(len(frame), np.nan)
    all_markets = frame["market"].to_numpy()
    all_origins = frame["origin"].to_numpy()
    all_prices = frame["p0"].to_numpy()
    lookback = np.timedelta64(lookback_days, "D")
    tolerance = np.timedelta64(tolerance_days, "D")

    for market in pd.unique(all_markets):
        # Row numbers of this market, sorted by origin date (stable sort keeps ties in their original order).
        rows = np.flatnonzero(all_markets == market)
        order = np.argsort(all_origins[rows], kind="stable")
        rows = rows[order]
        history_dates = all_origins[rows]
        history_prices = all_prices[rows]

        for row in rows:
            # The date `lookback_days` before this origin.
            earlier_date = all_origins[row] - lookback

            # Position of the last quote on or before that date (-1 if there is none).
            position = np.searchsorted(history_dates, earlier_date, side="right") - 1
            if position < 0:
                continue  # no quote that early

            # The quote must be close enough to the earlier date.
            if earlier_date - history_dates[position] > tolerance:
                continue

            # Return from that earlier price to today's price.
            result[row] = np.exp(all_prices[row] - history_prices[position]) - 1
    return result


def with_entry_lag(frame: pd.DataFrame) -> pd.DataFrame:
    """Enter at the market's NEXT quote after the origin instead of the origin quote itself (a one-quote execution lag).

    The signal is still computed at the origin; the trade starts at the next observed quote (usually 1-2 days later), so its
    return is realised price at the target over that later price. Trades whose next quote falls on or after the target date are dropped.
    """
    table = frame.sort_values(["market", "origin"]).copy()

    # Price and date of the NEXT quote in the same market (missing for the last quote).
    next_price = table.groupby("market")["p0"].shift(-1)
    next_date = table.groupby("market")["origin"].shift(-1)

    # Keep the signal-day price, and trade at the next quote's price.
    table["p0_signal"] = table["p0"]
    table["p0"] = next_price

    # Keep only trades whose next quote exists and comes before the exit date.
    keep = next_date.notna() & (next_date < table["target_date"])
    return table[keep].reset_index(drop=True)


def concentration(trades: pd.DataFrame, n_markets: int, share: float = 0.05) -> dict[str, float]:
    """How much of the total P&L the best `share` of trades carries, and the P&L (in % of capital) left without them."""
    # Trade returns sorted from best to worst.
    returns_best_first = np.flip(np.sort(trades["ret"].to_numpy()))
    n_trades = len(returns_best_first)

    # Number of "top" trades (at least one, unless there are no trades).
    n_top = 0
    if n_trades > 0:
        n_top = int(math.ceil(share * n_trades))
        if n_top < 1:
            n_top = 1

    # Total P&L of all trades, and of the top trades.
    total_pnl = 0.0
    if n_trades > 0:
        total_pnl = float(returns_best_first.sum())
    top_pnl = float(returns_best_first[:n_top].sum())

    # Share of the total coming from the top trades (NaN if the total is zero).
    if total_pnl:
        top_share_pct = 100 * top_pnl / total_pnl
    else:
        top_share_pct = np.nan

    return {
        "top_trades": n_top,
        "top_share_of_pnl_pct": top_share_pct,
        "pnl_without_top_pct_of_capital": 100 * (total_pnl - top_pnl) / max(n_markets, 1),
    }


def adverse_excursion(trades: pd.DataFrame, frame: pd.DataFrame) -> pd.Series:
    """Worst unrealised loss inside each trade: the lowest observed quote between entry and exit relative to the entry price.

    The P&L and drawdown figures book a trade only at its exit, so a position that fell 40% and recovered shows no loss there; this
    is that hidden path risk. Uses the quotes in `frame` (one per market and origin day); the exit quote is the realised target.
    """
    worst_moves = []
    for market, market_trades in trades.groupby("market"):
        # All quotes of this market, sorted by date, as rupee prices.
        quotes = frame[frame["market"] == market].sort_values("origin")
        quote_dates = quotes["origin"].to_numpy()
        quote_prices = np.exp(quotes["p0"].to_numpy())

        entries = market_trades["entry"].to_numpy()
        exits = market_trades["exit"].to_numpy()
        trade_returns = market_trades["ret"].to_numpy()

        for i in range(len(entries)):
            entry = entries[i]
            exit_date = exits[i]

            # Price on the entry day (NaN if there is no quote exactly on that day).
            entry_position = np.searchsorted(quote_dates, entry)
            if entry_position < len(quote_prices) and quote_dates[entry_position] == entry:
                entry_price = quote_prices[entry_position]
            else:
                entry_price = np.nan

            # Without an entry price we cannot measure the move.
            if not np.isfinite(entry_price):
                worst_moves.append(np.nan)
                continue

            # All quotes during the trade, plus the exit price.
            during_trade = (quote_dates >= entry) & (quote_dates <= exit_date)
            prices_inside = quote_prices[during_trade]
            exit_price = entry_price * (1 + trade_returns[i])
            price_path = np.append(prices_inside, exit_price)

            # Worst point of the path compared with the entry price.
            worst_moves.append(float(price_path.min() / entry_price - 1))
    return pd.Series(worst_moves, dtype=float)


def backtest_long_flat(frame: pd.DataFrame, threshold: float = 0.03, cost_bps: float = 0.0, always_long: bool = False,
                       momentum_days: int | None = None, entry_lag: bool = False) -> TradingResult:
    """Score the long/flat rule on a forecast frame (columns market, origin, target_date, y_log, p0, f_log).

    `always_long=True` ignores the forecast and enters whenever allowed: the buy-and-hold-style benchmark with the same trade schedule.
    `momentum_days=k` replaces the forecast by "the price rose more than `threshold` over the last k days": the no-model benchmark.
    `entry_lag=True` trades at the next quote after the signal instead of the signal-day quote (see `with_entry_lag`).
    """
    # Only rows that have a forecast, an actual value and a starting price.
    table = frame.dropna(subset=["f_log", "y_log", "p0"]).reset_index(drop=True)

    # Price on the day the signal is made.
    signal_price = table["p0"].to_numpy()
    if entry_lag:
        table = with_entry_lag(table)
        signal_price = table["p0_signal"].to_numpy()

    # Expected return according to the forecast.
    expected_return = np.exp(table["f_log"].to_numpy() - signal_price) - 1

    # Momentum benchmark: use the past return instead of the forecast (missing -> -100%, so no trade).
    if momentum_days is not None:
        past = past_return(table, momentum_days)
        expected_return = np.nan_to_num(past, nan=-1.0)

    # Decide on which origins we want to enter.
    if always_long:
        enter = np.ones(len(table), dtype=bool)
    else:
        enter = expected_return > threshold

    # Build the trades and the equity curve.
    trades = trade_schedule(table, enter, cost_bps)
    markets = sorted(table["market"].unique())
    start = table["origin"].min()
    end = table["target_date"].max()
    equity = equity_curve(trades, markets, start, end)

    settings = {"threshold": threshold, "cost_bps": cost_bps, "always_long": always_long}
    return TradingResult(trades, equity, len(markets), settings)
