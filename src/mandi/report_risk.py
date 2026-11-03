"""Tail-risk and illustrative trading tables from the final backtest forecasts.

Usage (needs results/fair_forecasts.parquet from `python -m mandi.run_phase3 final`; no refitting):
    python -m mandi.report_risk

Writes results/tail_risk.csv, results/trading_summary.csv, results/trading_sensitivity.csv and results/figures/trading_equity.png.
The README blocks "tail_risk", "trading" and "key_results" are generated from these files by `python -m mandi.readme_tables`.

Mandi prices are not directly tradable; the trading table illustrates the method (see mandi/trading.py).
The long/flat threshold (3%) was fixed before looking at the result; the sensitivity table shows other values rather than picking the best.
"""
from __future__ import annotations

import pandas as pd

from mandi.config import ROOT
from mandi.framework import BacktestResult
from mandi.trading import backtest_long_flat

R = ROOT / "results"
MODELS = {"lgbm": "LightGBM", "base_naive": "naive (last price)", "base_mean3": "mean of last 3 quotes", "base_seasonal": "seasonal naive (365 d)"}
HEADLINE_THRESHOLD = 0.03
SENSITIVITY_THRESHOLDS = [0.0, 0.03, 0.06]
SENSITIVITY_COSTS_BPS = [0.0, 100.0]


def tail_table(fair: pd.DataFrame) -> pd.DataFrame:
    """Pooled (all five markets) error tails per horizon and model, on the rows every model forecast."""
    rows = []
    for h, g in fair.groupby("h"):
        for col, name in MODELS.items():
            tr = BacktestResult.from_columns(g, col, name).tail_risk()
            rows.append({"h": int(h), "model": col, **tr})
    return pd.DataFrame(rows)


def trading_tables(fair: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, dict[str, pd.Series]], pd.DataFrame]:
    """(headline summary, sensitivity grid, equity curves, P&L by year) of the long/flat rule on the LightGBM forecasts and of the always-long benchmark."""
    summ, sens, curves, years = [], [], {}, []
    for h, g in fair.groupby("h"):
        res = BacktestResult.from_columns(g, "lgbm", "LightGBM")
        model = res.trading(HEADLINE_THRESHOLD)
        years.append(model.by_year().assign(h=int(h)))
        bench = backtest_long_flat(res.frame, always_long=True)
        mom = backtest_long_flat(res.frame, HEADLINE_THRESHOLD, momentum_days=int(h))
        for label, tr in (("long/flat on LightGBM forecast", model), (f"momentum rule: price up >3% over the last {int(h)} days (no model)", mom),
                          ("always long (benchmark, same trade schedule)", bench)):
            summ.append({"h": int(h), "strategy": label, **tr.summary()})
        curves[int(h)] = {"LightGBM long/flat": model.equity, "momentum rule": mom.equity, "always long": bench.equity}
        for thr in SENSITIVITY_THRESHOLDS:
            for cost in SENSITIVITY_COSTS_BPS:
                s = res.trading(thr, cost).summary()
                sens.append({"h": int(h), "threshold_pct": 100 * thr, "round_trip_cost_bps": cost, **s})
    return pd.DataFrame(summ), pd.DataFrame(sens), curves, pd.concat(years, ignore_index=True)[["h", "year", "trades", "pnl_pct_of_capital", "win_rate_pct"]]


def plot_equity(curves: dict[int, dict[str, pd.Series]], path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(curves), figsize=(11, 3.6), sharey=False)
    for ax, (h, cs) in zip(axes if len(curves) > 1 else [axes], curves.items()):
        for name, s in cs.items():
            ax.plot(s.index, s.values, label=name, lw=1.6)
        ax.axhline(1.0, color="grey", lw=0.8, ls=":")
        ax.set_title(f"{h}-day horizon: equal-weight equity (start = 1)")
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Illustration only: mandi prices are not directly tradable", fontsize=9, y=0.01, color="grey")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    fair = pd.read_parquet(R / "fair_forecasts.parquet")
    tail = tail_table(fair)
    summ, sens, curves, years = trading_tables(fair)
    years.round(4).to_csv(R / "trading_by_year.csv", index=False)
    tail.round(4).to_csv(R / "tail_risk.csv", index=False)
    summ.round(4).to_csv(R / "trading_summary.csv", index=False)
    sens.round(4).to_csv(R / "trading_sensitivity.csv", index=False)
    plot_equity(curves, R / "figures" / "trading_equity.png")
    print(tail.round(2).to_string(index=False))
    print(summ.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
