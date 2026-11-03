"""End-to-end demo of the backtesting framework on SYNTHETIC prices (no data download needed).

    python -m mandi.demo

It simulates five seasonal random-walk "mandis", builds the origin tables, runs the naive baseline and LightGBM through
`Backtester`, and prints point metrics, tail-risk metrics and the long/flat trading evaluation. The numbers mean nothing
about onions: they only show that the pipeline (refit schedule, label embargo, fairness alignment, metrics) runs end to
end on a fresh clone. The real results come from `python -m mandi.run_phase3 final` on the Mendeley data (see README).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mandi import backtest as B
from mandi import features as F
from mandi.config import load_config
from mandi.framework import Backtester, ColumnModel, LGBMModel


def synthetic_panel(n_markets: int = 5, start: str = "2017-01-01", end: str = "2021-06-30", seed: int = 0) -> pd.DataFrame:
    """Daily log-price = yearly cycle + slow trend shocks + noise; ~15% of days missing. Prices in rupees per quintal."""
    rng = np.random.default_rng(seed)
    days = pd.date_range(start, end, freq="D")
    season = 0.35 * np.sin(2 * np.pi * (days.dayofyear.to_numpy() - 60) / 365.25)
    common = np.cumsum(rng.normal(0, 0.02, len(days)))
    frames = []
    for m in range(n_markets):
        own = np.cumsum(rng.normal(0, 0.015, len(days)))
        logp = np.log(1500) + season + 0.5 * common + 0.5 * own
        price = np.round(np.exp(logp) / 50) * 50
        keep = rng.random(len(days)) > 0.15
        frames.append(pd.DataFrame({"market_key": f"Synthetic_{m}", "date": days[keep], "modal_price": price[keep],
                                    "arrivals_tonnes": rng.uniform(50, 500, keep.sum()), "is_stale": False}))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    cfg = load_config()
    panel = synthetic_panel()
    arrays = F.build_arrays(panel, "2017-01-01", "2021-06-30", sorted(panel["market_key"].unique()))
    tables = B.build_tables(arrays, {}, cfg)
    params = {"objective": "huber", "num_leaves": 8, "min_child_samples": 20, "n_estimators": 60, "learning_rate": cfg["forecast"]["models"]["lgbm"]["learning_rate"]}
    window = ("2019-07-01", "2021-06-30")
    for h in (7, 30):
        bt = Backtester(tables[h], window, min_train_rows=300)
        naive, lgbm = bt.compare([ColumnModel("base_naive", "naive"), LGBMModel(params, F.FEATURE_COLUMNS, cfg["seed"], "lgbm")])
        print(f"\n=== horizon {h} days: {len(lgbm.frame)} forecasts, {lgbm.n_fits} monthly refits (synthetic data) ===")
        for r in (naive, lgbm):
            pm, tr = r.point_metrics(), r.tail_risk()
            print(f"{r.name:6s} MAE {pm['MAE']:7.1f}  p95 error {tr['p95_abs_error']:7.1f}  worst-5% mean {tr['cvar5_abs_error']:7.1f}  max {tr['max_abs_error']:7.1f}")
        s = lgbm.trading(threshold=0.03).summary()
        print(f"long/flat (illustration): trades {s['trades']}, P&L {s['total_return_pct']:+.0f}% of capital, Sharpe {s['sharpe']:.2f}, "
              f"max drawdown {s['max_drawdown_pct']:.0f}%, win rate {s['win_rate_pct']:.0f}%, mean of worst 5% trades {s['cvar5_trade_return_pct']:.0f}%")


if __name__ == "__main__":
    main()
