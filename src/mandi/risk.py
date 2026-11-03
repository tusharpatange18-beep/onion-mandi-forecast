"""Tail-risk summaries of forecast errors and of trade outcomes.

Averages hide the cases that hurt. These functions describe the bad end of a distribution:

* ``worst_q`` : the value at the 95th percentile of the loss (the "worst 5%" threshold, VaR-style);
* ``cvar``    : the mean of the worst 5% of losses (expected shortfall / CVaR-style);
* ``max``     : the single worst loss.

For forecast errors the loss is the absolute error; for trades it is the negative return (see ``mandi.trading``).
"""
from __future__ import annotations

import numpy as np


def cvar(loss: np.ndarray, q: float = 0.05) -> float:
    """Mean of the worst `q` share of `loss` (larger = worse). At least one observation is always included."""
    x = np.sort(np.asarray(loss, dtype=float))[::-1]
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return float("nan")
    k = max(1, int(np.ceil(q * len(x))))
    return float(x[:k].mean())


def tail_risk(errors: np.ndarray, q: float = 0.05) -> dict[str, float]:
    """MAE next to its tail: 95th-percentile absolute error, CVaR-style mean of the worst 5%, maximum error, and the share of total absolute error they carry.

    `errors` are signed forecast errors (actual - forecast); the loss is |error|, so over- and under-forecasts both count.
    """
    a = np.abs(np.asarray(errors, dtype=float))
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return {"n": 0, "MAE": np.nan, "p95_abs_error": np.nan, "cvar5_abs_error": np.nan, "max_abs_error": np.nan, "worst5_share_of_total_error": np.nan}
    k = max(1, int(np.ceil(q * len(a))))
    worst = np.sort(a)[::-1][:k]
    return {"n": int(len(a)), "MAE": float(a.mean()), "p95_abs_error": float(np.quantile(a, 1 - q)), "cvar5_abs_error": float(worst.mean()),
            "max_abs_error": float(a.max()), "worst5_share_of_total_error": float(worst.sum() / a.sum())}
