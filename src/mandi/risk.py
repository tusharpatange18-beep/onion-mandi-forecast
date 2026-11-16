"""Tail-risk summaries of forecast errors and of trade outcomes.

Averages hide the cases that hurt. These functions describe the bad end of a distribution:

* ``worst_q`` : the value at the 95th percentile of the loss (the "worst 5%" threshold, VaR-style);
* ``cvar``    : the mean of the worst 5% of losses (expected shortfall / CVaR-style);
* ``max``     : the single worst loss.

For forecast errors the loss is the absolute error; for trades it is the negative return (see ``mandi.trading``).
"""
from __future__ import annotations

import math

import numpy as np


def _remove_missing(values: np.ndarray) -> np.ndarray:
    """Return the values without NaN (missing) entries."""
    is_missing = np.isnan(values)
    return values[~is_missing]


def _sort_largest_first(values: np.ndarray) -> np.ndarray:
    """Return the values sorted from largest to smallest."""
    sorted_small_to_large = np.sort(values)
    # Flip the order so the largest value comes first.
    return np.flip(sorted_small_to_large)


def _tail_count(q: float, total_count: int) -> int:
    """How many observations make up the worst `q` share (always at least one)."""
    count = int(math.ceil(q * total_count))
    if count < 1:
        count = 1
    return count


def cvar(loss: np.ndarray, q: float = 0.05) -> float:
    """Mean of the worst `q` share of `loss` (larger = worse). At least one observation is always included."""
    # Clean the input: float array without missing values.
    losses = np.asarray(loss, dtype=float)
    losses = _remove_missing(losses)

    # Nothing left means there is nothing to average.
    if len(losses) == 0:
        return float("nan")

    # Sort so the worst (largest) losses come first.
    losses_worst_first = _sort_largest_first(losses)

    # Take the worst q share and average it.
    tail_size = _tail_count(q, len(losses))
    worst_losses = losses_worst_first[:tail_size]
    return float(worst_losses.mean())


def tail_risk(errors: np.ndarray, q: float = 0.05) -> dict[str, float]:
    """MAE next to its tail: 95th-percentile absolute error, CVaR-style mean of the worst 5%, maximum error, and the share of total absolute error they carry.

    `errors` are signed forecast errors (actual - forecast); the loss is |error|, so over- and under-forecasts both count.
    """
    # The loss is the size of the error, ignoring its sign.
    abs_errors = np.abs(np.asarray(errors, dtype=float))
    abs_errors = _remove_missing(abs_errors)

    # No errors at all: return an empty summary.
    if len(abs_errors) == 0:
        return {
            "n": 0,
            "MAE": np.nan,
            "p95_abs_error": np.nan,
            "cvar5_abs_error": np.nan,
            "max_abs_error": np.nan,
            "worst5_share_of_total_error": np.nan,
        }

    # The worst q share of the errors.
    tail_size = _tail_count(q, len(abs_errors))
    errors_worst_first = _sort_largest_first(abs_errors)
    worst_errors = errors_worst_first[:tail_size]

    # Summary numbers.
    mae = abs_errors.mean()  # average error
    p95 = np.quantile(abs_errors, 1 - q)  # error at the 95th percentile
    cvar5 = worst_errors.mean()  # average of the worst 5%
    max_error = abs_errors.max()  # single biggest error
    worst_share = worst_errors.sum() / abs_errors.sum()  # share of all error coming from the worst 5%

    return {
        "n": int(len(abs_errors)),
        "MAE": float(mae),
        "p95_abs_error": float(p95),
        "cvar5_abs_error": float(cvar5),
        "max_abs_error": float(max_error),
        "worst5_share_of_total_error": float(worst_share),
    }
