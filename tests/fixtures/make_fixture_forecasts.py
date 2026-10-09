"""SYNTHETIC TEST DATA ONLY: write tests/fixtures/forecasts_fixture.parquet.

Made-up numbers in the schema of results/forecasts.parquet, used by CI to start the API container and check its healthcheck.
Nothing here is derived from the onion dataset. Run:  python tests/fixtures/make_fixture_forecasts.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MARKETS = ["Solapur", "Pune", "Vashi", "Kolhapur", "Sambhajinagar"]
HERE = Path(__file__).resolve().parent


def make() -> pd.DataFrame:
    rows = []
    for m in MARKETS:
        for h in (7, 30):
            for k in range(3):                                   # k = 2 is the latest origin
                origin = pd.Timestamp("2024-12-31") - pd.Timedelta(days=2 - k)
                point = 1000.0 + 10 * k
                rows.append({"market": m, "horizon": h, "origin_date": origin, "target_date": origin + pd.Timedelta(days=h), "last_price": 1000.0, "point": point,
                             "lo80": point * 0.8, "hi80": point * 1.2, "lo95": point * 0.6, "hi95": point * 1.5, "naive": 1000.0, "n_calibration": 200,
                             "model_spec": "SYNTHETIC-FIXTURE", "trained_through": pd.Timestamp("2024-12-31"), "generated_at": "2025-01-01T00:00:00",
                             "git_commit": "fixture", "as_of": pd.Timestamp("2025-01-01"), "data_through": pd.Timestamp("2024-12-31"), "is_latest": k == 2,
                             "in_sample": False, "actual_price": np.nan, "gap_before_origin_days": 1.0,
                             "synthetic_test_data": "SYNTHETIC TEST DATA ONLY - not from the onion dataset"})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out = HERE / "forecasts_fixture.parquet"
    make().to_parquet(out, index=False)
    print(f"wrote {out.name} (synthetic test data only)")
