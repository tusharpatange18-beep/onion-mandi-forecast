"""Load the raw Mendeley parquet, reading only the columns and rows we need."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from mandi.config import ROOT

logger = logging.getLogger(__name__)


def raw_path(cfg: dict[str, Any]) -> Path:
    """Absolute path of the raw parquet file, from the config."""
    return ROOT / cfg["paths"]["raw_file"]


def load_state_rows(cfg: dict[str, Any]) -> pd.DataFrame:
    """Load every row of the configured state, with only the raw columns we use.

    The file's own lag, rolling, climate and price_spike columns are never read.
    """
    path = raw_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"Raw data not found at {path}. Download it; do not fabricate data.")
    df = pd.read_parquet(
        path,
        columns=cfg["data"]["keep_columns"],
        filters=[("state", "==", cfg["data"]["state"])],
    )
    logger.info("Loaded %d %s rows, %d markets", len(df), cfg["data"]["state"], df["market_key"].nunique())
    return df


def load_calendar(cfg: dict[str, Any]) -> tuple[pd.DataFrame, int]:
    """Build one calendar row per date (festival and crop-season flags) from the whole file.

    The whole file is used so that dates on which none of our markets reported still get
    their flags. Returns the table and the number of dates whose flags disagree between rows
    (the first row is kept for those; the caller logs the count).
    """
    cols = ["date"] + cfg["data"]["calendar_columns"]
    cal = pd.read_parquet(raw_path(cfg), columns=cols)
    n_conflict = int((cal.groupby("date")[cfg["data"]["calendar_columns"]].nunique() > 1).any(axis=1).sum())
    cal = cal.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    return cal, n_conflict
