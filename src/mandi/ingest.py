"""Load the raw data, reading only the columns and rows we need.

Onion: the Mendeley parquet file. Tomato and potato: the CEDA district CSV files (see mandi.fetch_ceda).
"""
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


def district_of(market_key: str) -> str:
    """District name inside a CEDA market key, e.g. "Maharashtra_Pune_Pune district" -> "Pune"."""
    parts = market_key.split("_")  # [state, district, market name]
    return parts[1]


def ceda_file(cfg: dict[str, Any], district: str) -> Path:
    """Path of the downloaded CEDA file of this crop and district (written by `python -m mandi.fetch_ceda`)."""
    commodity = cfg["data"]["commodity"]
    file_name = f"{commodity.lower()}_{district.replace(' ', '_')}.csv"
    return ROOT / cfg["paths"]["raw_dir"] / file_name


def load_ceda_rows(cfg: dict[str, Any]) -> pd.DataFrame:
    """Load the CEDA district files of this crop and give them the same columns as the onion file.

    CEDA has one row per (district, day): prices are already averaged over the district's markets and all
    varieties, so the variety is set to one label for all rows. Days with arrivals but no price keep a missing
    price here; the cleaning step drops them and logs how many.
    """
    state = cfg["data"]["state"]
    tables = []
    for market_key in cfg["markets"]:
        district = district_of(market_key)
        path = ceda_file(cfg, district)
        if not path.exists():
            raise FileNotFoundError(f"CEDA data not found at {path}. Run: python -m mandi.fetch_ceda {cfg['data']['commodity']}")

        raw = pd.read_csv(path, parse_dates=["date"])

        # Build a table with the onion file's column names, one column at a time.
        table = pd.DataFrame()
        table["date"] = raw["date"]
        table["state"] = state
        table["district"] = district
        table["market"] = market_key.split("_")[2]  # e.g. "Pune district"
        table["market_key"] = market_key
        table["variety"] = "all varieties (district average)"
        table["arrivals_tonnes"] = raw["arrivals_tonnes"]
        table["min_price_rs_qtl"] = raw["price_min"]
        table["max_price_rs_qtl"] = raw["price_max"]
        table["modal_price_rs_qtl"] = raw["price_modal"]
        tables.append(table)
        logger.info("Loaded %d rows for %s from %s", len(table), market_key, path.name)

    return pd.concat(tables, ignore_index=True)


def load_rows(cfg: dict[str, Any]) -> pd.DataFrame:
    """Load the raw rows from the crop's data source: the onion file, or the CEDA district files."""
    source = cfg["data"].get("source", "mendeley")
    if source == "ceda":
        return load_ceda_rows(cfg)
    return load_state_rows(cfg)


def empty_calendar(cfg: dict[str, Any]) -> tuple[pd.DataFrame, int]:
    """A calendar with only the dates of the study period and no flag columns (for crops without the onion file)."""
    dates = pd.date_range(cfg["data"]["start_date"], cfg["data"]["end_date"], freq="D", name="date")
    calendar = pd.DataFrame({"date": dates})
    return calendar, 0


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
