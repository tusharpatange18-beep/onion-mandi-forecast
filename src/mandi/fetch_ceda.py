"""Download daily district-level prices and arrivals from the CEDA Agri Market Data portal and save them as CSV files.

Source: CEDA (Centre for Economic Data and Analysis, Ashoka University), https://agmarknet.ceda.ashoka.edu.in/
CEDA compiles the Agmarknet data of the Ministry of Agriculture and Farmers Welfare and maps markets to Census 2011
districts. Please cite CEDA when using this data.

What one row means: one day, one commodity, one district.
* price_min / price_max / price_modal: average over the markets in the district that reported that day (Rs per quintal);
* arrivals_tonnes: total arrivals in the district that day.

Note: these are DISTRICT values (several markets together), not single-market values like the onion data.

Usage:
    python -m mandi.fetch_ceda Tomato Potato

One CSV file per (commodity, district) is written to data/raw/ceda/ (not in git, like the onion file).
"""
from __future__ import annotations

import csv
import json
import logging
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from mandi.config import ROOT

logger = logging.getLogger(__name__)

BASE_URL = "https://agmarknet.ceda.ashoka.edu.in"
START_DATE = "2014-01-01"  # same period as the onion study
END_DATE = "2024-12-31"
STATE_ID = 27  # Maharashtra
COMMODITY_IDS = {"Potato": 24, "Tomato": 78}
# Census 2011 district numbers used by CEDA.
DISTRICT_IDS = {
    "Pune": 521,
    "Solapur": 526,
    "Kolhapur": 530,
    "Aurangabad": 515,  # Chhatrapati Sambhajinagar (renamed in 2023; Census 2011 name is Aurangabad)
    "Mumbai": 519,  # includes the Vashi (New Mumbai) APMC, which Agmarknet lists under Mumbai
}
# Districts used per crop (chosen from the data-quality report of 2014-2024):
# * tomato leaves out Mumbai (only 2-12 days with a price per year in 2015-2019);
# * potato leaves out Kolhapur (25% of days covered, no data from mid-2018 to mid-2021).
# Thane and Mumbai Suburban were checked too and left out (patchy / empty).
CROP_DISTRICTS = {
    "Tomato": ["Pune", "Solapur", "Kolhapur", "Aurangabad"],
    "Potato": ["Pune", "Solapur", "Aurangabad", "Mumbai"],
}
MAX_ATTEMPTS = 5  # tries per request before giving up
PAUSE_SECONDS = 1.0  # pause between requests, to be polite to the server
FIELDS = ["date", "commodity", "state", "district", "district_id",
          "price_min", "price_max", "price_modal", "arrivals_tonnes"]
OUT_DIR = ROOT / "data" / "raw" / "ceda"


def post_json(path: str, body: dict) -> dict:
    """Send `body` as JSON to BASE_URL + path and return the JSON answer. Retries a few times on errors."""
    url = BASE_URL + path
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, method="POST",
                                     headers={"Content-type": "application/json", "User-Agent": "Mozilla/5.0"})

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            reason = f"HTTP {error.code}"
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            reason = type(error).__name__
        # Wait a bit longer after each failed attempt (30, 60, 90, ... seconds).
        wait_seconds = 30 * attempt
        logger.warning("%s: attempt %d failed (%s), waiting %d s", path, attempt, reason, wait_seconds)
        time.sleep(wait_seconds)
    raise RuntimeError(f"giving up on {path} {body}")


def fetch_district(commodity: str, district: str) -> list[dict]:
    """Daily prices and arrivals of one commodity in one district, joined by date, oldest day first."""
    body = {
        "state_id": STATE_ID,
        "commodity_id": COMMODITY_IDS[commodity],
        "district_id": DISTRICT_IDS[district],
        "calculation_type": "d",  # "d" = daily values
        "start_date": START_DATE,
        "end_date": END_DATE,
    }
    price_rows = post_json("/api/prices", body).get("data", [])
    time.sleep(PAUSE_SECONDS)
    arrival_rows = post_json("/api/quantities", body).get("data", [])
    time.sleep(PAUSE_SECONDS)

    # One output row per day. Start with the price days ...
    rows_by_date = {}
    for price in price_rows:
        rows_by_date[price["t"]] = {
            "date": price["t"],
            "commodity": commodity,
            "state": "Maharashtra",
            "district": district,
            "district_id": DISTRICT_IDS[district],
            "price_min": price["p_min"],
            "price_max": price["p_max"],
            "price_modal": price["p_modal"],
            "arrivals_tonnes": "",
        }

    # ... then add the arrivals (a day can have arrivals without prices, or prices without arrivals).
    for arrival in arrival_rows:
        day = arrival["t"]
        if day not in rows_by_date:
            rows_by_date[day] = {
                "date": day, "commodity": commodity, "state": "Maharashtra", "district": district,
                "district_id": DISTRICT_IDS[district], "price_min": "", "price_max": "", "price_modal": "",
            }
        rows_by_date[day]["arrivals_tonnes"] = arrival["qty"]

    # Oldest day first (dates are written year-month-day, so sorting the text sorts by date).
    rows = []
    for day in sorted(rows_by_date):
        rows.append(rows_by_date[day])
    logger.info("%s %s: %d price days, %d arrival days, %d days in total",
                commodity, district, len(price_rows), len(arrival_rows), len(rows))
    return rows


def save_csv(rows: list[dict], path: Path) -> None:
    """Write the rows to a CSV file with the columns in FIELDS."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    commodities = sys.argv[1:]
    if not commodities:
        commodities = ["Tomato", "Potato"]

    for commodity in commodities:
        for district in CROP_DISTRICTS[commodity]:
            file_name = f"{commodity.lower()}_{district.replace(' ', '_')}.csv"
            path = OUT_DIR / file_name

            # Skip files that were downloaded before.
            if path.exists():
                logger.info("%s %s: already downloaded, skipping (%s)", commodity, district, file_name)
                continue

            rows = fetch_district(commodity, district)
            save_csv(rows, path)


if __name__ == "__main__":
    main()
