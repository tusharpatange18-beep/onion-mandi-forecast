"""Shared test configuration.

Marker `local_data(*names)`: the test needs data or result files that are NOT in git (raw data, the processed panel, the backtest
forecasts, the saved tuning run, the serving forecasts). When any of them is absent (for example in CI, or on a fresh clone) the
test is skipped with a reason that lists exactly which files are missing. Run `pytest -rs` to see the list.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mandi.config import ROOT, load_config

_CFG = load_config()


def _tuning_run_dir() -> Path:
    sel = yaml.safe_load((ROOT / _CFG["forecast"]["selected_models_file"]).read_text(encoding="utf-8"))
    return Path(_CFG["forecast"]["runs_dir"]) / sel["tuning_run"]


LOCAL_FILES: dict[str, Path] = {
    "raw": Path(_CFG["paths"]["raw_file"]),                         # data/raw/... (downloaded by hand, never in git)
    "panel": Path(_CFG["paths"]["panel"]),                          # data/processed/panel.parquet (python -m mandi.run_phase1)
    "fair": Path("results/fair_forecasts.parquet"),                  # final backtest forecasts (python -m mandi.run_phase3 final)
    "tuning_run": _tuning_run_dir() / "forecasts.parquet",          # saved tuning-run forecasts (python -m mandi.run_phase3 tune)
    "interval_forecasts": Path("results/interval_forecasts.parquet"),  # python -m mandi.intervals
    "forecasts": Path(_CFG["serving"]["forecasts_file"]),           # python -m mandi.forecast
}


def pytest_runtest_setup(item: pytest.Item) -> None:
    m = item.get_closest_marker("local_data")
    if m is None:
        return
    unknown = [n for n in m.args if n not in LOCAL_FILES]
    if unknown:
        raise ValueError(f"unknown local_data name(s) {unknown}; known: {sorted(LOCAL_FILES)}")
    missing = [str(LOCAL_FILES[n]).replace("\\", "/") for n in m.args if not (ROOT / LOCAL_FILES[n]).exists()]
    if missing:
        pytest.skip("local data not in git, absent here: " + ", ".join(missing))
