"""Tests for the crop setting (onion / tomato / potato) and the CEDA data loader."""
from __future__ import annotations

import pandas as pd
import pytest
import yaml

from mandi import ingest
from mandi.config import ROOT, _merge, crop_of, load_config, results_dir


def _onion_part_of_file() -> dict:
    """The config file as written, without the `crops` section (= the onion settings)."""
    with open(ROOT / "configs" / "config.yaml", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    raw.pop("crops")
    return raw


def test_onion_settings_are_unchanged(monkeypatch):
    # No MANDI_CROP: exactly the onion settings, with nothing added.
    monkeypatch.delenv("MANDI_CROP", raising=False)
    cfg = load_config()
    assert cfg == _onion_part_of_file()
    assert crop_of(cfg) == "onion"
    assert results_dir(cfg) == ROOT / "results"


def test_crop_from_environment_variable(monkeypatch):
    monkeypatch.setenv("MANDI_CROP", "Tomato ")  # case and spaces do not matter
    cfg = load_config()
    assert crop_of(cfg) == "tomato"
    assert results_dir(cfg) == ROOT / "results" / "tomato"


@pytest.mark.parametrize("crop", ["tomato", "potato"])
def test_crop_changes_only_its_own_settings(crop):
    onion = _onion_part_of_file()
    cfg = load_config(crop=crop)

    # Settings the crop does not list stay exactly as for onion.
    assert cfg["cleaning"] == onion["cleaning"]
    assert cfg["backtest"] == onion["backtest"]
    assert cfg["forecast"]["models"] == onion["forecast"]["models"]
    assert cfg["forecast"]["features"] == onion["forecast"]["features"]
    assert cfg["forecast"]["tuning_folds"] == onion["forecast"]["tuning_folds"]

    # Settings the crop lists are its own: no onion file, folder or selection is shared.
    assert cfg["data"]["source"] == "ceda"
    assert cfg["paths"]["results"] == f"results/{crop}"
    assert cfg["forecast"]["selected_models_file"] == f"configs/selected_models_{crop}.yaml"
    assert cfg["forecast"]["runs_dir"] == f"results/{crop}/runs"
    assert cfg["corrections"] == []
    assert len(cfg["markets"]) == 4


def test_unknown_crop_is_refused():
    with pytest.raises(ValueError, match="Unknown crop"):
        load_config(crop="carrot")


def test_merge_replaces_lists_merges_groups_and_keeps_inputs():
    base = {"group": {"a": 1, "b": [1, 2]}, "plain": 5}
    changes = {"group": {"b": [9]}, "plain": 6}
    merged = _merge(base, changes)
    assert merged == {"group": {"a": 1, "b": [9]}, "plain": 6}
    # The inputs are not changed.
    assert base == {"group": {"a": 1, "b": [1, 2]}, "plain": 5}


def test_district_of_market_key():
    assert ingest.district_of("Maharashtra_Pune_Pune district") == "Pune"
    assert ingest.district_of("Maharashtra_Aurangabad_Aurangabad district") == "Aurangabad"


def _write_ceda_file(folder, commodity, district, rows):
    """Write a small CEDA-style CSV (the format of mandi.fetch_ceda)."""
    table = pd.DataFrame(rows, columns=["date", "price_min", "price_max", "price_modal", "arrivals_tonnes"])
    table["commodity"] = commodity
    table["state"] = "Maharashtra"
    table["district"] = district
    table["district_id"] = 0
    table.to_csv(folder / f"{commodity.lower()}_{district}.csv", index=False)


def test_load_ceda_rows_gives_the_onion_columns(tmp_path):
    cfg = load_config(crop="tomato")
    cfg["paths"]["raw_dir"] = str(tmp_path)  # an absolute path: ROOT / absolute path = that path
    cfg["markets"] = ["Maharashtra_Pune_Pune district", "Maharashtra_Solapur_Solapur district"]
    _write_ceda_file(tmp_path, "Tomato", "Pune", [["2020-01-01", 500, 900, 700, 120.5],
                                                  ["2020-01-02", None, None, None, 80.0]])  # arrivals but no price
    _write_ceda_file(tmp_path, "Tomato", "Solapur", [["2020-01-01", 400, 800, 600, None]])  # price but no arrivals

    rows = ingest.load_rows(cfg)

    # Same columns as the onion file (the ones the cleaning step reads).
    assert list(rows.columns) == cfg["data"]["keep_columns"]
    assert len(rows) == 3
    pune = rows[rows["market_key"] == "Maharashtra_Pune_Pune district"]
    assert list(pune["district"]) == ["Pune", "Pune"]
    assert list(pune["market"]) == ["Pune district", "Pune district"]
    assert pune["modal_price_rs_qtl"].iloc[0] == 700
    assert pune["arrivals_tonnes"].iloc[0] == 120.5
    # A day without a price keeps a missing price (the cleaning step drops it and logs it).
    assert pd.isna(pune["modal_price_rs_qtl"].iloc[1])
    assert rows["date"].dtype.kind == "M"  # real dates, not text


def test_load_ceda_rows_says_how_to_download_missing_files(tmp_path):
    cfg = load_config(crop="potato")
    cfg["paths"]["raw_dir"] = str(tmp_path)
    with pytest.raises(FileNotFoundError, match="mandi.fetch_ceda Potato"):
        ingest.load_rows(cfg)


def test_empty_calendar_covers_the_study_period():
    cfg = load_config(crop="tomato")
    calendar, n_conflict = ingest.empty_calendar(cfg)
    assert n_conflict == 0
    assert list(calendar.columns) == ["date"]
    assert calendar["date"].iloc[0] == pd.Timestamp(cfg["data"]["start_date"])
    assert calendar["date"].iloc[-1] == pd.Timestamp(cfg["data"]["end_date"])
    assert len(calendar) == 4018  # 2014-01-01 to 2024-12-31
