import pytest

from mandi.config import ROOT, load_config


def test_config_loads_and_has_required_sections():
    cfg = load_config()
    for key in ("seed", "paths", "data", "markets", "cleaning", "backtest"):
        assert key in cfg
    assert cfg["cleaning"]["max_ffill_gap_days"] == 3


@pytest.mark.local_data("raw")
def test_raw_data_is_where_config_says():
    assert (ROOT / load_config()["paths"]["raw_file"]).exists()
