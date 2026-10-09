"""The committed CI fixture is synthetic, reproducible from its script, and served by the API."""
import importlib.util

import pandas as pd
from fastapi.testclient import TestClient

from api.main import create_app
from mandi.config import ROOT

FIX = ROOT / "tests" / "fixtures"


def _make():
    spec = importlib.util.spec_from_file_location("make_fixture_forecasts", FIX / "make_fixture_forecasts.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.make()


def test_committed_fixture_equals_what_its_script_produces_and_is_labelled_synthetic():
    committed = pd.read_parquet(FIX / "forecasts_fixture.parquet")
    pd.testing.assert_frame_equal(committed, _make(), check_dtype=False)
    assert committed["synthetic_test_data"].str.contains("SYNTHETIC TEST DATA ONLY").all()
    assert (committed["model_spec"] == "SYNTHETIC-FIXTURE").all()


def test_api_serves_the_fixture_as_the_container_check_does(tmp_path):
    c = TestClient(create_app(FIX / "forecasts_fixture.parquet", tmp_path))     # no result tables, as in the CI container
    h = c.get("/health").json()
    assert h["status"] == "ok" and h["forecast_file_exists"] is True
    for m in ["Solapur", "Pune", "Vashi", "Kolhapur", "Sambhajinagar"]:
        for hz in (7, 30):
            r = c.get("/forecast", params={"market": m, "horizon": hz})
            assert r.status_code == 200 and r.json()["model_spec"] == "SYNTHETIC-FIXTURE"
