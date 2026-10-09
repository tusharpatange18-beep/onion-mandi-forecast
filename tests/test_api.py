import os
import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import ForecastResponse, create_app
from mandi.config import ROOT, load_config

CFG = load_config()
MARKETS = ["Solapur", "Pune", "Vashi", "Kolhapur", "Sambhajinagar"]


def _forecast_frame(point_scale: float = 1.0, drop: tuple[str, int] | None = None, gap: float = 1.0, origin_back: int = 0, in_sample_latest: bool = False) -> pd.DataFrame:
    rows = []
    for m in MARKETS:
        for h in (7, 30):
            if drop == (m, h):
                continue
            for k in range(3):                                              # k = 2 is the latest origin
                origin = pd.Timestamp("2024-12-31") - pd.Timedelta(days=2 - k + (origin_back if k == 2 else 0))
                p = 1000.0 * point_scale
                rows.append({"market": m, "horizon": h, "origin_date": origin, "target_date": origin + pd.Timedelta(days=h), "last_price": 1000.0, "point": p,
                             "lo80": p * 0.8, "hi80": p * 1.2, "lo95": p * 0.6, "hi95": p * 1.5, "naive": 1000.0, "n_calibration": 200,
                             "model_spec": f"lgbm|test{h}", "trained_through": pd.Timestamp("2024-12-31"), "generated_at": (pd.Timestamp.now() - pd.Timedelta(days=2)).isoformat(timespec="seconds"),
                             "git_commit": "abc1234", "as_of": pd.Timestamp("2025-01-01"), "data_through": pd.Timestamp("2024-12-31"), "is_latest": k == 2,
                             "in_sample": (k < 2) or in_sample_latest, "actual_price": 990.0 if k < 2 else np.nan, "gap_before_origin_days": gap if k == 2 else 1.0})
    return pd.DataFrame(rows)


def _results(tmp_path):
    d = tmp_path / "results"
    d.mkdir(exist_ok=True)
    pd.DataFrame({"h": [7, 30], "point_estimate_pct": [2.71, 7.74], "ci95_low_strongest": [-0.07, -1.84], "ci95_high_strongest": [5.03, 14.05]}).to_csv(d / "bootstrap_ci.csv", index=False)
    pd.DataFrame({"h": [7, 30], "method": ["lgbm_abs"] * 2, "level": [0.95, 0.95], "macro_avg_rel_width": [0.803, 1.516]}).to_csv(d / "interval_coverage_macro.csv", index=False)
    pd.DataFrame({"group_type": ["event_group"], "group": ["export_ban"], "h": [30], "method": ["lgbm_abs"], "level": [0.8], "coverage": [0.723]}).to_csv(d / "interval_breakdowns.csv", index=False)
    pd.DataFrame({"split_by": ["gap_before_origin"] * 2, "h": [7, 30], "segment": ["7+ days"] * 2, "n": [25, 21], "log_impr_vs_strongest_pct": [-29.9, -13.8]}).to_csv(d / "error_analysis.csv", index=False)
    return d


@pytest.fixture()
def env(tmp_path):
    f = tmp_path / "forecasts.parquet"
    _forecast_frame().to_parquet(f, index=False)
    res = _results(tmp_path)
    return f, res, TestClient(create_app(f, res))


# ---------------------------------------------------------------- /health and /markets
def test_health_reports_the_file_the_dates_the_age_and_the_version(env):
    f, res, c = env
    r = c.get("/health")
    assert r.status_code == 200
    j = r.json()
    assert j["status"] == "ok" and j["forecast_file_exists"] is True
    assert j["data_through"] == "2024-12-31" and j["as_of"] == "2025-01-01"
    assert j["api_version"] == CFG["serving"]["api"]["version"] and 1.99 < j["age_days"] < 2.1 and "not live" in j["note"]


def test_health_is_degraded_but_200_when_the_file_is_missing(tmp_path):
    c = TestClient(create_app(tmp_path / "missing.parquet", _results(tmp_path)))
    r = c.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "degraded" and r.json()["forecast_file_exists"] is False and r.json()["data_through"] is None


def test_markets_lists_the_five_configured_markets_with_their_latest_origin(env):
    _, _, c = env
    j = c.get("/markets").json()["markets"]
    assert [m["market"] for m in j] == MARKETS
    assert all(m["horizons"] == [7, 30] and m["latest_origin_date"] == "2024-12-31" for m in j)


# ---------------------------------------------------------------- /forecast
def test_forecast_response_has_every_field_of_the_schema_and_validates(env):
    _, _, c = env
    r = c.get("/forecast", params={"market": "Pune", "horizon": 7})
    assert r.status_code == 200
    j = r.json()
    assert set(j) == set(ForecastResponse.model_fields)
    ForecastResponse.model_validate(j)
    for k in ["market", "horizon", "origin_date", "target_date", "last_price", "point", "lo80", "hi80", "lo95", "hi95", "naive", "n_calibration", "model_spec",
              "trained_through", "generated_at", "git_commit", "caveats"]:
        assert k in j
    assert j["lo95"] < j["lo80"] < j["point"] < j["hi80"] < j["hi95"] and j["target_date"] == "2025-01-07" and j["origin_date"] == "2024-12-31"


@pytest.mark.parametrize("name", ["pune", "PUNE", " Pune ", "Pune"])
def test_market_is_case_insensitive(env, name):
    _, _, c = env
    r = c.get("/forecast", params={"market": name, "horizon": 30})
    assert r.status_code == 200 and r.json()["market"] == "Pune"


def test_latest_row_is_returned_not_the_history(env):
    _, _, c = env
    assert c.get("/forecast", params={"market": "Solapur", "horizon": 7}).json()["origin_date"] == "2024-12-31"


@pytest.mark.parametrize("params", [{"market": "Nashik", "horizon": 7}, {"market": "Pune", "horizon": 14}, {"market": "Pune", "horizon": 0},
                                    {"market": "Pune", "horizon": "abc"}, {"market": "Pune"}, {"horizon": 7}, {}])
def test_invalid_input_gives_422(env, params):
    _, _, c = env
    r = c.get("/forecast", params=params)
    assert r.status_code == 422


def test_422_message_lists_the_allowed_markets_and_horizons(env):
    _, _, c = env
    assert all(m in c.get("/forecast", params={"market": "x", "horizon": 7}).json()["detail"] for m in MARKETS)
    assert "[7, 30]" in c.get("/forecast", params={"market": "Pune", "horizon": 9}).json()["detail"]


def test_404_when_a_valid_market_and_horizon_have_no_forecast(tmp_path):
    f = tmp_path / "forecasts.parquet"
    _forecast_frame(drop=("Vashi", 30)).to_parquet(f, index=False)
    c = TestClient(create_app(f, _results(tmp_path)))
    assert c.get("/forecast", params={"market": "Vashi", "horizon": 30}).status_code == 404
    assert c.get("/forecast", params={"market": "Vashi", "horizon": 7}).status_code == 200


def test_503_when_the_forecast_file_is_missing(tmp_path):
    c = TestClient(create_app(tmp_path / "nope.parquet", _results(tmp_path)))
    r = c.get("/forecast", params={"market": "Pune", "horizon": 7})
    assert r.status_code == 503 and "mandi.forecast" in r.json()["detail"]
    assert c.get("/forecast/history", params={"market": "Pune", "horizon": 7}).status_code == 503
    assert c.get("/markets").status_code == 200                     # the market list does not need the file


def test_file_is_reloaded_when_its_mtime_changes(env):
    f, res, c = env
    assert c.get("/forecast", params={"market": "Pune", "horizon": 7}).json()["point"] == 1000.0
    _forecast_frame(point_scale=2.0).to_parquet(f, index=False)
    new = time.time() + 5
    os.utime(f, (new, new))
    assert c.get("/forecast", params={"market": "Pune", "horizon": 7}).json()["point"] == 2000.0


# ---------------------------------------------------------------- caveats
def test_caveats_always_say_the_data_ends_and_the_forecast_is_not_live(env):
    _, _, c = env
    for h in (7, 30):
        cav = c.get("/forecast", params={"market": "Pune", "horizon": h}).json()["caveats"]
        assert "2024-12-31" in cav[0] and "not live" in cav[0] and "2025-01-01" in cav[0]


def test_horizon_7_caveat_reads_its_numbers_from_the_results_table(env):
    _, res, c = env
    cav = " ".join(c.get("/forecast", params={"market": "Pune", "horizon": 7}).json()["caveats"])
    b = pd.read_csv(res / "bootstrap_ci.csv").set_index("h").loc[7]
    assert f"{b['point_estimate_pct']:+.1f}%" in cav and f"{b['ci95_low_strongest']:+.1f}%" in cav and f"{b['ci95_high_strongest']:+.1f}%" in cav
    assert "statistically a tie with naive" in cav                    # the interval includes zero in the table
    pd.DataFrame({"h": [7], "point_estimate_pct": [4.0], "ci95_low_strongest": [1.0], "ci95_high_strongest": [7.0]}).to_csv(res / "bootstrap_ci.csv", index=False)
    new = time.time() + 5
    os.utime(res / "bootstrap_ci.csv", (new, new))
    cav2 = " ".join(c.get("/forecast", params={"market": "Pune", "horizon": 7}).json()["caveats"])
    assert "statistically a tie" not in cav2 and "excludes zero" in cav2 and "+4.0%" in cav2        # the wording follows the table, not a fixed claim


def test_horizon_30_caveat_reads_interval_width_and_shock_coverage_from_the_results(env):
    _, res, c = env
    cav = " ".join(c.get("/forecast", params={"market": "Pune", "horizon": 30}).json()["caveats"])
    assert "152%" in cav and "72%" in cav and "export-ban" in cav


def test_long_gap_and_stale_quote_and_in_sample_caveats(tmp_path):
    f = tmp_path / "f.parquet"
    _forecast_frame(gap=10.0, origin_back=9, in_sample_latest=True).to_parquet(f, index=False)
    c = TestClient(create_app(f, _results(tmp_path)))
    cav = " ".join(c.get("/forecast", params={"market": "Pune", "horizon": 7}).json()["caveats"])
    assert "10 days earlier" in cav and "-30%" in cav and "25 forecasts" in cav                     # gap caveat with the table's numbers
    assert "9 days older than the end of the dataset" in cav
    assert "not an out-of-sample test" in cav


def test_caveats_degrade_gracefully_when_the_result_tables_are_missing(tmp_path):
    f = tmp_path / "f.parquet"
    _forecast_frame().to_parquet(f, index=False)
    c = TestClient(create_app(f, tmp_path / "empty"))
    r = c.get("/forecast", params={"market": "Pune", "horizon": 7})
    assert r.status_code == 200 and len(r.json()["caveats"]) >= 2 and "not live" in r.json()["caveats"][0]


# ---------------------------------------------------------------- /forecast/history
def test_history_returns_the_rows_in_date_order_with_in_sample_flags_and_nulls(env):
    _, _, c = env
    j = c.get("/forecast/history", params={"market": "Pune", "horizon": 7}).json()
    assert j["market"] == "Pune" and len(j["rows"]) == 3 and "in_sample" in j["note"]
    dates = [r["origin_date"] for r in j["rows"]]
    assert dates == sorted(dates)
    assert [r["in_sample"] for r in j["rows"]] == [True, True, False] and j["rows"][-1]["actual_price"] is None
    assert c.get("/forecast/history", params={"market": "x", "horizon": 7}).status_code == 422


# ---------------------------------------------------------------- against the real file
real = ROOT / CFG["serving"]["forecasts_file"]


@pytest.mark.skipif(not real.exists(), reason="results/forecasts.parquet not generated")
def test_real_forecast_file_serves_every_market_and_horizon():
    c = TestClient(create_app())
    for m in MARKETS:
        for h in (7, 30):
            r = c.get("/forecast", params={"market": m, "horizon": h})
            assert r.status_code == 200, (m, h, r.text)
            j = r.json()
            assert j["lo95"] < j["lo80"] < j["point"] < j["hi80"] < j["hi95"] and j["horizon"] == h and j["market"] == m
    assert c.get("/health").json()["status"] == "ok"
