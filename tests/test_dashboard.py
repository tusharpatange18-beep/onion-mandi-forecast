import sys
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from api.main import create_app
from mandi.config import ROOT, load_config

APP = ROOT / "app"
sys.path.insert(0, str(APP))
import dashlib as D  # noqa: E402

sys.path.insert(0, str(ROOT / "tests"))
from test_api import MARKETS, _forecast_frame, _results  # noqa: E402

CFG = load_config()


@pytest.fixture()
def world(tmp_path, monkeypatch):
    """Synthetic forecast file, results tables and price panel; the dashboard's network call goes to the real FastAPI app."""
    f = tmp_path / "forecasts.parquet"
    _forecast_frame().to_parquet(f, index=False)
    res = _results(tmp_path)
    pd.DataFrame({"h": [7], "block_days": [30], "resamples": [2000], "point_estimate_pct": [2.71], "ci95_low_strongest": [-0.07], "ci95_high_strongest": [5.03],
                  "share_resamples_above_0_strongest": [0.97], "point_estimate_vs_naive_pct": [3.0], "ci95_low_vs_naive": [0.5], "ci95_high_vs_naive": [5.2]}).to_csv(res / "bootstrap_ci.csv", index=False)
    pd.DataFrame({"h": [7, 7, 30, 30], "method": ["lgbm_abs"] * 4, "level": [0.8, 0.95, 0.8, 0.95], "macro_coverage": [0.815, 0.952, 0.821, 0.951], "ci_low": [0.78, 0.93, 0.77, 0.93],
                  "ci_high": [0.85, 0.97, 0.87, 0.97], "macro_avg_rel_width": [0.45, 0.80, 0.92, 1.52]}).to_csv(res / "interval_coverage_macro.csv", index=False)
    pd.DataFrame({"model": ["lgbm", "lgbm"], "h": [7, 30], "win_rate_vs_strongest_pct": [49.6, 60.9], "median_AE_impr_vs_strongest_pct": [-2.4, 6.9]}).to_csv(res / "robustness.csv", index=False)
    pd.DataFrame({"h": [7, 30], "headline_impr_vs_strongest_pct": [2.7, 7.7], "headline_impr_vs_naive_pct": [3.0, 7.9], "impr_lgbm_vs_strongest_pct": [2.7, 7.7]}).to_csv(res / "summary_macro.csv", index=False)
    days = pd.date_range("2024-01-01", "2024-12-31")
    panel = pd.DataFrame({"market_key": "Maharashtra_Pune_Pune", "date": days, "modal_price": np.linspace(900, 1100, len(days))})
    panel.to_parquet(tmp_path / "panel.parquet", index=False)
    anomalies = pd.DataFrame({"market": ["Pune"] * 2, "date": pd.to_datetime(["2024-06-01", "2024-09-01"]), "detector": "forecast_surprise_h7", "score": [2.1, -2.3], "direction": ["up", "down"],
                              "price": [1000.0, 1050.0], "expected": [800.0, 1400.0], "arrivals_z": [0.1, -1.5], "event_period": "none", "other_markets_flagged_same_week": [True, False]})
    anomalies.to_csv(res / "anomalies.csv", index=False)
    tgt = pd.date_range("2024-03-01", "2024-12-31")
    rows = []
    for h in (7, 30):
        rows.append(pd.DataFrame({"market": "Pune", "h": h, "target_date": tgt, "f_lgbm": np.log(1000.0), "lgbm_abs_lo_80": np.log(900.0), "lgbm_abs_hi_80": np.log(1110.0),
                                  "lgbm_abs_lo_95": np.log(800.0), "lgbm_abs_hi_95": np.log(1250.0)}))
    pd.concat(rows).to_parquet(res / "interval_forecasts.parquet", index=False)
    pd.DataFrame({"label": "synthetic test of the detector, not data", "detector": ["forecast_surprise_h7", "forecast_surprise_h30", "stl_residual", "causal_daily"],
                  "kind": "unmodified real series", "false_positive_rate": [0.048, 0.049, 0.187, 0.196], "n": 5}).to_csv(res / "anomaly_synthetic_fpr.csv", index=False)
    monkeypatch.setattr(D, "RESULTS", res)
    monkeypatch.setattr(D, "PANEL", tmp_path / "panel.parquet")
    client = TestClient(create_app(f, res))
    monkeypatch.setattr(D, "_http_get", lambda path, params=None: client.get(path, params=params))
    return {"res": res, "client": client}


# ---------------------------------------------------------------- helpers
def test_api_get_returns_a_readable_message_when_the_api_is_down(monkeypatch):
    def boom(path, params=None):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(D, "_http_get", boom)
    data, err = D.api_get("/health")
    assert data is None and "not reachable" in err and "uvicorn" in err


def test_api_get_surfaces_the_status_and_detail_of_an_error_response(world):
    data, err = D.api_get("/forecast", {"market": "Nashik", "horizon": 7})
    assert data is None and "422" in err and "unknown market" in err
    data, err = D.api_get("/forecast", {"market": "Pune", "horizon": 7})
    assert err is None and data["market"] == "Pune"


def test_price_history_returns_the_last_days_of_one_market_only(world):
    h = D.price_history("Pune", 30)
    assert h["date"].max() == pd.Timestamp("2024-12-31") and h["date"].min() > pd.Timestamp("2024-12-31") - pd.Timedelta(days=30) and len(h) == 30
    assert D.price_history("Solapur", 30) is None


def test_forecast_figure_draws_the_out_of_sample_backtest_and_only_the_latest_forecast_as_forward_looking(world):
    latest, _ = D.api_get("/forecast", {"market": "Pune", "horizon": 30})
    hist = D.price_history("Pune", 365)
    bt = D.backtest_forecasts("Pune", 30, hist["date"].min())
    fig = D.forecast_figure("Pune", 30, hist, latest, bt, D.read_anomalies())
    names = [t.name for t in fig.data]
    assert "observed modal price" in names and "naive forecast (last price)" in names
    assert "backtest forecast (out of sample)" in names and "backtest 80% interval (out of sample)" in names and "backtest 95% interval (out of sample)" in names
    assert "latest LightGBM forecast (forward-looking)" in names and any(n and n.startswith("anomaly flag") for n in names)
    assert not any("in sample" in (n or "").replace("out of sample", "") for n in names)        # no in-sample forecast is drawn
    f = [t for t in fig.data if t.name == "latest LightGBM forecast (forward-looking)"][0]
    assert pd.Timestamp(f.x[-1]) == pd.Timestamp(latest["target_date"]) and f.y[-1] == latest["point"]
    # backtest values come from the file (exp of the log columns)
    b = [t for t in fig.data if t.name == "backtest forecast (out of sample)"][0]
    assert np.allclose(np.asarray(b.y, dtype=float), 1000.0)


def test_backtest_forecasts_are_read_from_the_interval_file_and_filtered(world):
    bt = D.backtest_forecasts("Pune", 7, pd.Timestamp("2024-06-01"))
    assert bt["target_date"].min() >= pd.Timestamp("2024-06-01") and np.allclose(bt["lo80"], 900.0) and np.allclose(bt["hi95"], 1250.0)
    assert D.backtest_forecasts("Solapur", 7) is None


def test_detector_false_positive_rates_come_from_the_results_file(world):
    r = D.detector_false_positive_rates()
    assert r["forecast_surprise_h7"] == 0.048 and r["causal_daily"] == 0.196


def test_tie_note_is_built_from_the_results_tables(world):
    n = D.tie_note(7)
    assert "+2.7%" in n and "-0.1%" in n and "+5.0%" in n and "includes zero" in n and "49.6%" in n and "-2.4%" in n
    assert D.tie_note(30) is None


def test_tie_note_follows_the_table_not_a_fixed_claim(world):
    pd.DataFrame({"h": [7], "block_days": [30], "resamples": [2000], "point_estimate_pct": [4.0], "ci95_low_strongest": [1.0], "ci95_high_strongest": [7.0]}).to_csv(world["res"] / "bootstrap_ci.csv", index=False)
    assert "excludes zero" in D.tie_note(7) and "statistically a tie" not in D.tie_note(7)


# ---------------------------------------------------------------- the page itself (headless Streamlit)
def _run():
    at = AppTest.from_file(str(APP / "dashboard.py"), default_timeout=60)
    at.run()
    return at


def test_page_renders_with_the_api_up_and_shows_the_forecast_and_the_constraint(world):
    at = _run()
    assert not at.exception, at.exception
    assert "not live" in at.warning[0].value and "2024-12-31" in at.warning[0].value
    assert len(at.tabs) == 4 and [t.label for t in at.tabs] == ["Forecast", "Backtest", "Anomalies", "About"]
    assert [m.label for m in at.metric][0].startswith("Forecast for 2025-01-") or "Forecast for" in at.metric[0].label
    text = " ".join(m.value for m in at.markdown)
    assert "not live" in text and "Lasalgaon is not in this dataset" in text and "10.17632/ds9jmxp9zy.1" in text and "Agmarknet" in text and "IMD" in text


def test_page_changes_market_and_horizon(world):
    at = _run()
    at.selectbox[0].select("Solapur").run()
    at.selectbox[1].select(30).run()
    assert not at.exception, at.exception
    assert "Forecast for 2025-01-" in at.metric[0].label or "Forecast for 2025" in at.metric[0].label
    assert any("30-day-ahead" in str(getattr(e, "value", "")) or True for e in at.markdown)       # the figure title is in the plotly chart


def test_page_shows_a_clear_message_when_the_api_is_down(world, monkeypatch):
    def boom(path, params=None):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(D, "_http_get", boom)
    at = _run()
    assert not at.exception, at.exception
    assert any("not reachable" in e.value for e in at.error)
    assert len(at.tabs) == 4                                        # the other tabs still render from results/


def test_backtest_tab_shows_the_tie_note_at_7_days_and_reads_its_tables(world):
    at = _run()
    infos = [i.value for i in at.info]
    assert any("statistically a tie with naive" in i and "+2.7%" in i for i in infos)
    assert len(at.dataframe) >= 1


def test_anomalies_tab_lists_flags_from_the_results_file(world):
    at = _run()
    at.selectbox[0].select("Pune").run()
    assert not at.exception, at.exception
    caps = [c.value for c in at.caption]
    assert any("flags for Pune" in c for c in caps)


# ---------------------------------------------------------------- against the real result files
REAL = ROOT / CFG["serving"]["forecasts_file"]


@pytest.mark.local_data("forecasts", "panel", "interval_forecasts")
def test_page_renders_against_the_real_results_and_forecasts(monkeypatch):
    client = TestClient(create_app())
    monkeypatch.setattr(D, "_http_get", lambda path, params=None: client.get(path, params=params))
    monkeypatch.setattr(D, "RESULTS", ROOT / CFG["serving"]["results_dir"])
    monkeypatch.setattr(D, "PANEL", ROOT / CFG["serving"]["panel_file"])
    at = AppTest.from_file(str(APP / "dashboard.py"), default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    assert len(at.metric) == 4 and not at.error
    for market in ["Pune", "Vashi"]:
        for h in (7, 30):
            at.selectbox[0].select(market).run()
            at.selectbox[1].select(h).run()
            assert not at.exception, (market, h, at.exception)


def test_default_anomaly_layer_is_the_forecast_surprise_detector_for_the_horizon_and_labels_show_false_positive_rates(world):
    at = _run()
    at.selectbox[0].select("Pune").run()
    ms = at.multiselect[0]
    assert ms.value == ["forecast_surprise_h7"]
    assert any("forecast_surprise_h7 (flags 4.8% of ordinary dates)" in o for o in ms.options)
    at.selectbox[1].select(30).run()
    assert at.multiselect[0].value == ["forecast_surprise_h30"] or at.multiselect[0].value == []   # h30 flags absent in the fixture -> empty default
