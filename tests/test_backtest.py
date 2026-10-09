import numpy as np
import pandas as pd
import pytest

from mandi import backtest as B
from mandi import metrics as MET
from mandi import models as M


def test_blocks_partition_the_window_without_overlap():
    refits = B.refit_schedule("2020-01-01", "2020-12-31")
    blocks = B._blocks(refits, pd.Timestamp("2020-12-31"))
    assert len(blocks) == 12
    assert blocks[0][0] == pd.Timestamp("2020-01-01") and blocks[-1][1] == pd.Timestamp("2020-12-31")
    for (r1, e1), (r2, e2) in zip(blocks[:-1], blocks[1:]):
        assert e1 + pd.Timedelta(days=1) == r2


def test_lightgbm_training_rows_only_use_targets_known_before_the_refit_date(monkeypatch):
    seen = []

    def fake(train, test, features, params, seed):
        seen.append((train["target_date"].max(), test["origin"].min(), test["origin"].max()))
        return np.zeros(len(test))

    monkeypatch.setattr(M, "lgbm_fit_predict", fake)
    rng = np.random.default_rng(0)
    origins = pd.date_range("2019-01-01", "2020-06-30")
    tab = pd.DataFrame({"market_idx": rng.integers(0, 5, 5 * len(origins)), "origin": np.tile(origins, 5), "p0": 7.0})
    tab["target_date"] = tab["origin"] + pd.Timedelta(days=30)
    tab["y_log"], tab["y_change"] = 7.0, 0.0
    blocks = B._blocks(B.refit_schedule("2020-01-01", "2020-06-30"), pd.Timestamp("2020-06-30"))
    B._task_lgbm((B.Spec("lgbm", "x", ()), 30, tab, blocks, 1))
    assert len(seen) == 6
    for (max_target, o_min, o_max), (r, e) in zip(seen, blocks):
        assert max_target < r             # no target from the future of the refit date was used for training
        assert o_min >= r and o_max <= e  # and the model only forecasts origins of its own block


def test_fairness_filter_keeps_only_triples_every_model_can_forecast():
    t = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0], "b": [1.0, np.nan, 3.0, 4.0]})
    kept, rep = B.fairness_filter(t, ["a", "b"])
    assert len(kept) == 2
    assert rep.set_index("column")["cannot_forecast"].to_dict() == {"a": 1, "b": 1}
    assert (rep["dropped_for_all_models_total"] == 2).all()


@pytest.mark.parametrize("kind,spec", [("ets", {"trend": True, "damped": True}), ("sarima", {"order": (1, 1, 1), "seasonal_order": (0, 0, 0, 0)})])
def test_state_space_h_step_forecast_equals_statsmodels_forecast(kind, spec):
    rng = np.random.default_rng(0)
    idx = pd.date_range("2019-01-01", periods=300)
    y = pd.Series(7 + np.cumsum(rng.normal(0, 0.05, 300)), index=idx)
    y[rng.random(300) < 0.15] = np.nan  # missing days stay missing
    params = M.fit_state_space(kind, spec, y.iloc[:200])
    fc = M.state_space_forecasts(kind, spec, y, params, [7, 30])
    for i in (210, 250, 280):
        ref = np.asarray(M._build(kind, spec, y.iloc[: i + 1]).filter(params).forecast(30))
        assert abs(ref[6] - fc[7][i]) < 1e-8 and abs(ref[29] - fc[30][i]) < 1e-8


def test_state_space_forecast_at_t_ignores_later_data():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2019-01-01", periods=250)
    y = pd.Series(7 + np.cumsum(rng.normal(0, 0.05, 250)), index=idx)
    spec = {"order": (1, 1, 1), "seasonal_order": (0, 0, 0, 0)}
    params = M.fit_state_space("sarima", spec, y.iloc[:150])
    a = M.state_space_forecasts("sarima", spec, y.iloc[:200], params, [7])[7]
    y2 = y.copy()
    y2.iloc[200:] = 100.0
    b = M.state_space_forecasts("sarima", spec, y2, params, [7])[7]
    np.testing.assert_allclose(a[:200], b[:200])


def test_point_metrics_on_a_hand_checked_example():
    y, f = np.log([100.0, 200.0]), np.log([110.0, 180.0])
    m = MET.point_metrics(y, f, mase_scale=10.0)
    assert np.isclose(m["MAE"], 15.0) and np.isclose(m["MASE"], 1.5)
    assert np.isclose(m["RMSE"], np.sqrt((100 + 400) / 2))
    assert np.isclose(m["sMAPE"], 100 * np.mean([2 * 10 / 210, 2 * 20 / 380]))
    assert np.isclose(m["MAE_log"], np.mean([np.log(1.1), np.log(200 / 180)]))


def test_diebold_mariano_matches_a_hand_computation_and_detects_a_better_model():
    rng = np.random.default_rng(0)
    base = rng.uniform(5, 15, 400)
    model = base - 1.0 + rng.normal(0, 0.5, 400)  # clearly better on average
    out = MET.diebold_mariano(base, model, h=7)
    assert out["DM"] > 5 and out["p_value"] < 0.001 and out["mean_loss_diff"] > 0.5
    d = base - model
    n, h = len(d), 7
    dc = d - d.mean()
    var = sum((1 if k == 0 else 2) * np.dot(dc[k:], dc[: n - k]) / n for k in range(h))
    dm = d.mean() / np.sqrt(var / n) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    assert np.isclose(out["DM"], dm)
    assert np.isnan(MET.diebold_mariano(base, base, h=7)["DM"])  # identical forecasts: no variance, no test


def test_diebold_mariano_is_not_significant_for_equally_good_forecasts():
    rng = np.random.default_rng(3)
    a, b = rng.uniform(5, 15, 600), rng.uniform(5, 15, 600)
    assert MET.diebold_mariano(a, b, h=7)["p_value"] > 0.01


def test_naive_scale_is_mean_absolute_rupee_change():
    assert np.isclose(MET.naive_scale(np.log([110.0, 90.0]), np.log([100.0, 100.0])), 10.0)
