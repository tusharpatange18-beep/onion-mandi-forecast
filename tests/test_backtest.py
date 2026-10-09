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


def test_damped_trend_ets_fit_is_finite_when_a_long_gap_follows_the_first_quote():
    """Regression: Kolhapur has a 71-day gap right after its first quote; the library's default start values gave a NaN
    initial trend, so every forecast was NaN and the fairness filter silently erased the whole market."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2014-06-18", periods=400)
    y = pd.Series(7 + np.cumsum(rng.normal(0, 0.04, 400)), index=idx)
    y.iloc[1:72] = np.nan
    spec = {"trend": True, "damped": True}
    params = M.fit_state_space("ets", spec, y.iloc[:300])
    assert np.all(np.isfinite(params))
    fc = M.state_space_forecasts("ets", spec, y, params, [7, 30])
    assert np.isfinite(fc[7][-50:]).all() and np.isfinite(fc[30][-50:]).all()


def test_fit_state_space_refuses_non_finite_parameters(monkeypatch):
    class Bad:
        params = np.array([0.1, np.nan])
        mle_retvals = {"converged": True}

    class FakeModel:
        param_names = ["smoothing_level", "initial_level"]

        def fit(self, **kw):
            return Bad()

    monkeypatch.setattr(M, "_build", lambda kind, spec, endog: FakeModel())
    y = pd.Series(np.arange(10.0), index=pd.date_range("2020-01-01", periods=10))
    with pytest.raises(ValueError, match="non-finite"):
        M.fit_state_space("ets", {"trend": False, "damped": False}, y)


def test_coverage_guard_aborts_when_a_model_cannot_forecast_a_whole_market():
    t = pd.DataFrame({"market": ["A"] * 50 + ["B"] * 50, "h": 7, "base_seasonal": 1.0, "m1": 1.0})
    t.loc[t["market"] == "B", "m1"] = np.nan  # model m1 silently fails for market B
    with pytest.raises(RuntimeError, match="m1 missing 100.0% of 50 triples for B"):
        B.check_forecast_coverage(t, ["m1"])
    t["m1"] = 1.0
    B.check_forecast_coverage(t, ["m1"])  # healthy table passes
    t.loc[t.index[:3], "m1"] = np.nan   # 3 of 50 = 6% in market A -> still caught
    with pytest.raises(RuntimeError):
        B.check_forecast_coverage(t, ["m1"])


def test_anchored_forecast_is_last_price_plus_predicted_change():
    out = B.anchored_forecast(np.array([7.0, 6.0]), np.array([7.5, 5.0]), np.array([7.2, 5.5]))
    np.testing.assert_allclose(out, [7.3, 5.5])


def test_prophet_anchored_task_uses_the_origin_price_and_predicted_movement(monkeypatch):
    """With a stub whose 'Prophet' is a straight line of slope 0.01 per day, the anchored forecast must be
    last observed log price + 0.01 * h, independent of the line's level, while plain Prophet returns the line itself."""
    def stub(y_train, target_dates, spec, seed=0):
        return 100.0 + 0.01 * (pd.DatetimeIndex(target_dates) - pd.Timestamp("2019-01-01")).days.to_numpy()

    monkeypatch.setattr(M, "prophet_forecast", stub)
    days = pd.date_range("2018-01-01", "2020-03-31")
    rng = np.random.default_rng(0)
    y = 7 + np.cumsum(rng.normal(0, 0.02, len(days)))
    ys = pd.Series(y, index=days)
    blocks = B._blocks(B.refit_schedule("2020-01-01", "2020-03-31"), pd.Timestamp("2020-03-31"))
    for fam in ("prophet_anchored", "prophet"):
        col, _, df, _, n_fits, _ = B._task_prophet((B.Spec(fam, "x", (("changepoint_prior_scale", 0.05),)), 0, y, days, blocks, [7, 30]))
        assert n_fits == 3 and len(df) > 100
        for h in (7, 30):
            d = df[df["h"] == h]
            if fam == "prophet_anchored":
                expected = ys.loc[d["origin"]].to_numpy() + 0.01 * h
            else:
                expected = 100.0 + 0.01 * ((d["origin"] + pd.Timedelta(days=h)) - pd.Timestamp("2019-01-01")).dt.days.to_numpy()
            np.testing.assert_allclose(d["f"].to_numpy(), expected, atol=1e-9)


def test_specs_include_the_declared_prophet_anchored_candidates():
    from mandi.config import load_config

    cols = [s.col for s in B.specs_from_config(load_config())]
    assert "prophet_anchored|cps0.05" in cols and "prophet_anchored|cps0.3" in cols and "prophet|cps0.3" in cols
    assert len(cols) == 25
