import numpy as np
import pandas as pd
import pytest

from mandi import forecast as FC
from mandi import intervals as IV
from mandi.config import ROOT, load_config

CFG = load_config()
FAIR = ROOT / "results" / "fair_forecasts.parquet"
needs_results = pytest.mark.skipif(not (FAIR.exists() and (ROOT / CFG["paths"]["panel"]).exists()), reason="panel / backtest results not built")

# Serving and the backtest call the same training function on the same rows with the same seed and thread count, so the log forecasts
# should agree to floating-point precision. The tolerance is stated here: 1e-6 in log space (about 0.0001% in rupees).
LOG_TOLERANCE = 1e-6


@pytest.fixture(scope="module")
def inputs():
    return FC.load_inputs(CFG)


@pytest.fixture(scope="module")
def latest_run():
    return FC.build_forecasts(CFG)


# ---------------------------------------------------------------- training embargo
def test_training_rows_have_a_target_strictly_before_the_as_of_date_synthetic():
    as_of = pd.Timestamp("2024-06-01")
    tgt = pd.to_datetime(["2024-05-30", "2024-05-31", "2024-06-01", "2024-06-02", "2024-07-01"])
    t = pd.DataFrame({"target_date": tgt, "y_log": [7.0, 7.0, 7.0, 7.0, np.nan], "origin": tgt - pd.Timedelta(days=7)})
    tr = FC.training_rows(t, as_of)
    assert list(tr["target_date"]) == list(tgt[:2])                       # the row whose target is the as-of date itself is excluded
    assert (tr["target_date"] < as_of).all() and tr["y_log"].notna().all()


@needs_results
@pytest.mark.parametrize("as_of", ["2020-03-01", "2024-06-01", "2025-01-01"])
def test_no_training_row_has_a_target_on_or_after_as_of_on_real_tables(inputs, as_of):
    _, tables, _ = inputs
    d = pd.Timestamp(as_of)
    for h, t in tables.items():
        tr = FC.training_rows(t, d)
        assert len(tr) > 1000 and tr["target_date"].max() < d
        assert tr["origin"].max() < d - pd.Timedelta(days=h) + pd.Timedelta(days=1)   # origin + h < as_of


@needs_results
def test_default_as_of_is_the_day_after_the_last_observed_quote(inputs, latest_run):
    arrays, _, last = inputs
    assert latest_run["as_of"].iloc[0] == last + pd.Timedelta(days=1)
    assert latest_run["data_through"].iloc[0] == last == pd.Timestamp(CFG["data"]["end_date"]) or last <= pd.Timestamp(CFG["data"]["end_date"])
    assert (latest_run["origin_date"] < latest_run["as_of"]).all()
    assert (latest_run["trained_through"] < latest_run["as_of"]).all()


# ---------------------------------------------------------------- serving reproduces the backtest
@needs_results
def test_serving_reproduces_the_backtest_forecasts_of_the_june_2024_refit(inputs):
    """With as-of equal to the 2024-06-01 monthly refit date of the backtest, forecasts for the origins of June 2024 must match
    results/fair_forecasts.parquet (the backtest) to within LOG_TOLERANCE in log space."""
    _, tables, _ = inputs
    as_of = pd.Timestamp("2024-06-01")
    select = {h: (t["origin"] >= as_of) & (t["origin"] <= "2024-06-30") & t["y_log"].notna() for h, t in tables.items()}
    preds, trained = FC.predict_origins(CFG, tables, as_of, select)
    assert all(v < as_of for v in trained.values())
    fair = pd.read_parquet(FAIR)
    m = preds.merge(fair[["market", "origin", "h", "lgbm"]], on=["market", "origin", "h"], how="inner")
    assert len(m) > 100 and set(m["h"]) == {7, 30}
    diff = (m["f_log"] - m["lgbm"]).abs()
    assert diff.max() < LOG_TOLERANCE, f"max log difference {diff.max():.3e} over {len(m)} forecasts"


# ---------------------------------------------------------------- the served table
@needs_results
def test_output_schema_and_one_latest_row_per_market_and_horizon(latest_run):
    assert list(latest_run.columns) == FC.OUTPUT_COLUMNS
    for c in ["market", "horizon", "origin_date", "target_date", "last_price", "point", "lo80", "hi80", "lo95", "hi95", "naive", "n_calibration",
              "model_spec", "trained_through", "generated_at", "git_commit"]:
        assert c in latest_run.columns
    lat = latest_run[latest_run["is_latest"]]
    assert len(lat) == len(CFG["markets"]) * 2 and not lat.duplicated(["market", "horizon"]).any()
    assert latest_run.groupby(["market", "horizon"]).size().eq(CFG["serving"]["history_origins"]).all()      # latest + the 59 before it
    assert (latest_run["target_date"] - latest_run["origin_date"]).dt.days.eq(latest_run["horizon"]).all()


@needs_results
def test_latest_origin_is_each_markets_last_observed_quote(inputs, latest_run):
    arrays, _, _ = inputs
    for i, m in enumerate(arrays.short):
        last = arrays.days[~np.isnan(arrays.logp[i])][-1]
        g = latest_run[(latest_run["market"] == m) & latest_run["is_latest"]]
        assert (g["origin_date"] == last).all() and len(g) == 2


@needs_results
def test_intervals_are_ordered_lo95_lt_lo80_lt_point_lt_hi80_lt_hi95(latest_run):
    d = latest_run.dropna(subset=["lo95", "lo80", "hi80", "hi95"])
    assert len(d) == len(latest_run)                                      # seeded calibration: every row has an interval
    assert (d["lo95"] < d["lo80"]).all() and (d["lo80"] < d["point"]).all() and (d["point"] < d["hi80"]).all() and (d["hi80"] < d["hi95"]).all()
    assert (d["n_calibration"] >= CFG["intervals"]["min_errors"]).all()
    assert (d["naive"] == d["last_price"]).all()


@needs_results
def test_forecasts_for_unresolved_targets_are_out_of_sample_and_flagged_as_such(latest_run):
    unresolved = latest_run[latest_run["target_date"] >= latest_run["as_of"]]        # the target lies in the future: genuinely out of sample
    assert len(unresolved) > 0 and unresolved["actual_price"].isna().all() and not unresolved["in_sample"].any()
    resolved = latest_run[latest_run["in_sample"]]
    assert resolved["actual_price"].notna().all() and (resolved["target_date"] < resolved["as_of"]).all()
    passed_without_quote = latest_run[(latest_run["target_date"] < latest_run["as_of"]) & ~latest_run["in_sample"]]
    assert passed_without_quote["actual_price"].isna().all()                 # no quote that day (a closure): nothing to compare, not in the training set


# ---------------------------------------------------------------- calibration uses only resolved errors
def test_serving_calibration_ignores_errors_not_yet_resolved_at_the_origin():
    rng = np.random.default_rng(0)
    cal_t = pd.date_range("2023-01-01", periods=400).to_numpy()
    score = np.abs(rng.normal(0, 0.1, 400))
    origin = np.array(["2023-12-01"], dtype="datetime64[ns]")
    base, n0 = IV.quantiles_for(origin, cal_t, score, [0.95], 365, 100)
    changed = score.copy()
    changed[cal_t >= origin[0]] = 99.0                                     # errors that become known on or after the origin
    new, n1 = IV.quantiles_for(origin, cal_t, changed, [0.95], 365, 100)
    assert n0[0] == n1[0] and np.isclose(base[0.95][0], new[0.95][0])


def test_quantiles_for_matches_the_phase_4_rolling_procedure():
    rng = np.random.default_rng(1)
    origin = pd.date_range("2018-01-01", periods=600).to_numpy()
    target = origin + np.timedelta64(7, "D")
    score = np.abs(rng.normal(0, 0.1, 600))
    a, na = IV.conformal_halfwidths(origin, target, score, [0.8, 0.95], 365, 100)
    b, nb = IV.quantiles_for(origin, target, score, [0.8, 0.95], 365, 100)
    assert (na == nb).all() and np.allclose(a[0.8], b[0.8], equal_nan=True) and np.allclose(a[0.95], b[0.95], equal_nan=True)
