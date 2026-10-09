"""Embargo check: no training row may have a target date on or after its refit date.

"Only data before the refit date" is not enough when labels look ahead: a row whose origin is before the refit date
but whose h-day-ahead target is after it would leak the future into training. The training rule in
backtest._task_lgbm keeps a row only if target_date < refit date, which for horizon h means origin <= refit - h - 1 day.
"""
import numpy as np
import pandas as pd
import pytest

from mandi import backtest as B
from mandi import features as F
from mandi import models as M
from mandi.config import ROOT, load_config

CFG = load_config()


@pytest.fixture(scope="module")
def tables():
    path = ROOT / CFG["paths"]["panel"]
    if not path.exists():
        pytest.skip("panel not built")
    p = pd.read_parquet(path)
    a = F.build_arrays(p, CFG["data"]["start_date"], CFG["data"]["end_date"], CFG["markets"])
    return B.build_tables(a, F.closure_weekdays(p, CFG["eda"]["train_end"]), CFG)


def _windows():
    folds = CFG["forecast"]["tuning_folds"]
    return {"tuning": (f"{folds[0]}-01-01", f"{folds[-1]}-12-31"),
            "final": (CFG["backtest"]["first_test_date"], CFG["backtest"]["last_test_date"])}


@pytest.mark.local_data("panel")
@pytest.mark.parametrize("window_name", ["tuning", "final"])
@pytest.mark.parametrize("h", [7, 30])
def test_every_lightgbm_training_row_has_a_target_before_the_refit_date(monkeypatch, tables, window_name, h):
    seen = []

    def spy(train, test, features, params, seed):
        seen.append({"max_target": train["target_date"].max(), "max_origin": train["origin"].max(),
                     "n_train": len(train), "test_min": test["origin"].min()})
        return np.zeros(len(test))

    monkeypatch.setattr(M, "lgbm_fit_predict", spy)
    start, end = _windows()[window_name]
    blocks = B._blocks(B.refit_schedule(start, end), pd.Timestamp(end))
    B._task_lgbm((B.Spec("lgbm", "spy", ()), h, tables[h], blocks, 0))
    assert len(seen) == len(blocks)  # a model was fitted at every refit date
    for s, (refit, _) in zip(seen, blocks):
        assert s["max_target"] < refit, f"training target {s['max_target']} is not before refit {refit}"
        assert s["max_origin"] <= refit - pd.Timedelta(days=h + 1)  # for h = 30: origins no later than refit - 31 days
        assert s["test_min"] >= refit


def test_the_embargo_test_can_fail():
    """Guard against a vacuous test: a training set that includes a later target must be caught by the same assertion."""
    refit = pd.Timestamp("2020-02-01")
    origins = pd.date_range("2019-12-01", "2020-01-31")
    leaky = pd.DataFrame({"origin": origins, "target_date": origins + pd.Timedelta(days=30)})
    assert leaky["target_date"].max() >= refit  # an origin one day before the refit has its target after it
    clean = leaky[leaky["target_date"] < refit]
    assert clean["target_date"].max() < refit and clean["origin"].max() <= refit - pd.Timedelta(days=31)


@pytest.mark.parametrize("window_name", ["tuning", "final"])
def test_windows_stay_inside_their_allowed_periods(window_name):
    start, end = map(pd.Timestamp, _windows()[window_name])
    train_end = pd.Timestamp(CFG["eda"]["train_end"])
    if window_name == "tuning":
        assert end <= train_end                      # tuning never touches 2020-2024
    else:
        assert start > train_end                     # the final evaluation starts after the tuning period
