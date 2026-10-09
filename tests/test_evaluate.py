import subprocess

import numpy as np
import pandas as pd
import pytest

from mandi import backtest as B
from mandi import evaluate as E
from mandi import features as F
from mandi import models as M


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def test_dirty_flag_looks_at_source_config_tests_only_and_reports_the_selection_commit(tmp_path, monkeypatch):
    for d in ("src", "configs", "tests", "results", "docs"):
        (tmp_path / d).mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n")
    (tmp_path / "configs" / "selected_models.yaml").write_text("a: 1\n")
    (tmp_path / "project.md").write_text("brief\n")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "-c", "user.email=t@t", "-c", "user.name=t", "add", "-A")
    _git(tmp_path, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "first")
    monkeypatch.setattr(B, "ROOT", tmp_path)
    st = B.git_state()
    assert st["dirty"] is False and st["selection_commit"] == st["commit"]
    (tmp_path / "results" / "new_table.csv").write_text("x\n")          # outputs do not count
    (tmp_path / "docs" / "report.md").write_text("x\n")
    assert B.git_state()["dirty"] is False
    (tmp_path / "src" / "a.py").write_text("x = 2\n")                    # source changes do
    assert B.git_state()["dirty"] is True
    _git(tmp_path, "checkout", "--", "src/a.py")
    (tmp_path / "tests" / "t.py").write_text("y = 1\n")                  # untracked test file counts
    assert B.git_state()["dirty"] is True


def _tab(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = np.log(rng.uniform(500, 1500, n))
    t = pd.DataFrame({"market": "A", "h": 7, "y_log": y})
    t["base_naive"] = y + rng.normal(0, 0.20, n)
    t["base_mean3"] = y + rng.normal(0, 0.15, n)       # the strongest baseline by construction
    t["base_seasonal"] = y + rng.normal(0, 0.60, n)
    for fam, sd in [("ets", 0.19), ("sarima", 0.19), ("prophet", 0.5), ("prophet_anchored", 0.3), ("lgbm", 0.10)]:
        t[fam] = y + rng.normal(0, sd, n)
    t["origin"] = pd.date_range("2020-01-01", periods=n)
    t["target_date"] = t["origin"] + pd.Timedelta(days=7)
    t["gap_prev"] = 1.0
    return t


def test_summary_picks_the_strongest_baseline_by_rupee_mae_and_reports_both_improvements():
    t = _tab()
    met = E.final_metrics(t, {("A", 7): 100.0})
    s = E.summary_table(met, {7: "lgbm"}, {("A", 7): "ets"}).iloc[0]
    assert s["strongest_baseline"] == "base_mean3"
    mae = met.set_index("model")["MAE"]
    assert np.isclose(s["headline_impr_vs_strongest_pct"], 100 * (mae["base_mean3"] - mae["lgbm"]) / mae["base_mean3"])
    assert np.isclose(s["headline_impr_vs_naive_pct"], 100 * (mae["base_naive"] - mae["lgbm"]) / mae["base_naive"])
    assert s["headline_model"] == "lgbm" and s["per_cell_selected_model"] == "ets"
    assert s["headline_impr_vs_strongest_pct"] > 0
    for fam in E.FAMILIES:                              # every family is always shown
        assert f"impr_{fam}_vs_strongest_pct" in s.index and f"impr_{fam}_vs_naive_pct" in s.index


def test_headline_is_not_replaced_by_the_test_best_model():
    t = _tab()
    t["ets"] = t["y_log"]                                # a perfect model: it becomes the test-best, but not the headline
    met = E.final_metrics(t, {("A", 7): 100.0})
    s = E.summary_table(met, {7: "lgbm"}, {("A", 7): "sarima"}).iloc[0]
    assert s["test_best_model_(optimistic)"] == "ets" and s["headline_model"] == "lgbm" and s["per_cell_selected_model"] == "sarima"


def test_win_rate_median_error_and_ties():
    t = _tab()
    met = E.final_metrics(t, {("A", 7): 100.0})
    summ = E.summary_table(met, {7: "lgbm"}, {("A", 7): "lgbm"})
    rb = E.robustness_table(t, summ).set_index("model")
    assert rb.loc["base_mean3", "win_rate_vs_strongest_pct"] == 50.0       # a model against itself: all ties count half
    assert rb.loc["lgbm", "win_rate_vs_strongest_pct"] > 50 and rb.loc["prophet", "win_rate_vs_strongest_pct"] < 50
    assert rb.loc["lgbm", "median_AE"] < rb.loc["lgbm", "median_AE_strongest_baseline"]
    assert rb.loc["lgbm", "median_AE_impr_vs_strongest_pct"] > 0


def test_per_year_table_splits_by_target_year():
    t = _tab(n=500)
    met = E.final_metrics(t, {("A", 7): 100.0})
    summ = E.summary_table(met, {7: "lgbm"}, {("A", 7): "lgbm"})
    py = E.per_year_table(t, summ, {7: "lgbm"})
    assert set(py["year"]) == {2020, 2021} and py["n"].sum() == len(t)


def test_ablation_drops_exactly_the_requested_features_and_keeps_the_market_id(monkeypatch):
    seen = {}

    def spy(train, test, features, params, seed):
        seen["features"] = list(features)
        return np.zeros(len(test))

    monkeypatch.setattr(M, "lgbm_fit_predict", spy)
    origins = pd.date_range("2016-01-01", "2020-03-31")  # enough rows (the task needs >= 500 to train)
    tab = pd.DataFrame({"market_idx": 0, "origin": origins, "p0": 7.0, "target_date": origins + pd.Timedelta(days=7), "y_log": 7.0, "y_change": 0.0})
    blocks = B._blocks(B.refit_schedule("2020-01-01", "2020-03-31"), pd.Timestamp("2020-03-31"))
    dropped = ("target_month", "target_woy", "target_dow", "origin_dow", "target_expected_closed")
    B._task_lgbm((B.Spec("lgbm", "x", (("drop", dropped),)), 7, tab, blocks, 1))
    assert set(seen["features"]) == set(F.FEATURE_COLUMNS) - set(dropped)
    assert "market_idx" in seen["features"]
    B._task_lgbm((B.Spec("lgbm", "y", ()), 7, tab, blocks, 1))
    assert seen["features"] == F.FEATURE_COLUMNS  # no drop: every normal run uses all features


def test_selection_file_declares_the_headline_and_every_horizon():
    import yaml
    from mandi.config import ROOT

    sel = yaml.safe_load((ROOT / "configs" / "selected_models.yaml").read_text(encoding="utf-8"))
    assert sel["headline_model_by_horizon"] == {7: "lgbm", 30: "lgbm"}
    assert set(sel["chosen"]) == {7, 30} and all(set(d) == set(E.FAMILIES) for d in sel["chosen"].values())
