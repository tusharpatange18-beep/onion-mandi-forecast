"""LightGBM thread count: 4 by default (as in the onion runs), or the value of MANDI_LGBM_THREADS."""
from mandi import models as M


def test_default_is_four_threads(monkeypatch):
    # no setting -> the onion default
    monkeypatch.delenv("MANDI_LGBM_THREADS", raising=False)
    assert M.lgbm_threads() == 4


def test_environment_variable_sets_the_threads(monkeypatch):
    # the crop workflow sets 1 thread per worker
    monkeypatch.setenv("MANDI_LGBM_THREADS", "1")
    assert M.lgbm_threads() == 1
