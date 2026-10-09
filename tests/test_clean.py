import numpy as np
import pandas as pd
import pytest

from mandi import clean
from mandi.config import ROOT, load_config

CFG = load_config()
PANEL_PATH = ROOT / CFG["paths"]["panel"]
MAX_GAP = CFG["cleaning"]["max_ffill_gap_days"]


# ------------------------------------------------------------------ unit tests (small made-up cases)
def _series(values):
    return pd.Series(values, index=pd.date_range("2020-01-01", periods=len(values), freq="D"), dtype=float)


def test_fill_short_gaps_fills_only_gaps_up_to_limit():
    s = _series([1, np.nan, np.nan, 2, np.nan, np.nan, np.nan, np.nan, 3])
    filled, mask = clean.fill_short_gaps(s, max_gap=3)
    assert filled.iloc[1] == 1 and filled.iloc[2] == 1  # gap of 2: filled
    assert filled.iloc[4:8].isna().all()  # gap of 4: not filled at all, not even partly
    assert mask.sum() == 2


def test_fill_short_gaps_never_touches_observed_values():
    s = _series([5, np.nan, 7])
    filled, _ = clean.fill_short_gaps(s, max_gap=3)
    assert filled.iloc[0] == 5 and filled.iloc[2] == 7


def test_stale_runs_flagged_but_values_kept():
    n = 8
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=n),
                       "min_price_rs_qtl": [100] * 5 + [90, 95, 99], "max_price_rs_qtl": [200] * 5 + [210, 220, 230],
                       "modal_price_rs_qtl": [150] * 5 + [160, 170, 180]})
    df[["min_price_rs_qtl", "max_price_rs_qtl", "modal_price_rs_qtl"]] = df[["min_price_rs_qtl", "max_price_rs_qtl", "modal_price_rs_qtl"]].astype(float)
    df["market_key"] = "m"
    out = clean.flag_stale_runs(df, 5, clean.CleaningLog())
    assert out["is_stale"].tolist() == [True] * 5 + [False] * 3
    assert out["modal_price_rs_qtl"].tolist() == df["modal_price_rs_qtl"].tolist()


def test_outlier_flag_catches_planted_error_and_leaves_values():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=120)
    price = 1000 + rng.integers(-60, 60, size=120).astype(float)
    price[60] = 10000  # planted error
    g = pd.DataFrame({"date": idx, "market_key": "m", "modal_price": price})
    params = CFG["cleaning"]["outlier"]
    res = clean.flag_outliers(g, params, clean.CleaningLog())
    assert res.loc[res["date"] == idx[60], "is_outlier_flag"].item()
    assert res["is_outlier_flag"].sum() == 1
    assert res["modal_price"].tolist() == price.tolist()


def test_outlier_flag_does_not_fire_with_too_few_observations():
    idx = pd.date_range("2020-01-01", periods=40)
    price = np.full(40, np.nan)
    price[[5, 20, 35]] = [1000, 9000, 1000]  # only 3 observations in any window
    g = pd.DataFrame({"date": idx, "market_key": "m", "modal_price": price})
    res = clean.flag_outliers(g, CFG["cleaning"]["outlier"], clean.CleaningLog())
    assert not res["is_outlier_flag"].any()


def test_primary_variety_is_most_frequent_and_others_dropped():
    df = pd.DataFrame({"market_key": "m", "variety": ["A", "A", "B"], "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-01"])})
    prim = clean.choose_primary_variety(df)
    assert prim == {"m": "A"}
    out = clean.select_primary_rows(df, prim, clean.CleaningLog())
    assert out["variety"].tolist() == ["A", "A"]


def test_non_positive_prices_dropped():
    df = pd.DataFrame({"min_price_rs_qtl": [100.0, 0.0], "max_price_rs_qtl": [200.0, 50.0],
                       "modal_price_rs_qtl": [150.0, 20.0], "arrivals_tonnes": [10.0, 5.0]})
    out = clean.drop_invalid_prices(df, clean.CleaningLog())
    assert len(out) == 1


# ------------------------------------------------------------------ checks on the real panel
@pytest.fixture(scope="module")
def panel():
    if not PANEL_PATH.exists():
        pytest.skip("panel.parquet not built; run python -m mandi.run_phase1")
    return pd.read_parquet(PANEL_PATH)


@pytest.mark.local_data("panel")
def test_panel_markets_match_config(panel):
    assert set(panel["market_key"]) == set(CFG["markets"])


@pytest.mark.local_data("panel")
def test_no_duplicate_market_date(panel):
    assert not panel.duplicated(["market_key", "date"]).any()


@pytest.mark.local_data("panel")
def test_dates_sorted_and_daily_within_each_market(panel):
    for _, g in panel.groupby("market_key"):
        assert g["date"].is_monotonic_increasing
        assert (g["date"].diff().dropna() == pd.Timedelta(days=1)).all()


@pytest.mark.local_data("panel")
def test_observed_prices_positive_and_ordered(panel):
    obs = panel[~panel["was_missing"]]
    assert (obs[["modal_price", "min_price", "max_price"]] > 0).all().all()
    assert not obs["is_order_violation"].any()


@pytest.mark.local_data("panel")
def test_was_missing_matches_missing_target(panel):
    assert (panel["was_missing"] == panel["modal_price"].isna()).all()


@pytest.mark.local_data("panel")
def test_targets_are_never_filled(panel):
    # a day that was missing must have a missing target, whatever the *_ffill columns say
    assert panel.loc[panel["was_missing"], "modal_price"].isna().all()


@pytest.mark.local_data("panel")
def test_no_fill_beyond_gap_limit(panel):
    for _, g in panel.groupby("market_key"):
        run = (g["was_missing"] != g["was_missing"].shift()).cumsum()
        run_len = g.groupby(run)["was_missing"].transform("size")
        long_gap = g["was_missing"] & (run_len > MAX_GAP)
        assert g.loc[long_gap, "modal_price_ffill"].isna().all()
        assert not g.loc[long_gap, "was_filled"].any()
        short_gap = g["was_missing"] & (run_len <= MAX_GAP)
        assert g.loc[short_gap, "was_filled"].all()
        assert g.loc[short_gap, "modal_price_ffill"].notna().all()


@pytest.mark.local_data("panel")
def test_filled_values_equal_last_observation(panel):
    for _, g in panel.groupby("market_key"):
        last_obs = g["modal_price"].ffill()
        f = g["was_filled"]
        assert (g.loc[f, "modal_price_ffill"] == last_obs[f]).all()


def _raw_state_rows():
    return pd.read_parquet(ROOT / CFG["paths"]["raw_file"],
                           columns=["market_key", "date", "variety", "modal_price_rs_qtl", "min_price_rs_qtl", "max_price_rs_qtl"],
                           filters=[("state", "==", CFG["data"]["state"])])


@pytest.mark.local_data("panel", "raw")
def test_raw_columns_equal_the_source_file_and_are_never_modified(panel):
    raw = _raw_state_rows()
    obs = panel.dropna(subset=["modal_price_raw"])
    m = obs.merge(raw, on=["market_key", "date", "variety"], how="left", validate="one_to_one")
    assert (m["modal_price_raw"] == m["modal_price_rs_qtl"]).all()
    assert (m["min_price_raw"] == m["min_price_rs_qtl"]).all()
    assert (m["max_price_raw"] == m["max_price_rs_qtl"]).all()
    # every source row of a chosen market's primary variety appears in the panel's raw column
    assert panel["modal_price_raw"].notna().sum() == len(obs)


@pytest.mark.local_data("panel")
def test_only_approved_corrections_changed_and_only_to_missing(panel):
    corrected = panel[panel["was_corrected"]]
    approved = {(c["market_key"], pd.Timestamp(c["date"])) for c in CFG["corrections"]}
    assert set(zip(corrected["market_key"], corrected["date"])) == approved
    assert corrected["modal_price"].isna().all() and corrected["modal_price_raw"].notna().all()
    assert corrected["was_missing"].all()
    untouched = panel[~panel["was_corrected"]]
    same = (untouched["modal_price"] == untouched["modal_price_raw"]) | (untouched["modal_price"].isna() & untouched["modal_price_raw"].isna())
    assert same.all()


@pytest.mark.local_data("panel")
def test_corrected_days_are_not_in_the_filled_target(panel):
    corrected = panel[panel["was_corrected"]]
    assert corrected["modal_price"].isna().all()  # target stays missing even though a *_ffill value may exist


def test_apply_corrections_refuses_unflagged_target():
    g = pd.DataFrame({"market_key": "m", "date": pd.date_range("2020-01-01", periods=3), "modal_price": [1.0, 2.0, 3.0],
                      "min_price": 1.0, "max_price": 3.0, "was_missing": False, "was_corrected": False,
                      "is_outlier_flag": [False, False, False]})
    with pytest.raises(ValueError):
        clean.apply_corrections(g, [{"market_key": "m", "date": "2020-01-02"}], clean.CleaningLog())


def test_apply_corrections_sets_missing_and_keeps_raw():
    g = pd.DataFrame({"market_key": "m", "date": pd.date_range("2020-01-01", periods=3), "modal_price_raw": [1.0, 99.0, 3.0],
                      "modal_price": [1.0, 99.0, 3.0], "min_price": 1.0, "max_price": 3.0, "was_missing": False,
                      "was_corrected": False, "is_outlier_flag": [False, True, False]})
    out = clean.apply_corrections(g, [{"market_key": "m", "date": "2020-01-02"}], clean.CleaningLog())
    assert np.isnan(out.loc[1, "modal_price"]) and out.loc[1, "modal_price_raw"] == 99.0
    assert out["was_corrected"].tolist() == [False, True, False]
    assert g.loc[1, "modal_price"] == 99.0  # input frame is not mutated
