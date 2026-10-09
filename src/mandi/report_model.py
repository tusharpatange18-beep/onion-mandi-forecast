"""Phase 3: write docs/model_report.md from the files in results/. Every number is computed here, none is typed by hand.

Usage: python -m mandi.report_model
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import yaml

from mandi.config import ROOT, load_config
from mandi.report3 import _fmt
from mandi.run_phase1 import df_to_md

logger = logging.getLogger(__name__)
FAM = ["ets", "sarima", "prophet", "prophet_anchored", "lgbm"]
FAM_LABEL = {"ets": "ETS (= ARIMA(0,1,1), one result)", "sarima": "SARIMA(0,1,1)", "prophet": "Prophet (off the shelf)",
             "prophet_anchored": "Prophet anchored", "lgbm": "LightGBM"}
R = ROOT / "results"


def _csv(name: str) -> pd.DataFrame:
    return pd.read_csv(R / f"{name}.csv")


def _runs() -> dict[str, dict]:
    out = {}
    for kind in ("final", "final_error", "ablation_post_hoc"):
        runs = sorted((R / "runs").glob({"final": "final_2*", "final_error": "final_error_*", "ablation_post_hoc": "ablation_*"}[kind]))
        if runs:
            out[kind] = json.loads((runs[-1] / "run.json").read_text(encoding="utf-8")) | {"dir": runs[-1].name}
    return out


def _md(df: pd.DataFrame, nd: int = 2) -> str:
    return df_to_md(_fmt(df))


def build() -> str:
    """Assemble the report."""
    cfg = load_config()
    runs = _runs()
    fin = runs["final"]
    summ, macro, dm, dmall = _csv("summary_table"), _csv("summary_macro"), _csv("dm_tests"), _csv("dm_all_models")
    met, py, rob, err, rt, drop = _csv("backtest_metrics"), _csv("per_year"), _csv("robustness"), _csv("error_analysis"), _csv("runtime"), _csv("dropped_triples")
    stale = _csv("sensitivity_stale")
    sel = yaml.safe_load((ROOT / cfg["forecast"]["selected_models_file"]).read_text(encoding="utf-8"))
    mh = macro.set_index("h")
    hd = dm[dm["which"] == "headline"]
    sig = {h: int((g["p_value"] < 0.05).sum()) for h, g in hd.groupby("h")}
    pos = {h: int((g["headline_impr_vs_strongest_pct"] > 0).sum()) for h, g in summ.groupby("h")}
    rb_l = rob[rob["model"] == "lgbm"].groupby("h")[["win_rate_vs_strongest_pct", "win_rate_vs_naive_pct", "median_AE_impr_vs_strongest_pct"]].mean()
    pyh = py.groupby(["h", "year"])["impr_vs_strongest_pct"].mean().unstack(0)
    neg_years = {h: [int(y) for y in pyh.index[pyh[h] < 0]] for h in (7, 30)}
    n_tr, n_drop = int(drop["total_triples"].iloc[0]), int(drop["dropped_for_all_models_total"].iloc[0])
    d: list[str] = []
    a = d.append

    a("# Model report (Phase 3)")
    a("")
    a("## Rules declared before the final run")
    a("")
    a("- **Primary metric:** MAE in rupees per quintal. MAE on the log scale is secondary. RMSE, MASE (scaled by the in-sample naive error of 2014-2019) and sMAPE are in section 4.")
    a("- **Strongest baseline** for each market and horizon = whichever of (a) naive (last observed quote), (b) the mean of the last 3 observed quotes and (c) the 365-day seasonal-naive has the "
      "lowest rupee MAE on the final window 2020-2024. This is a conservative, hindsight choice: it makes the benchmark as hard as possible. Improvement over plain naive is reported next to it.")
    a("- **Headline model:** at each horizon, the family with the best pooled score on the 2017-2019 tuning folds, using the frozen configurations. That is "
      f"**LightGBM at both horizons** (7 days: `{sel['chosen'][7]['lgbm']}`; 30 days: `{sel['chosen'][30]['lgbm']}`). The per-market tuning-selected model and the model that happens to be best on the test window are "
      "secondary rows (section 3); the latter is optimistic because it is chosen after seeing the results. Every family is shown on every market and horizon.")
    a("- **Task:** a forecast origin is a day with an observed quote; the target is the quote 7 or 30 calendar days later and is scored only if it exists. "
      f"Every model is scored on the same {n_tr - n_drop:,} (market, origin, horizon) triples; {n_drop} of {n_tr:,} were dropped for all models because baseline (c) has no quote within 3 days of the target date one year earlier.")
    a("- **Protocol:** models are refit on the first day of every month using only rows whose target date is before that day (embargo; tested). Hyperparameters were chosen on 2017-2019 folds only. "
      "The final run was made once, on a clean tree. The sensitivity runs (section 7) and the ablation (section 9) are never used to choose between models.")
    a(f"- **Provenance:** final run `{fin['dir']}`, code commit `{fin['git']['commit']}`, working tree dirty = `{fin['git']['dirty']}`, selection file commit `{fin['git']['selection_commit']}`, "
      f"configuration hash `{fin['config_sha1']}`.")
    a("- ETS local level and ARIMA(0,1,1) are mathematically the same model, so the `ets` and `sarima` columns count as one result (their numbers agree).")
    a("")

    a("## 1. Headline result")
    a("")
    a(f"LightGBM improves rupee MAE over the strongest baseline by **{mh.loc[7, 'headline_impr_vs_strongest_pct']:.1f}% at 7 days and {mh.loc[30, 'headline_impr_vs_strongest_pct']:.1f}% at 30 days** "
      f"(equal-weight average over the five markets). Against plain naive the figures are {mh.loc[7, 'headline_impr_vs_naive_pct']:.1f}% and {mh.loc[30, 'headline_impr_vs_naive_pct']:.1f}%. "
      f"It beats the strongest baseline in {pos[7]} of 5 markets at 7 days and {pos[30]} of 5 at 30 days. "
      f"Diebold-Mariano tests against the strongest baseline (absolute error, Harvey correction, h-1 lags) reject equality at the 5% level in {sig[7]} of 5 markets at 7 days and {sig[30]} of 5 at 30 days; "
      "the forecasts overlap, so these p-values are approximate, and with ten cells some significance is expected by chance.")
    a("")
    cols = ["market", "h", "n", "strongest_baseline", "baseline_MAE", "naive_MAE", "headline_MAE", "headline_impr_vs_strongest_pct", "headline_impr_vs_naive_pct"]
    a(_md(summ[cols].rename(columns={"headline_impr_vs_strongest_pct": "impr_vs_strongest_%", "headline_impr_vs_naive_pct": "impr_vs_naive_%"})))
    a("")
    a("Macro average over the five markets (equal weight), improvement in rupee MAE in %:")
    a("")
    a(_md(macro[["h", "headline_impr_vs_strongest_pct", "headline_impr_vs_naive_pct"]]))
    a("")
    a("Diebold-Mariano test of the headline model against the strongest baseline (positive DM = LightGBM better; `mean_loss_diff` in rupees):")
    a("")
    a(_md(hd[["market", "h", "strongest_baseline", "DM", "p_value", "mean_loss_diff", "n"]]))
    a("")
    if (R / "bootstrap_ci.csv").exists():
        ci, holm = _csv("bootstrap_ci"), _csv("holm_dm")
        a("### 1.1 Uncertainty of the macro improvement: block bootstrap (analysis only)")
        a("")
        b = ci.iloc[0]
        a(f"Origin days are resampled in blocks of {int(b['block_days'])} days, with the same days used for all five markets (so cross-market correlation, serial correlation and the overlap of the horizons stay inside each block); "
          f"{int(b['resamples'])} resamples, seed {int(b['seed'])}. The strongest baseline of each market is held at its full-window identity; the `reselected` columns re-choose it in every resample (more conservative). "
          "Intervals are 95% percentile intervals for the macro improvement in rupee MAE, in %.")
        a("")
        a(_md(ci[["h", "point_estimate_pct", "ci95_low_strongest", "ci95_high_strongest", "share_resamples_above_0_strongest", "ci95_low_strongest_reselected",
                  "ci95_high_strongest_reselected", "point_estimate_vs_naive_pct", "ci95_low_vs_naive", "ci95_high_vs_naive"]]))
        a("")
        a("### 1.2 Multiple testing: Holm adjustment of the ten Diebold-Mariano p-values (analysis only)")
        a("")
        a(f"Ten tests (5 markets x 2 horizons) are made for the headline model. The Bonferroni threshold at 5% is {holm['bonferroni_threshold'].iloc[0]:.4f}. "
          f"Raw p < 0.05 in {int(holm['significant_raw_5pct'].sum())} cells; after Holm adjustment {int(holm['significant_holm_5pct'].sum())} remain significant at 5% "
          f"(Bonferroni: {int(holm['significant_bonferroni'].sum())}).")
        a("")
        a(_md(holm[["market", "h", "strongest_baseline", "DM", "p_value", "p_holm", "significant_raw_5pct", "significant_holm_5pct", "significant_bonferroni"]]))
        a("")

    a("## 2. Every family on every market and horizon")
    a("")
    a("Improvement in rupee MAE over the strongest baseline, in % (positive = better). `prophet` is off the shelf; `prophet_anchored` is the declared variant.")
    a("")
    a(_md(summ[["market", "h"] + [f"impr_{f}_vs_strongest_pct" for f in FAM]]))
    a("")
    a("Macro average:")
    a("")
    a(_md(macro[["h"] + [f"impr_{f}_vs_strongest_pct" for f in FAM]]))
    a("")
    a("Same, against plain naive:")
    a("")
    a(_md(macro[["h"] + [f"impr_{f}_vs_naive_pct" for f in FAM]]))
    a("")
    a("Diebold-Mariano p-values of every family against the strongest baseline (no model selection involved):")
    a("")
    a(_md(dmall.pivot(index=["market", "h"], columns="model", values="p_value").reset_index()[["market", "h"] + FAM]))
    a("")

    a("## 3. Secondary rows: per-cell tuning-selected and test-best")
    a("")
    a("The per-cell model is the family with the lowest rupee MAE on the 2017-2019 folds in that market and horizon (fixed before the final run). "
      "The test-best model is the family with the lowest MAE on 2020-2024 in that cell; it is chosen after seeing the results, so its improvement is optimistic and is shown only for transparency.")
    a("")
    a(_md(summ[["market", "h", "per_cell_selected_model", "per_cell_impr_vs_strongest_pct", "test_best_model_(optimistic)", "test_best_impr_vs_strongest_pct", "headline_impr_vs_strongest_pct"]]))
    a("")
    a("Macro average of these three rows:")
    a("")
    a(_md(macro[["h", "headline_impr_vs_strongest_pct", "per_cell_impr_vs_strongest_pct", "test_best_impr_vs_strongest_pct"]]))
    a("")

    a("## 4. All metrics per market, horizon and model")
    a("")
    a("Rupee scale except `MAE_log`. MASE is scaled by the in-sample naive error of 2014-2019 for that market and horizon.")
    a("")
    mm = met.copy()
    a("Macro average over markets:")
    a("")
    a(_md(mm.groupby(["h", "model"])[["MAE", "RMSE", "MASE", "sMAPE", "MAE_log"]].mean().reset_index()))
    a("")
    a("Per market:")
    a("")
    a(_md(mm[["market", "h", "model", "n", "MAE", "RMSE", "MASE", "sMAPE", "MAE_log"]]))
    a("")

    a("## 5. Robustness: per year, win rate, median error")
    a("")
    a("### 5.1 By target year (headline model vs the strongest baseline, macro average over markets, %)")
    a("")
    cnt = py.assign(pos=py["impr_vs_strongest_pct"] > 0).groupby(["h", "year"])["pos"].sum().unstack(0)
    t = pyh.copy()
    t.columns = [f"impr_h{h}_%" for h in t.columns]
    t[[f"markets_positive_h{h}_of_5" for h in cnt.columns]] = cnt.to_numpy()
    a(_md(t.reset_index()))
    a("")
    a(f"Years in which the macro improvement is negative: 7 days {neg_years[7] or 'none'}; 30 days {neg_years[30] or 'none'}.")
    a("")
    a("Per market and year:")
    a("")
    a(_md(py))
    a("")
    if (R / "loss_2021_by_month.csv").exists():
        bm, bk = _csv("loss_2021_by_month"), _csv("loss_2021_by_market")
        a("### 5.3 The 2021 loss year, described (analysis only; no cause is claimed)")
        a("")
        a("Improvement of the headline model over the strongest baseline in 2021 by target month and market, in % (negative = the model loses). "
          "Nothing is changed in the model; this only shows where the losses are.")
        a("")
        for h in (7, 30):
            piv = bm[bm["h"] == h].pivot(index="month", columns="market", values="impr_vs_strongest_pct")
            piv["all markets (mean)"] = piv.mean(axis=1)
            a(f"**h = {h} days**")
            a("")
            a(_md(piv.reset_index()))
            a("")
        ex = bm.groupby(["h", "month"]).apply(lambda g: np.average(g["excess_AE_per_forecast"], weights=g["n"]), include_groups=False).rename("excess_rupees_per_forecast").reset_index()
        tot = bm.groupby("h").apply(lambda g: (g["excess_AE_per_forecast"] * g["n"]).sum(), include_groups=False)
        ex["share_of_year_excess_%"] = [100 * (g.excess_rupees_per_forecast * bm[(bm["h"] == g.h) & (bm["month"] == g.month)]["n"].sum()) / tot[g.h] for g in ex.itertuples()]
        a("Excess absolute error of the headline model over the strongest baseline (rupees per forecast; positive = worse), by month, and each month's share of the year's total excess:")
        a("")
        a(_md(ex))
        a("")
        a("2021 compared with the other years, by market (movement = mean absolute log change from the origin quote to the target; signed error = forecast minus actual on the log scale):")
        a("")
        a(_md(bk))
        a("")
    a("### 5.2 Win rate and median error (macro average over markets)")
    a("")
    a("Win rate = share of triples in which the model's absolute rupee error is smaller than the comparator's (ties count half). "
      "A model can win on the mean and lose on the median when its gain comes from a minority of large moves.")
    a("")
    g = rob.groupby(["h", "model"])[["win_rate_vs_strongest_pct", "win_rate_vs_naive_pct", "median_AE", "median_AE_impr_vs_strongest_pct"]].mean().reset_index()
    a(_md(g))
    a("")
    a(f"LightGBM: win rate against the strongest baseline {rb_l.loc[7, 'win_rate_vs_strongest_pct']:.1f}% at 7 days and {rb_l.loc[30, 'win_rate_vs_strongest_pct']:.1f}% at 30 days; "
      f"change in median absolute error {rb_l.loc[7, 'median_AE_impr_vs_strongest_pct']:+.1f}% and {rb_l.loc[30, 'median_AE_impr_vs_strongest_pct']:+.1f}% (positive = smaller error).")
    a("")
    a("Per market:")
    a("")
    a(_md(rob[rob["model"].isin(["lgbm", "base_naive"])][["market", "h", "model", "n", "median_AE", "median_AE_strongest_baseline", "win_rate_vs_strongest_pct", "win_rate_vs_naive_pct"]]))
    a("")

    if (R / "gain_concentration.csv").exists():
        gc = _csv("gain_concentration")
        a("### 5.4 Where does the mean gain come from? (analysis only)")
        a("")
        a("Saving on each forecast = |strongest-baseline error| - |headline error| in rupees. The table shows how much of the total saving comes from the 10% of forecasts with the largest baseline errors (the large moves), "
          "the share of forecasts on which the model is better, and the typical (median) saving. A mean gain that comes from the large moves with a zero median saving means the model is tied on ordinary days.")
        a("")
        a(_md(gc))
        a("")
    a("## 6. Error analysis")
    a("")
    a("Segments are defined by the **target date** (the date the price is realised) for event periods, and by the days since the previous observed quote before the origin for gaps. "
      "Pooled on the log scale (equal weight per forecast); rupee MAE is shown too. Event periods follow the sourced dates in `docs/eda_findings.md` (export bans E5-E6, E9-E10, E12-E13; COVID lockdown E7-E8; MEP-only E11-E12).")
    a("")
    for name, title in [("event_group", "6.1 Export-ban periods vs normal periods"), ("event_period", "6.2 Each event period"), ("gap_before_origin", "6.3 By gap since the previous observed quote")]:
        a(f"### {title}")
        a("")
        a(_md(err[err["split_by"] == name].drop(columns="split_by")))
        a("")
    a("Small segments (the COVID lockdown, and gaps of 7+ days) have few forecasts; read them as indications only.")
    a("")

    a("## 7. Sensitivity checks (side by side, never used to choose between models)")
    a("")
    a("### 7.1 Sensitivity (a): every 'likely error' flagged point set to missing")
    a("")
    if "final_error" in runs:
        ex = runs["final_error"]
        a(f"Run `{ex['dir']}` (commit `{ex['git']['commit']}`, dirty `{ex['git']['dirty']}`), same frozen models refit on the modified panel. "
          f"The review table lists {ex.get('likely_error_points_in_review')} 'likely error' points; the ones not already set to missing were: {ex.get('newly_set_to_missing')}.")
        a("")
        s_err = _csv("sens_error_summary_table")
        m_err = _csv("sens_error_summary_macro")
        cmp = summ[["market", "h", "n", "strongest_baseline", "headline_impr_vs_strongest_pct"]].merge(
            s_err[["market", "h", "n", "strongest_baseline", "headline_impr_vs_strongest_pct"]], on=["market", "h"], suffixes=("_main", "_error_set_missing"))
        a(_md(cmp))
        a("")
        a("Macro average, improvement over the strongest baseline in %:")
        a("")
        a(_md(macro[["h", "headline_impr_vs_strongest_pct"]].merge(m_err[["h", "headline_impr_vs_strongest_pct"]], on="h", suffixes=("_main", "_error_set_missing"))))
    else:
        a("(not run)")
    a("")
    a("### 7.2 Sensitivity (b): with and without `is_stale` dates")
    a("")
    a("Triples whose origin or target date lies in a run of 5 or more identical (min, max, modal) quotes are excluded in the second scoring set; the strongest baseline is recomputed on each set.")
    a("")
    st = stale.pivot_table(index=["market", "h"], columns="scoring_set", values=["n", "headline_impr_vs_strongest_pct"], aggfunc="first")
    st.columns = [f"{a_}|{b_}" for a_, b_ in st.columns]
    a(_md(st.reset_index()))
    a("")
    sm = stale.groupby(["scoring_set", "h"])["headline_impr_vs_strongest_pct"].mean().unstack(0)
    a("Macro average improvement over the strongest baseline, in %:")
    a("")
    a(_md(sm.reset_index()))
    a("")

    a("## 8. Runtime and dropped triples")
    a("")
    a(f"Wall-clock for the final run: {fin['timings']['wall_seconds']} s on 12 worker processes (so per-fit seconds include contention; LightGBM used 4 threads per fit).")
    a("")
    a(_md(rt))
    a("")
    a("Triples dropped for all models (a triple any model cannot forecast is dropped for everyone):")
    a("")
    a(_md(drop))
    a("")

    a("## 9. Post-hoc: feature-group ablation of the frozen LightGBM (not used for any choice)")
    a("")
    ab_path = R / "ablation_lgbm.csv"
    if ab_path.exists():
        ab = pd.read_csv(ab_path)
        ar = runs.get("ablation_post_hoc", {})
        a("**This section is post-hoc and was not used for any choice.** The frozen LightGBM was refit with one feature group removed at a time, with the same hyperparameters, "
          "the same monthly refits and the same triples as the final run. Groups: calendar (target month, week of year, weekday, origin weekday, expected-closure flag); "
          "other-market (other markets' 7- and 28-day changes and the origin's price relative to them); mean-reversion (deviation of the last quote from its 3-quote mean and from its 30-day mean); "
          "days-since-last-quote (gap before the origin). "
          f"Run `{ar.get('dir', '?')}` (commit `{ar.get('git', {}).get('commit', '?')}`, dirty `{ar.get('git', {}).get('dirty', '?')}`).")
        a("")
        a("Macro average over markets. `change_vs_full_%` is the change in rupee MAE when the group is dropped (positive = the group was helping); the last column is the remaining improvement over the strongest baseline:")
        a("")
        a(_md(ab.groupby(["h", "variant"])[["MAE", "change_vs_full_MAE", "change_vs_full_pct", "impr_vs_strongest_pct"]].mean().reset_index()))
        a("")
        a("Per market:")
        a("")
        a(_md(ab[["market", "h", "variant", "n", "MAE", "change_vs_full_pct", "impr_vs_strongest_pct"]]))
    else:
        a("(not run)")
    a("")
    return "\n".join(d)


def main() -> None:
    """Write docs/model_report.md (numbers and tables; commentary is appended by hand below the marker)."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config()
    path = ROOT / cfg["paths"]["docs"] / "model_report.md"
    marker = "<!-- COMMENTARY BELOW IS WRITTEN BY HAND; EVERYTHING ABOVE IS GENERATED -->"
    commentary = ""
    if path.exists() and marker in path.read_text(encoding="utf-8"):
        commentary = path.read_text(encoding="utf-8").split(marker, 1)[1]
    path.write_text(build() + "\n" + marker + commentary, encoding="utf-8")
    logger.info("wrote %s", path)


if __name__ == "__main__":
    main()
