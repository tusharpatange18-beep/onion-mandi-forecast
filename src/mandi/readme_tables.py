"""Generate the result blocks of README.md from results/*.csv (no number in them is typed by hand).

Usage:
    python -m mandi.readme_tables           # rewrite the blocks between the marker comments in README.md
    python -m mandi.readme_tables --check   # exit 1 if README.md is not what the result files produce (used in CI)

A block is everything between `<!-- BEGIN GENERATED: name -->` and `<!-- END GENERATED: name -->`.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from mandi.config import ROOT, load_config

R = ROOT / "results"
FAM = {"ets": "ETS (= ARIMA(0,1,1))", "sarima": "SARIMA(0,1,1)", "prophet": "Prophet (off the shelf)", "prophet_anchored": "Prophet anchored", "lgbm": "LightGBM"}


def _csv(name: str) -> pd.DataFrame:
    return pd.read_csv(R / f"{name}.csv")


def _md(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        out.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in row) + " |")
    return "\n".join(out)


def _pct(x: float, nd: int = 1, sign: bool = True) -> str:
    return f"{x:+.{nd}f}%" if sign else f"{x:.{nd}f}%"


# --------------------------------------------------------------------------- blocks
def block_headline() -> str:
    b = _csv("bootstrap_ci").set_index("h")
    m = _csv("summary_macro").set_index("h")
    py = _csv("per_year").groupby(["h", "year"])["impr_vs_strongest_pct"].mean().unstack(0)
    br = _csv("interval_breakdowns")
    cov = _csv("interval_coverage_macro")
    ban = br[(br["group_type"] == "event_group") & (br["group"] == "export_ban") & (br["method"] == "lgbm_abs")].set_index(["h", "level"])["coverage"]
    nominal = cov[cov["method"] == "lgbm_abs"].set_index(["h", "level"])["macro_coverage"]
    neg = [int(y) for y in py.index if (py.loc[y] < 0).all()]
    tie7 = b.loc[7, "ci95_low_strongest"] <= 0 <= b.loc[7, "ci95_high_strongest"]
    lines = [
        f"- **30-day forecasts:** LightGBM improves the mean absolute error (rupees) over the strongest simple baseline by **{_pct(m.loc[30, 'headline_impr_vs_strongest_pct'])}** "
        f"(macro average of five markets; 95% block-bootstrap interval {_pct(b.loc[30, 'ci95_low_strongest'])} to {_pct(b.loc[30, 'ci95_high_strongest'])}).",
        f"- **7-day forecasts:** {_pct(m.loc[7, 'headline_impr_vs_strongest_pct'])} (interval {_pct(b.loc[7, 'ci95_low_strongest'])} to {_pct(b.loc[7, 'ci95_high_strongest'])})"
        + (": **a statistical tie with naive** (the last observed price)." if tie7 else "."),
        f"- **Not uniform over time:** {', '.join(map(str, neg)) or 'no year'} {'is a loss year' if len(neg) == 1 else 'are loss years'} at both horizons "
        f"(macro {', '.join(_pct(py.loc[y, h]) for y in neg for h in (7, 30))} at 7 and 30 days).",
        f"- **Prediction intervals** are calibrated on average (80% intervals cover {_pct(100 * nominal.loc[(7, 0.8)], sign=False)} at 7 days and {_pct(100 * nominal.loc[(30, 0.8)], sign=False)} at 30 days) "
        f"but **under-cover in shocks**: in export-ban periods the 80% interval covers {_pct(100 * ban.loc[(7, 0.8)], sign=False)} and {_pct(100 * ban.loc[(30, 0.8)], sign=False)}.",
    ]
    return "\n".join(lines)


def block_macro() -> str:
    m = _csv("summary_macro")
    rows = []
    for _, r in m.iterrows():
        row = {"horizon (days)": int(r["h"]), "headline LightGBM vs strongest baseline": _pct(r["headline_impr_vs_strongest_pct"]), "LightGBM vs naive": _pct(r["headline_impr_vs_naive_pct"])}
        for f, name in FAM.items():
            if f != "lgbm":
                row[f"{name} vs strongest"] = _pct(r[f"impr_{f}_vs_strongest_pct"])
        rows.append(row)
    return "Improvement in rupee MAE over the strongest baseline (positive = better), equal-weight average of the five markets:\n\n" + _md(pd.DataFrame(rows))


def block_per_market() -> str:
    s = _csv("summary_table")
    dm = _csv("dm_tests")
    dm = dm[dm["which"] == "headline"].set_index(["market", "h"])
    holm = _csv("holm_dm").set_index(["market", "h"])
    rows = []
    for _, r in s.sort_values(["h", "market"]).iterrows():
        k = (r["market"], r["h"])
        rows.append({"market": r["market"], "h": int(r["h"]), "strongest baseline": r["strongest_baseline"].replace("base_", ""), "baseline MAE (Rs)": f"{r['baseline_MAE']:.0f}",
                     "LightGBM MAE (Rs)": f"{r['headline_MAE']:.0f}", "vs strongest": _pct(r["headline_impr_vs_strongest_pct"]), "vs naive": _pct(r["headline_impr_vs_naive_pct"]),
                     "DM p": f"{dm.loc[k, 'p_value']:.3f}", "Holm p": f"{holm.loc[k, 'p_holm']:.3f}", "forecasts": int(r["n"])})
    return ("Per market (Diebold-Mariano test against the strongest baseline, Harvey correction; Holm adjustment over the ten tests):\n\n" + _md(pd.DataFrame(rows)))


def block_bootstrap() -> str:
    b = _csv("bootstrap_ci")
    rows = [{"horizon (days)": int(r["h"]), "improvement vs strongest": _pct(r["point_estimate_pct"]),
             "95% interval": f"{_pct(r['ci95_low_strongest'])} to {_pct(r['ci95_high_strongest'])}", "share of resamples above 0": f"{100 * r['share_resamples_above_0_strongest']:.0f}%",
             "improvement vs naive": _pct(r["point_estimate_vs_naive_pct"]), "95% interval (vs naive)": f"{_pct(r['ci95_low_vs_naive'])} to {_pct(r['ci95_high_vs_naive'])}"} for _, r in b.iterrows()]
    first = b.iloc[0]
    return (f"Block bootstrap of origin days ({int(first['block_days'])}-day blocks, the same days for all markets, {int(first['resamples'])} resamples, seed {int(first['seed'])}):\n\n" + _md(pd.DataFrame(rows)))


def block_coverage() -> str:
    c = _csv("interval_coverage_macro")
    c = c[c["method"].isin(["lgbm_abs", "naive_abs"])]
    rows = [{"method": "LightGBM conformal" if r["method"] == "lgbm_abs" else "naive conformal (comparator)", "h": int(r["h"]), "nominal": f"{100 * r['level']:.0f}%",
             "coverage": f"{100 * r['macro_coverage']:.1f}%", "95% interval": f"{100 * r['ci_low']:.1f}% to {100 * r['ci_high']:.1f}%",
             "mean width (Rs)": f"{r['macro_avg_width_rs']:.0f}", "width / forecast": f"{100 * r['macro_avg_rel_width']:.0f}%"} for _, r in c.sort_values(["h", "level", "method"]).iterrows()]
    return "Rolling split-conformal intervals, 2020-2024 (macro over markets; block-bootstrap interval of the coverage):\n\n" + _md(pd.DataFrame(rows))


def block_per_year() -> str:
    p = _csv("per_year").groupby(["year", "h"])["impr_vs_strongest_pct"].mean().unstack("h")
    df = pd.DataFrame({"target year": p.index.astype(int), "7 days": [_pct(v) for v in p[7]], "30 days": [_pct(v) for v in p[30]]})
    return "LightGBM vs the strongest baseline by target year (macro over markets):\n\n" + _md(df)


def block_events() -> str:
    e = _csv("error_analysis")
    e = e[e["split_by"] == "event_group"]
    br = _csv("interval_breakdowns")
    br = br[(br["group_type"] == "event_group") & (br["method"] == "lgbm_abs") & (br["level"] == 0.8)].set_index(["h", "group"])["coverage"]
    rows = [{"period of the target date": r["segment"], "h": int(r["h"]), "forecasts": int(r["n"]), "log-MAE improvement vs strongest": _pct(r["log_impr_vs_strongest_pct"]),
             "80% interval coverage": f"{100 * br.loc[(r['h'], r['segment'])]:.1f}%"} for _, r in e.sort_values(["h", "segment"]).iterrows()]
    return "By event period (high-confidence event dates; see docs/eda_findings.md):\n\n" + _md(pd.DataFrame(rows))


def block_gaps() -> str:
    e = _csv("error_analysis")
    e = e[e["split_by"] == "gap_before_origin"]
    rows = [{"days since the previous quote": r["segment"], "h": int(r["h"]), "forecasts": int(r["n"]), "log-MAE improvement vs strongest": _pct(r["log_impr_vs_strongest_pct"])}
            for _, r in e.sort_values(["h", "segment"]).iterrows()]
    return _md(pd.DataFrame(rows))


def block_ablation() -> str:
    a = _csv("ablation_lgbm").groupby(["h", "variant"])[["change_vs_full_pct", "impr_vs_strongest_pct"]].mean().reset_index()
    rows = [{"h": int(r["h"]), "feature group dropped": r["variant"].replace("drop_", "").replace("_", " ") if r["variant"] != "full" else "(none: full model)",
             "change in MAE": _pct(r["change_vs_full_pct"]), "remaining improvement vs strongest": _pct(r["impr_vs_strongest_pct"])} for _, r in a.iterrows()]
    return "**Post-hoc, not used for any choice.** Refitting the frozen LightGBM without one feature group (macro over markets):\n\n" + _md(pd.DataFrame(rows))


def block_anomalies() -> str:
    rec, fpr, cnt = _csv("anomaly_synthetic_recall"), _csv("anomaly_synthetic_fpr"), _csv("anomaly_counts")
    fpr = fpr[fpr["kind"].str.startswith("unmodified")].set_index("detector")["false_positive_rate"]
    flags = cnt.groupby("detector")["n_flags"].sum()
    kind = {"stl_residual": "retrospective large-swing detector (not a spike detector)", "forecast_surprise_h7": "real time: outside the 95% interval issued 7 days earlier",
            "forecast_surprise_h30": "real time: outside the 95% interval issued 30 days earlier", "causal_daily": "real time: distance from a 28-quote median (noisy comparator)"}
    rows = []
    for d in ["forecast_surprise_h7", "forecast_surprise_h30", "stl_residual", "causal_daily"]:
        g = rec[rec["detector"] == d]
        r = g.groupby("magnitude").apply(lambda x: 100 * (x["recall"] * x["spikes"]).sum() / x["spikes"].sum(), include_groups=False)
        rows.append({"detector": d, "what it is": kind[d], "flags": int(flags.get(d, 0)), "false positives (ordinary dates)": f"{100 * fpr[d]:.1f}%",
                     **{f"recall, {int(100 * k)}% spike": f"{v:.0f}%" for k, v in r.items()}})
    return ("Recall and false-positive rate from injecting synthetic spikes into an in-memory copy of the series (a test of the detectors, not data):\n\n" + _md(pd.DataFrame(rows)))


BLOCKS = {"headline": block_headline, "macro": block_macro, "per_market": block_per_market, "bootstrap": block_bootstrap, "coverage": block_coverage,
          "per_year": block_per_year, "events": block_events, "gaps": block_gaps, "ablation": block_ablation, "anomalies": block_anomalies}
PATTERN = re.compile(r"(<!-- BEGIN GENERATED: (\w+) -->\r?\n)(.*?)(<!-- END GENERATED: \2 -->)", re.S)


def render(text: str) -> str:
    """Replace every generated block in `text` with freshly generated content."""
    found = set()

    def sub(m: re.Match) -> str:
        name = m.group(2)
        if name not in BLOCKS:
            raise KeyError(f"unknown generated block '{name}'")
        found.add(name)
        return m.group(1) + BLOCKS[name]() + "\n" + m.group(4)

    out = PATTERN.sub(sub, text)
    missing = set(BLOCKS) - found
    if missing:
        raise KeyError(f"README.md lacks generated blocks: {sorted(missing)}")
    return out


def main() -> None:
    """Rewrite or check README.md."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="exit 1 if README.md differs from what results/ produce")
    args = ap.parse_args()
    load_config()
    path = ROOT / "README.md"
    text = path.read_text(encoding="utf-8")
    new = render(text)
    if args.check:
        if new != text:
            print("README.md result blocks are out of date with results/*.csv: run `python -m mandi.readme_tables`", file=sys.stderr)
            sys.exit(1)
        print("README.md result blocks match results/*.csv")
        return
    path.write_text(new, encoding="utf-8")
    print(f"wrote {len(BLOCKS)} generated blocks into README.md")


if __name__ == "__main__":
    main()
