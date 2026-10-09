"""Phase 2: figures. All functions save a PNG and return its path."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mandi.eda import WEEKDAYS  # noqa: E402

# Colour-blind-safe palette (Okabe-Ito), one fixed colour per market in every figure.
COLORS = {"Solapur": "#0072B2", "Pune": "#D55E00", "Vashi": "#009E73", "Kolhapur": "#CC79A7", "Sambhajinagar": "#E69F00"}
EVENT_STYLE = {"export": dict(color="#555555", ls="--", lw=0.8), "covid": dict(color="#7B3294", ls="-", lw=1.0)}
SPLIT = pd.Timestamp("2020-01-01")

plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})


def _save(fig: plt.Figure, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, bbox_inches="tight", dpi=130)
    plt.close(fig)
    return path


def _events(ax: plt.Axes, events: list[dict[str, Any]], label: bool = True, xmin=None, xmax=None) -> None:
    """Draw event lines (grey dashed = export policy, purple = COVID) with their ids on top."""
    for e in events:
        d = pd.Timestamp(e["date"])
        if xmin is not None and not (xmin <= d <= xmax):
            continue
        ax.axvline(d, **EVENT_STYLE[e["kind"]], zorder=1)
        if label:
            ax.annotate(e["id"], (d, 1.0), xycoords=("data", "axes fraction"), fontsize=6, ha="center", va="bottom", color=EVENT_STYLE[e["kind"]]["color"])


def price_trend(panel: pd.DataFrame, events: list[dict[str, Any]], out: Path) -> Path:
    """Daily modal price (log scale) with a 30-observed-day rolling median, events marked."""
    mk = list(COLORS)
    fig, axes = plt.subplots(len(mk), 1, figsize=(11, 10), sharex=True)
    for ax, m in zip(axes, mk):
        g = panel[panel["m"] == m].set_index("date")["modal_price"].dropna()
        ax.scatter(g.index, g.values, s=2, alpha=0.25, color=COLORS[m], linewidths=0)
        ax.plot(g.rolling(30, min_periods=15).median(), color="black", lw=1.0)
        ax.set_yscale("log")
        ax.axvline(SPLIT, color="black", lw=0.8, ls=":")
        _events(ax, events, label=(m == mk[0]))
        ax.set_ylabel(f"{m}\nRs/quintal", fontsize=8)
    axes[0].set_title("Modal price by market (log scale). Black line = 30-observed-day rolling median. "
                      "Dotted = start of 2020 (test period). Grey dashed = export policy, purple = COVID (ids in docs)", fontsize=9)
    return _save(fig, out, "01_price_trend.png")


def yearly_seasonality(panel: pd.DataFrame, out: Path) -> Path:
    """Month-of-year profile of mean log price (minus the year's own mean), one line per year."""
    mk = list(COLORS)
    fig, axes = plt.subplots(1, len(mk), figsize=(15, 3.2), sharey=True)
    for ax, m in zip(axes, mk):
        g = panel[panel["m"] == m].dropna(subset=["modal_price"]).copy()
        g["lp"] = np.log(g["modal_price"])
        g["y"], g["mo"] = g["date"].dt.year, g["date"].dt.month
        prof = g.groupby(["y", "mo"])["lp"].agg(["mean", "count"])
        prof = prof[prof["count"] >= 10]["mean"].unstack(0)
        prof = prof - prof.mean()
        train_years = [y for y in prof.columns if y <= 2019]
        for y in prof.columns:
            if y >= 2020:
                ax.plot(prof.index, 100 * (np.exp(prof[y]) - 1), color="#bbbbbb", lw=0.8)
        for y in train_years:
            ax.plot(prof.index, 100 * (np.exp(prof[y]) - 1), color=COLORS[m], lw=1.0, alpha=0.5)
        ax.plot(prof.index, 100 * (np.exp(prof[train_years].mean(axis=1)) - 1), color="black", lw=2)
        ax.set_title(m, fontsize=9)
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(list("JFMAMJJASOND"))
    axes[0].set_ylabel("% vs the year's mean level")
    fig.suptitle("Yearly seasonality. Coloured = 2014-2019 (drives conclusions), black = 2014-2019 average, grey = 2020-2024 (description only)", y=1.04, fontsize=9)
    return _save(fig, out, "02_yearly_seasonality.png")


def weekday_effect(table: pd.DataFrame, out: Path) -> Path:
    """Price and arrivals effect by weekday, relative to the same week's mean."""
    mk = list(COLORS)
    fig, axes = plt.subplots(2, len(mk), figsize=(15, 5), sharex=True)
    for j, m in enumerate(mk):
        t = table[table["market"] == m].set_index("weekday").reindex(WEEKDAYS)
        ax = axes[0, j]
        err = [t["price_effect_pct"] - t["price_ci_lo_pct"], t["price_ci_hi_pct"] - t["price_effect_pct"]]
        ax.bar(WEEKDAYS, t["price_effect_pct"].fillna(0), yerr=np.nan_to_num(np.array(err, dtype=float)), color=COLORS[m], capsize=2)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_title(m, fontsize=9)
        axes[1, j].bar(WEEKDAYS, t["arrivals_effect_pct"].fillna(0), color="#888888")
        axes[1, j].axhline(0, color="black", lw=0.6)
    axes[0, 0].set_ylabel("price, % vs week mean\n(bars: 95% interval)")
    axes[1, 0].set_ylabel("arrivals, % vs week mean")
    fig.suptitle("Weekday effects, 2014-2019 (closed weekday has no bar)", y=1.0, fontsize=9)
    return _save(fig, out, "03_weekday_effects.png")


def price_vs_arrivals(wp: pd.DataFrame, wa: pd.DataFrame, lag_table: pd.DataFrame, out: Path) -> Path:
    """Weekly log price vs log arrivals (2014-2019) and lagged cross-correlation of their weekly changes."""
    mk = list(COLORS)
    fig, axes = plt.subplots(2, len(mk), figsize=(15, 6))
    for j, m in enumerate(mk):
        ok = wp[m].notna() & wa[m].notna()
        axes[0, j].scatter(wa[m][ok], wp[m][ok], s=6, alpha=0.5, color=COLORS[m], linewidths=0)
        axes[0, j].set_title(f"{m}  r={wp[m][ok].corr(wa[m][ok]):.2f}", fontsize=9)
        axes[0, j].set_xlabel("log weekly arrivals")
        t = lag_table[lag_table["market"] == m]
        axes[1, j].bar(t["arrivals_lag_weeks"], t["corr_price_change_vs_lagged_arrivals_change"], color=COLORS[m])
        axes[1, j].axhline(0, color="black", lw=0.6)
        axes[1, j].set_xlabel("arrivals change lagged k weeks")
        axes[1, j].set_ylim(-0.5, 0.5)
    axes[0, 0].set_ylabel("log weekly price")
    axes[1, 0].set_ylabel("corr(price change, arrivals change k wk earlier)")
    fig.suptitle("Price vs arrivals, 2014-2019 (k=0 is the same week and is not usable at forecast time)", y=1.0, fontsize=9)
    return _save(fig, out, "04_price_vs_arrivals.png")


def comovement(corr_train: dict[str, pd.DataFrame], wp: pd.DataFrame, out: Path) -> Path:
    """Heatmaps of weekly price-change and price-level correlations (2014-2019) and the five weekly price series."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), gridspec_kw={"width_ratios": [1, 1, 1.8]})
    for ax, key, title in [(axes[0], "changes", "weekly log price CHANGES"), (axes[1], "levels", "weekly log price LEVELS")]:
        c = corr_train[key]
        im = ax.imshow(c.values, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(c)), c.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(c)), c.index if key == "changes" else [""] * len(c))
        for i in range(len(c)):
            for k in range(len(c)):
                ax.text(k, i, f"{c.values[i, k]:.2f}", ha="center", va="center", color="white" if c.values[i, k] < 0.6 else "black", fontsize=8)
        ax.set_title(f"Correlation of {title}\n2014-2019", fontsize=9)
    for m in COLORS:
        axes[2].plot(np.exp(wp[m]), color=COLORS[m], lw=0.8, label=m)
    axes[2].set_yscale("log")
    axes[2].axvline(SPLIT, color="black", lw=0.8, ls=":")
    axes[2].legend(fontsize=7, ncol=5, loc="upper left")
    axes[2].set_title("Weekly mean price of observed days (Rs/quintal, log scale)", fontsize=9)
    return _save(fig, out, "05_comovement.png")


def acf_pacf(acfs: dict[str, pd.DataFrame], title: str, name: str, out: Path) -> Path:
    """ACF (top) and PACF (bottom) per market with approximate 95% bands."""
    mk = [m for m in COLORS if m in acfs]
    fig, axes = plt.subplots(2, len(mk), figsize=(15, 5), sharex=True)
    for j, m in enumerate(mk):
        t = acfs[m]
        band = 1.96 / np.sqrt(t["n_valid"].iloc[0])
        for i, col in enumerate(["acf", "pacf"]):
            ax = axes[i, j]
            ax.bar(t["lag"][1:], t[col][1:], color=COLORS[m], width=0.7)
            ax.axhline(band, color="black", lw=0.6, ls="--")
            ax.axhline(-band, color="black", lw=0.6, ls="--")
            ax.set_ylim(-1, 1) if col == "acf" and t[col][1:].abs().max() > 0.5 else ax.set_ylim(-0.6, 0.6)
        axes[0, j].set_title(f"{m} (n={t['n_valid'].iloc[0]} wk)", fontsize=9)
        axes[1, j].set_xlabel("lag (weeks)")
    axes[0, 0].set_ylabel("ACF")
    axes[1, 0].set_ylabel("PACF")
    fig.suptitle(title, y=1.0, fontsize=9)
    return _save(fig, out, name)


def daily_acf(t: pd.DataFrame, out: Path) -> Path:
    """ACF of daily log price changes from observed pairs only."""
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for m in COLORS:
        s = t[t["market"] == m]
        ax.plot(s["lag_days"], s["acf"], marker="o", ms=3, color=COLORS[m], label=m)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("lag (days)")
    ax.set_ylabel("autocorrelation of daily log changes")
    ax.legend(fontsize=7)
    ax.set_title("Daily log price changes, 2014-2019, observed pairs only (no filled values)", fontsize=9)
    return _save(fig, out, "06c_daily_acf.png")


def missingness(miss: dict[str, pd.DataFrame], out: Path) -> Path:
    """Heatmaps: missing share by weekday, and unexpected missing share (closure weekday excluded) by year and by month."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 3.6))
    items = [(miss["by_weekday"], "Missing share by weekday (all missing days)", list(WEEKDAYS)),
             (miss["by_year"], "Unexpected missing share by year\n(closure weekday excluded)", None),
             (miss["by_month"], "Unexpected missing share by month\n(closure weekday excluded)", None)]
    for ax, (df, title, labels) in zip(axes, items):
        df = df.reindex([m for m in COLORS if m in df.index])
        im = ax.imshow(df.values, aspect="auto", cmap="magma_r", vmin=0, vmax=max(0.5, np.nanmax(df.values) if df.size else 0.5) if labels is None else 1)
        ax.set_yticks(range(len(df)), df.index if ax is axes[0] else [""] * len(df))
        ax.set_xticks(range(df.shape[1]), [str(c) for c in df.columns], rotation=90 if df.shape[1] > 8 else 0)
        for i in range(df.shape[0]):
            for k in range(df.shape[1]):
                v = df.values[i, k]
                if pd.notna(v):
                    ax.text(k, i, f"{v:.2f}" if labels else f"{100 * v:.0f}", ha="center", va="center", fontsize=6.5,
                            color="white" if v > 0.5 * im.get_clim()[1] else "black")
        ax.set_title(title, fontsize=9)
    fig.suptitle("Missingness (numbers in the year and month panels are %)", y=1.05, fontsize=9)
    return _save(fig, out, "07_missingness.png")


def missing_timeline(p: pd.DataFrame, events: list[dict[str, Any]], out: Path) -> Path:
    """One strip per market: every day from 2014 to 2024, coloured by observed / closure day / unexpected missing."""
    mk = list(COLORS)
    days = pd.date_range("2014-01-01", "2024-12-31")
    img = np.full((len(mk), len(days)), np.nan)
    for i, m in enumerate(mk):
        g = p[p["m"] == m].set_index("date").reindex(days)
        img[i] = np.where(g["unexpected_missing"].eq(True), 2, np.where(g["is_closure"].eq(True), 1, np.where(g["was_missing"].eq(False), 0, np.nan)))
    from matplotlib.colors import ListedColormap

    fig, ax = plt.subplots(figsize=(14, 3.2))
    ax.imshow(img, aspect="auto", cmap=ListedColormap(["#d9d9d9", "#9ecae1", "#d7191c"]), vmin=0, vmax=2, interpolation="nearest",
              extent=[matplotlib.dates.date2num(days[0]), matplotlib.dates.date2num(days[-1]), len(mk), 0])
    ax.xaxis_date()
    ax.set_yticks(np.arange(len(mk)) + 0.5, mk)
    _events(ax, events, label=True, xmin=days[0], xmax=days[-1])
    ax.set_title("Each day: grey = observed, blue = regular closure weekday, red = unexpected missing, white = before the market's first quote", fontsize=9)
    return _save(fig, out, "08_missing_timeline.png")


def volatility(panel: pd.DataFrame, wp: pd.DataFrame, events: list[dict[str, Any]], out: Path) -> Path:
    """Rolling 26-week standard deviation of weekly log price changes, with events."""
    fig, ax = plt.subplots(figsize=(11, 3.8))
    d = wp.diff()
    for m in COLORS:
        ax.plot(d[m].rolling(26, min_periods=15).std(), color=COLORS[m], lw=1, label=m)
    ax.axvline(SPLIT, color="black", lw=0.8, ls=":")
    _events(ax, events, label=True)
    ax.set_ylabel("std of weekly log change\n(26-week window)")
    ax.legend(fontsize=7, ncol=5, loc="upper left", bbox_to_anchor=(0, 0.98))
    ax.set_title("Volatility over time", fontsize=9, pad=14)
    return _save(fig, out, "09_volatility.png")


def transform_check(panel: pd.DataFrame, out: Path) -> Path:
    """Rolling std of daily price changes against rolling mean price, log-log."""
    fig, axes = plt.subplots(1, len(COLORS), figsize=(15, 3), sharey=False)
    for ax, m in zip(axes, COLORS):
        x = panel[panel["m"] == m].set_index("date")["modal_price"].dropna()
        mean, sd = x.rolling(60, min_periods=40).mean(), x.diff().rolling(60, min_periods=40).std()
        ok = (mean > 0) & (sd > 0)
        ax.scatter(mean[ok], sd[ok], s=3, alpha=0.4, color=COLORS[m], linewidths=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        k = np.polyfit(np.log(mean[ok]), np.log(sd[ok]), 1)
        xs = np.linspace(mean[ok].min(), mean[ok].max(), 50)
        ax.plot(xs, np.exp(k[1]) * xs ** k[0], color="black", lw=1)
        ax.set_title(f"{m}  slope={k[0]:.2f}", fontsize=9)
        ax.set_xlabel("rolling mean price")
    axes[0].set_ylabel("rolling std of daily price change")
    fig.suptitle("Does variability grow with the price level? (2014-2019; slope near 1 = changes are proportional to the level)", y=1.02, fontsize=9)
    return _save(fig, out, "10_variance_vs_level.png")
