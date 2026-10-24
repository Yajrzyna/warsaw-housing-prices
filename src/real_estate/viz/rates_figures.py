"""Figures for Phase 8: price-rate dependence (rt_*) and forecasts (fc_*)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from real_estate.config import REPORTS_TABLES_DIR as T
from real_estate.viz import style


def _q(path) -> pd.DataFrame:
    d = pd.read_csv(path, index_col=0)
    d.index = pd.PeriodIndex(d.index, freq="Q")
    return d


def _x(idx):
    return pd.PeriodIndex(idx, freq="Q").to_timestamp(how="start") + pd.Timedelta(days=45)


def connected_scatter(lag: int = 4):
    """Price growth vs the rate `lag` quarters earlier: nominal (left) and real (right)."""
    q = _q(T / "rt_quarterly.csv")
    growth = (q["hedonic"] / q["hedonic"].shift(4) - 1) * 100
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), sharey=True)
    years = np.array([p.year + (p.quarter - 1) / 4 for p in q.index])
    tables = []
    for ax, col, title in [(axes[0], "mortgage_rate", "Nominal mortgage rate"),
                           (axes[1], "real_rate", "Real mortgage rate (minus inflation)")]:
        x = q[col].shift(lag)
        d = pd.DataFrame({"x": x, "y": growth, "t": years}, index=q.index).dropna()
        d = d.loc[pd.Period("2008Q1", "Q"):]      # skip the 2006-07 boom (+59% y/y), as in the timeline
        r = d["x"].corr(d["y"])
        ax.plot(d["x"], d["y"], color=style.GRID, lw=1.2, zorder=1)
        sc = ax.scatter(d["x"], d["y"], c=d["t"], cmap=style.SEQ_CMAP, s=34, edgecolor=style.SURFACE,
                        linewidth=1.2, zorder=2, vmin=2008, vmax=2026)
        b = np.polyfit(d["x"], d["y"], 1)
        xs = np.linspace(d["x"].min(), d["x"].max(), 10)
        ax.plot(xs, np.polyval(b, xs), color=style.INK_2, lw=1.2, zorder=1)
        for lab in ["2008Q4", "2013Q1", "2016Q4", "2020Q4", "2022Q4", "2024Q2", "2026Q1"]:
            p = pd.Period(lab, "Q")
            if p in d.index:
                ax.annotate(lab[:4], (d.loc[p, "x"], d.loc[p, "y"]), xytext=(5, 5), textcoords="offset points",
                            fontsize=8.5, color=style.INK_2)
        ax.axhline(0, color=style.AXIS, lw=1)
        ax.set_xlabel(f"{title}, {lag} quarters earlier (%)")
        ax.set_title(f"{title}: correlation {r:+.2f}", fontsize=11, pad=6)
        tables.append(d.rename(columns={"x": f"{col}_lag{lag}", "y": "price_growth_yoy"}))
    axes[0].set_ylabel("Resale price growth, year on year (%)")
    cb = fig.colorbar(sc, ax=axes, orientation="horizontal", fraction=0.04, pad=0.14, aspect=50)
    cb.outline.set_visible(False)
    cb.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}"))
    cb.set_label("Year (lighter = earlier)", color=style.INK_2)
    fig.suptitle("Prices follow the real interest rate, not the nominal one", x=0.01, y=1.0, ha="left", va="top",
                 fontsize=13, fontweight="bold", color=style.INK)
    fig.text(0.01, 0.955, "Each dot is a quarter, 2008–2026, joined in time order. In 2022 the nominal rate hit 9% "
             "but inflation was higher, so the real rate was deeply negative.", fontsize=9.5, color=style.INK_2, va="top")
    fig.subplots_adjust(top=0.84, bottom=0.3, wspace=0.08)
    style.source_note(fig, "Source: RCN Warszawa (hedonic index), Eurostat (3M rate, HICP). Mortgage rate = 3M + 2 pp.")
    t = tables[0].join(tables[1].drop(columns=["price_growth_yoy", "t"]))
    t.index = t.index.astype(str)
    return style.save(fig, "rt_01_rates_vs_growth", t.round(2))


def capacity():
    q = _q(T / "rt_quarterly.csv")
    x = _x(q.index)
    price_idx = q["hedonic"]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True, gridspec_kw={"height_ratios": [1.6, 1]})
    ax.axhline(100, color=style.AXIS, lw=1)
    ax.plot(x, q["capacity_index"], color=style.SLOTS[1], label="Borrowing capacity of an average wage")
    ax.plot(x, price_idx, color=style.SLOTS[0], label="Resale price index (hedonic)")
    for col, lab in [("capacity_index", "Capacity"), ("hedonic", "Prices")]:
        style.end_label(ax, x[-1], q[col].iloc[-1], f"{lab} {q[col].iloc[-1]:.0f}")
    ax.set_ylim(0)
    ax.set_ylabel("Index, 2015 = 100")
    ax.legend(loc="upper left")
    ax.set_title("What an average wage can borrow swings far more than prices")
    style.subtitle(ax, "Max loan with a payment of 40% of the gross wage over 25 years at the mortgage rate; 2015 = 100")
    ax2.axhline(1, color=style.AXIS, lw=1)
    ax2.plot(x, q["price_to_capacity"], color=style.SLOTS[0])
    style.end_label(ax2, x[-1], q["price_to_capacity"].iloc[-1], f"{q['price_to_capacity'].iloc[-1]:.2f}")
    ax2.set_title("Median 50 m² flat ÷ what an average wage can borrow (80% loan)", fontsize=11, pad=6)
    ax2.set_xlim(x[0], x[-1] + pd.Timedelta(days=600))
    ax2.set_xticks(pd.to_datetime([f"{y}-01-01" for y in range(2008, 2027, 2)]), range(2008, 2027, 2))
    fig.tight_layout()
    style.source_note(fig, "Source: RCN Warszawa, GUS BDL (wages), Eurostat. Above 1: one average wage cannot finance the flat.")
    t = q[["capacity_index", "hedonic", "price_to_capacity", "max_loan", "mortgage_rate"]].round(3)
    t.index = t.index.astype(str)
    return style.save(fig, "rt_02_borrowing_capacity", t)


def impulse_responses():
    irf = pd.read_csv(T / "rt_impulse_responses.csv")
    panels = [("Price (nominal)", "nominal mortgage rate"), ("Price (nominal)", "real mortgage rate"),
              ("Resale volume", "nominal mortgage rate"), ("Resale volume", "real mortgage rate")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    for ax, (o, s) in zip(axes.flat, panels):
        g = irf[(irf["outcome"] == o) & (irf["shock"] == s)]
        color = style.SLOTS[0] if o.startswith("Price") else style.SLOTS[1]
        ax.axhline(0, color=style.AXIS, lw=1)
        ax.fill_between(g["h"], g["lo90"], g["hi90"], color=color, alpha=0.12, lw=0)
        ax.plot(g["h"], g["beta"], color=color, marker="o", ms=5, markeredgecolor=style.SURFACE, markeredgewidth=1.2)
        last = g.iloc[-1]
        ax.annotate(f"{last.beta:+.1f}%", (last.h, last.beta), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=style.INK_2)
        ax.set_title(f"{o} after +1 pp {s}", fontsize=10.5, pad=6)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0f}%" if v else "0%"))
        ax.set_xlim(-0.3, 13.2)
    for ax in axes[1]:
        ax.set_xlabel("Quarters after the rate change")
    fig.suptitle("How prices and sales respond to a rate rise", x=0.01, y=0.995, ha="left", va="top",
                 fontsize=13, fontweight="bold", color=style.INK)
    fig.text(0.01, 0.955, "Local projections, 2008–2026: cumulative change after a +1 pp change in the mortgage rate, "
             "controlling for inflation, wage growth and past moves. Band: 90% confidence.", fontsize=9.5,
             color=style.INK_2, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    style.source_note(fig, "Source: RCN Warszawa, Eurostat, GUS. Rate changes respond to the economy, so these are associations, not clean causal effects.")
    return style.save(fig, "rt_03_impulse_responses", irf.round(3).set_index(["outcome", "shock", "h"]))


def rolling_correlation():
    rc = _q(T / "rt_rolling_correlation.csv").dropna()
    x = _x(rc.index)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.axhline(0, color=style.AXIS, lw=1)
    for col, c in zip(rc.columns, [style.SLOTS[0], style.SLOTS[1]]):
        ax.plot(x, rc[col], color=c, label=f"Price growth vs {col}")
    ax.set_ylim(-1, 1)
    ax.legend(loc="upper right")
    ax.set_xticks(pd.to_datetime([f"{y}-01-01" for y in range(2012, 2027, 2)]), range(2012, 2027, 2))
    ax.set_title("The link between rates and prices changes sign from one period to the next")
    style.subtitle(ax, "Correlation over the previous 5 years (20 quarters), resale price growth year on year")
    style.source_note(fig, "Source: RCN Warszawa, Eurostat.")
    t = rc.round(3)
    t.index = t.index.astype(str)
    return style.save(fig, "rt_04_rolling_correlation", t)


def backtest_accuracy():
    acc = pd.read_csv(T / "fc_accuracy.csv")
    names = {"naive": "No change", "drift": "Trend (last 5 yrs)", "ar": "Autoregressive",
             "arx": "Autoregressive + rates & wages", "combo": "Average of the two"}
    colors = {"naive": style.DEEMPH, "drift": style.SLOTS[3], "ar": style.SLOTS[2], "arx": style.SLOTS[1],
              "combo": style.SLOTS[0]}
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for m in ["naive", "drift", "ar", "arx", "combo"]:
        g = acc[acc["model"] == m]
        ax.plot(g["h"], g["rmse"], color=colors[m], label=names[m], marker="o", ms=5,
                markeredgecolor=style.SURFACE, markeredgewidth=1.2, lw=2.4 if m == "combo" else 1.8)
    ax.set_ylim(0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_xlabel("Forecast horizon (quarters ahead)")
    ax.legend(loc="upper left")
    ax.set_title("How far off were the forecasts? Backtest 2012–2026")
    style.subtitle(ax, "Root mean squared error of the forecast index level; each model re-estimated at every origin "
                       "with data available then")
    style.source_note(fig, "Source: RCN Warszawa hedonic index. Drivers held at their last value in the backtest (no look-ahead).")
    return style.save(fig, "fc_01_backtest_accuracy", acc.round(3).set_index(["model", "h"]))


def fan_chart():
    fc = pd.read_csv(T / "fc_forecast.csv")
    hist = _q(T / "rt_quarterly.csv")["hedonic"].loc[pd.Period("2019Q1", "Q"):]
    base = fc[fc["scenario"] == "rates hold"]
    xq = pd.PeriodIndex(base["q"], freq="Q")
    xh, xf = _x(hist.index), _x(xq)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(xh, hist, color=style.SLOTS[0], label="Hedonic index (observed)")
    link = lambda s: np.concatenate([[hist.iloc[-1]], s])
    xl = np.concatenate([[xh[-1]], xf])
    ax.fill_between(xl, link(base["lo80"]), link(base["hi80"]), color=style.SLOTS[0], alpha=0.12, lw=0,
                    label="80% range (spread of past errors)")
    ax.plot(xl, link(base["index"]), color=style.SLOTS[0], lw=2, ls=(0, (1, 1.5)), label="Forecast: rates hold")
    for name, c in [("real rate −1 pp", style.SLOTS[2]), ("real rate +1 pp", style.SLOTS[1])]:
        s = fc[fc["scenario"] == name]
        ax.plot(xl, link(s["index"]), color=c, lw=1.6, label=f"Forecast: {name}")
    end = base.iloc[-1]
    style.end_label(ax, xf[-1], end["index"], f"{end['index']:.0f} ({end['change_vs_last_pct']:+.1f}%)")
    ax.set_ylabel("Index, 2015 = 100")
    ax.legend(loc="upper left")
    ax.set_xlim(xh[0], xf[-1] + pd.Timedelta(days=300))
    ax.set_title("Outlook: roughly flat, with wide uncertainty")
    style.subtitle(ax, f"Resale hedonic index, forecast {base['q'].iloc[0]}–{base['q'].iloc[-1]} (average of AR and "
                       f"AR-with-rates models); scenarios shift the real mortgage rate")
    style.source_note(fig, "Dotted line = forecast. 80% range = spread of the model's past errors at each horizon (their average bias removed).")
    return style.save(fig, "fc_02_forecast", fc.round(2).set_index(["scenario", "q"]))


ALL = [connected_scatter, capacity, impulse_responses, rolling_correlation, backtest_accuracy, fan_chart]
