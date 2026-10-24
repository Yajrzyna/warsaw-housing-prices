"""Figures for the price indices (reports/figures/idx_*.png, table twins in reports/tables/)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from real_estate.indices.build import DISTRICT_PATH, QUARTERLY_PATH
from real_estate.viz import style

# method -> categorical slot (fixed; validated adjacent 4-slot set, light mode)
METHOD_STYLE = {
    "hedonic": (style.SLOTS[0], "Hedonic (headline)"),
    "repeat_sales": (style.SLOTS[1], "Repeat sales"),
    "mix_adjusted": (style.SLOTS[2], "Mix-adjusted"),
    "median": (style.SLOTS[3], "Median"),
}


def load_quarterly() -> pd.DataFrame:
    long = pd.read_parquet(QUARTERLY_PATH)
    long["q"] = pd.PeriodIndex(long["q"], freq="Q")
    return long


def _series(long, segment, method) -> pd.Series:
    s = long[(long["segment"] == segment) & (long["method"] == method)].set_index("q")["index"]
    return s.sort_index()


def _x(idx) -> pd.DatetimeIndex:
    return pd.PeriodIndex(idx, freq="Q").to_timestamp(how="start") + pd.Timedelta(days=45)


def _year_ticks(ax, start=2008, stop=2026, step=2):
    ax.set_xticks(pd.to_datetime([f"{y}-01-01" for y in range(start, stop + 1, step)]), range(start, stop + 1, step))


# ------------------------------------------------------------------ 1 methods
def methods_comparison(long):
    t = pd.DataFrame({m: _series(long, "secondary", m) for m in METHOD_STYLE})
    t["nbp_hedonic"] = _series(long, "secondary", "nbp_hedonic")
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    x = _x(t.index)
    ax.plot(x, t["nbp_hedonic"], color=style.DEEMPH, lw=1.5, label="NBP hedonic (benchmark)")
    for m, (c, lab) in list(METHOD_STYLE.items())[::-1]:
        ax.plot(x, t[m], color=c, label=lab, lw=2.4 if m == "hedonic" else 1.8)
    last = t.index[-1]
    style.end_label(ax, x[-1], t.loc[last, "hedonic"], f"Hedonic {t.loc[last, 'hedonic']:.0f}")
    style.end_label(ax, x[-1], t.loc[last, "nbp_hedonic"], f"NBP {t.loc[last, 'nbp_hedonic']:.0f}")
    ax.axhline(100, color=style.AXIS, lw=1)
    ax.set_ylim(0)
    ax.set_ylabel("Index, 2015 = 100")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="upper left")
    ax.set_title("Four methods, one story: secondary-market price index, Warsaw")
    style.subtitle(ax, "Quarterly, 2015 = 100. All RCN-based methods agree within a few points; "
                       "NBP's hedonic index (agent-reported sample) rises less after 2016")

    dev = (np.log(t[[m for m in METHOD_STYLE if m != "hedonic"] + ["nbp_hedonic"]].div(t["hedonic"], axis=0)) * 100)
    ax2.axhline(0, color=style.AXIS, lw=1)
    ax2.plot(x, dev["nbp_hedonic"], color=style.DEEMPH, lw=1.5)
    for m in ["repeat_sales", "mix_adjusted", "median"]:
        c, lab = METHOD_STYLE[m]
        ax2.plot(x, dev[m], color=c, lw=1.6)
    # RCN methods converge at the end -> legend (top panel) carries identity; label only NBP
    style.end_label(ax2, x[-1], dev["nbp_hedonic"].iloc[-1], f"NBP {dev['nbp_hedonic'].iloc[-1]:+.0f}")
    ax2.set_ylabel("vs hedonic (log pts ×100)")
    ax2.set_xlim(x[0], x[-1] + pd.Timedelta(days=900))
    _year_ticks(ax2)
    fig.tight_layout()
    style.source_note(fig, "Source: RCN Warszawa (cleaned), NBP BaRN. Hedonic = rolling 5-quarter time-dummy "
                           "regression with H3-cell fixed effects.")
    t.index = t.index.astype(str)
    return style.save(fig, "idx_01_methods_secondary", t.round(2)), t


# ------------------------------------------------------------------ 2 real vs nominal
def nominal_vs_real(long):
    t = pd.DataFrame({"nominal": _series(long, "secondary", "hedonic"),
                      "real": _series(long, "secondary", "hedonic_real")}).dropna()
    fig, ax = plt.subplots(figsize=(10, 5))
    x = _x(t.index)
    ax.axhline(100, color=style.AXIS, lw=1)
    ax.plot(x, t["nominal"], color=style.SLOTS[0], label="Nominal")
    ax.plot(x, t["real"], color=style.SLOTS[1], label="Real (deflated by HICP)")
    for col, lab in [("nominal", "Nominal"), ("real", "Real")]:
        style.end_label(ax, x[-1], t[col].iloc[-1], f"{lab} {t[col].iloc[-1]:.0f}")
    pk = t.loc[:pd.Period("2012Q4", "Q"), "real"].idxmax()
    ax.annotate(f"2008 real peak {t.loc[pk, 'real']:.0f}", (_x([pk])[0], t.loc[pk, "real"]),
                xytext=(0, 12), textcoords="offset points", ha="center", fontsize=9, color=style.INK_2)
    ax.set_ylim(0)
    ax.set_ylabel("Index, 2015 = 100")
    ax.set_xlim(x[0], x[-1] + pd.Timedelta(days=700))
    _year_ticks(ax)
    ax.legend(loc="upper left")
    ax.set_title("After inflation, flat prices went sideways for a decade, then rose ~50%")
    style.subtitle(ax, "Hedonic index, secondary market, 2015 = 100; real = nominal ÷ HICP Poland (2015 = 100)")
    style.source_note(fig, "Source: RCN Warszawa (cleaned), Eurostat prc_hicp_minr.")
    t.index = t.index.astype(str)
    return style.save(fig, "idx_02_nominal_vs_real", t.round(2)), t


# ------------------------------------------------------------------ 3 annual growth
def yoy_growth(long):
    t = pd.DataFrame({"hedonic": _series(long, "secondary", "hedonic"),
                      "nbp_hedonic": _series(long, "secondary", "nbp_hedonic")})
    yoy = (t / t.shift(4) - 1) * 100
    yoy = yoy.loc[pd.Period("2008Q1", "Q"):]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = _x(yoy.index)
    ax.axhline(0, color=style.AXIS, lw=1)
    ax.plot(x, yoy["nbp_hedonic"], color=style.DEEMPH, lw=1.5, label="NBP hedonic (benchmark)")
    ax.plot(x, yoy["hedonic"], color=style.SLOTS[0], label="RCN hedonic")
    style.end_label(ax, x[-1], yoy["hedonic"].iloc[-1], f"{yoy['hedonic'].iloc[-1]:+.1f}%")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0f}%" if v else "0%"))
    ax.set_xlim(x[0], x[-1] + pd.Timedelta(days=500))
    _year_ticks(ax)
    ax.legend(loc="upper left")
    ax.set_title("Annual price growth, secondary market")
    style.subtitle(ax, "Year-on-year change of the quality-adjusted (hedonic) index")
    style.source_note(fig, "Source: RCN Warszawa (cleaned), NBP BaRN.")
    yoy.index = yoy.index.astype(str)
    return style.save(fig, "idx_03_yoy_growth", yoy.round(2)), yoy


# ------------------------------------------------------------------ 4 primary market
def primary_market(long):
    t = pd.DataFrame({"secondary_hedonic": _series(long, "secondary", "hedonic"),
                      "primary_hedonic": _series(long, "primary_developer", "hedonic"),
                      "primary_median": _series(long, "primary_developer", "median")})
    fig, ax = plt.subplots(figsize=(10, 5))
    x = _x(t.index)
    ax.axhline(100, color=style.AXIS, lw=1)
    ax.plot(x, t["primary_median"], color=style.DEEMPH, lw=1.5, label="Developer – raw median")
    ax.plot(x, t["secondary_hedonic"], color=style.SEGMENT_COLORS["secondary"], label="Secondary – hedonic")
    ax.plot(x, t["primary_hedonic"], color=style.SEGMENT_COLORS["primary_developer"], label="Developer – hedonic")
    ends = t.iloc[-1]
    # the gray median ends on top of the secondary line -> legend only for it
    for col, lab in [("secondary_hedonic", "Secondary"), ("primary_hedonic", "Dev. hedonic")]:
        style.end_label(ax, x[-1], ends[col], f"{lab} {ends[col]:.0f}")
    ax.set_ylim(0)
    ax.set_ylabel("Index, 2015 = 100")
    ax.set_xlim(x[0], x[-1] + pd.Timedelta(days=900))
    _year_ticks(ax)
    ax.legend(loc="upper left")
    ax.set_title("New-build prices: the raw median swings with the mix of flats completed")
    style.subtitle(ax, "Developer sales, 2015 = 100. In 2025Q4 a wave of small-flat completions lifted the median "
                       "per m² 18% in one quarter; the hedonic index rose 4%.")
    style.source_note(fig)
    t.index = t.index.astype(str)
    return style.save(fig, "idx_04_primary_market", t.round(2)), t


# ------------------------------------------------------------------ 5 districts
def district_small_multiples():
    t = pd.read_csv(DISTRICT_PATH, index_col="year")
    city = t["Warszawa"]
    districts = t.drop(columns="Warszawa").iloc[-1].sort_values(ascending=False).index
    fig, axes = plt.subplots(6, 3, figsize=(10, 14), sharex=True, sharey=True)
    for ax, d in zip(axes.flat, districts):
        ax.plot(t.index, city, color=style.DEEMPH, lw=1.3)
        ax.plot(t.index, t[d], color=style.SLOTS[0], lw=2)
        ax.axhline(100, color=style.AXIS, lw=0.8)
        ax.set_title(f"{d}  {t[d].iloc[-1] - 100:+.0f}% since 2015", fontsize=10, pad=6)
        ax.set_xticks([2008, 2015, 2025])
    for ax in axes[:, 0]:
        ax.set_ylabel("2015 = 100")
    fig.suptitle("Price index by district vs Warsaw, secondary market", x=0.01, y=0.995, ha="left", va="top",
                 fontsize=13, fontweight="bold", color=style.INK)
    fig.text(0.01, 0.972, "Annual hedonic index (3-year rolling window, H3 fixed effects), 2015 = 100. "
             "Blue = district, gray = Warsaw. Sorted by 2025 level.", fontsize=9.5, color=style.INK_2, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    style.source_note(fig)
    return style.save(fig, "idx_05_districts", t.round(2)), t


ALL = [methods_comparison, nominal_vs_real, yoy_growth, primary_market]
