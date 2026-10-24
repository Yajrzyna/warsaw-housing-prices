"""Figures for Phase 7 (reports/figures/mac_*.png, table twins in reports/tables/)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from real_estate.config import REPORTS_TABLES_DIR as T
from real_estate.macro.analysis import EVENTS
from real_estate.viz import style


def _load_q() -> pd.DataFrame:
    q = pd.read_csv(T / "mac_quarterly.csv", index_col=0)
    q.index = pd.PeriodIndex(q.index, freq="Q")
    return q


def _x(idx):
    return pd.PeriodIndex(idx, freq="Q").to_timestamp(how="start") + pd.Timedelta(days=45)


def _shade(ax, label_row: bool = False):
    for name, (a, b) in EVENTS.items():
        x0 = pd.Period(a, "Q").to_timestamp(how="start")
        x1 = pd.Period(b, "Q").to_timestamp(how="end")
        ax.axvspan(x0, x1, color=style.GRID, alpha=0.6, lw=0, zorder=0)
        if label_row:
            ax.text(x0, 1.0, " " + name, transform=ax.get_xaxis_transform(), fontsize=8, color=style.MUTED,
                    va="top", ha="left")


def _years(ax, start=2008, stop=2026, step=2):
    ax.set_xticks(pd.to_datetime([f"{y}-01-01" for y in range(start, stop + 1, step)]), range(start, stop + 1, step))


def affordability():
    q = _load_q()
    x = _x(q.index)
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    specs = [("years_of_wages", "Years of gross wages to buy a 50 m² flat", "{:.1f} yrs"),
             ("m2_per_wage", "m² bought with one monthly gross wage", "{:.2f} m²"),
             ("payment_share", "Mortgage payment as % of a gross monthly wage (80% loan, 25 yrs)", "{:.0f}%")]
    for i, (ax, (col, title, fmt)) in enumerate(zip(axes, specs)):
        _shade(ax, label_row=(i == 0))
        ax.plot(x, q[col], color=style.SLOTS[0])
        ax.scatter([x[-1]], [q[col].iloc[-1]], s=40, color=style.SLOTS[0], edgecolor=style.SURFACE, linewidth=1.5, zorder=3)
        style.end_label(ax, x[-1], q[col].iloc[-1], fmt.format(q[col].iloc[-1]))
        ax.set_title(title, fontsize=11, pad=6)
        ax.set_xlim(x[0], x[-1] + pd.Timedelta(days=500))
    _years(axes[-1])
    fig.suptitle("Affordability of a typical Warsaw flat", x=0.01, y=0.995, ha="left", va="top",
                 fontsize=13, fontweight="bold", color=style.INK)
    fig.text(0.01, 0.962, "Median resale price per m² × 50 m²; Warsaw average gross wage (GUS, interpolated to quarters); "
             "mortgage rate = 3M money-market rate + 2 pp", fontsize=9.5, color=style.INK_2, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    style.source_note(fig, "Source: RCN Warszawa, GUS BDL, Eurostat irt_st_m. Shaded: COVID-19, 2021–22 rate hikes, "
                           "Bezpieczny Kredyt 2%.")
    t = q[[c for c, _, _ in specs]].round(3)
    t.index = t.index.astype(str)
    return style.save(fig, "mac_01_affordability", t)


def timeline():
    q = _load_q()
    yoy = (q[["hedonic", "hedonic_real"]] / q[["hedonic", "hedonic_real"]].shift(4) - 1) * 100
    start = pd.Period("2008Q1", "Q")          # the 2006-07 boom (+59% y/y) would flatten everything after it
    q, yoy = q.loc[start:], yoy.loc[start:]
    x = _x(q.index)
    fig, axes = plt.subplots(3, 1, figsize=(10, 9.5), sharex=True, gridspec_kw={"height_ratios": [1.3, 1, 1]})
    ax = axes[0]
    _shade(ax, label_row=True)
    ax.axhline(0, color=style.AXIS, lw=1)
    ax.plot(x, yoy["hedonic"], color=style.SLOTS[0], label="Nominal")
    ax.plot(x, yoy["hedonic_real"], color=style.SLOTS[1], label="Real")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0f}%" if v else "0%"))
    ax.set_title("Resale price growth, year on year (hedonic index)", fontsize=11, pad=6)
    ax.legend(loc="upper left", ncols=2)
    ax = axes[1]
    _shade(ax)
    ax.plot(x, q["mortgage_rate"], color=style.SLOTS[0], label="Mortgage rate proxy (3M + 2 pp)")
    ax.plot(x, q["reference_rate"], color=style.DEEMPH, lw=1.6, label="NBP reference rate")
    ax.set_ylim(0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_title("Interest rates", fontsize=11, pad=6)
    ax.legend(loc="upper left")
    ax = axes[2]
    _shade(ax)
    ax.plot(x, q["sales_secondary"], color=style.SEGMENT_COLORS["secondary"], label="Resales")
    ax.plot(x, q["sales_developer"], color=style.SEGMENT_COLORS["primary_developer"], label="Developer deeds")
    ax.set_ylim(0)
    style.thousands(ax)
    ax.set_title("Apartment sales per quarter", fontsize=11, pad=6)
    ax.legend(loc="upper left")
    _years(axes[-1])
    fig.suptitle("Prices, rates and sales, 2008–2026", x=0.01, y=0.995, ha="left", va="top",
                 fontsize=13, fontweight="bold", color=style.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    style.source_note(fig, "Source: RCN Warszawa, NBP, Eurostat. Developer deeds lag contracts by 1–3 years.")
    t = pd.concat([yoy.add_prefix("yoy_"), q[["mortgage_rate", "reference_rate", "sales_secondary", "sales_developer"]]], axis=1).round(2)
    t.index = t.index.astype(str)
    return style.save(fig, "mac_02_timeline", t)


def bk2_size_bands():
    b = pd.read_csv(T / "mac_size_band_indices.csv", index_col=0)
    b.index = pd.PeriodIndex(b.index, freq="Q")
    b = b.loc[pd.Period("2021Q1", "Q"):]
    x = _x(b.index)
    fig, ax = plt.subplots(figsize=(10, 5))
    a0, a1 = EVENTS["Bezpieczny Kredyt 2%"]
    ax.axvspan(pd.Period(a0, "Q").to_timestamp(how="start"), pd.Period(a1, "Q").to_timestamp(how="end"),
               color=style.GRID, alpha=0.6, lw=0, zorder=0)
    ax.text(pd.Period(a0, "Q").to_timestamp(how="start"), 1.0, " Bezpieczny Kredyt 2%", transform=ax.get_xaxis_transform(),
            fontsize=8.5, color=style.MUTED, va="top")
    ax.axhline(100, color=style.AXIS, lw=1)
    for col, c in zip(b.columns, style.SLOTS[:3]):
        ax.plot(x, b[col], color=c, label=col.replace("m²", "m²"))
        style.end_label(ax, x[-1], b[col].iloc[-1], f"{b[col].iloc[-1]:.0f}")
    ax.set_xlim(x[0], x[-1] + pd.Timedelta(days=150))
    ax.set_ylabel("Index, 2023Q2 = 100")
    ax.legend(loc="upper left")
    did = pd.read_csv(T / "mac_bk2_did.csv", index_col=0)
    gap = did.loc["small − large (pp)"]
    ax.set_title("Small flats rose faster during the programme – then gave it back")
    style.subtitle(ax, f"Resale hedonic index by flat size. Small minus large: {gap.iloc[0]:+.1f} pp before, "
                       f"{gap.iloc[1]:+.1f} pp during (2023Q2–2024Q2), {gap.iloc[2]:+.1f} pp after")
    style.source_note(fig)
    t = b.round(2)
    t.index = t.index.astype(str)
    return style.save(fig, "mac_03_bk2_size_bands", t)


def supply():
    a = pd.read_csv(T / "mac_annual.csv", index_col=0)
    a = a.loc[2007:2025]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(a.index, a["dwellings_started"], color=style.SLOTS[1], label="Dwellings started (GUS)")
    ax.plot(a.index, a["dwellings_completed"], color=style.SLOTS[0], label="Dwellings completed (GUS)")
    ax.plot(a.index, a["sales_developer"], color=style.SLOTS[2], label="Developer sale deeds (RCN, clean)")
    for col, lab in [("dwellings_completed", "completed"), ("sales_developer", "deeds")]:
        style.end_label(ax, a.index[-1], a[col].iloc[-1], f"{a[col].iloc[-1]:,.0f}")
    ax.set_ylim(0)
    style.thousands(ax)
    ax.set_xlim(a.index[0], a.index[-1] + 1.5)
    ax.set_xticks(range(2008, 2026, 2))
    ax.legend(loc="upper left")
    ax.set_title("Housing supply in Warsaw")
    style.subtitle(ax, "Per year. Starts fell after the 2022 rate shock; completions and developer deeds jumped in 2025")
    style.source_note(fig, "Source: GUS BDL (Warsaw), RCN Warszawa.")
    return style.save(fig, "mac_04_supply", a[["dwellings_started", "dwellings_completed", "sales_developer", "completions_per_1000"]].round(2))


def district_affordability():
    t = pd.read_csv(T / "mac_district_affordability.csv", index_col=0).sort_values("years_of_wages")
    fig, ax = plt.subplots(figsize=(9, 6.5))
    y = np.arange(len(t))
    ax.barh(y, t["years_of_wages"], height=0.6, color=style.SLOTS[0])
    for yi, (v, p) in enumerate(zip(t["years_of_wages"], t["payment_share"])):
        ax.text(v + 0.08, yi, f"{v:.1f} yrs · {p:.0f}%", va="center", fontsize=8.5, color=style.INK_2)
    ax.set_yticks(y, t.index)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, t["years_of_wages"].max() * 1.25)
    ax.set_xlabel("Years of Warsaw average gross wage for a 50 m² flat (2025 median resale price)")
    ax.set_title("Affordability by district, 2025")
    style.subtitle(ax, "Labels: years of wages · mortgage payment as % of one gross monthly wage (80% loan, 25 yrs, 2025 rate)")
    style.source_note(fig, "Source: RCN Warszawa, GUS BDL, Eurostat.")
    return style.save(fig, "mac_05_district_affordability", t.round(2))


ALL = [affordability, timeline, bk2_size_bands, supply, district_affordability]
