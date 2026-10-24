"""Exploratory figures for the Warsaw apartment transaction data.

Each function returns (figure path, table) and writes a CSV table twin next to
reports/tables/. Run all via scripts/04_eda.py.
"""

from __future__ import annotations

import geopandas as gpd
import h3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from shapely.geometry import Polygon

from real_estate.features.build_features import FEATURES_PATH
from real_estate.geo import bdot
from real_estate.geo.districts import load_districts
from real_estate.ingest.nbp import warsaw_transaction_prices
from real_estate.viz import style

COLUMNS = ["date", "year", "quarter", "segment", "price_m2", "area_m2", "rooms", "floor", "district",
           "dist_centre_km", "dist_metro_km", "h3_r8", "vintage"]
MAIN_SEGMENTS = ["secondary", "primary_developer", "primary_other"]
FIRST_QUARTER = pd.Period("2006Q2", "Q")      # register effectively starts in April 2006
LAST_FULL_QUARTER = pd.Period("2025Q4", "Q")   # 2026 deeds are still being registered
LAST_PRICE_QUARTER = pd.Period("2026Q1", "Q")  # 2026Q2: only ~36 resales registered so far
DIVERGING = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#104281", "#3987e5", "#9ec5f4", "#f0efec", "#f3b2b1", "#e34948", "#a8282a"])


def load() -> pd.DataFrame:
    df = pd.read_parquet(FEATURES_PATH, columns=COLUMNS)
    df["q"] = pd.PeriodIndex(df["quarter"], freq="Q")
    return df[df["q"] >= FIRST_QUARTER]


def _qx(periods) -> pd.DatetimeIndex:
    return pd.PeriodIndex(periods, freq="Q").to_timestamp(how="start") + pd.Timedelta(days=45)


# ------------------------------------------------------------------ 1 volume
def volume_by_segment(df: pd.DataFrame):
    t = (df[df["segment"].isin(MAIN_SEGMENTS) & (df["q"] <= LAST_FULL_QUARTER)]
         .groupby(["q", "segment"]).size().unstack()[MAIN_SEGMENTS])
    fig, ax = plt.subplots(figsize=(10, 5))
    x = _qx(t.index)
    for seg in MAIN_SEGMENTS:
        ax.plot(x, t[seg], color=style.SEGMENT_COLORS[seg], label=style.SEGMENT_LABELS[seg])
        style.end_label(ax, x[-1], t[seg].iloc[-1], f"{style.SEGMENT_LABELS[seg]} {int(t[seg].iloc[-1]):,}")
    ax.set_title("Apartment sales per quarter, by market segment")
    style.subtitle(ax, "Deeds registered in RCN, clean sample, 2006Q2–2025Q4 (2026 still being registered)")
    ax.set_ylim(0)
    style.thousands(ax)
    ax.set_ylabel("Sales per quarter")
    ax.legend(loc="upper left", ncols=3, bbox_to_anchor=(0, 1.0))
    ax.set_xlim(x[0], x[-1] + pd.Timedelta(days=1400))
    ax.set_xticks(pd.to_datetime([f"{y}-01-01" for y in range(2008, 2026, 2)]), range(2008, 2026, 2))
    style.source_note(fig)
    t.index = t.index.astype(str)
    return style.save(fig, "eda_01_volume_by_segment", t), t


# ------------------------------------------------------------------ 2 price trend
def price_trend(df: pd.DataFrame):
    segs = ["secondary", "primary_developer"]
    t = (df[df["segment"].isin(segs) & (df["q"] <= LAST_PRICE_QUARTER)]
         .groupby(["q", "segment"])["price_m2"].median().unstack()[segs])
    nbp = warsaw_transaction_prices()["secondary"].rename("nbp_secondary_mean")
    t = t.join(nbp, how="left")
    fig, ax = plt.subplots(figsize=(10, 5))
    x = _qx(t.index)
    ax.plot(x, t["nbp_secondary_mean"], color=style.DEEMPH, lw=1.5, label="NBP secondary (average, benchmark)")
    for seg in segs:
        ax.plot(x, t[seg], color=style.SEGMENT_COLORS[seg], label=style.SEGMENT_LABELS[seg] + " (median)")
        style.end_label(ax, x[-1], t[seg].iloc[-1], f"{t[seg].iloc[-1]:,.0f}")
    ax.set_title("Median transaction price per m² (PLN)")
    style.subtitle(ax, "Quarterly, Warsaw apartments. Primary = price at final deed (completion), lagging contract prices")
    style.thousands(ax)
    ax.set_ylim(0)
    ax.legend(loc="upper left")
    style.source_note(fig, "Source: RCN Warszawa (cleaned), NBP BaRN. Prices nominal, gross.")
    t.index = t.index.astype(str)
    return style.save(fig, "eda_02_price_trend", t.round(0)), t


# ------------------------------------------------------------------ 3 district level
def district_level(df: pd.DataFrame, year: int = 2025):
    s = df[(df["segment"] == "secondary") & (df["year"] == year)]
    t = s.groupby("district")["price_m2"].agg(median="median", n="size").sort_values("median")
    fig, ax = plt.subplots(figsize=(9, 6.5))
    y = np.arange(len(t))
    ax.barh(y, t["median"], height=0.6, color=style.SLOTS[0])
    for yi, v in zip(y, t["median"]):
        ax.text(v + 150, yi, f"{v:,.0f}", va="center", fontsize=8.5, color=style.INK_2)
    ax.set_yticks(y, t.index)
    ax.grid(axis="y", visible=False)
    style.thousands(ax, "x")
    ax.set_xlim(0, t["median"].max() * 1.12)
    ax.set_title(f"Median secondary-market price per m² by district, {year} (PLN)")
    style.subtitle(ax, f"{int(t['n'].sum()):,} secondary-market sales; smallest district sample: "
                       f"{t['n'].idxmin()} (n={t['n'].min()})")
    style.source_note(fig)
    return style.save(fig, "eda_03_district_price_level", t.round(0)), t


# ------------------------------------------------------------------ 4 district growth
def district_growth(df: pd.DataFrame, y0: int = 2016, y1: int = 2025):
    s = df[(df["segment"] == "secondary") & df["year"].isin([y0, y1])]
    t = s.groupby(["district", "year"])["price_m2"].median().unstack()
    t["growth_pct"] = (t[y1] / t[y0] - 1) * 100
    t = t.sort_values("growth_pct")
    fig, ax = plt.subplots(figsize=(9, 6.5))
    y = np.arange(len(t))
    ax.hlines(y, t[y0], t[y1], color=style.GRID, lw=2)
    ax.scatter(t[y0], y, s=60, color="#86b6ef", edgecolor=style.SURFACE, linewidth=2, zorder=3, label=str(y0))
    ax.scatter(t[y1], y, s=60, color="#1c5cab", edgecolor=style.SURFACE, linewidth=2, zorder=3, label=str(y1))
    for yi, (v, g) in enumerate(zip(t[y1], t["growth_pct"])):
        ax.text(v + 400, yi, f"+{g:.0f}%", va="center", fontsize=8.5, color=style.INK_2)
    ax.set_yticks(y, t.index)
    ax.grid(axis="y", visible=False)
    style.thousands(ax, "x")
    ax.set_xlim(0, t[y1].max() * 1.15)
    ax.legend(loc="upper right", ncols=2)
    city = s.groupby("year")["price_m2"].median()
    ax.set_title(f"Secondary-market price per m² by district, {y0} vs {y1} (PLN)")
    style.subtitle(ax, f"Median; sorted by growth. Warsaw overall: {city[y0]:,.0f} to {city[y1]:,.0f} "
                       f"(+{(city[y1] / city[y0] - 1) * 100:.0f}%)")
    style.source_note(fig)
    return style.save(fig, "eda_04_district_growth", t.round(1)), t


# ------------------------------------------------------------------ maps
def _hex_frame(cells: pd.Index) -> gpd.GeoDataFrame:
    polys = [Polygon([(lng, lat) for lat, lng in h3.cell_to_boundary(c)]) for c in cells]
    return gpd.GeoDataFrame({"h3_r8": cells}, geometry=polys, crs=4326).to_crs(2180)


def _map_base(ax, districts, river):
    districts.boundary.plot(ax=ax, color=style.INK_2, linewidth=0.6, zorder=3)
    river.plot(ax=ax, color="#9ec5f4", edgecolor="none", zorder=2, alpha=0.9)
    ax.set_axis_off()
    ax.set_aspect("equal")


def map_price_level(df: pd.DataFrame, years=(2024, 2025), min_n: int = 10):
    s = df[(df["segment"] == "secondary") & df["year"].isin(years)]
    t = s.groupby("h3_r8")["price_m2"].agg(median="median", n="size")
    t = t[t["n"] >= min_n]
    hexes = _hex_frame(t.index).join(t, on="h3_r8")
    districts, river = load_districts(), bdot.vistula()
    fig, ax = plt.subplots(figsize=(9, 9.5))
    vmin, vmax = np.percentile(hexes["median"], [2, 98])
    hexes.plot(ax=ax, column="median", cmap=style.SEQ_CMAP, vmin=vmin, vmax=vmax,
               edgecolor=style.SURFACE, linewidth=0.4, zorder=1)
    _map_base(ax, districts, river)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin, vmax), cmap=style.SEQ_CMAP),
                      ax=ax, orientation="horizontal", fraction=0.035, pad=0.02, aspect=40)
    cb.outline.set_visible(False)
    cb.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    cb.set_label("Median price per m² (PLN)", color=style.INK_2)
    ax.set_title(f"Where apartments are expensive: secondary market {years[0]}–{years[-1]}")
    style.subtitle(ax, f"H3 hexagons (~0.7 km²) with ≥{min_n} sales; lines = district boundaries; blue = Vistula")
    style.source_note(fig)
    return style.save(fig, "eda_05_map_price_level", t.round(0)), t


def map_relative_growth(df: pd.DataFrame, base=(2016, 2017), recent=(2024, 2025), min_n: int = 15):
    s = df[df["segment"] == "secondary"]
    a = s[s["year"].isin(base)].groupby("h3_r8")["price_m2"].agg(p0="median", n0="size")
    b = s[s["year"].isin(recent)].groupby("h3_r8")["price_m2"].agg(p1="median", n1="size")
    t = a.join(b, how="inner")
    t = t[(t["n0"] >= min_n) & (t["n1"] >= min_n)]
    t["growth_pct"] = (t["p1"] / t["p0"] - 1) * 100
    city = (s[s["year"].isin(recent)]["price_m2"].median() / s[s["year"].isin(base)]["price_m2"].median() - 1) * 100
    t["vs_city_pp"] = t["growth_pct"] - city
    hexes = _hex_frame(t.index).join(t, on="h3_r8")
    districts, river = load_districts(), bdot.vistula()
    lim = np.percentile(np.abs(hexes["vs_city_pp"]), 95)
    fig, ax = plt.subplots(figsize=(9, 9.5))
    hexes.plot(ax=ax, column="vs_city_pp", cmap=DIVERGING, vmin=-lim, vmax=lim,
               edgecolor=style.SURFACE, linewidth=0.4, zorder=1)
    _map_base(ax, districts, river)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-lim, lim), cmap=DIVERGING),
                      ax=ax, orientation="horizontal", fraction=0.035, pad=0.02, aspect=40)
    cb.outline.set_visible(False)
    cb.set_label(f"Growth minus Warsaw-wide growth (+{city:.0f}%), percentage points  (blue = slower, red = faster)",
                 color=style.INK_2)
    ax.set_title(f"Which areas outpaced the city: {base[0]}–{str(base[-1])[-2:]} to {recent[0]}–{str(recent[-1])[-2:]}")
    style.subtitle(ax, f"Secondary-market median price per m², hexagons with ≥{min_n} sales in both periods")
    style.source_note(fig)
    return style.save(fig, "eda_06_map_relative_growth", t.round(1)), t


# ------------------------------------------------------------------ 7 gradients
def price_gradients(df: pd.DataFrame, years=(2023, 2024, 2025)):
    """Price relative to the quarter's segment median - removes the time trend."""
    s = df[(df["segment"] == "secondary") & df["year"].isin(years)].copy()
    s["rel"] = s["price_m2"] / s.groupby("q")["price_m2"].transform("median")
    specs = [
        ("dist_centre_km", "Distance to centre (km)", np.arange(0, 17, 1)),
        ("dist_metro_km", "Distance to nearest open metro station (km)", [0, .25, .5, .75, 1, 1.5, 2, 3, 4, 6, 10, 16]),
        ("floor", "Floor (1 = ground)", [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 8.5, 10.5, 13.5, 16.5, 20.5, 60]),
        ("area_m2", "Usable area (m²)", [15, 25, 30, 35, 40, 50, 60, 70, 80, 100, 130, 300]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharey=True)
    tables = []
    for ax, (col, label, bins) in zip(axes.flat, specs):
        cut = pd.cut(s[col], bins)
        g = s.groupby(cut, observed=True).agg(median=("rel", "median"), n=("rel", "size"), x=(col, "median"))
        g = g[g["n"] >= 100]
        ax.axhline(1, color=style.AXIS, lw=1)
        ax.plot(g["x"], g["median"], color=style.SLOTS[0], marker="o", ms=7,
                markeredgecolor=style.SURFACE, markeredgewidth=1.5)
        ax.set_xlabel(label)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        tables.append(g.assign(feature=col).reset_index().rename(columns={col: "bin"}))
    axes[0, 0].set_ylabel("Price/m² vs quarter median")
    axes[1, 0].set_ylabel("Price/m² vs quarter median")
    fig.suptitle(f"How price per m² varies — secondary market {years[0]}–{years[-1]}", x=0.01, y=0.995,
                 ha="left", va="top", fontsize=13, fontweight="bold", color=style.INK)
    fig.text(0.01, 0.955, "Median of each sale's price/m² ÷ its quarter's median (100% = typical). "
             "Raw gradients — not controlled for other attributes.", fontsize=9.5, color=style.INK_2, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    style.source_note(fig)
    t = pd.concat(tables)
    t["bin"] = t["bin"].astype(str)
    return style.save(fig, "eda_07_price_gradients", t.set_index(["feature", "bin"]).round(3)), t


# ------------------------------------------------------------------ 8 composition
def composition(df: pd.DataFrame):
    d = df[df["segment"].isin(MAIN_SEGMENTS[:2]) & (df["year"] <= 2025)]
    area = d.groupby(["year", "segment"])["area_m2"].median().unstack()[MAIN_SEGMENTS[:2]]
    small = d.assign(s=d["area_m2"] < 40).groupby(["year", "segment"])["s"].mean().unstack()[MAIN_SEGMENTS[:2]] * 100
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    # left panel: lines converge in 2025 -> no end labels (legend carries identity)
    for ax, t, ttl, fmt in [(axes[0], area, "Median usable area (m²)", None),
                            (axes[1], small, "Share of sales under 40 m² (%)", "{:.0f}%")]:
        for seg in t.columns:
            ax.plot(t.index, t[seg], color=style.SEGMENT_COLORS[seg], label=style.SEGMENT_LABELS[seg])
            if fmt:
                style.end_label(ax, t.index[-1], t[seg].iloc[-1], fmt.format(t[seg].iloc[-1]))
        ax.set_title(ttl, fontsize=11)
        ax.set_ylim(0)
        ax.set_xlim(t.index[0], t.index[-1] + 2.5)
        ax.set_xticks(range(2008, 2026, 4))
    axes[0].legend(loc="lower left")
    fig.suptitle("Flats sold are getting smaller — especially new-builds", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", color=style.INK)
    fig.tight_layout()
    style.source_note(fig)
    t = area.add_prefix("median_area_").join(small.add_prefix("pct_under_40m2_"))
    return style.save(fig, "eda_08_composition", t.round(1)), t


# ------------------------------------------------------------------ 9 primary/secondary gap
def primary_secondary_gap(df: pd.DataFrame):
    t = (df[df["segment"].isin(["secondary", "primary_developer"]) & (df["q"] <= LAST_PRICE_QUARTER)]
         .groupby(["q", "segment"])["price_m2"].median().unstack())
    t["ratio"] = t["primary_developer"] / t["secondary"]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = _qx(t.index)
    ax.axhline(1, color=style.AXIS, lw=1)
    ax.plot(x, t["ratio"], color=style.SLOTS[1])
    style.end_label(ax, x[-1], t["ratio"].iloc[-1], f"{t['ratio'].iloc[-1]:.0%}")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title("Developer (primary) median price/m² as % of secondary-market median")
    style.subtitle(ax, "Quarterly. Below 100% = new-builds cheaper per m² at the deed date "
                       "(outer locations, prices fixed at contract 1–3 years earlier)")
    style.source_note(fig)
    t.index = t.index.astype(str)
    return style.save(fig, "eda_09_primary_secondary_gap", t.round(3)), t
