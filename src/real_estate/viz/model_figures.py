"""Figures for Phase 6 (reports/figures/mdl_*.png, table twins in reports/tables/)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from real_estate.config import REPORTS_TABLES_DIR as T
from real_estate.viz import style
from real_estate.viz.eda_figures import DIVERGING, _hex_frame, _map_base
from real_estate.geo import bdot
from real_estate.geo.districts import load_districts

MODEL_LABELS = {"hedonic_A": "Hedonic – location variables", "hedonic_B": "Hedonic – H3 fixed effects",
                "gbm": "Gradient boosting", "gbm_nbhd": "Gradient boosting + local prices"}


def _dot_whisker(ax, t: pd.DataFrame, color: str):
    y = np.arange(len(t))[::-1]
    ax.axvline(0, color=style.AXIS, lw=1)
    ax.hlines(y, t["ci_low_pct"], t["ci_high_pct"], color=color, lw=2, alpha=0.45)
    ax.scatter(t["premium_pct"], y, s=55, color=color, edgecolor=style.SURFACE, linewidth=1.5, zorder=3)
    for yi, v in zip(y, t["premium_pct"]):
        ax.annotate(f"{v:+.1f}%", (v, yi), xytext=(0, 7), textcoords="offset points", ha="center",
                    fontsize=8, color=style.INK_2)
    ax.set_yticks(y, t["label"])
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0f}%" if v else "0%"))


def unit_premia():
    t = pd.read_csv(T / "mdl_premia_spec_b.csv").dropna(subset=["premium_pct"])
    fig, ax = plt.subplots(figsize=(9, 8))
    _dot_whisker(ax, t, style.SLOTS[0])
    ax.set_title("What a flat's features are worth (same location)")
    style.subtitle(ax, "Effect on price per m², hedonic model with H3-cell fixed effects, 2023Q1–2026Q1, "
                       "95% CI clustered by building")
    style.source_note(fig)
    return style.save(fig, "mdl_01_unit_premia", t.set_index("label")[["premium_pct", "ci_low_pct", "ci_high_pct"]].round(2))


def location_premia():
    t = pd.read_csv(T / "mdl_premia_spec_a.csv")
    loc = t[t["term"].str.match(r"^(centre|metro|park|forest|vistula|stops|right)")].dropna(subset=["premium_pct"])
    fig, ax = plt.subplots(figsize=(9, 6.5))
    _dot_whisker(ax, loc, style.SLOTS[1])
    ax.set_title("What location is worth")
    style.subtitle(ax, "Effect on price per m² holding flat features fixed; reference: 6–8 km from centre, "
                       "metro > 2 km. Hedonic model with location variables, 2023Q1–2026Q1")
    style.source_note(fig)
    return style.save(fig, "mdl_02_location_premia", loc.set_index("label")[["premium_pct", "ci_low_pct", "ci_high_pct"]].round(2))


def location_value_map():
    t = pd.read_csv(T / "mdl_location_value_h3.csv")
    hexes = _hex_frame(pd.Index(t["h3_r8"])).merge(t, on="h3_r8")
    lim = np.percentile(np.abs(hexes["premium_pct"]), 95)
    fig, ax = plt.subplots(figsize=(9, 9.5))
    hexes.plot(ax=ax, column="premium_pct", cmap=DIVERGING, vmin=-lim, vmax=lim,
               edgecolor=style.SURFACE, linewidth=0.4, zorder=1)
    _map_base(ax, load_districts(), bdot.vistula())
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-lim, lim), cmap=DIVERGING), ax=ax,
                      orientation="horizontal", fraction=0.035, pad=0.02, aspect=40)
    cb.outline.set_visible(False)
    cb.set_label("Location premium vs Warsaw average, % (blue = cheaper, red = dearer)", color=style.INK_2)
    ax.set_title("Location value, with flat features held constant")
    style.subtitle(ax, "H3-cell fixed effects from the hedonic model, 2023Q1–2026Q1; cells with ≥ 20 sales")
    style.source_note(fig)
    return style.save(fig, "mdl_03_location_value_map", t.round(2))


def avm_accuracy():
    t = pd.read_csv(T / "mdl_avm_metrics.csv")
    t["scheme_short"] = np.where(t["scheme"].str.startswith("temporal"), "Future sales (2025–26)", "Unseen areas")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    order = [m for m in MODEL_LABELS if m in t["model"].unique()]
    y = np.arange(len(order))[::-1]
    for scheme, color, dy in [("Future sales (2025–26)", style.SLOTS[0], 0.14), ("Unseen areas", style.SLOTS[1], -0.14)]:
        s = t[t["scheme_short"] == scheme].set_index("model").reindex(order)
        ok = s["median_ape_pct"].notna().to_numpy()
        ax.barh(y[ok] + dy, s["median_ape_pct"][ok], height=0.26, color=color, label=scheme)
        for yi, v in zip(y[ok] + dy, s["median_ape_pct"][ok]):
            ax.text(v + 0.15, yi, f"{v:.1f}%", va="center", fontsize=8.5, color=style.INK_2)
    ax.set_yticks(y, [MODEL_LABELS[m] for m in order])
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, t["median_ape_pct"].max() * 1.2)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.legend(loc="lower right")
    ax.set_title("Valuation model accuracy – median absolute error")
    style.subtitle(ax, "Lower is better. Future sales: trained to 2024Q4. Unseen areas: 5-fold CV on ~36 km² blocks")
    style.source_note(fig)
    return style.save(fig, "mdl_04_avm_accuracy", t.drop(columns="scheme_short").round(2))


def avm_importance(top: int = 14):
    t = pd.read_csv(T / "mdl_avm_importance.csv").head(top)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    y = np.arange(len(t))[::-1]
    ax.barh(y, t["mae_increase"] * 100, height=0.6, color=style.SLOTS[0])
    ax.set_yticks(y, t["feature"])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Increase in mean absolute error when the feature is shuffled (log points ×100)")
    ax.set_title("What the valuation model relies on")
    style.subtitle(ax, "Permutation importance, gradient boosting + local prices, test sales 2025Q1–2026Q1")
    style.source_note(fig)
    return style.save(fig, "mdl_05_avm_importance", t.round(5))


ALL = [unit_premia, location_premia, location_value_map, avm_accuracy, avm_importance]
