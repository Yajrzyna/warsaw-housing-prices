"""Shared matplotlib style for report figures (light mode, static PNG).

Palette = dataviz reference instance. Categorical slots are assigned in fixed
order and follow the entity (segment -> slot is fixed below). Slots 1-3 were
validated all-pairs (validate_palette.js, light): CVD dE >= 9.2, normal >= 24.
Aqua (slot 3) is < 3:1 on the surface -> charts using it carry direct labels,
and every figure is saved with a CSV table twin.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from real_estate.config import REPORTS_DIR

FIG_DIR = REPORTS_DIR / "figures"
TABLE_DIR = REPORTS_DIR / "tables"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
DEEMPH = "#c3c2b7"  # context / de-emphasis gray

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEGMENT_COLORS = {"secondary": SLOTS[0], "primary_developer": SLOTS[1], "primary_other": SLOTS[2]}
SEGMENT_LABELS = {"secondary": "Secondary", "primary_developer": "Primary (developer)",
                  "primary_other": "Primary (other seller)"}

BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
             "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)


def apply() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10, "text.color": INK,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK_2,
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlecolor": INK, "axes.titlepad": 28,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "xtick.major.size": 0, "ytick.major.size": 0,
        "lines.linewidth": 2, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "legend.frameon": False, "legend.fontsize": 9, "legend.labelcolor": INK_2,
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def subtitle(ax, text: str) -> None:
    """Secondary line under the (left-aligned) title."""
    ax.text(0, 1.015, text, transform=ax.transAxes, color=INK_2, fontsize=9.5, va="bottom")


def source_note(fig, text: str = "Source: RCN Warszawa (notarial deeds), cleaned sample. Prices nominal, gross.") -> None:
    fig.text(0.01, -0.02, text, color=MUTED, fontsize=8, ha="left", va="top")


def thousands(ax, axis: str = "y") -> None:
    fmt = mpl.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def end_label(ax, x, y, text: str, dx: float = 6) -> None:
    """Direct label at a line end - text in ink, never the series color."""
    ax.annotate(text, (x, y), xytext=(dx, 0), textcoords="offset points",
                va="center", ha="left", fontsize=9, color=INK_2)


def save(fig, name: str, table: pd.DataFrame | None = None) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    if table is not None:  # table-view twin of the chart
        TABLE_DIR.mkdir(parents=True, exist_ok=True)
        table.to_csv(TABLE_DIR / f"{name}.csv")
    return path
