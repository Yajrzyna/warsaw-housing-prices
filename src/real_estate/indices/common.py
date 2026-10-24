"""Shared inputs and helpers for the price indices.

Every index function returns a DataFrame indexed by period with at least a
`log_index` column (natural log, arbitrary base); `rebase` turns it into an
index with base year = 100.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from real_estate.features.build_features import FEATURES_PATH

FIRST_Q = pd.Period("2006Q2", "Q")   # register effectively starts in April 2006
LAST_Q = pd.Period("2026Q1", "Q")    # 2026Q2 has only ~36 registered resales (registration lag)
BASE_YEAR = 2015

COLUMNS = [
    "transaction_id", "date", "year", "quarter", "segment", "district", "h3_r8",
    "price", "price_m2", "log_price_m2", "area_m2", "rooms", "floor", "street", "house_no",
    "is_top_floor", "height_class", "vintage", "age_at_sale", "has_ancillary_area", "dist_metro_km",
    "bdot_building_id", "building_key",
]


def load_sales(segments: tuple[str, ...] | None = None) -> pd.DataFrame:
    df = pd.read_parquet(FEATURES_PATH, columns=COLUMNS)
    df["q"] = pd.PeriodIndex(df["quarter"], freq="Q")
    df = df[(df["q"] >= FIRST_Q) & (df["q"] <= LAST_Q)]
    if segments is not None:
        df = df[df["segment"].isin(segments)]
    return df.reset_index(drop=True)


def rebase(log_index: pd.Series, base_year: int = BASE_YEAR) -> pd.Series:
    """exp(log index) scaled so the base year's average level = 100."""
    level = np.exp(log_index)
    years = [p.year if isinstance(p, pd.Period) else int(p) for p in log_index.index]
    base = level[np.array(years) == base_year].mean()
    return 100 * level / base


def growth(index: pd.Series, lag: int) -> pd.Series:
    """Percentage change over `lag` periods."""
    return (index / index.shift(lag) - 1) * 100
