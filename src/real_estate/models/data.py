"""Modelling frame: recent market sales with features and an index-adjusted target."""

from __future__ import annotations

import numpy as np
import pandas as pd

from real_estate.features.build_features import FEATURES_PATH
from real_estate.indices.build import QUARTERLY_PATH

SEGMENTS = ("secondary", "primary_developer")
FIRST_Q, LAST_Q = pd.Period("2016Q1", "Q"), pd.Period("2026Q1", "Q")

COLUMNS = [
    "transaction_id", "date", "year", "quarter", "segment", "district", "h3_r8", "building_key",
    "price_m2", "log_price_m2", "area_m2", "rooms", "floor", "storeys", "floor_rel", "is_top_floor",
    "height_class", "footprint_m2", "building_function", "vintage", "age_at_sale", "has_ancillary_area",
    "dist_centre_km", "dist_metro_km", "dist_park_km", "dist_forest_km", "dist_vistula_km", "dist_rail_km",
    "n_stops_500m", "right_bank", "x_2180", "y_2180", "lon", "lat",
    "nbhd_log_price_m2", "nbhd_n", "nbhd_median_dist_m", "t_quarter",
]


def load_frame(first_q: pd.Period = FIRST_Q, last_q: pd.Period = LAST_Q) -> pd.DataFrame:
    df = pd.read_parquet(FEATURES_PATH, columns=COLUMNS)
    df["q"] = pd.PeriodIndex(df["quarter"], freq="Q")
    df = df[df["segment"].isin(SEGMENTS) & df["q"].between(first_q, last_q)]
    df = df.dropna(subset=["h3_r8", "x_2180", "log_price_m2"]).reset_index(drop=True)
    df["log_index"] = segment_log_index(df)
    df["y_adj"] = df["log_price_m2"] - df["log_index"]   # price level net of market movement
    return df


def segment_log_index(df: pd.DataFrame) -> pd.Series:
    """log of the segment's hedonic index (2015 = 100) in each sale's quarter."""
    idx = pd.read_parquet(QUARTERLY_PATH)
    idx = idx[idx["method"] == "hedonic"][["segment", "q", "index"]]
    idx["log_index"] = np.log(idx["index"] / 100)
    key = pd.MultiIndex.from_frame(idx[["segment", "q"]])
    lookup = pd.Series(idx["log_index"].to_numpy(), index=key)
    return pd.Series(lookup.reindex(pd.MultiIndex.from_arrays([df["segment"], df["q"].astype(str)])).to_numpy(),
                     index=df.index)
