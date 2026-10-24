"""Feature engineering for the clean RCN apartment sample.

Input : data/processed/rcn_apartments_clean.parquet
Output: data/processed/rcn_apartments_features.parquet
Docs  : docs/data_dictionaries/features.md

Feature groups
  unit          size bands, area per room, floor position
  building      BDOT10k match: storeys, function, footprint; vintage proxy from RCN
  location      distance to centre, Vistula, parks, forests, rail, stop density, H3 cells
  metro         time-aware: only stations open on the deed date count
  neighbourhood leakage-safe local price level: k nearest sales in the 12 months
                before the quarter of the sale
  time          month, quarter index
"""

from __future__ import annotations

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

from real_estate.config import PROCESSED_DIR
from real_estate.geo import bdot

CLEAN_PATH = PROCESSED_DIR / "rcn_apartments_clean.parquet"
FEATURES_PATH = PROCESSED_DIR / "rcn_apartments_features.parquet"

CENTRE_WGS84 = (21.0067, 52.2317)  # Palace of Culture and Science (PKiN)
RIGHT_BANK = {"Białołęka", "Targówek", "Praga-Północ", "Praga-Południe", "Rembertów", "Wawer", "Wesoła"}
BUILDING_MAX_DIST_M = 40
STOP_RADIUS_M = 500
KNN_K, KNN_MAX_DIST_M, KNN_MIN_N, KNN_WINDOW_DAYS = 20, 1500, 5, 365
FIRST_FULL_YEAR = 2007  # first year a building's first sale can mark its completion (register starts 2006)


def _xy(df: pd.DataFrame) -> np.ndarray:
    return df[["x_2180", "y_2180"]].to_numpy()


def _points(df: pd.DataFrame) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(index=df.index, geometry=gpd.points_from_xy(df["x_2180"], df["y_2180"]), crs=2180)


def _nearest_distance(pts: gpd.GeoDataFrame, targets: gpd.GeoDataFrame) -> pd.Series:
    j = gpd.sjoin_nearest(pts, targets[["geometry"]], how="left", distance_col="_d")
    return j.groupby(level=0)["_d"].min().reindex(pts.index)


# --------------------------------------------------------------------------- unit
def unit_features(df: pd.DataFrame) -> pd.DataFrame:
    df["log_area"] = np.log(df["area_m2"])
    df["area_per_room"] = df["area_m2"] / df["rooms"]
    df["size_band"] = pd.cut(df["area_m2"], [0, 30, 40, 50, 60, 80, 100, np.inf],
                             labels=["<30", "30-40", "40-50", "50-60", "60-80", "80-100", "100+"], right=False)
    df["month"] = df["date"].dt.month
    q = pd.PeriodIndex(df["quarter"], freq="Q")
    df["t_quarter"] = (q.year - 2006) * 4 + q.quarter - 1  # 0 = 2006Q1
    return df


# --------------------------------------------------------------------------- building
def match_buildings(df: pd.DataFrame, buildings: gpd.GeoDataFrame) -> pd.DataFrame:
    """Attach the BDOT building containing the sale point, else the nearest one within 40 m."""
    cand = buildings[~buildings["building_function"].isin(bdot.NON_RESIDENTIAL)]
    cols = ["bdot_building_id", "storeys", "building_function", "footprint_m2"]
    pts = _points(df)

    within = gpd.sjoin(pts, cand[cols + ["geometry"]], how="left", predicate="within")
    within = within[~within.index.duplicated()]
    out = within[cols].copy()
    out["building_match"] = np.where(out["bdot_building_id"].notna(), "within", None)
    out["building_match_dist_m"] = np.where(out["bdot_building_id"].notna(), 0.0, np.nan)

    miss = out["bdot_building_id"].isna()
    near = gpd.sjoin_nearest(pts[miss], cand[cols + ["geometry"]], how="left", distance_col="d")
    near = near[~near.index.duplicated()]
    ok = near["d"] <= BUILDING_MAX_DIST_M
    idx = near.index[ok]
    out.loc[idx, cols] = near.loc[idx, cols].to_numpy()
    out.loc[idx, "building_match"] = "nearest"
    out.loc[idx, "building_match_dist_m"] = near.loc[idx, "d"]
    out["storeys"] = pd.to_numeric(out["storeys"])
    out["footprint_m2"] = pd.to_numeric(out["footprint_m2"])
    return df.join(out)


def building_vintage(df: pd.DataFrame) -> pd.DataFrame:
    """Completion-year proxy from the register itself.

    A building counts as built in year Y (>= 2007) if its first recorded sale is a
    developer primary-market sale in Y and it has >= 2 developer sales. Old blocks
    have secondary sales from 2006 on, so their first sale is not primary - this
    also protects against cooperative conversions recorded as "primary market".
    Everything else is 'older_or_unknown' (no public source of building age found).
    """
    addr = "addr:" + df["street"].str.lower().str.strip() + "|" + df["house_no"].astype(str).str.lower().str.replace(" ", "")
    df["building_key"] = df["bdot_building_id"].fillna(addr)

    g = df.sort_values("date").groupby("building_key")
    first = g.first()[["year", "segment"]].rename(columns={"year": "bld_first_sale_year", "segment": "bld_first_segment"})
    first["bld_n_developer_sales"] = g["segment"].apply(lambda s: (s == "primary_developer").sum())
    first["bld_n_sales"] = g.size()
    is_new = (first["bld_first_segment"] == "primary_developer") & (first["bld_n_developer_sales"] >= 2) \
        & (first["bld_first_sale_year"] >= FIRST_FULL_YEAR)
    first["build_year_proxy"] = first["bld_first_sale_year"].where(is_new)

    df = df.join(first[["bld_first_sale_year", "bld_n_sales", "build_year_proxy"]], on="building_key")
    df["vintage"] = np.where(df["build_year_proxy"].notna(), "new_2007plus", "older_or_unknown")
    df["age_at_sale"] = (df["year"] - df["build_year_proxy"]).clip(lower=0)
    return df


def floor_features(df: pd.DataFrame) -> pd.DataFrame:
    """RCN floor numbering: 1 = ground floor (parter)."""
    storeys = df["storeys"].where(df["storeys"] >= df["floor"])  # BDOT storeys inconsistent with floor -> unknown
    df["is_ground_floor"] = (df["floor"] == 1).where(df["floor"].notna())
    df["is_top_floor"] = ((df["floor"] == storeys) & (storeys > 1)).where(storeys.notna() & df["floor"].notna())
    df["floor_rel"] = ((df["floor"] - 1) / (storeys - 1)).where(storeys > 1).clip(0, 1)
    df["height_class"] = pd.cut(df["storeys"], [0, 4, 10, np.inf], labels=["low_1-4", "mid_5-10", "high_11+"])
    return df


# --------------------------------------------------------------------------- location
def location_features(df: pd.DataFrame) -> pd.DataFrame:
    to_2180 = Transformer.from_crs("EPSG:4326", "EPSG:2180", always_xy=True)
    cx, cy = to_2180.transform(*CENTRE_WGS84)
    df["dist_centre_km"] = np.hypot(df["x_2180"] - cx, df["y_2180"] - cy) / 1000
    df["right_bank"] = df["district"].isin(RIGHT_BANK)

    pts = _points(df)
    df["dist_vistula_km"] = pts.distance(bdot.vistula().geometry.iloc[0]).to_numpy() / 1000
    df["dist_park_km"] = _nearest_distance(pts, bdot.parks()) / 1000
    df["dist_forest_km"] = _nearest_distance(pts, bdot.forests()) / 1000

    xy = _xy(df)
    rail = bdot.rail_stations()
    df["dist_rail_km"] = cKDTree(np.c_[rail.geometry.x, rail.geometry.y]).query(xy)[0] / 1000
    stops = bdot.bus_tram_stops()
    df["n_stops_500m"] = cKDTree(np.c_[stops.geometry.x, stops.geometry.y]).query_ball_point(
        xy, r=STOP_RADIUS_M, return_length=True)

    df["h3_r8"] = [h3.latlng_to_cell(la, lo, 8) for la, lo in zip(df["lat"], df["lon"])]
    df["h3_r9"] = [h3.latlng_to_cell(la, lo, 9) for la, lo in zip(df["lat"], df["lon"])]
    return df


def metro_features(df: pd.DataFrame, stations: gpd.GeoDataFrame) -> pd.DataFrame:
    """Distance to the nearest metro station open on the deed date (time-aware)."""
    sxy = np.c_[stations.geometry.x, stations.geometry.y]
    d = np.hypot(df["x_2180"].to_numpy()[:, None] - sxy[:, 0], df["y_2180"].to_numpy()[:, None] - sxy[:, 1]) / 1000
    opened = stations["opened"].to_numpy()
    is_open_now = (stations["status"] == "open").to_numpy()

    open_at_sale = df["date"].to_numpy()[:, None] >= opened[None, :]
    d_sale = np.where(open_at_sale, d, np.inf)
    d_now = np.where(is_open_now[None, :], d, np.inf)

    df["dist_metro_km"] = d_sale.min(axis=1)
    df["nearest_metro"] = stations["station"].to_numpy()[d_sale.argmin(axis=1)]
    df["nearest_metro_line"] = stations["line"].to_numpy()[d_sale.argmin(axis=1)]
    df["dist_metro_2026_km"] = d_now.min(axis=1)  # network as of today (anticipation / DiD analyses)
    df["dist_metro_incl_planned_km"] = d.min(axis=1)
    df["metro_within_1km"] = df["dist_metro_km"] <= 1
    df["metro_opened_later_within_1km"] = (df["dist_metro_2026_km"] <= 1) & ~df["metro_within_1km"]
    return df


# --------------------------------------------------------------------------- neighbourhood
def neighbourhood_price(df: pd.DataFrame) -> pd.DataFrame:
    """Median log price/m2 of the k nearest sales in the 365 days before the sale's quarter.

    Only sales strictly before the quarter start are used, so the feature never sees
    the sale itself or any contemporaneous / future sale (safe for AVM training).
    """
    med = np.full(len(df), np.nan)
    n_used = np.zeros(len(df), dtype=int)
    dist = np.full(len(df), np.nan)
    q_start = pd.PeriodIndex(df["quarter"], freq="Q").start_time
    dates, xy, y = df["date"].to_numpy(), _xy(df), df["log_price_m2"].to_numpy()

    for qs in np.unique(q_start):
        target = np.flatnonzero(q_start == qs)
        ref = np.flatnonzero((dates < qs) & (dates >= qs - np.timedelta64(KNN_WINDOW_DAYS, "D")))
        if len(ref) < KNN_MIN_N:
            continue
        k = min(KNN_K, len(ref))
        dd, ii = cKDTree(xy[ref]).query(xy[target], k=k)
        dd, ii = dd.reshape(len(target), k), ii.reshape(len(target), k)
        vals = np.where(dd <= KNN_MAX_DIST_M, y[ref][ii], np.nan)
        cnt = np.isfinite(vals).sum(axis=1)
        ok = cnt >= KNN_MIN_N
        med[target[ok]] = np.nanmedian(vals[ok], axis=1)
        n_used[target] = cnt
        dist[target[ok]] = np.nanmedian(np.where(np.isfinite(vals), dd, np.nan)[ok], axis=1)

    df["nbhd_log_price_m2"] = med
    df["nbhd_n"] = n_used
    df["nbhd_median_dist_m"] = dist
    return df


def _on_located(df: pd.DataFrame, func, *args) -> pd.DataFrame:
    """Apply a spatial step to rows with coordinates only; others get NaN features."""
    has_xy = df["x_2180"].notna() & df["y_2180"].notna()
    part = func(df[has_xy].copy(), *args)
    new_cols = [c for c in part.columns if c not in df.columns]
    return df.join(part[new_cols])


def build() -> pd.DataFrame:
    df = pd.read_parquet(CLEAN_PATH)
    df = unit_features(df)
    df = _on_located(df, match_buildings, bdot.buildings())
    df = building_vintage(df)
    df = floor_features(df)
    df = _on_located(df, location_features)
    df = _on_located(df, metro_features, bdot.metro_stations())
    df = _on_located(df, neighbourhood_price)
    return df


def main() -> None:
    df = build()
    df.to_parquet(FEATURES_PATH, index=False)
    print(f"features: {len(df):,} rows x {df.shape[1]} columns -> {FEATURES_PATH}")


if __name__ == "__main__":
    main()
