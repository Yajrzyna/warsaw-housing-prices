"""BDOT10k (national topographic database, GUGiK) layers for Warsaw.

Download: https://opendata.geoportal.gov.pl/bdot10k/schemat2021/GPKG/14/1465_GPKG.zip
(one GeoPackage per object class, EPSG:2180, state as of the file date - 2025/26).

Used layers
  BUBD_A  buildings: storeys (LICZBAKONDYGNACJI), function, EGiB id
  OIKM_P  transport points: metro entrances, rail stations, bus/tram stops
  KUSK_A  parks, squares, sport & recreation complexes
  PTLZ_A  forests / wooded land
  PTWP_A  surface water (the Vistula = largest flowing-water polygon)

Note: these are *current* layers - applied to past sales they describe today's
surroundings (except metro, which is made time-aware with opening dates).
"""

from __future__ import annotations

import zipfile
from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

from real_estate.config import EXTERNAL_DIR, HTTP_HEADERS, RAW_DIR

BDOT_URL = "https://opendata.geoportal.gov.pl/bdot10k/schemat2021/GPKG/14/1465_GPKG.zip"
BDOT_DIR = RAW_DIR / "geo" / "bdot10k"
BDOT_ZIP = BDOT_DIR / "1465_GPKG.zip"
METRO_CSV = EXTERNAL_DIR / "metro_stations.csv"

# building functions that cannot contain apartments - excluded when matching sales to buildings
# (BDOT draws underground garages under courtyards as separate footprints)
NON_RESIDENTIAL = {
    "garaż", "budynek gospodarczy", "stacja transformatorowa", "magazyn", "parking wielopoziomowy",
    "szklarnia lub cieplarnia", "stacja pomp", "kotłownia", "stacja gazowa", "silos",
    "zbiornik na ciecz", "toaleta publiczna", "myjnia samochodowa", "stacja paliw",
    "warsztat remontowo-naprawczy", "produkcyjny",
}


def download(dest: Path = BDOT_ZIP) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(BDOT_URL, headers=HTTP_HEADERS, stream=True, timeout=600) as r:
        r.raise_for_status()
        with dest.open("wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
    return dest


def _layer(code: str) -> gpd.GeoDataFrame:
    if not BDOT_ZIP.exists():
        download()
    with zipfile.ZipFile(BDOT_ZIP) as zf:
        name = next(n for n in zf.namelist() if f"__OT_{code}.gpkg" in n)
        target = BDOT_DIR / name
        if not target.exists():
            zf.extract(name, BDOT_DIR)
    return gpd.read_file(target)


@lru_cache
def buildings() -> gpd.GeoDataFrame:
    b = _layer("BUBD_A")
    b = b[b["KATEGORIAISTNIENIA"] != "zniszczony"]
    out = b.rename(columns={
        "IDEGIB": "bdot_building_egib_id", "LICZBAKONDYGNACJI": "storeys",
        "PRZEWAZAJACAFUNKCJABUDYNKU": "building_function", "LOKALNYID": "bdot_building_id",
    })[["bdot_building_id", "bdot_building_egib_id", "storeys", "building_function", "geometry"]]
    out["footprint_m2"] = out.area
    return out.reset_index(drop=True)


@lru_cache
def transport_points() -> gpd.GeoDataFrame:
    return _layer("OIKM_P")[["RODZAJ", "NAZWA", "KATEGORIAISTNIENIA", "geometry"]]


def metro_stations() -> gpd.GeoDataFrame:
    """One point per station (mean of its entrances) with opening date."""
    meta = pd.read_csv(METRO_CSV, parse_dates=["opened"])
    ent = transport_points()
    ent = ent[ent["RODZAJ"] == "wejście do stacji metra"]
    rows = []
    for rec in meta.itertuples():
        pts = ent[ent["NAZWA"].isin(rec.bdot_names.split("|"))]
        if pts.empty:
            raise ValueError(f"No BDOT entrances for metro station {rec.station}")
        c = pts.geometry.union_all().centroid
        rows.append({"station": rec.station, "line": rec.line, "opened": rec.opened,
                     "status": rec.status, "geometry": c})
    return gpd.GeoDataFrame(rows, crs=ent.crs)


def rail_stations() -> gpd.GeoDataFrame:
    p = transport_points()
    return p[p["RODZAJ"] == "stacja lub przystanek kolejowy"].reset_index(drop=True)


def bus_tram_stops() -> gpd.GeoDataFrame:
    p = transport_points()
    return p[p["RODZAJ"] == "przystanek autobusowy lub tramwajowy"].reset_index(drop=True)


def parks() -> gpd.GeoDataFrame:
    k = _layer("KUSK_A")
    return k[k["RODZAJ"].isin(["park lub skwer", "ogród botaniczny", "ogród zoologiczny"])].reset_index(drop=True)


def forests(min_area_m2: float = 20_000) -> gpd.GeoDataFrame:
    f = _layer("PTLZ_A")
    return f[f.area >= min_area_m2].reset_index(drop=True)


def vistula() -> gpd.GeoDataFrame:
    w = _layer("PTWP_A")
    w = w[w["RODZAJ"] == "woda płynąca"]
    return w.loc[[w.area.idxmax()]].reset_index(drop=True)
