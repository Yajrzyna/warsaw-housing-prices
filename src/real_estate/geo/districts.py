"""Warsaw district (dzielnica) boundaries from the national boundary register (PRG, GUGiK).

In PRG the 18 Warsaw districts are cadastral units (jednostki ewidencyjne)
with codes 146502_8 (Bemowo) ... 146519_8 (Zoliborz). The same 6-digit prefix
starts the EGiB ids of units / buildings / parcels in the RCN data (2016+).
"""

from __future__ import annotations

import io
from pathlib import Path

import geopandas as gpd
import requests

from real_estate.config import HTTP_HEADERS, RAW_DIR

PRG_WFS = "https://mapy.geoportal.gov.pl/wss/service/PZGIK/PRG/WFS/AdministrativeBoundaries"
WARSAW_BBOX_2180 = "470000,620000,505000,660000,urn:ogc:def:crs:EPSG::2180"  # northing first
DISTRICTS_PATH = RAW_DIR / "geo" / "warsaw_districts.gpkg"


def download_districts(dest: Path = DISTRICTS_PATH) -> gpd.GeoDataFrame:
    params = {
        "SERVICE": "WFS", "VERSION": "2.0.0", "REQUEST": "GetFeature",
        "TYPENAMES": "ms:A05_Granice_jednostek_ewidencyjnych", "BBOX": WARSAW_BBOX_2180,
    }
    r = requests.get(PRG_WFS, params=params, headers=HTTP_HEADERS, timeout=300)
    r.raise_for_status()
    gdf = gpd.read_file(io.BytesIO(r.content))
    gdf = gdf[gdf["JPT_KOD_JE"].astype(str).str.startswith("1465")].copy()
    gdf["district_code"] = gdf["JPT_KOD_JE"].str[:6]
    gdf["district"] = gdf["JPT_NAZWA_"].str.replace(" - dzielnica", "", regex=False)
    gdf = gdf[["district_code", "district", "geometry"]].sort_values("district_code")
    if len(gdf) != 18:
        raise RuntimeError(f"Expected 18 Warsaw districts, got {len(gdf)}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(dest, driver="GPKG")
    return gdf


def load_districts(path: Path = DISTRICTS_PATH) -> gpd.GeoDataFrame:
    if not path.exists():
        return download_districts(path)
    return gpd.read_file(path)


if __name__ == "__main__":
    print(download_districts())
