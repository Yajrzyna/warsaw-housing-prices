"""Stage 2: resolve RCN references into one row per unit (lokal) sold.

Input : data/interim/rcn/<feature>/*.parquet   (from rcn_parse)
Output: data/interim/rcn_units.parquet        (all unit functions)
        data/interim/rcn_apartments.parquet   (funkcjaLokalu == 1, mieszkalna)

No rows are filtered out here apart from the function filter - market /
non-market, outlier and bundle rules belong to the cleaning stage. Instead
this stage adds deed-level context (how many transactions / units share one
notarial deed) that cleaning needs.
"""

from __future__ import annotations

from fractions import Fraction

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer

from real_estate.config import INTERIM_DIR
from real_estate.geo.districts import load_districts
from real_estate.ingest import rcn_codes as codes
from real_estate.ingest.rcn_parse import load_table

UNITS_PATH = INTERIM_DIR / "rcn_units.parquet"
APARTMENTS_PATH = INTERIM_DIR / "rcn_apartments.parquet"


def _latest_version(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Keep one row per key; newer source files win (sorted by file name, 2026 'rciwn' last)."""
    order = df["source_file"].str.replace("rciwn", "rcn")  # make 2026 snapshot sort after 2025
    return df.assign(_o=order).sort_values("_o").drop_duplicates(key, keep="last").drop(columns="_o")


def _share(s: pd.Series) -> pd.Series:
    def conv(v):
        try:
            return float(Fraction(str(v).replace(" ", "")))
        except (ValueError, ZeroDivisionError):
            return np.nan
    return s.map(conv, na_action="ignore")


def _first(refs: pd.Series) -> pd.Series:
    return refs.map(lambda r: r[0] if len(r) else None)


def build_units() -> pd.DataFrame:
    T = _latest_version(load_table("transakcje"), "lokalnyId")
    N = _latest_version(load_table("nieruchomosci"), "gml_id")
    L = _latest_version(load_table("lokale"), "gml_id")
    A = _latest_version(load_table("adresy"), "gml_id")
    D = _latest_version(load_table("dokumenty"), "gml_id")
    B = _latest_version(load_table("budynki"), "gml_id")

    t = T.assign(doc_ref=_first(T["podstawaPrawna"])).explode("nieruchomosc")
    n = N.explode("lokal").dropna(subset=["lokal"])

    df = (
        t.merge(n, left_on="nieruchomosc", right_on="gml_id", suffixes=("", "_n"))
        .merge(L, left_on="lokal", right_on="gml_id", suffixes=("", "_l"))
        .merge(D, left_on="doc_ref", right_on="gml_id", how="left", suffixes=("", "_d"))
    )
    df["addr_ref"] = _first(df["adresBudynkuZLokalem"])
    df = df.merge(A[["gml_id", "ulica", "numerPorzadkowy", "miejscowosc"]],
                  left_on="addr_ref", right_on="gml_id", how="left", suffixes=("", "_a"))

    out = pd.DataFrame({
        "transaction_id": df["lokalnyId"],
        "transaction_no": df["oznaczenieTransakcji"],
        "source_file": df["source_file"],
        "doc_ref": df["doc_ref"],
        "doc_number": df["oznaczenieDokumentu"],
        "date": df["dataSporzadzeniaDokumentu"],
        "transaction_type": df["rodzajTransakcji"],
        "market": df["rodzajRynku"],
        "seller_type": df["stronaSprzedajaca"],
        "buyer_type": df["stronaKupujaca"],
        "price_transaction": df["cenaTransakcjiBrutto"],
        "vat_transaction": df["kwotaPodatkuVAT"],
        "property_type": df["rodzajNieruchomosci"],
        "right_type": df["rodzajPrawaDoNieruchomosci"],
        "share_raw": df["udzialWPrawieDoNieruchomosci"],
        "share": _share(df["udzialWPrawieDoNieruchomosci"]),
        "price_property": df["cenaNieruchomosciBrutto"],
        "unit_ref": df["lokal"],
        "building_ref": _first(df["budynek"]),
        "unit_id": df["idLokalu"],
        "unit_function": df["funkcjaLokalu"],
        "rooms": df["liczbaIzb"],
        "floor": df["nrKondygnacji"],
        "area_m2": df["powUzytkowaLokalu"],
        "ancillary_area_m2": df.get("powUzytkowaPomieszczenPrzynal"),
        "price_unit": df["cenaLokaluBrutto"],
        "vat_unit": df.get("kwotaPodatkuVAT_l"),
        "unit_info": df.get("dodatkoweInformacje"),
        "street": df["ulica"],
        "house_no": df["numerPorzadkowy"],
        "x_2180": df["x_2180"],
        "y_2180": df["y_2180"],
    })

    # decoded labels next to codes
    for col, mapping in [
        ("transaction_type", codes.RODZAJ_TRANSAKCJI), ("market", codes.RODZAJ_RYNKU),
        ("seller_type", codes.STRONA), ("buyer_type", codes.STRONA),
        ("property_type", codes.RODZAJ_NIERUCHOMOSCI), ("right_type", codes.RODZAJ_PRAWA),
        ("unit_function", codes.FUNKCJA_LOKALU),
    ]:
        out[f"{col}_label"] = out[col].map(mapping)

    out = _fill_coordinates(out, B)
    out = _add_location(out)
    out = _add_deed_context(out)

    out["year"] = out["date"].dt.year
    out["quarter"] = out["date"].dt.to_period("Q").astype(str)
    # Unit price when the register gives one, else the transaction price
    # (only meaningful when the transaction concerns this single unit).
    out["price"] = out["price_unit"].fillna(out["price_transaction"])
    out["price_source"] = np.where(out["price_unit"].notna(), "unit", "transaction")
    out["price_m2"] = out["price"] / out["area_m2"].where(out["area_m2"] > 0)
    return out.sort_values(["date", "transaction_no"]).reset_index(drop=True)


def _fill_coordinates(df: pd.DataFrame, buildings: pd.DataFrame) -> pd.DataFrame:
    """Unit point -> building centroid -> median point of the same address elsewhere.

    Unit georeference points are missing for most of 2017-2020 and building /
    parcel geometry is missing for all of 2006-2015, so the fallbacks matter.
    """
    df["coord_source"] = np.where(df["x_2180"].notna(), "unit_point", None)
    b = buildings.dropna(subset=["x_2180"])

    # building named in the EGiB unit id first (a property can also link garages / outbuildings
    # on the same parcel), then the first building linked to the property
    by_ref = b.set_index("gml_id")[["x_2180", "y_2180"]]
    by_id = b.drop_duplicates("idBudynku").set_index("idBudynku")[["x_2180", "y_2180"]]
    building_id = df["unit_id"].str.rsplit(".", n=1).str[0]
    for keys, lookup in ((building_id, by_id), (df["building_ref"], by_ref)):
        fill = df["x_2180"].isna() & keys.isin(lookup.index)
        df.loc[fill, ["x_2180", "y_2180"]] = lookup.loc[keys[fill]].to_numpy()
        df.loc[fill, "coord_source"] = "building_centroid"

    addr = df["street"].str.lower().str.strip() + "|" + df["house_no"].str.lower().str.replace(" ", "")
    known = df[df["coord_source"].notna()].assign(_k=addr).groupby("_k")[["x_2180", "y_2180"]].median()
    fill = df["x_2180"].isna() & addr.isin(known.index)
    df.loc[fill, ["x_2180", "y_2180"]] = known.loc[addr[fill]].to_numpy()
    df.loc[fill, "coord_source"] = "address_lookup"
    return df


def _add_location(df: pd.DataFrame) -> pd.DataFrame:
    to_wgs = Transformer.from_crs("EPSG:2180", "EPSG:4326", always_xy=True)
    df["lon"], df["lat"] = to_wgs.transform(df["x_2180"].to_numpy(), df["y_2180"].to_numpy())

    districts = load_districts()
    pts = gpd.GeoDataFrame(df[["x_2180"]], geometry=gpd.points_from_xy(df["x_2180"], df["y_2180"]), crs="EPSG:2180")
    joined = gpd.sjoin(pts, districts, how="left", predicate="within")
    joined = joined[~joined.index.duplicated()]
    df["district_code"] = joined["district_code"]
    df["district_source"] = np.where(df["district_code"].notna(), "coordinates", None)

    # fallback: EGiB id prefix (146502..146519) when coordinates are missing
    prefix = df["unit_id"].str[:6]
    fill = df["district_code"].isna() & prefix.isin(districts["district_code"])
    df.loc[fill, "district_code"] = prefix[fill]
    df.loc[fill, "district_source"] = "unit_id"
    df["district"] = df["district_code"].map(dict(zip(districts["district_code"], districts["district"])))
    return df


def _add_deed_context(df: pd.DataFrame) -> pd.DataFrame:
    """How many transactions / units of each function share the same notarial deed."""
    g = df.groupby("doc_ref")
    df["deed_n_transactions"] = g["transaction_id"].transform("nunique")
    df["deed_n_units"] = g["unit_ref"].transform("size")
    for code, name in [(1, "residential"), (5, "garage")]:
        df[f"deed_n_{name}"] = (df["unit_function"] == code).groupby(df["doc_ref"]).transform("sum")
    df["deed_n_other"] = df["deed_n_units"] - df["deed_n_residential"] - df["deed_n_garage"]
    df["deed_price_sum_units"] = g["price_unit"].transform("sum", min_count=1)
    df["deed_n_distinct_tx_prices"] = g["price_transaction"].transform("nunique")
    return df


def main() -> None:
    units = build_units()
    UNITS_PATH.parent.mkdir(parents=True, exist_ok=True)
    units.to_parquet(UNITS_PATH, index=False)
    apartments = units[units["unit_function"] == 1].reset_index(drop=True)
    apartments.to_parquet(APARTMENTS_PATH, index=False)
    print(f"units: {len(units):,}  apartments: {len(apartments):,}")


if __name__ == "__main__":
    main()
