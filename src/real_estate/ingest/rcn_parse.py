"""Parse Warsaw RCN GML archives into flat parquet tables.

The GML follows the RCN conceptual model (zalacznik nr 9 do rozporzadzenia
w sprawie EGiB, 2021; docs/references/RCN_model_pojeciowy_zalacznik9.pdf).
Every object is a separate gml:featureMember and objects reference each
other through xlink:href:

    RCN_Transakcja --podstawaPrawna--> RCN_Dokument   (notarial deed, date)
                   --nieruchomosc--->  RCN_Nieruchomosc (1..*)
    RCN_Nieruchomosc --lokal-->   RCN_Lokal   --adresBudynkuZLokalem--> RCN_Adres
                     --budynek--> RCN_Budynek --adresBudynku-->         RCN_Adres
                     --dzialka--> RCN_Dzialka --adresDzialki-->         RCN_Adres

Stage 1 (parse_archive) streams one zip and writes one parquet per feature
type to data/interim/rcn/<feature>/<archive>.parquet. Reference fields are
stored as lists of gml ids. Geometry is reduced to a point (units: the
georeference point; buildings / parcels: polygon centroid) in EPSG:2180.
"""

from __future__ import annotations

import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd
from lxml import etree
from pyproj import Transformer

from real_estate.config import RCN_INTERIM_DIR, RCN_RAW_DIR

GML = "http://www.opengis.net/gml/3.2"
XLINK_HREF = "{http://www.w3.org/1999/xlink}href"
GML_ID = f"{{{GML}}}id"

FEATURE_TABLES = {
    "RCN_Transakcja": "transakcje",
    "RCN_Nieruchomosc": "nieruchomosci",
    "RCN_Lokal": "lokale",
    "RCN_Budynek": "budynki",
    "RCN_Dzialka": "dzialki",
    "RCN_Adres": "adresy",
    "RCN_Dokument": "dokumenty",
}

# Fields that are references (xlink:href) - always stored as lists of ids.
REF_FIELDS = {
    "RCN_Transakcja": ["podstawaPrawna", "nieruchomosc"],
    "RCN_Nieruchomosc": ["lokal", "budynek", "dzialka"],
    "RCN_Lokal": ["adresBudynkuZLokalem"],
    "RCN_Budynek": ["adresBudynku"],
    "RCN_Dzialka": ["adresDzialki"],
}

NUMERIC_FIELDS = {
    "cenaTransakcjiBrutto", "kwotaPodatkuVAT", "cenaNieruchomosciBrutto",
    "polePowierzchniNieruchomosciGruntowej", "cenaLokaluBrutto", "powUzytkowaLokalu",
    "powUzytkowaPomieszczenPrzynal", "cenaBudynkuBrutto", "powierzchniaUzytkowaBudynku",
    "cenaDzialkiEwidencyjnejBrutto", "polePowierzchniEwidencyjnej",
}
INT_FIELDS = {
    "rodzajTransakcji", "rodzajRynku", "stronaSprzedajaca", "stronaKupujaca",
    "rodzajNieruchomosci", "rodzajPrawaDoNieruchomosci", "funkcjaLokalu", "liczbaIzb",
    "nrKondygnacji", "rodzajBudynku", "sposobUzytkowania",
}

# Source CRS is EPSG:2178 (PL-2000 zone 7) with northing-first axis order;
# output in EPSG:2180 (PL-1992, metres, used nationally) as x=easting, y=northing.
_TO_2180 = Transformer.from_crs("EPSG:2178", "EPSG:2180", always_xy=True)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _ring_centroid(coords: list[float]) -> tuple[float, float, float]:
    """Area-weighted centroid of a ring given as flat [n, e, n, e, ...]. Returns (e, n, area)."""
    n, e = coords[0::2], coords[1::2]
    a = cx = cy = 0.0
    for i in range(len(e) - 1):
        cross = e[i] * n[i + 1] - e[i + 1] * n[i]
        a += cross
        cx += (e[i] + e[i + 1]) * cross
        cy += (n[i] + n[i + 1]) * cross
    if abs(a) < 1e-9:  # degenerate ring - fall back to vertex mean
        return sum(e) / len(e), sum(n) / len(n), 0.0
    return cx / (3 * a), cy / (3 * a), abs(a) / 2


def _geometry_point(geom_parent: etree._Element) -> tuple[float, float] | None:
    """Reduce a Point / Polygon / MultiSurface to a single (e, n) point in EPSG:2178."""
    pos = geom_parent.find(f".//{{{GML}}}pos")
    if pos is not None and pos.text:
        n, e = map(float, pos.text.split()[:2])
        return e, n
    best = None
    for ring in geom_parent.iterfind(f".//{{{GML}}}exterior//{{{GML}}}posList"):
        coords = list(map(float, ring.text.split()))
        e, n, area = _ring_centroid(coords)
        if best is None or area > best[2]:
            best = (e, n, area)
    return (best[0], best[1]) if best else None


def _parse_feature(el: etree._Element) -> dict:
    ftype = _local(el.tag)
    rec: dict = {"gml_id": el.get(GML_ID)}
    refs = defaultdict(list)
    multi = defaultdict(list)
    for child in el:
        name = _local(child.tag)
        href = child.get(XLINK_HREF)
        if href is not None:
            refs[name].append(href.lstrip("#"))
        elif name == "IdRCN":
            for sub in child.iter():
                if _local(sub.tag) in ("lokalnyId", "wersjaId") and sub.text:
                    rec[_local(sub.tag)] = sub.text.strip()
        elif name in ("geometria", "georeferencja"):
            pt = _geometry_point(child)
            if pt:
                rec["x_2180"], rec["y_2180"] = _TO_2180.transform(*pt)
        elif child.text and child.text.strip():
            multi[name].append(child.text.strip())
    for name, values in multi.items():
        rec[name] = values[0] if len(values) == 1 else ";".join(values)
    for name in REF_FIELDS.get(ftype, []):
        rec[name] = refs.get(name, [])
    return rec


def _typed_frame(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame.from_records(records)
    for col in df.columns:
        if col in NUMERIC_FIELDS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif col in INT_FIELDS:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    if "dataSporzadzeniaDokumentu" in df:
        df["dataSporzadzeniaDokumentu"] = pd.to_datetime(df["dataSporzadzeniaDokumentu"], errors="coerce")
    return df


def parse_archive(zip_path: Path, out_dir: Path = RCN_INTERIM_DIR) -> dict[str, int]:
    """Stream-parse one RCN zip archive; write one parquet per feature type."""
    records: dict[str, list[dict]] = defaultdict(list)
    source = zip_path.stem
    with zipfile.ZipFile(zip_path) as zf:
        gml_name = next(n for n in zf.namelist() if n.lower().endswith(".gml"))
        with zf.open(gml_name) as fh:
            for _, fm in etree.iterparse(fh, tag=f"{{{GML}}}featureMember", huge_tree=True):
                for feature in fm:
                    ftype = _local(feature.tag)
                    if ftype in FEATURE_TABLES:
                        records[ftype].append(_parse_feature(feature))
                # free memory: clear this element and already-processed siblings
                fm.clear()
                parent = fm.getparent()
                while fm.getprevious() is not None:
                    del parent[0]

    counts = {}
    for ftype, recs in records.items():
        table = FEATURE_TABLES[ftype]
        df = _typed_frame(recs)
        df.insert(0, "source_file", source)
        dest = out_dir / table / f"{source}.parquet"
        dest.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dest, index=False)
        counts[table] = len(df)
    return counts


def parse_all(raw_dir: Path = RCN_RAW_DIR, out_dir: Path = RCN_INTERIM_DIR, force: bool = False) -> None:
    for zip_path in sorted(raw_dir.glob("*.zip")):
        marker = out_dir / "transakcje" / f"{zip_path.stem}.parquet"
        if marker.exists() and not force:
            print(f"skip {zip_path.name} (already parsed)")
            continue
        counts = parse_archive(zip_path, out_dir)
        print(zip_path.name, counts, flush=True)


def load_table(table: str, in_dir: Path = RCN_INTERIM_DIR) -> pd.DataFrame:
    """Concatenate one feature table across all parsed archives."""
    files = sorted((in_dir / table).glob("*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


if __name__ == "__main__":
    parse_all()
