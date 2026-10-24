"""Macro data for Warsaw: interest rates, wages, housing supply, population.

Sources (raw copies in data/raw/macro/):
  NBP        reference rate, full decision history (static.nbp.pl archive XML)
  Eurostat   irt_st_m, Poland 3-month money-market rate (monthly; WIBOR-3M equivalent)
  GUS BDL    Warsaw (unit 071412865000), annual:
               64428   average gross monthly wage (PLN)
               72305   population, 31 December
               748585  dwellings completed (Jan-Dec)
               747729  dwellings started (Jan-Dec)

NBP's statistics on rates for *new* mortgages sit behind bot protection, so the
mortgage rate is proxied as 3M money-market rate + MORTGAGE_MARGIN_PP. Most
Polish mortgages are variable-rate (WIBOR + margin); margins have mostly been
around 2 pp.
"""

from __future__ import annotations

import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests

from real_estate.config import HTTP_HEADERS, RAW_DIR

MACRO_DIR = RAW_DIR / "macro"
NBP_RATES_URL = "https://static.nbp.pl/dane/stopy/stopy_procentowe_archiwum.xml"
EUROSTAT_IRT_URL = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/irt_st_m"
BDL_URL = "https://bdl.stat.gov.pl/api/v1/data/by-unit/{unit}"
WARSAW_UNIT = "071412865000"
MORTGAGE_MARGIN_PP = 2.0

BDL_VARS = {
    "wage_gross_monthly": [64428],
    "population": [72305],
    "dwellings_completed": [748585],
    "dwellings_started": [747729],
}
# Building permits (217329 / 567225) are not used: the post-2018 definition stops in 2019.
# Population has a level break in 2020 (GUS rebasing to the 2021 census), not a real jump.


def _get(url: str, **params) -> requests.Response:
    r = requests.get(url, params=params, headers=HTTP_HEADERS, timeout=120)
    r.raise_for_status()
    return r


def nbp_reference_rate(refresh: bool = False) -> pd.Series:
    """Daily NBP reference rate (%) from the full decision history."""
    path = MACRO_DIR / "nbp_rates_archive.xml"
    if refresh or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_get(NBP_RATES_URL).content)
    root = ET.fromstring(path.read_bytes())
    rows = []
    for block in root.iter("pozycje"):
        for pos in block.iter("pozycja"):
            if pos.get("id") == "ref":
                rows.append((pd.Timestamp(block.get("obowiazuje_od")), float(pos.get("oprocentowanie").replace(",", "."))))
    s = pd.Series(dict(rows)).sort_index()
    days = pd.date_range(s.index.min(), pd.Timestamp.today().normalize(), freq="D")
    return s.reindex(days).ffill().rename("reference_rate")


def interbank_3m(refresh: bool = False) -> pd.Series:
    """Monthly Poland 3-month money-market rate (%), Eurostat irt_st_m."""
    path = MACRO_DIR / "eurostat_irt_m3_pl.csv"
    if refresh or not path.exists():
        d = _get(EUROSTAT_IRT_URL, geo="PL", int_rt="IRT_M3", format="JSON", lang="EN",
                 sinceTimePeriod="2005-01").json()
        pos = {i: m for m, i in d["dimension"]["time"]["category"]["index"].items()}
        s = pd.Series({pos[int(k)]: v for k, v in d["value"].items()}).sort_index()
        path.parent.mkdir(parents=True, exist_ok=True)
        s.rename("rate_3m").to_csv(path, index_label="month")
    s = pd.read_csv(path, index_col=0)["rate_3m"]
    s.index = pd.PeriodIndex(s.index, freq="M")
    return s


def bdl_warsaw(refresh: bool = False) -> pd.DataFrame:
    """Annual GUS BDL series for Warsaw (one column per BDL_VARS key)."""
    path = MACRO_DIR / "bdl_warsaw_annual.json"
    if refresh or not path.exists():
        raw = {}
        for name, var_ids in BDL_VARS.items():
            raw[name] = {}
            for v in var_ids:
                d = _get(BDL_URL.format(unit=WARSAW_UNIT), **{"var-id": v, "format": "json", "page-size": 100}).json()
                for res in d.get("results", []):
                    for val in res["values"]:
                        raw[name].setdefault(val["year"], val["val"])   # first listed variable wins
                time.sleep(0.7)                                           # anonymous API rate limit
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(raw, indent=1))
    raw = json.loads(path.read_text())
    df = pd.DataFrame({k: pd.Series(v) for k, v in raw.items()})
    df.index = df.index.astype(int)
    df.index.name = "year"
    return df.sort_index()


def download_all() -> None:
    nbp_reference_rate(refresh=True)
    interbank_3m(refresh=True)
    bdl_warsaw(refresh=True)


if __name__ == "__main__":
    download_all()
    print(bdl_warsaw().tail(8))
    print(interbank_3m().tail(3))
    print(nbp_reference_rate().resample("YE").last().tail(8))
