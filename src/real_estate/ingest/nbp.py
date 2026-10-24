"""NBP BaRN housing price database - Warsaw transaction prices per m2 (quarterly).

Source: https://static.nbp.pl/dane/rynek-nieruchomosci/ceny_mieszkan.xlsx
Average transaction price per m2 incl. VAT, primary and secondary market,
from 2006Q3. NBP collects it from agents and developers (contract dates),
so it is an independent benchmark for the RCN (notarial deed) data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests

from real_estate.config import HTTP_HEADERS, RAW_DIR

NBP_URL = "https://static.nbp.pl/dane/rynek-nieruchomosci/ceny_mieszkan.xlsx"
NBP_PATH = RAW_DIR / "nbp" / "ceny_mieszkan.xlsx"
SHEETS = {"primary": "Rynek pierwotny", "secondary": "Rynek wtórny"}
ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4}


def download(dest: Path = NBP_PATH) -> Path:
    r = requests.get(NBP_URL, headers=HTTP_HEADERS, timeout=120)
    r.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(r.content)
    return dest


def warsaw_transaction_prices(path: Path = NBP_PATH) -> pd.DataFrame:
    """Quarterly Warsaw average transaction price per m2: columns primary, secondary."""
    if not path.exists():
        download(path)
    series = {}
    for market, sheet in SHEETS.items():
        d = pd.read_excel(path, sheet, header=None)
        block, city = d.iloc[3].ffill().fillna("").astype(str), d.iloc[6].fillna("").astype(str)
        col = next(i for i in range(d.shape[1])
                   if city[i].startswith("Warszawa") and block[i].startswith("Ceny transakcyjne"))
        s = d.iloc[7:, [0, col]].dropna()
        roman_q, year = s.iloc[:, 0].str.split().str[0], s.iloc[:, 0].str.split().str[1]
        idx = pd.PeriodIndex(year + "Q" + roman_q.map(ROMAN).astype(str), freq="Q")
        series[market] = pd.Series(s.iloc[:, 1].astype(float).to_numpy(), index=idx)
    out = pd.DataFrame(series)
    out.index.name = "quarter"
    return out


def warsaw_hedonic_index(path: Path = NBP_PATH) -> pd.Series:
    """NBP hedonic index, Warsaw secondary market, as a log level (2006Q3 = 0).

    NBP publishes quarter-on-quarter dynamics (previous quarter = 100); they are
    chained here.
    """
    if not path.exists():
        download(path)
    d = pd.read_excel(path, SHEETS["secondary"], header=None)
    block, city = d.iloc[3].ffill().fillna("").astype(str), d.iloc[6].fillna("").astype(str)
    col = next(i for i in range(d.shape[1]) if city[i].startswith("Warszawa")
               and "kwartał do kwartału poprzedniego" in block[i])
    s = d.iloc[7:, [0, col]].dropna(subset=[0])
    roman_q, year = s.iloc[:, 0].str.split().str[0], s.iloc[:, 0].str.split().str[1]
    idx = pd.PeriodIndex(year + "Q" + roman_q.map(ROMAN).astype(str), freq="Q")
    qoq = pd.Series(pd.to_numeric(s.iloc[:, 1], errors="coerce").to_numpy(), index=idx)
    log_level = np.log(qoq / 100).fillna(0).cumsum()
    log_level.name = "nbp_hedonic"
    return log_level
