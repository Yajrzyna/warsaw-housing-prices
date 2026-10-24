"""Eurostat HICP for Poland (all items, 2015 = 100) - used to deflate prices.

Dataset prc_hicp_minr (ECOICOP ver. 2, published from 2026; the older prc_hicp_midx
stops in 2025-12). Monthly index, averaged to quarters.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from real_estate.config import HTTP_HEADERS, RAW_DIR

HICP_URL = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/prc_hicp_minr"
HICP_PATH = RAW_DIR / "macro" / "hicp_pl_monthly.csv"


def download_hicp(dest: Path = HICP_PATH) -> pd.Series:
    params = {"geo": "PL", "unit": "I15", "coicop18": "TOTAL", "format": "JSON", "lang": "EN",
              "sinceTimePeriod": "2005-01"}
    r = requests.get(HICP_URL, params=params, headers=HTTP_HEADERS, timeout=120)
    r.raise_for_status()
    d = r.json()
    pos_to_month = {i: m for m, i in d["dimension"]["time"]["category"]["index"].items()}
    s = pd.Series({pos_to_month[int(k)]: v for k, v in d["value"].items()}).sort_index()
    s.index = pd.PeriodIndex(s.index, freq="M")
    s.name = "hicp_2015_100"
    dest.parent.mkdir(parents=True, exist_ok=True)
    s.to_csv(dest, index_label="month")
    return s


def hicp_quarterly(path: Path = HICP_PATH, refresh: bool = False) -> pd.Series:
    """Quarterly average HICP (2015 = 100). Quarters with < 3 months are dropped."""
    if refresh or not path.exists():
        m = download_hicp(path)
    else:
        m = pd.read_csv(path, index_col=0)["hicp_2015_100"]
        m.index = pd.PeriodIndex(m.index, freq="M")
    q = m.groupby(m.index.asfreq("Q"))
    out = q.mean()[q.size() == 3]
    out.index.name = "q"
    return out
