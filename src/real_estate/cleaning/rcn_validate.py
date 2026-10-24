"""Benchmark the cleaned RCN sample against NBP BaRN Warsaw transaction prices.

Writes reports/tables/rcn_vs_nbp_quarterly.csv and returns summary statistics.
NBP publishes average (not median) prices, so the comparison uses means.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from real_estate.cleaning.rcn_clean import CLEAN_PATH
from real_estate.config import REPORTS_TABLES_DIR
from real_estate.ingest.nbp import warsaw_transaction_prices

SEGMENT_TO_NBP = {"primary_developer": "primary", "secondary": "secondary"}


def compare(clean: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean = pd.read_parquet(CLEAN_PATH) if clean is None else clean
    nbp = warsaw_transaction_prices()
    q = pd.PeriodIndex(clean["quarter"], freq="Q")

    tables, stats = [], []
    for segment, market in SEGMENT_TO_NBP.items():
        sub = clean[clean["segment"] == segment]
        ours = sub.groupby(q[clean["segment"] == segment])["price_m2"].agg(["mean", "median", "size"])
        j = ours.join(nbp[market].rename("nbp_mean"), how="inner")
        j["ratio"] = j["mean"] / j["nbp_mean"]
        j.insert(0, "segment", segment)
        tables.append(j)

        yoy = np.log(j[["mean", "nbp_mean"]]).diff(4).dropna()
        qoq = np.log(j[["mean", "nbp_mean"]]).diff().dropna()
        stats.append({
            "segment": segment, "quarters": len(j),
            "median_level_ratio": j["ratio"].median(),
            "corr_qoq": qoq.corr().iloc[0, 1], "corr_yoy": yoy.corr().iloc[0, 1],
        })

    table = pd.concat(tables)
    REPORTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(REPORTS_TABLES_DIR / "rcn_vs_nbp_quarterly.csv")
    return table, pd.DataFrame(stats)


if __name__ == "__main__":
    print(compare()[1].round(3).to_string(index=False))
