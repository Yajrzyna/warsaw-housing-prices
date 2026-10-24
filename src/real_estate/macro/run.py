"""Phase 7 outputs: macro datasets, affordability, drivers, programme analysis.

  data/processed/macro_quarterly.parquet, macro_annual.parquet
  reports/tables/mac_*.csv
"""

from __future__ import annotations

import pandas as pd

from real_estate.config import PROCESSED_DIR, REPORTS_TABLES_DIR as T
from real_estate.features.build_features import FEATURES_PATH
from real_estate.macro import analysis as ma

Q_PATH = PROCESSED_DIR / "macro_quarterly.parquet"
A_PATH = PROCESSED_DIR / "macro_annual.parquet"


def cap_share(cap: int = 700_000) -> pd.DataFrame:
    """Share of resales with a total price <= cap (what a programme-sized loan plus deposit could buy)."""
    f = pd.read_parquet(FEATURES_PATH, columns=["quarter", "segment", "price", "area_m2"])
    s = f[f["segment"] == "secondary"]
    out = s.groupby("quarter").agg(sales=("price", "size"),
                                   share_under_cap_pct=("price", lambda p: (p <= cap).mean() * 100),
                                   share_small_pct=("area_m2", lambda a: (a < 45).mean() * 100))
    return out.round(2)


def main() -> None:
    q = ma.quarterly_frame()
    a = ma.annual_frame(q)
    q.assign(q=q.index.astype(str)).reset_index(drop=True).to_parquet(Q_PATH, index=False)
    a.reset_index().to_parquet(A_PATH, index=False)

    cols = ["hedonic", "hedonic_real", "median_price_m2", "sales_secondary", "sales_developer", "reference_rate",
            "rate_3m", "mortgage_rate", "hicp", "wage", "years_of_wages", "m2_per_wage", "payment_share"]
    qq = q[cols].round(3)
    qq.index = qq.index.astype(str)
    qq.to_csv(T / "mac_quarterly.csv")
    a.round(3).to_csv(T / "mac_annual.csv")
    ma.district_affordability(q).round(3).to_csv(T / "mac_district_affordability.csv")

    ll = ma.lead_lag(q)
    ll.round(3).to_csv(T / "mac_lead_lag.csv", index=False)

    rows = []
    for spec, kw in {"A: rate lag 2": {"rate_lag": 2}, "B: rate lag 3": {"rate_lag": 3},
                     "C: rate lag 2 + sales lag 2": {"rate_lag": 2, "sales_lag": 2}}.items():
        res, d = ma.driver_regression(q, **kw)
        for term in res.params.index:
            rows.append({"spec": spec, "term": term, "coef": res.params[term], "se": res.bse[term],
                         "p_value": res.pvalues[term], "n": int(res.nobs), "r2": res.rsquared,
                         "sample": f"{d.index.min()}–{d.index.max()}"})
    reg = pd.DataFrame(rows)
    reg.round(4).to_csv(T / "mac_regression.csv", index=False)

    bands = ma.size_band_indices()
    b = bands.round(2)
    b.index = b.index.astype(str)
    b.to_csv(T / "mac_size_band_indices.csv")
    did = ma.bk2_did(bands)
    did.to_csv(T / "mac_bk2_did.csv")
    cap_share().to_csv(T / "mac_resale_price_cap_share.csv")

    print(a[["median_price_m2", "wage", "years_of_wages", "m2_per_wage", "payment_share", "mortgage_rate"]].round(2).tail(6))
    print(reg[reg["term"] != "const"].round(3).to_string(index=False))
    print(did.to_string())


if __name__ == "__main__":
    main()
