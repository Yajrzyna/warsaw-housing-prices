"""Compute all price indices and write them to disk.

Outputs
  data/processed/price_indices_quarterly.parquet   long: segment, method, q, log_index, index (2015=100), n, ...
  reports/tables/price_indices_quarterly.csv        wide: one column per segment/method (+ real hedonic)
  reports/tables/price_indices_district_annual.csv  annual hedonic index per district (secondary, 2015=100)
  reports/tables/price_index_diagnostics.csv        growth, peak, volatility, agreement with NBP
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from real_estate.config import PROCESSED_DIR, REPORTS_TABLES_DIR
from real_estate.indices import common, hedonic, repeat_sales, simple
from real_estate.ingest.eurostat import hicp_quarterly
from real_estate.ingest.nbp import warsaw_hedonic_index

QUARTERLY_PATH = PROCESSED_DIR / "price_indices_quarterly.parquet"
DISTRICT_PATH = REPORTS_TABLES_DIR / "price_indices_district_annual.csv"
DIAG_PATH = REPORTS_TABLES_DIR / "price_index_diagnostics.csv"
HEADLINE = ("secondary", "hedonic")
LAST_FULL_YEAR = 2025


def _long(segment: str, method: str, t: pd.DataFrame) -> pd.DataFrame:
    out = t.copy()
    out.index = pd.PeriodIndex(out.index, freq="Q", name="q")
    out["index"] = common.rebase(out["log_index"])
    out = out.reset_index()
    out.insert(0, "method", method)
    out.insert(0, "segment", segment)
    return out


def quarterly_indices(sales: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    frames, info = [], {}
    for seg in ["secondary", "primary_developer"]:
        d = sales[sales["segment"] == seg]
        methods = {
            "median": simple.median_index(d),
            "mix_adjusted": simple.stratified_index(d),
            "hedonic": hedonic.rolling_time_dummy(d, window=5),
        }
        if seg == "secondary":
            methods["hedonic_adjacent"] = hedonic.rolling_time_dummy(d, window=2)
            pairs = repeat_sales.build_pairs(d)
            rs, info["repeat_sales"] = repeat_sales.case_shiller(pairs, sorted(d["q"].unique()))
            methods["repeat_sales"] = rs
        for name, t in methods.items():
            frames.append(_long(seg, name, t))

    nbp = warsaw_hedonic_index()
    nbp = nbp[(nbp.index >= common.FIRST_Q) & (nbp.index <= common.LAST_Q)]
    frames.append(_long("secondary", "nbp_hedonic", nbp.to_frame("log_index")))
    long = pd.concat(frames, ignore_index=True)

    # real (CPI-deflated) hedonic indices
    hicp = hicp_quarterly()
    real = []
    for seg in ["secondary", "primary_developer"]:
        h = long[(long["segment"] == seg) & (long["method"] == "hedonic")].set_index("q")
        r = (h["log_index"] - np.log(hicp.reindex(h.index))).dropna().to_frame("log_index")
        real.append(_long(seg, "hedonic_real", r))
    return pd.concat([long] + real, ignore_index=True), info


def district_indices(sales: pd.DataFrame) -> pd.DataFrame:
    """Annual hedonic (3-year rolling window, H3 fixed effects) per district, secondary market."""
    d = sales[(sales["segment"] == "secondary") & (sales["year"] <= LAST_FULL_YEAR)].copy()
    d["yr"] = d["year"].astype(int)
    out = {"Warszawa": hedonic.rolling_time_dummy(d, period_col="yr", window=3)["log_index"]}
    for name, g in d.groupby("district"):
        out[name] = hedonic.rolling_time_dummy(g, period_col="yr", window=3)["log_index"]
    idx = pd.DataFrame({k: common.rebase(v) for k, v in out.items()})
    idx.index.name = "year"
    return idx


def diagnostics(long: pd.DataFrame) -> pd.DataFrame:
    wide = long.pivot_table(index="q", columns=["segment", "method"], values="index")
    nbp = np.log(wide[("secondary", "nbp_hedonic")])
    rows = []
    for (seg, method), s in wide.items():
        s = s.dropna()
        ls = np.log(s)
        yearly = s.groupby(s.index.year).mean()   # annual averages - robust to a noisy base quarter
        peak_q = s.idxmax()
        rows.append({
            "segment": seg, "method": method,
            "growth_2007_2015_pct": (yearly[2015] / yearly[2007] - 1) * 100,
            "growth_2015_2025_pct": (yearly[2025] / yearly[2015] - 1) * 100,
            "cagr_2015_2025_pct": ((yearly[2025] / yearly[2015]) ** (1 / 10) - 1) * 100,
            "peak_quarter": str(peak_q), "change_peak_to_last_pct": (s.iloc[-1] / s.max() - 1) * 100,
            "last_quarter": str(s.index[-1]),
            "std_qoq_pp": ls.diff().std() * 100,
            "corr_qoq_vs_nbp": ls.diff().corr(nbp.diff()) if seg == "secondary" else np.nan,
            "corr_yoy_vs_nbp": ls.diff(4).corr(nbp.diff(4)) if seg == "secondary" else np.nan,
        })
    return pd.DataFrame(rows).round(3)


def main() -> None:
    sales = common.load_sales(("secondary", "primary_developer"))
    long, info = quarterly_indices(sales)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    long.assign(q=long["q"].astype(str)).to_parquet(QUARTERLY_PATH, index=False)

    wide = long.pivot_table(index="q", columns=["segment", "method"], values="index").round(2)
    wide.columns = [f"{s}__{m}" for s, m in wide.columns]
    wide.index = wide.index.astype(str)
    REPORTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    wide.to_csv(REPORTS_TABLES_DIR / "price_indices_quarterly.csv")

    district_indices(sales).round(2).to_csv(DISTRICT_PATH)
    diag = diagnostics(long)
    diag.to_csv(DIAG_PATH, index=False)
    print(diag.to_string(index=False))
    print("repeat sales:", info["repeat_sales"])


if __name__ == "__main__":
    main()
