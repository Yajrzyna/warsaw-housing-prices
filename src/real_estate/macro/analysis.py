"""Phase 7: macro drivers and affordability of Warsaw flats.

Datasets
  quarterly_frame()  2006Q2-2026Q1: price indices, median price/m2, sales volumes, rates, HICP,
                     wage (annual Warsaw series interpolated log-linearly), affordability
  annual_frame()     2006-2025: the same aggregated to years + housing supply and population

Affordability (typical flat = 50 m2 at the quarter's median resale price per m2)
  years_of_wages     price / (12 x average gross monthly wage in Warsaw)
  m2_per_wage        m2 bought with one monthly gross wage
  payment_share      monthly annuity on an 80% LTV, 25-year mortgage at the proxy
                     mortgage rate, as a share of one gross monthly wage

Drivers (kept deliberately modest: ~80 quarters, several regime changes)
  lead_lag()         correlation of y/y real price growth with lagged y/y changes in
                     mortgage rate and real wage growth
  driver_regression() OLS with Newey-West (HAC, 4 lags) standard errors
Events
  size_band_indices() hedonic indices for small / mid / large flats around
                     "Bezpieczny Kredyt 2%" (applications July 2023 - early 2024; loan caps
                     PLN 500k single / 600k couple -> demand tilted to cheaper, smaller flats)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from real_estate.features.build_features import FEATURES_PATH
from real_estate.indices import common, hedonic
from real_estate.indices.build import QUARTERLY_PATH
from real_estate.ingest import macro
from real_estate.ingest.eurostat import hicp_quarterly

FLAT_M2, LTV, YEARS = 50, 0.8, 25
EVENTS = {
    "COVID-19": ("2020Q2", "2020Q2"),
    "Rate hikes": ("2021Q4", "2022Q3"),
    "Bezpieczny Kredyt 2%": ("2023Q3", "2024Q1"),
}


def annuity(principal, annual_rate_pct, years=YEARS):
    r = np.asarray(annual_rate_pct) / 100 / 12
    n = years * 12
    return principal * r / (1 - (1 + r) ** -n)


def _quarterly_wage(annual: pd.Series, quarters: pd.PeriodIndex) -> pd.Series:
    """Log-linear interpolation of annual averages placed mid-year; extrapolated beyond both ends
    with the nearest observed annual growth."""
    a = annual.dropna()
    mid = np.array([y + 0.5 for y in a.index])
    t = np.array([q.year + (q.quarter - 0.5) / 4 for q in quarters])
    lw = np.interp(t, mid, np.log(a.to_numpy()))
    g_last, g_first = np.log(a.iloc[-1] / a.iloc[-2]), np.log(a.iloc[1] / a.iloc[0])
    lw = np.where(t > mid[-1], np.log(a.iloc[-1]) + g_last * (t - mid[-1]), lw)
    lw = np.where(t < mid[0], np.log(a.iloc[0]) + g_first * (t - mid[0]), lw)
    return pd.Series(np.exp(lw), index=quarters, name="wage")


def quarterly_frame() -> pd.DataFrame:
    idx = pd.read_parquet(QUARTERLY_PATH)
    idx["q"] = pd.PeriodIndex(idx["q"], freq="Q")
    w = idx.pivot_table(index="q", columns=["segment", "method"], values="index")
    q = pd.DataFrame({
        "hedonic": w[("secondary", "hedonic")], "hedonic_real": w[("secondary", "hedonic_real")],
        "hedonic_developer": w[("primary_developer", "hedonic")],
    })
    f = pd.read_parquet(FEATURES_PATH, columns=["quarter", "segment", "price_m2"])
    f["q"] = pd.PeriodIndex(f["quarter"], freq="Q")
    q["median_price_m2"] = f[f["segment"] == "secondary"].groupby("q")["price_m2"].median()
    vol = f.groupby(["q", "segment"]).size().unstack()
    q["sales_secondary"], q["sales_developer"] = vol["secondary"], vol["primary_developer"]
    q = q.loc[common.FIRST_Q:common.LAST_Q].copy()

    ref = macro.nbp_reference_rate()
    q["reference_rate"] = ref.groupby(ref.index.to_period("Q")).mean().reindex(q.index)
    m3 = macro.interbank_3m()
    q["rate_3m"] = m3.groupby(m3.index.asfreq("Q")).mean().reindex(q.index)
    q["mortgage_rate"] = q["rate_3m"] + macro.MORTGAGE_MARGIN_PP
    q["hicp"] = hicp_quarterly().reindex(q.index)
    q["wage"] = _quarterly_wage(macro.bdl_warsaw()["wage_gross_monthly"], q.index)
    q["real_wage"] = q["wage"] / q["hicp"] * 100
    return add_affordability(q)


def add_affordability(df: pd.DataFrame) -> pd.DataFrame:
    price = df["median_price_m2"] * FLAT_M2
    df["flat_price"] = price
    df["years_of_wages"] = price / (12 * df["wage"])
    df["m2_per_wage"] = df["wage"] / df["median_price_m2"]
    df["payment"] = annuity(price * LTV, df["mortgage_rate"])
    df["payment_share"] = df["payment"] / df["wage"] * 100
    return df


def annual_frame(q: pd.DataFrame) -> pd.DataFrame:
    yrs = q.index.year
    full = q.groupby(yrs).filter(lambda g: len(g) == 4)          # complete years only
    a = full.groupby(full.index.year).agg({
        "hedonic": "mean", "hedonic_real": "mean", "median_price_m2": "median",
        "sales_secondary": "sum", "sales_developer": "sum", "reference_rate": "mean",
        "mortgage_rate": "mean", "hicp": "mean"})
    bdl = macro.bdl_warsaw()
    a = a.join(bdl, how="left")
    a = a.rename(columns={"wage_gross_monthly": "wage"})
    a["real_wage"] = a["wage"] / a["hicp"] * 100
    a["completions_per_1000"] = a["dwellings_completed"] / a["population"] * 1000
    a = add_affordability(a)
    a.index.name = "year"
    return a


def district_affordability(q: pd.DataFrame, year: int = 2025) -> pd.DataFrame:
    f = pd.read_parquet(FEATURES_PATH, columns=["year", "segment", "district", "price_m2"])
    med = f[(f["segment"] == "secondary") & (f["year"] == year)].groupby("district")["price_m2"].median()
    wage = q.loc[[p for p in q.index if p.year == year], "wage"].mean()
    rate = q.loc[[p for p in q.index if p.year == year], "mortgage_rate"].mean()
    out = pd.DataFrame({"median_price_m2": med})
    out["flat_price"] = out["median_price_m2"] * FLAT_M2
    out["years_of_wages"] = out["flat_price"] / (12 * wage)
    out["payment_share"] = annuity(out["flat_price"] * LTV, rate) / wage * 100
    return out.sort_values("years_of_wages")


def lead_lag(q: pd.DataFrame, max_lag: int = 8) -> pd.DataFrame:
    """Nominal terms on purpose: deflating prices and wages by the same CPI would make an
    inflation shock move both "real" series together and manufacture a correlation."""
    y = np.log(q["hedonic"]).diff(4) * 100
    drivers = {
        "mortgage_rate_change_pp": q["mortgage_rate"].diff(4),
        "wage_growth_pct": np.log(q["wage"]).diff(4) * 100,
        "sales_growth_pct": np.log(q["sales_secondary"]).diff(4) * 100,
    }
    rows = []
    for name, x in drivers.items():
        for k in range(0, max_lag + 1):
            d = pd.concat([y, x.shift(k)], axis=1).dropna()
            rows.append({"driver": name, "lag_quarters": k, "corr": d.corr().iloc[0, 1], "n": len(d)})
    return pd.DataFrame(rows)


def driver_regression(q: pd.DataFrame, rate_lag: int, wage_lag: int = 0, sales_lag: int | None = None):
    """y/y nominal price growth on lagged y/y mortgage-rate change, wage growth, optionally lagged
    sales growth, and a programme dummy. Newey-West standard errors (4 lags)."""
    cols = {
        "price_growth": np.log(q["hedonic"]).diff(4) * 100,
        f"mortgage_rate_change_lag{rate_lag}": q["mortgage_rate"].diff(4).shift(rate_lag),
        f"wage_growth_lag{wage_lag}": (np.log(q["wage"]).diff(4) * 100).shift(wage_lag),
        "bk2_programme": [pd.Period("2023Q3", "Q") <= p <= pd.Period("2024Q2", "Q") for p in q.index],
    }
    if sales_lag is not None:
        cols[f"sales_growth_lag{sales_lag}"] = (np.log(q["sales_secondary"]).diff(4) * 100).shift(sales_lag)
    d = pd.DataFrame(cols, index=q.index).astype(float).dropna()
    X = sm.add_constant(d.drop(columns="price_growth"))
    res = sm.OLS(d["price_growth"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    return res, d


SIZE_BANDS = {"small (< 45 m²)": (0, 45), "mid (45–65 m²)": (45, 65), "large (≥ 65 m²)": (65, 1e9)}


def size_band_indices(first_q: str = "2019Q1") -> pd.DataFrame:
    """Resale hedonic index per size band (rolling 5-quarter time dummy, H3 fixed effects)."""
    s = common.load_sales(("secondary",))
    s = s[s["q"] >= pd.Period(first_q, "Q")]
    out = {}
    for name, (lo, hi) in SIZE_BANDS.items():
        band = s[(s["area_m2"] >= lo) & (s["area_m2"] < hi)]
        out[name] = hedonic.rolling_time_dummy(band, window=5)["log_index"]
    df = pd.DataFrame(out)
    return 100 * np.exp(df - df.loc[pd.Period("2023Q2", "Q")])    # 2023Q2 = 100 (last pre-programme quarter)


def bk2_did(bands: pd.DataFrame) -> pd.DataFrame:
    """Growth of each size band before and during the programme, and the small-minus-large gap."""
    g = lambda a, b: (bands.loc[pd.Period(b, "Q")] / bands.loc[pd.Period(a, "Q")] - 1) * 100
    out = pd.DataFrame({"pre: 2022Q2→2023Q2": g("2022Q2", "2023Q2"),
                        "programme: 2023Q2→2024Q2": g("2023Q2", "2024Q2"),
                        "after: 2024Q2→2025Q2": g("2024Q2", "2025Q2")})
    out.loc["small − large (pp)"] = out.iloc[0] - out.iloc[2]
    return out.round(2)
