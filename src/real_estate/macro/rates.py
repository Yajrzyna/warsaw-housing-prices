"""How do Warsaw flat prices depend on interest rates?

A mortgage is a long, large loan, so its payment is very sensitive to the rate:
at ~6% over 25 years a +1 pp rate cuts the loan a given payment can carry by ~9%.
If buyers borrowed to their limit and prices followed borrowing capacity, prices
would swing just as much. This module checks how far that holds.

  rate_frame()          quarterly data + inflation, real mortgage rate, borrowing capacity
  borrowing capacity    max loan = DSTI x wage x annuity factor(rate, 25y); DSTI = 40% of gross wage
  correlations()        levels and growth vs nominal / real rates, several lags
  rolling_correlation() 5-year rolling correlation -> is the link stable across regimes?
  granger()             do rate changes help predict price changes (and vice versa)?
  long_run()            log real price ~ log real wage + rate (Engle-Granger cointegration)
  local_projections()   cumulative response over 0-12 quarters to a +1 pp rate change (Jorda 2005),
                        controlling for inflation, wage growth and own lags; HAC standard errors
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, grangercausalitytests

from real_estate.macro import analysis as ma

DSTI = 0.40          # payment as a share of gross wage used for "borrowing capacity"
FLAT_M2, LTV, MONTHS = 50, 0.8, 300


def annuity_factor(rate_pct) -> np.ndarray:
    """Present value of 1 PLN a month for 25 years at the given annual rate."""
    i = np.asarray(rate_pct, dtype=float) / 100 / 12
    return (1 - (1 + i) ** -MONTHS) / i


def rate_frame() -> pd.DataFrame:
    q = ma.quarterly_frame()
    q["inflation"] = (q["hicp"] / q["hicp"].shift(4) - 1) * 100
    q["real_rate"] = q["mortgage_rate"] - q["inflation"]          # ex-post real mortgage rate
    q["max_loan"] = DSTI * q["wage"] * annuity_factor(q["mortgage_rate"])
    q["capacity_price_m2"] = q["max_loan"] / LTV / FLAT_M2           # price/m2 a 50 m2 flat could cost
    base = q.loc[[p for p in q.index if p.year == 2015]]
    q["capacity_index"] = q["max_loan"] / base["max_loan"].mean() * 100
    q["price_to_capacity"] = q["median_price_m2"] / q["capacity_price_m2"]
    return q


def correlations(q: pd.DataFrame, max_lag: int = 8) -> pd.DataFrame:
    y_nom = np.log(q["hedonic"]).diff(4) * 100
    y_real = np.log(q["hedonic_real"]).diff(4) * 100
    drivers = {
        "mortgage rate (level)": q["mortgage_rate"],
        "mortgage rate change (y/y)": q["mortgage_rate"].diff(4),
        "real mortgage rate (level)": q["real_rate"],
        "borrowing capacity growth (y/y)": np.log(q["capacity_index"]).diff(4) * 100,
    }
    rows = []
    for target, y in [("nominal price growth", y_nom), ("real price growth", y_real)]:
        for name, x in drivers.items():
            for k in range(max_lag + 1):
                d = pd.concat([y, x.shift(k)], axis=1).dropna()
                rows.append({"target": target, "driver": name, "lag": k, "corr": d.corr().iloc[0, 1], "n": len(d)})
    return pd.DataFrame(rows)


def rolling_correlation(q: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    y = np.log(q["hedonic"]).diff(4) * 100
    return pd.DataFrame({
        "nominal rate change, 2q earlier": y.rolling(window).corr(q["mortgage_rate"].diff(4).shift(2)),
        "real rate, 3q earlier": y.rolling(window).corr(q["real_rate"].shift(3)),
    })


def granger(q: pd.DataFrame, lags=(2, 4)) -> pd.DataFrame:
    d = pd.DataFrame({"price": np.log(q["hedonic"]).diff() * 100, "nominal rate": q["mortgage_rate"].diff(),
                      "real rate": q["real_rate"].diff()}).dropna()
    rows = []
    for lag in lags:
        for r in ["nominal rate", "real rate"]:
            fwd = grangercausalitytests(d[["price", r]], maxlag=[lag])[lag][0]["ssr_ftest"]
            back = grangercausalitytests(d[[r, "price"]], maxlag=[lag])[lag][0]["ssr_ftest"]
            rows.append({"rate": r, "lags": lag, "rate_to_price_F": fwd[0], "rate_to_price_p": fwd[1],
                         "price_to_rate_F": back[0], "price_to_rate_p": back[1]})
    return pd.DataFrame(rows)


def long_run(q: pd.DataFrame) -> pd.DataFrame:
    d = pd.DataFrame({"log_real_price": np.log(q["hedonic_real"]), "log_real_wage": np.log(q["real_wage"]),
                      "mortgage_rate": q["mortgage_rate"], "real_rate": q["real_rate"]}).dropna()
    rows = []
    for rate in ["mortgage_rate", "real_rate"]:
        X = sm.add_constant(d[["log_real_wage", rate]])
        res = sm.OLS(d["log_real_price"], X).fit()
        rows.append({"rate": rate, "wage_elasticity": res.params["log_real_wage"],
                     "rate_semi_elasticity_pct_per_pp": res.params[rate] * 100, "r2": res.rsquared,
                     "engle_granger_p": coint(d["log_real_price"], d[["log_real_wage", rate]])[1], "n": len(d)})
    return pd.DataFrame(rows)


def local_projections(y: pd.Series, shock: pd.Series, controls: list[pd.Series], horizons: int = 12,
                      n_lags: int = 2) -> pd.DataFrame:
    """beta_h from  (y_{t+h} - y_{t-1}) x100 = a + beta_h * shock_t + lags + controls_{t-1} + e."""
    rows = []
    for h in range(horizons + 1):
        dep = (y.shift(-h) - y.shift(1)) * 100
        X = pd.DataFrame({"shock": shock})
        for lag in range(1, n_lags + 1):
            X[f"dy_l{lag}"] = (y.diff() * 100).shift(lag)
            X[f"shock_l{lag}"] = shock.shift(lag)
        for c in controls:
            X[c.name] = c.shift(1)
        D = pd.concat([dep.rename("dep"), X], axis=1).dropna()
        res = sm.OLS(D["dep"], sm.add_constant(D.drop(columns="dep"))).fit(cov_type="HAC", cov_kwds={"maxlags": h + 1})
        rows.append({"h": h, "beta": res.params["shock"], "se": res.bse["shock"], "n": int(res.nobs)})
    out = pd.DataFrame(rows)
    out["lo90"], out["hi90"] = out["beta"] - 1.645 * out["se"], out["beta"] + 1.645 * out["se"]
    return out


def impulse_responses(q: pd.DataFrame) -> pd.DataFrame:
    controls = [q["inflation"].rename("inflation"), (np.log(q["wage"]).diff(4) * 100).rename("wage_growth")]
    outcomes = {"Price (nominal)": np.log(q["hedonic"]), "Price (real)": np.log(q["hedonic_real"]),
                "Resale volume": np.log(q["sales_secondary"])}
    shocks = {"nominal mortgage rate": q["mortgage_rate"].diff(), "real mortgage rate": q["real_rate"].diff()}
    frames = []
    for oname, y in outcomes.items():
        for sname, s in shocks.items():
            frames.append(local_projections(y, s, controls).assign(outcome=oname, shock=sname))
    return pd.concat(frames, ignore_index=True)
