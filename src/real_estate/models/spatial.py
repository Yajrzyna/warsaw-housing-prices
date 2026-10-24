"""Spatial diagnostics.

- Moran's I of regression residuals (k-nearest-neighbour weights, row-standardised,
  permutation p-value): does the model leave location signal unexplained?
- District heterogeneity: spec B re-estimated per district - do attribute premia
  differ across the city (a coarse, transparent alternative to GWR)?
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from real_estate.models import hedonic_ols as ho


def morans_i(xy: np.ndarray, values: np.ndarray, k: int = 10, n_perm: int = 199, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    z = values - values.mean()
    _, idx = cKDTree(xy).query(xy, k=k + 1)
    nb = idx[:, 1:]                                   # drop self
    denom = z @ z

    def stat(v):
        return float(v @ v[nb].mean(axis=1) / denom)

    obs = stat(z)
    perms = np.array([stat(rng.permutation(z)) for _ in range(n_perm)])
    return {"morans_i": obs, "expected": -1 / (len(z) - 1),
            "p_value": (np.sum(perms >= obs) + 1) / (n_perm + 1), "n": len(z), "k": k}


def residual_autocorrelation(df: pd.DataFrame, res_a, parts_b: dict, sample: int = 20_000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for name, resid in [("Spec A (location variables)", pd.Series(res_a.resid, index=df.index)),
                        ("Spec B (H3-cell fixed effects)", parts_b["resid"])]:
        s = resid.sample(min(sample, len(resid)), random_state=seed)
        xy = df.loc[s.index, ["x_2180", "y_2180"]].to_numpy() + rng.normal(0, 0.01, (len(s), 2))
        rows.append({"model": name, **morans_i(xy, s.to_numpy())})
    return pd.DataFrame(rows)


HETERO_TERMS = ["floor_ground", "floor_5_7", "height_high", "new_age_2_4", "primary_developer"]


def district_heterogeneity(df: pd.DataFrame, min_rows: int = 800) -> pd.DataFrame:
    rows = []
    for district, g in df.groupby("district"):
        if len(g) < min_rows:
            continue
        res, prem, _ = ho.fit_spec_b(g)
        prem = prem.set_index("term")
        row = {"district": district, "n": len(g)}
        for t in HETERO_TERMS:
            if t in prem.index:
                row[f"{t}_pct"] = prem.loc[t, "premium_pct"]
                row[f"{t}_lo"], row[f"{t}_hi"] = prem.loc[t, "ci_low_pct"], prem.loc[t, "ci_high_pct"]
        row["area_elasticity"] = res.params.get("log_area", np.nan)
        rows.append(row)
    return pd.DataFrame(rows)
