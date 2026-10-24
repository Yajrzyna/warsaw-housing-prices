"""Simple median and mix-adjusted (stratified) price indices."""

from __future__ import annotations

import numpy as np
import pandas as pd

SIZE_BANDS = [0, 35, 45, 60, 80, np.inf]
SIZE_LABELS = ["<35", "35-45", "45-60", "60-80", "80+"]


def median_index(df: pd.DataFrame, period_col: str = "q") -> pd.DataFrame:
    """Median price per m2 per period - no quality or mix adjustment."""
    g = df.groupby(period_col)["price_m2"]
    out = pd.DataFrame({"median_price_m2": g.median(), "n": g.size()})
    out["log_index"] = np.log(out["median_price_m2"])
    return out


def stratified_index(df: pd.DataFrame, period_col: str = "q", weight_years: tuple[int, int] = (2016, 2025),
                     min_cell: int = 3) -> pd.DataFrame:
    """Mix-adjusted index: fixed-weight chained geometric index of stratum medians.

    Strata = district x size band. Weights = each stratum's share of sales in
    `weight_years`. The log change between consecutive periods is the weighted
    mean of stratum log-median changes over strata with >= `min_cell` sales in
    both periods (weights re-normalised). When a stratum is present throughout,
    its chained changes telescope, so the chain equals the direct fixed-weight
    geometric index - drift comes only from missing cells (`coverage` < 1).
    """
    d = df.assign(stratum=df["district"].astype(str) + "|" +
                  pd.cut(df["area_m2"], SIZE_BANDS, labels=SIZE_LABELS, right=False).astype(str))
    g = d.groupby([period_col, "stratum"])["log_price_m2"]
    med = g.median().unstack()
    cnt = g.size().unstack().reindex_like(med).fillna(0)

    in_w = d["year"].between(*weight_years)
    w = d[in_w]["stratum"].value_counts(normalize=True).reindex(med.columns).fillna(0)

    med = med.where(cnt >= min_cell)
    diff = med.diff()
    avail = diff.notna()
    wsum = avail.mul(w, axis=1).sum(axis=1)
    step = diff.fillna(0).mul(w, axis=1).sum(axis=1) / wsum.replace(0, np.nan)
    step.iloc[0] = 0.0
    return pd.DataFrame({"log_index": step.fillna(0).cumsum(), "coverage": wsum.round(3),
                         "n": cnt.sum(axis=1).astype(int)})
