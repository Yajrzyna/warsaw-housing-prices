"""Repeat-sales price index (Case-Shiller weighted).

Pairs = consecutive sales of the same flat. The official unit id is missing
before 2016 and changes when buildings are renumbered, so the flat key is
street + house number + usable area + floor + rooms. Rarely it pools two
identical flats (same building, floor, area, rooms) - near-perfect quality
matches, so this adds noise but no bias.

Estimation (Case & Shiller 1987):
  1. OLS of log(P2/P1) on period dummies (+1 at second sale, -1 at first);
     pairs with robust |z| of the residual > `trim_z` are dropped (renovations,
     non-market prices) and stage 1 is re-run.
  2. Squared residuals regressed on the holding period -> error variance
     grows with time between sales.
  3. WLS with weights 1 / fitted variance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def flat_key(df: pd.DataFrame) -> pd.Series:
    return (df["street"].str.lower().str.strip() + "|" +
            df["house_no"].astype(str).str.lower().str.replace(" ", "") + "|" +
            df["area_m2"].round(2).astype(str) + "|" +
            df["floor"].astype("Int64").astype(str) + "|" +
            df["rooms"].astype("Int64").astype(str))


def build_pairs(df: pd.DataFrame, first_segments=("secondary",), second_segments=("secondary",),
                min_gap_days: int = 180, period_col: str = "q") -> pd.DataFrame:
    """Consecutive sales of the same flat, at least `min_gap_days` apart."""
    d = df.assign(key=flat_key(df)).dropna(subset=["key"])
    d = d[d["rooms"].notna() & d["floor"].notna()].sort_values(["key", "date"])
    prev = d.groupby("key")[["date", "price", "segment", period_col]].shift()
    p = pd.DataFrame({
        "key": d["key"], "p1": prev[period_col], "p2": d[period_col],
        "date1": prev["date"], "date2": d["date"], "seg1": prev["segment"], "seg2": d["segment"],
        "lr": np.log(d["price"] / prev["price"]),
    }).dropna(subset=["date1"])
    p["gap_years"] = (p["date2"] - p["date1"]).dt.days / 365.25
    p = p[(p["gap_years"] * 365.25 >= min_gap_days) & p["seg1"].isin(first_segments) & p["seg2"].isin(second_segments)]
    return p.reset_index(drop=True)


def _design(pairs: pd.DataFrame, periods: list) -> np.ndarray:
    pos = {p: i for i, p in enumerate(periods)}
    D = np.zeros((len(pairs), len(periods)))
    D[np.arange(len(pairs)), pairs["p2"].map(pos).to_numpy()] += 1
    D[np.arange(len(pairs)), pairs["p1"].map(pos).to_numpy()] -= 1
    return D[:, 1:]  # first period = base (log index 0)


def case_shiller(pairs: pd.DataFrame, periods: list, trim_z: float = 4.0) -> tuple[pd.DataFrame, dict]:
    y = pairs["lr"].to_numpy()
    D = _design(pairs, periods)

    b1, *_ = np.linalg.lstsq(D, y, rcond=None)
    e = y - D @ b1
    z = (e - np.median(e)) / (1.4826 * np.median(np.abs(e - np.median(e))))
    keep = np.abs(z) <= trim_z
    y, D, gap = y[keep], D[keep], pairs["gap_years"].to_numpy()[keep]
    b1, *_ = np.linalg.lstsq(D, y, rcond=None)
    e = y - D @ b1

    A = np.column_stack([np.ones_like(gap), gap])
    c, *_ = np.linalg.lstsq(A, e ** 2, rcond=None)
    var = np.clip(A @ c, np.percentile(e ** 2, 5) + 1e-6, None)
    w = 1 / var

    sw = np.sqrt(w)
    b, *_ = np.linalg.lstsq(D * sw[:, None], y * sw, rcond=None)
    r = (y - D @ b) * sw
    sigma2 = r @ r / max(len(y) - D.shape[1], 1)
    cov = sigma2 * np.linalg.pinv((D * w[:, None]).T @ D)

    kept = pairs[keep]
    n_in = pd.concat([kept["p1"], kept["p2"]]).value_counts()
    out = pd.DataFrame({
        "log_index": np.concatenate([[0.0], b]),
        "se": np.concatenate([[0.0], np.sqrt(np.clip(np.diag(cov), 0, None))]),
        "n_pairs": [int(n_in.get(p, 0)) for p in periods],
    }, index=pd.Index(periods, name="q"))
    info = {"pairs_total": len(pairs), "pairs_used": int(keep.sum()), "trimmed": int((~keep).sum()),
            "var_intercept": c[0], "var_slope_per_year": c[1]}
    return out, info
