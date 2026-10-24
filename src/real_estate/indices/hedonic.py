"""Hedonic price index: rolling time-dummy regression with location fixed effects.

For each window of `window` consecutive periods ending at t:

    log(price/m2)_i = sum_s delta_s * D_s(i) + X_i' beta + alpha_{cell(i)} + e_i

with D_s time dummies (first period of the window = base), X unit/building
controls and alpha an H3-cell fixed effect (absorbed by demeaning within cell).
The index change from t-1 to t is taken from the window ending at t
(delta_t - delta_{t-1}, "movement splice") and chained - the rolling time
dummy (RTD) method used by statistical offices. Coefficients may drift over
time, but a published period is never revised. The first window supplies the
first `window` levels directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

CONTROL_NAMES = [
    "log_area", "log_area_sq", "rooms_1", "rooms_3", "rooms_4", "rooms_5p", "rooms_na",
    "floor_ground", "floor_5_7", "floor_8_10", "floor_11_15", "floor_16p", "floor_na", "top_floor",
    "height_mid", "height_high", "height_na",
    "new_age_0_1", "new_age_2_4", "new_age_5_9", "new_age_10p", "ancillary", "log_dist_metro",
]


def controls(df: pd.DataFrame) -> pd.DataFrame:
    """Hedonic controls. Base categories: 2 rooms, floors 2-4, low-rise, older/unknown vintage."""
    la = np.log(df["area_m2"])
    rooms = df["rooms"].clip(upper=5)
    fl = df["floor"]
    hc = df["height_class"].astype(object)
    new = df["vintage"].eq("new_2007plus")
    age = df["age_at_sale"]
    metro = df["dist_metro_km"].fillna(df["dist_metro_km"].median())
    X = {
        "log_area": la, "log_area_sq": (la - np.log(50)) ** 2,
        "rooms_1": rooms.eq(1), "rooms_3": rooms.eq(3), "rooms_4": rooms.eq(4), "rooms_5p": rooms.eq(5),
        "rooms_na": rooms.isna(),
        "floor_ground": fl.eq(1), "floor_5_7": fl.between(5, 7), "floor_8_10": fl.between(8, 10),
        "floor_11_15": fl.between(11, 15), "floor_16p": fl.ge(16), "floor_na": fl.isna(),
        "top_floor": df["is_top_floor"].astype(object).eq(True),
        "height_mid": hc.eq("mid_5-10"), "height_high": hc.eq("high_11+"), "height_na": df["height_class"].isna(),
        "new_age_0_1": new & age.le(1), "new_age_2_4": new & age.between(2, 4),
        "new_age_5_9": new & age.between(5, 9), "new_age_10p": new & age.ge(10),
        "ancillary": df["has_ancillary_area"].astype(object).eq(True),
        "log_dist_metro": np.log(metro + 0.2),
    }
    return pd.DataFrame(X, index=df.index)[CONTROL_NAMES].astype(float)


def _demean(A: np.ndarray, codes: np.ndarray, n_groups: int) -> np.ndarray:
    """Subtract group means (one-way fixed-effect within transformation)."""
    G = sp.csr_matrix((np.ones(len(codes)), (np.arange(len(codes)), codes)), shape=(len(codes), n_groups))
    counts = np.asarray(G.sum(axis=0)).ravel()
    means = (G.T @ A) / counts[:, None]
    return A - G @ means


def _window_regression(y: np.ndarray, T: np.ndarray, X: np.ndarray, groups: np.ndarray):
    """Within-cell OLS of y on [T, X]. Returns time coefficients and their covariance."""
    codes, uniq = pd.factorize(groups)
    sizes = np.bincount(codes)
    keep = sizes[codes] > 1                      # singletons carry no within-cell information
    codes = pd.factorize(codes[keep])[0]
    y, T, X = y[keep], T[keep], X[keep]
    n_groups = codes.max() + 1

    Z = _demean(np.hstack([T, X]), codes, n_groups)
    yd = _demean(y[:, None], codes, n_groups).ravel()
    k_t = T.shape[1]
    var_ok = Z.std(axis=0) > 1e-10
    var_ok[:k_t] = True                          # time dummies are always kept (present by construction)
    Z = Z[:, var_ok]

    beta, *_ = np.linalg.lstsq(Z, yd, rcond=None)
    resid = yd - Z @ beta
    dof = max(len(yd) - Z.shape[1] - n_groups, 1)
    sigma2 = resid @ resid / dof
    cov = sigma2 * np.linalg.pinv(Z.T @ Z)
    r2_within = 1 - (resid @ resid) / (yd @ yd)
    return beta[:k_t], cov[:k_t, :k_t], len(yd), r2_within


AGE_CONTROLS = ("new_age_0_1", "new_age_2_4", "new_age_5_9", "new_age_10p")


def rolling_time_dummy(df: pd.DataFrame, period_col: str = "q", window: int = 5,
                       fe_col: str = "h3_r8", y_col: str = "log_price_m2",
                       exclude_controls: tuple[str, ...] = ()) -> pd.DataFrame:
    """Chained rolling-window time-dummy index.

    With a building fixed effect (fe_col="building_key") pass
    exclude_controls=AGE_CONTROLS: within a building age moves one-for-one with
    time, so age and time effects are not separately identified.
    """
    df = df.dropna(subset=[fe_col, y_col])
    periods = sorted(df[period_col].unique())
    X_all = controls(df).drop(columns=list(exclude_controls)).to_numpy()
    y_all = df[y_col].to_numpy()
    per_all = df[period_col].to_numpy()
    fe_all = df[fe_col].to_numpy()

    log_idx, step_se, n_obs, r2 = {}, {}, {}, {}
    window = min(window, len(periods))
    for end in range(window - 1, len(periods)):
        win = periods[end - window + 1:end + 1]
        m = np.isin(per_all, win)
        T = np.column_stack([(per_all[m] == p).astype(float) for p in win[1:]])
        b, cov, n, r2w = _window_regression(y_all[m], T, X_all[m], fe_all[m])
        delta = np.concatenate([[0.0], b])
        if end == window - 1:                    # first window: take all levels
            for p, dlt in zip(win, delta):
                log_idx[p] = dlt
                step_se[p] = np.nan
        else:
            log_idx[win[-1]] = log_idx[win[-2]] + delta[-1] - delta[-2]
            if window > 2:                        # var(d_t - d_{t-1})
                v = cov[-1, -1] + cov[-2, -2] - 2 * cov[-1, -2]
            else:
                v = cov[-1, -1]
            step_se[win[-1]] = np.sqrt(max(v, 0))
        n_obs[win[-1]] = n
        r2[win[-1]] = r2w

    out = pd.DataFrame({"log_index": pd.Series(log_idx), "step_se": pd.Series(step_se),
                        "n_window": pd.Series(n_obs), "r2_within": pd.Series(r2)})
    out.index.name = period_col
    return out.sort_index()


def pooled_coefficients(df: pd.DataFrame, period_col: str = "q", fe_col: str = "h3_r8",
                        y_col: str = "log_price_m2") -> pd.Series:
    """Full-sample hedonic coefficients (time dummies + cell FE) - for interpretation only."""
    df = df.dropna(subset=[fe_col, y_col])
    periods = sorted(df[period_col].unique())
    per = df[period_col].to_numpy()
    T = np.column_stack([(per == p).astype(float) for p in periods[1:]])
    X = controls(df).to_numpy()
    codes, _ = pd.factorize(df[fe_col])
    sizes = np.bincount(codes)
    keep = sizes[codes] > 1
    codes = pd.factorize(codes[keep])[0]
    Z = _demean(np.hstack([T, X])[keep], codes, codes.max() + 1)
    yd = _demean(df[y_col].to_numpy()[keep][:, None], codes, codes.max() + 1).ravel()
    beta, *_ = np.linalg.lstsq(Z, yd, rcond=None)
    return pd.Series(beta[T.shape[1]:], index=CONTROL_NAMES)
