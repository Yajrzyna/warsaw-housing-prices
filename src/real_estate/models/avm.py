"""Automated valuation model (AVM): gradient boosting vs hedonic baselines.

Target: y_adj = log(price/m2) - log(segment hedonic index in the sale's quarter),
i.e. the price level net of market movement. Predictions are converted back
with the index *published one quarter earlier* (q-1), as a production AVM would
- so the test never sees the index of the quarter it predicts.

Validation
  temporal  train 2016Q1-2024Q4, test 2025Q1-2026Q1 (future sales)
  spatial   2023Q1-2026Q1, 5-fold GroupKFold on H3 resolution-6 parents
            (~36 km2 blocks): the model never sees the area it predicts, and the
            neighbourhood-price feature is withheld (no local sales would exist)

Models
  hedonic_A    OLS, unit + location variables (spec A without time effects)
  hedonic_B    OLS + H3-cell fixed effects (temporal only - unseen cells can't be priced)
  gbm          HistGradientBoosting, unit + location + coordinates
  gbm_nbhd     gbm + leakage-safe neighbourhood price level (past 12 months, k=20)
"""

from __future__ import annotations

import h3
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold

from real_estate.indices.hedonic import _demean
from real_estate.models import hedonic_ols as ho

NUM = ["area_m2", "rooms", "floor", "storeys", "floor_rel", "footprint_m2", "age_at_sale",
       "dist_centre_km", "dist_metro_km", "dist_park_km", "dist_forest_km", "dist_vistula_km", "dist_rail_km",
       "n_stops_500m", "x_2180", "y_2180"]
FLAGS = ["is_top_floor", "has_ancillary_area", "right_bank"]
CAT = ["segment", "district", "height_class", "building_function", "vintage"]
NBHD = ["nbhd_rel", "nbhd_n", "nbhd_median_dist_m"]
TEST_START = pd.Period("2025Q1", "Q")


def gbm_frame(df: pd.DataFrame, with_nbhd: bool) -> pd.DataFrame:
    X = df[NUM].astype(float).copy()
    for c in FLAGS:
        X[c] = df[c].astype(object).eq(True).astype(float)
    for c in CAT:
        X[c] = df[c].astype(str).astype("category")
    if with_nbhd:
        X[NBHD] = df[NBHD].astype(float)
    return X


def add_nbhd_rel(df: pd.DataFrame) -> pd.DataFrame:
    """Neighbourhood log price/m2 net of the market level over the 4 quarters it covers."""
    sec = df[df["segment"] == "secondary"].groupby("q")["log_index"].first().sort_index()
    lag_mean = sec.shift(1).rolling(4, min_periods=1).mean()
    df = df.copy()
    df["nbhd_rel"] = df["nbhd_log_price_m2"] - df["q"].map(lag_mean).astype(float)
    return df


def _gbm(seed: int = 0) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=800, learning_rate=0.06, max_leaf_nodes=63, min_samples_leaf=40, l2_regularization=1.0,
        categorical_features="from_dtype", early_stopping=True, validation_fraction=0.1, n_iter_no_change=30,
        random_state=seed)


def _ols_a(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    def mat(d):
        return pd.concat([ho.unit_matrix(d), ho.location_matrix(d)], axis=1).assign(const=1.0)
    Xtr, Xte = mat(train), mat(test)
    beta, *_ = np.linalg.lstsq(Xtr.to_numpy(), train["y_adj"].to_numpy(), rcond=None)
    return Xte.to_numpy() @ beta


def _ols_b(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """OLS with H3-cell FE; test rows in cells unseen in training fall back to spec A."""
    codes, cells = pd.factorize(train["h3_r8"])
    U = ho.unit_matrix(train)
    Ud = _demean(U.to_numpy(), codes, len(cells))
    keep = Ud.std(axis=0) > 1e-10
    yd = _demean(train[["y_adj"]].to_numpy(), codes, len(cells)).ravel()
    beta, *_ = np.linalg.lstsq(Ud[:, keep], yd, rcond=None)
    fe = pd.Series(train["y_adj"].to_numpy() - U.to_numpy()[:, keep] @ beta).groupby(codes).mean()
    fe.index = cells
    pred = ho.unit_matrix(test).to_numpy()[:, keep] @ beta + test["h3_r8"].map(fe).to_numpy()
    fallback = np.isnan(pred)
    if fallback.any():
        pred[fallback] = _ols_a(train, test[fallback])
    return pred


def metrics(y_true_log: np.ndarray, y_pred_log: np.ndarray, price_m2: np.ndarray) -> dict:
    ratio = np.exp(y_pred_log - y_true_log)
    ape = np.abs(ratio - 1)
    return {"n_test": len(ape), "median_ape_pct": np.median(ape) * 100, "mean_ape_pct": ape.mean() * 100,
            "mae_pln_m2": np.mean(np.abs(price_m2 * ratio - price_m2)),
            "within_10_pct": np.mean(ape <= 0.10) * 100, "within_20_pct": np.mean(ape <= 0.20) * 100,
            "median_bias_pct": (np.median(ratio) - 1) * 100}


def temporal_validation(df: pd.DataFrame, seed: int = 0):
    df = add_nbhd_rel(df)
    train, test = df[df["q"] < TEST_START], df[df["q"] >= TEST_START].copy()
    # index known at prediction time = previous quarter's value
    idx = df.groupby(["segment", "q"])["log_index"].first()
    prev = pd.Series([idx.get((s, q - 1), np.nan) for s, q in zip(test["segment"], test["q"])], index=test.index)
    offset = prev.to_numpy()

    preds = {"hedonic_A": _ols_a(train, test), "hedonic_B": _ols_b(train, test)}
    models = {}
    for name, nb in [("gbm", False), ("gbm_nbhd", True)]:
        m = _gbm(seed).fit(gbm_frame(train, nb), train["y_adj"])
        preds[name] = m.predict(gbm_frame(test, nb))
        models[name] = m

    rows, per_row = [], {}
    for name, p in preds.items():
        pred_log = p + offset
        rows.append({"scheme": "temporal (test 2025Q1-2026Q1)", "model": name,
                     **metrics(test["log_price_m2"].to_numpy(), pred_log, test["price_m2"].to_numpy())})
        per_row[name] = pred_log
    out = pd.DataFrame(rows)
    test = test.assign(**{f"pred_{k}": v for k, v in per_row.items()})
    return out, models, test


def spatial_validation(df: pd.DataFrame, first_q: pd.Period = pd.Period("2023Q1", "Q"), n_folds: int = 5,
                       seed: int = 0) -> pd.DataFrame:
    d = df[df["q"] >= first_q].reset_index(drop=True)
    groups = d["h3_r8"].map(lambda c: h3.cell_to_parent(c, 6))
    preds = {"hedonic_A": np.zeros(len(d)), "gbm": np.zeros(len(d))}
    for tr, te in GroupKFold(n_splits=n_folds).split(d, groups=groups):
        train, test = d.iloc[tr], d.iloc[te]
        preds["hedonic_A"][te] = _ols_a(train, test)
        m = _gbm(seed).fit(gbm_frame(train, False), train["y_adj"])
        preds["gbm"][te] = m.predict(gbm_frame(test, False))
    rows = []
    for name, p in preds.items():
        rows.append({"scheme": f"spatial ({n_folds}-fold, unseen ~36 km2 areas)", "model": name,
                     **metrics(d["log_price_m2"].to_numpy(), p + d["log_index"].to_numpy(), d["price_m2"].to_numpy())})
    return pd.DataFrame(rows)


def importance(model, test: pd.DataFrame, with_nbhd: bool, n: int = 8000, seed: int = 0) -> pd.DataFrame:
    s = test.sample(min(n, len(test)), random_state=seed)
    X = gbm_frame(s, with_nbhd)
    r = permutation_importance(model, X, s["y_adj"], n_repeats=3, random_state=seed,
                               scoring="neg_mean_absolute_error")
    return (pd.DataFrame({"feature": X.columns, "mae_increase": r.importances_mean, "sd": r.importances_std})
            .sort_values("mae_increase", ascending=False).reset_index(drop=True))
