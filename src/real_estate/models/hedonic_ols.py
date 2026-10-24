"""Cross-sectional hedonic regressions (recent market, 2023Q1-2026Q1).

Spec A - "what location is worth": unit/building controls + explicit location
         variables (distance bands to centre / metro, park, Vistula, stops, bank of
         the river) + quarter effects. Location effects are interpretable
         but confounded with each other and with unobserved neighbourhood quality.
Spec B - "what the flat is worth": the same unit/building controls + an H3-cell
         fixed effect that absorbs all location. Its cell effects form a location-
         value surface, and its coefficients drive the dashboard's price calculator.

Standard errors are clustered by building. Premia are reported as
exp(b) - 1 (percent change in price per m2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from real_estate.indices.hedonic import _demean

WINDOW = (pd.Period("2023Q1", "Q"), pd.Period("2026Q1", "Q"))

UNIT_TERMS = {
    "log_area": "log(area) [elasticity]", "log_area_sq": "log(area) squared",
    "rooms_1": "1 room (vs 2)", "rooms_3": "3 rooms (vs 2)", "rooms_4": "4 rooms (vs 2)", "rooms_5p": "5+ rooms (vs 2)",
    "floor_ground": "Ground floor (vs 2-4)", "floor_5_7": "Floor 5-7 (vs 2-4)", "floor_8_10": "Floor 8-10 (vs 2-4)",
    "floor_11_15": "Floor 11-15 (vs 2-4)", "floor_16p": "Floor 16+ (vs 2-4)", "top_floor": "Top floor",
    "height_mid": "Building 5-10 storeys (vs 1-4)", "height_high": "Building 11+ storeys (vs 1-4)",
    "new_age_0_1": "New building, 0-1 yrs old (vs older)", "new_age_2_4": "New building, 2-4 yrs",
    "new_age_5_9": "New building, 5-9 yrs", "new_age_10p": "New building, 10+ yrs",
    "ancillary": "Has cellar / storage room", "primary_developer": "Developer sale (vs resale)",
}
LOCATION_TERMS = {
    "centre_0_2": "Centre 0-2 km (vs 6-8)", "centre_2_4": "Centre 2-4 km", "centre_4_6": "Centre 4-6 km",
    "centre_8_10": "Centre 8-10 km", "centre_10_12": "Centre 10-12 km", "centre_12p": "Centre 12+ km",
    "metro_0_05": "Metro < 500 m (vs > 2 km)", "metro_05_1": "Metro 500 m-1 km", "metro_1_2": "Metro 1-2 km",
    "park_300": "Park within 300 m", "forest_300": "Forest within 300 m", "vistula_1km": "Vistula within 1 km",
    "stops_per_10": "Per 10 bus/tram stops within 500 m", "right_bank": "Right bank of the Vistula",
}


def unit_matrix(df: pd.DataFrame) -> pd.DataFrame:
    la = np.log(df["area_m2"])
    rooms = df["rooms"].clip(upper=5)
    fl = df["floor"]
    hc = df["height_class"].astype(object)
    new = df["vintage"].eq("new_2007plus")
    age = df["age_at_sale"]
    X = pd.DataFrame({
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
        "primary_developer": df["segment"].eq("primary_developer"),
    }, index=df.index)
    return X.astype(float)


def location_matrix(df: pd.DataFrame) -> pd.DataFrame:
    c, m = df["dist_centre_km"], df["dist_metro_km"]
    X = pd.DataFrame({
        "centre_0_2": c.lt(2), "centre_2_4": c.between(2, 4, "left"), "centre_4_6": c.between(4, 6, "left"),
        "centre_8_10": c.between(8, 10, "left"), "centre_10_12": c.between(10, 12, "left"), "centre_12p": c.ge(12),
        "metro_0_05": m.lt(0.5), "metro_05_1": m.between(0.5, 1, "left"), "metro_1_2": m.between(1, 2, "left"),
        "park_300": df["dist_park_km"].lt(0.3), "forest_300": df["dist_forest_km"].lt(0.3),
        "vistula_1km": df["dist_vistula_km"].lt(1), "stops_per_10": df["n_stops_500m"] / 10,
        "right_bank": df["right_bank"].astype(bool),
    }, index=df.index)
    return X.astype(float)


def time_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """Common quarter effects (base = first quarter); the developer dummy is then an average gap."""
    D = pd.get_dummies(df["q"].astype(str), dtype=float, prefix="q")
    return D.drop(columns=D.columns[0])


def time_segment_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """Quarter effects per segment (base: first quarter of each segment) - used for prediction,
    because developer deed prices and resale prices did not move together in 2023-26."""
    key = df["segment"].astype(str) + "|" + df["q"].astype(str)
    D = pd.get_dummies(key, dtype=float, prefix="q")
    base = [f"q_{s}|{df.loc[df['segment'] == s, 'q'].min()}" for s in df["segment"].unique()]
    return D.drop(columns=[b for b in base if b in D.columns])


def _premia(res, names: dict, terms: list[str]) -> pd.DataFrame:
    b, se = res.params[terms], res.bse[terms]
    out = pd.DataFrame({"term": terms, "label": [names.get(t, t) for t in terms], "coef": b.values, "se": se.values})
    is_log = out["term"].isin(["log_area", "log_area_sq"])
    out["premium_pct"] = np.where(is_log, np.nan, (np.exp(out["coef"]) - 1) * 100)
    out["ci_low_pct"] = np.where(is_log, np.nan, (np.exp(out["coef"] - 1.96 * out["se"]) - 1) * 100)
    out["ci_high_pct"] = np.where(is_log, np.nan, (np.exp(out["coef"] + 1.96 * out["se"]) - 1) * 100)
    out["p_value"] = res.pvalues[terms].values
    return out


def fit_spec_a(df: pd.DataFrame):
    X = pd.concat([unit_matrix(df), location_matrix(df), time_dummies(df)], axis=1)
    X = sm.add_constant(X)
    groups = pd.factorize(df["building_key"])[0]
    res = sm.OLS(df["log_price_m2"].to_numpy(), X).fit(cov_type="cluster", cov_kwds={"groups": groups})
    terms = [t for t in list(UNIT_TERMS) + list(LOCATION_TERMS) if t in X.columns]
    return res, _premia(res, {**UNIT_TERMS, **LOCATION_TERMS}, terms)


def fit_spec_b(df: pd.DataFrame, min_cell: int = 2, segment_time: bool = False):
    """H3-cell fixed effects via within transformation; returns results, premia, model parts.

    segment_time=True gives each segment its own quarter effects (for prediction); the
    developer dummy is then the gap in the first quarter only, so premia use False.
    """
    cells = df["h3_r8"].to_numpy()
    sizes = df.groupby("h3_r8")["h3_r8"].transform("size").to_numpy()
    d = df[sizes >= min_cell]
    codes = pd.factorize(d["h3_r8"])[0]
    U, T = unit_matrix(d), (time_segment_dummies(d) if segment_time else time_dummies(d))
    X = pd.concat([U, T], axis=1)
    Xd = pd.DataFrame(_demean(X.to_numpy(), codes, codes.max() + 1), columns=X.columns, index=d.index)
    Xd = Xd.loc[:, Xd.std() > 1e-10]
    yd = _demean(d[["log_price_m2"]].to_numpy(), codes, codes.max() + 1).ravel()
    groups = pd.factorize(d["building_key"])[0]
    res = sm.OLS(yd, Xd).fit(cov_type="cluster", cov_kwds={"groups": groups})

    beta = res.params
    fitted_no_fe = X[beta.index].to_numpy() @ beta.to_numpy()
    cell_fe = pd.Series(d["log_price_m2"].to_numpy() - fitted_no_fe, index=d.index).groupby(d["h3_r8"]).mean()
    resid = d["log_price_m2"].to_numpy() - fitted_no_fe - cell_fe.reindex(d["h3_r8"]).to_numpy()
    parts = {
        "beta": beta, "cell_fe": cell_fe, "cell_n": d.groupby("h3_r8").size(),
        "smearing": float(np.mean(np.exp(resid))), "resid": pd.Series(resid, index=d.index),
        "r2_within": float(res.rsquared),
    }
    terms = [t for t in UNIT_TERMS if t in beta.index]
    return res, _premia(res, UNIT_TERMS, terms), parts


def location_value(parts: dict, df: pd.DataFrame, min_n: int = 20) -> pd.DataFrame:
    """Cell fixed effects as % premium over the sales-weighted Warsaw average (cells with >= min_n sales)."""
    fe, n = parts["cell_fe"], parts["cell_n"]
    ref = np.average(fe, weights=n)
    out = pd.DataFrame({"h3_r8": fe.index, "fe": fe.values - ref, "n": n.reindex(fe.index).values})
    out["premium_pct"] = (np.exp(out["fe"]) - 1) * 100
    district = df.groupby("h3_r8")["district"].agg(lambda s: s.mode().iat[0])
    out["district"] = out["h3_r8"].map(district)
    return out[out["n"] >= min_n].reset_index(drop=True)
