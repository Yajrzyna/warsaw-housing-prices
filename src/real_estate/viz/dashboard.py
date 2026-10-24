"""Build the interactive dashboard: gather results into one JSON bundle and inject it
into reports/dashboard/template.html -> reports/dashboard/warsaw_flat_prices.html.

Every number shown on the page comes from the pipeline outputs in reports/tables/
and data/processed/ - nothing is typed in by hand.
"""

from __future__ import annotations

import json
from datetime import date

import h3
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, mapping

from real_estate.config import PROCESSED_DIR, REPORTS_DIR, REPORTS_TABLES_DIR as T
from real_estate.geo import bdot
from real_estate.geo.districts import load_districts
from real_estate.indices.build import QUARTERLY_PATH
from real_estate.models.run import CALCULATOR_PATH

DASH_DIR = REPORTS_DIR / "dashboard"
TEMPLATE = DASH_DIR / "template.html"
OUTPUT = DASH_DIR / "warsaw_flat_prices.html"


def _r(x, nd=1):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


def _geo(gdf, tolerance_m: float) -> dict:
    g = gdf.to_crs(2180).copy()
    g["geometry"] = g.geometry.simplify(tolerance_m)
    g = g.to_crs(4326)

    def rnd(coords):
        if isinstance(coords[0], (float, int)):
            return [round(coords[0], 5), round(coords[1], 5)]
        return [rnd(c) for c in coords]

    feats = []
    for _, row in g.iterrows():
        geom = mapping(row.geometry)
        feats.append({"type": "Feature", "properties": {k: row[k] for k in g.columns if k != "geometry"},
                      "geometry": {"type": geom["type"], "coordinates": rnd(geom["coordinates"])}})
    return {"type": "FeatureCollection", "features": feats}


def indices() -> dict:
    long = pd.read_parquet(QUARTERLY_PATH)
    wide = long.pivot_table(index="q", columns=["segment", "method"], values="index").sort_index()
    col = lambda s, m: [_r(v, 2) for v in wide[(s, m)].to_numpy()]
    hed, nbp = wide[("secondary", "hedonic")], wide[("secondary", "nbp_hedonic")]
    yoy = lambda s: [_r(v, 2) for v in ((s / s.shift(4) - 1) * 100).to_numpy()]
    return {
        "q": list(wide.index),
        "hedonic": col("secondary", "hedonic"), "repeat_sales": col("secondary", "repeat_sales"),
        "mix_adjusted": col("secondary", "mix_adjusted"), "median": col("secondary", "median"),
        "nbp_hedonic": col("secondary", "nbp_hedonic"), "hedonic_real": col("secondary", "hedonic_real"),
        "primary_hedonic": col("primary_developer", "hedonic"), "primary_median": col("primary_developer", "median"),
        "yoy_hedonic": yoy(hed), "yoy_nbp": yoy(nbp),
    }


def kpis(ix: dict) -> dict:
    diag = pd.read_csv(T / "price_index_diagnostics.csv").set_index(["segment", "method"])
    eda02 = pd.read_csv(T / "eda_02_price_trend.csv", index_col=0)
    vol = pd.read_csv(T / "eda_01_volume_by_segment.csv", index_col=0)
    v25 = vol.loc[[q for q in vol.index if q.startswith("2025")]].sum()
    h = diag.loc[("secondary", "hedonic")]
    return {
        "median_2025": _r(median_2025(), 0), "last_q": ix["q"][-1],
        "last_median": _r(eda02["secondary"].iloc[-1], 0),
        "growth_2015_2025": _r(h["growth_2015_2025_pct"]), "cagr_2015_2025": _r(h["cagr_2015_2025_pct"]),
        "real_growth_2015_2025": _r(diag.loc[("secondary", "hedonic_real"), "growth_2015_2025_pct"]),
        "peak_q": h["peak_quarter"], "since_peak": _r(h["change_peak_to_last_pct"]),
        "real_since_peak": _r(diag.loc[("secondary", "hedonic_real"), "change_peak_to_last_pct"]),
        "yoy_last": ix["yoy_hedonic"][-1],
        "sales_2025": int(v25.sum()), "sales_2025_secondary": int(v25["secondary"]),
        "sales_2025_developer": int(v25["primary_developer"]),
        "nbp_growth_2015_2025": _r(diag.loc[("secondary", "nbp_hedonic"), "growth_2015_2025_pct"]),
    }


def median_2025() -> float:
    from real_estate.features.build_features import FEATURES_PATH
    f = pd.read_parquet(FEATURES_PATH, columns=["segment", "year", "price_m2"])
    return float(f[(f["segment"] == "secondary") & (f["year"] == 2025)]["price_m2"].median())


def map_layers() -> dict:
    price = pd.read_csv(T / "eda_05_map_price_level.csv")
    growth = pd.read_csv(T / "eda_06_map_relative_growth.csv")
    loc = pd.read_csv(T / "mdl_location_value_h3.csv")
    cells = sorted(set(price["h3_r8"]) | set(growth["h3_r8"]) | set(loc["h3_r8"]))
    hexes = {c: [[round(lng, 5), round(lat, 5)] for lat, lng in h3.cell_to_boundary(c)] for c in cells}
    dist = load_districts().to_crs(4326)
    centres = gpd.GeoDataFrame({"h3": cells}, geometry=[Point(h3.cell_to_latlng(c)[::-1]) for c in cells], crs=4326)
    cell_district = gpd.sjoin(centres, dist[["district", "geometry"]], how="left", predicate="within").drop_duplicates("h3")
    return {
        "hexes": hexes,
        "cell_district": {r.h3: r.district for r in cell_district.itertuples() if isinstance(r.district, str)},
        "price": {r.h3_r8: [_r(r.median, 0), int(r.n)] for r in price.itertuples()},
        "growth": {r.h3_r8: [_r(r.vs_city_pp), _r(r.growth_pct), int(r.n0), int(r.n1)] for r in growth.itertuples()},
        "location": {r.h3_r8: [_r(r.premium_pct), int(r.n), r.district] for r in loc.itertuples()},
        "districts": _geo(load_districts()[["district", "geometry"]], 25),
        "vistula": _geo(bdot.vistula()[["geometry"]], 15),
    }


def districts() -> list[dict]:
    lvl = pd.read_csv(T / "eda_03_district_price_level.csv", index_col=0)
    gr = pd.read_csv(T / "eda_04_district_growth.csv", index_col=0)
    ann = pd.read_csv(T / "price_indices_district_annual.csv", index_col=0)
    out = []
    for d in lvl.index:
        out.append({"district": d, "median_2025": _r(lvl.loc[d, "median"], 0), "n_2025": int(lvl.loc[d, "n"]),
                    "median_2016": _r(gr.loc[d, "2016.0"], 0), "median_growth": _r(gr.loc[d, "growth_pct"]),
                    "hedonic_growth": _r(ann.loc[2025, d] - 100), "series": [_r(v, 1) for v in ann[d]]})
    return out


def models() -> dict:
    pb = pd.read_csv(T / "mdl_premia_spec_b.csv").dropna(subset=["premium_pct"])
    pa = pd.read_csv(T / "mdl_premia_spec_a.csv")
    pa = pa[pa["term"].str.match(r"^(centre|metro|park|forest|vistula|stops|right)")]
    rec = lambda t: [{"label": r.label, "pct": _r(r.premium_pct, 2), "lo": _r(r.ci_low_pct, 2),
                      "hi": _r(r.ci_high_pct, 2)} for r in t.itertuples()]
    met = pd.read_csv(T / "mdl_avm_metrics.csv")
    imp = pd.read_csv(T / "mdl_avm_importance.csv").head(12)
    fit = pd.read_csv(T / "mdl_fit_stats.csv")
    err = pd.read_csv(T / "mdl_avm_errors.csv")
    err = err[err["segment"] == "secondary"].sort_values("median_ape_pct")
    return {
        "unit": rec(pb), "location": rec(pa),
        "avm": [{"scheme": r.scheme, "model": r.model, "n": int(r.n_test), "mdape": _r(r.median_ape_pct, 2),
                 "mape": _r(r.mean_ape_pct, 2), "mae": _r(r.mae_pln_m2, 0), "w10": _r(r.within_10_pct),
                 "w20": _r(r.within_20_pct), "bias": _r(r.median_bias_pct, 2)} for r in met.itertuples()],
        "importance": [{"feature": r.feature, "value": _r(r.mae_increase * 100, 3)} for r in imp.itertuples()],
        "fit": [{"model": r.model, "n": int(r.n), "r2": _r(r.r2, 3), "moran": _r(r.morans_i, 3)} for r in fit.itertuples()],
        "errors_by_district": [{"district": r.district, "n": int(r.n), "mdape": _r(r.median_ape_pct),
                                "w10": _r(r.within_10_pct)} for r in err.itertuples()],
    }


def calculator() -> dict:
    c = json.loads(CALCULATOR_PATH.read_text())
    cells = pd.DataFrame(c["cells"])
    typical = cells.groupby("district").apply(lambda g: np.average(g["fe"], weights=g["n"]), include_groups=False)
    met = pd.read_csv(T / "mdl_avm_metrics.csv")
    hb = met[(met["model"] == "hedonic_B") & met["scheme"].str.startswith("temporal")].iloc[0]
    return {"quarter": c["quarter"], "delta_last": c["delta_last"], "beta": c["beta"], "smearing": c["smearing"],
            "fe_ref": c["fe_ref"], "window": c["window"],
            "cells": {r.h3: [r.fe, r.n, r.district, r.metro_km] for r in cells.itertuples()},
            "district_fe": {k: round(float(v), 5) for k, v in typical.items()},
            "accuracy": {"mdape": _r(hb.median_ape_pct), "w10": _r(hb.within_10_pct), "w20": _r(hb.within_20_pct)}}


def macro() -> dict:
    from real_estate.macro.analysis import EVENTS
    q = pd.read_csv(T / "mac_quarterly.csv", index_col=0)
    col = lambda c, nd=2: [_r(v, nd) for v in q[c]]
    dist = pd.read_csv(T / "mac_district_affordability.csv", index_col=0).sort_values("years_of_wages", ascending=False)
    bands = pd.read_csv(T / "mac_size_band_indices.csv", index_col=0)
    bands = bands.loc["2021Q1":]
    did = pd.read_csv(T / "mac_bk2_did.csv", index_col=0)
    reg = pd.read_csv(T / "mac_regression.csv")
    cap = pd.read_csv(T / "mac_resale_price_cap_share.csv", index_col=0)
    return {
        "q": list(q.index), "years_of_wages": col("years_of_wages"), "m2_per_wage": col("m2_per_wage", 3),
        "payment_share": col("payment_share", 1), "mortgage_rate": col("mortgage_rate"),
        "wage_last": _r(q["wage"].iloc[-1], 0),
        "districts": [{"district": d, "years": _r(r.years_of_wages, 2), "payment": _r(r.payment_share, 1)}
                      for d, r in dist.iterrows()],
        "bands": {"q": list(bands.index), **{c: [_r(v, 2) for v in bands[c]] for c in bands.columns}},
        "did": {row: [_r(v, 1) for v in did.loc[row]] for row in did.index}, "did_periods": list(did.columns),
        "regression": [{"spec": r.spec, "term": r.term, "coef": _r(r.coef, 2), "se": _r(r.se, 2), "p": _r(r.p_value, 3),
                        "n": int(r.n), "r2": _r(r.r2, 2)} for r in reg.itertuples()],
        "cap_share": {k: _r(cap.loc[k, "share_under_cap_pct"], 0) for k in ["2023Q2", "2024Q2"]},
        "events": [{"name": k, "from": a, "to": b} for k, (a, b) in EVENTS.items()],
    }


def rates() -> dict:
    sc = pd.read_csv(T / "rt_01_rates_vs_growth.csv", index_col=0)
    q = pd.read_csv(T / "rt_quarterly.csv", index_col=0)
    irf = pd.read_csv(T / "rt_impulse_responses.csv")
    fc = pd.read_csv(T / "fc_forecast.csv")
    acc = pd.read_csv(T / "fc_accuracy.csv")
    gr = pd.read_csv(T / "rt_granger.csv")
    hist = q.loc["2019Q1":, "hedonic"]
    fq = list(fc["q"].unique())
    base = fc[fc["scenario"] == "rates hold"].set_index("q")
    return {
        "scatter": [{"q": k, "y": _r(r.price_growth_yoy, 2), "nominal": _r(r.mortgage_rate_lag4, 2),
                     "real": _r(r.real_rate_lag4, 2)} for k, r in sc.iterrows()],
        "capacity": {"q": list(q.index), "capacity_index": [_r(v, 1) for v in q["capacity_index"]],
                     "hedonic": [_r(v, 1) for v in q["hedonic"]], "ratio": [_r(v, 3) for v in q["price_to_capacity"]]},
        "irf": [{"outcome": r.outcome, "shock": r.shock, "h": int(r.h), "beta": _r(r.beta, 2), "lo": _r(r.lo90, 2),
                 "hi": _r(r.hi90, 2)} for r in irf.itertuples()],
        "granger": [{"rate": r.rate, "lags": int(r.lags), "p_fwd": _r(r.rate_to_price_p, 3), "p_back": _r(r.price_to_rate_p, 3)}
                    for r in gr.itertuples()],
        "forecast": {"q": list(hist.index) + fq, "n_hist": len(hist), "hist": [_r(v, 2) for v in hist],
                     "scenarios": {s: [_r(v, 2) for v in g["index"]] for s, g in fc.groupby("scenario", sort=False)},
                     "lo": [_r(v, 2) for v in base["lo80"]], "hi": [_r(v, 2) for v in base["hi80"]],
                     "change": {s: _r(g["change_vs_last_pct"].iloc[-1], 1) for s, g in fc.groupby("scenario", sort=False)}},
        "accuracy": [{"model": r.model, "h": int(r.h), "rmse": _r(r.rmse, 2), "bias": _r(r.bias, 2)}
                     for r in acc.itertuples() if r.h in (1, 4, 8)],
    }


RULE_LABELS = {
    "excl_invalid_date": "Invalid deed date", "excl_duplicate": "Duplicate records",
    "excl_non_market_type": "Auction, bailiff or discounted municipal sale",
    "excl_public_party": "State or city as buyer or seller", "excl_partial_share": "Only a share of the flat sold",
    "excl_not_unit_ownership": "Right sold is not flat ownership",
    "excl_bundle_price": "Sold with other units, no separate price",
    "excl_multi_apartment_deed": "Several flats in one deed (investor deals)",
    "excl_area": "Implausible floor area", "excl_no_district": "No location",
    "excl_price_m2_bounds": "Implausible price per m²",
    "excl_outlier_low": "Price far below similar flats nearby", "excl_outlier_high": "Price far above similar flats nearby",
}


def cleaning() -> list[dict]:
    log = pd.read_csv(T / "rcn_cleaning_log.csv")
    return [{"rule": RULE_LABELS.get(r.rule, r.description), "removed": int(r.removed_sequential),
             "remaining": int(r.remaining_after)} for r in log.itertuples()]


def build() -> dict:
    ix = indices()
    log = pd.read_csv(T / "rcn_cleaning_log.csv")
    return {
        "meta": {"built": date.today().isoformat(), "n_apartments_raw": int(log["remaining_after"].iloc[0] + log["removed_sequential"].iloc[0]),
                 "n_clean": int(log["remaining_after"].iloc[-1])},
        "kpis": kpis(ix), "indices": ix, "map": map_layers(), "districts": districts(),
        "districts_city": [_r(v, 1) for v in pd.read_csv(T / "price_indices_district_annual.csv", index_col=0)["Warszawa"]],
        "models": models(), "calculator": calculator(), "cleaning": cleaning(), "macro": macro(), "rates": rates(),
    }


def main() -> None:
    bundle = build()
    payload = json.dumps(bundle, ensure_ascii=False, separators=(",", ":"))
    html = TEMPLATE.read_text().replace("/*__DATA__*/null", payload)
    OUTPUT.write_text(html)
    print(f"dashboard: {OUTPUT} ({len(html) / 1e6:.2f} MB, data {len(payload) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
