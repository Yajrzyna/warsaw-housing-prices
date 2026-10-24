"""Phase 6: hedonic & spatial models and the AVM. Writes tables + the calculator model.

Outputs (reports/tables/mdl_*.csv, data/processed/calculator_model.json):
  mdl_premia_spec_a / _spec_b      premia with 95% CI (building-clustered SE)
  mdl_fit_stats                    R2, n, Moran's I of residuals
  mdl_location_value_h3            location premium per H3 cell (spec B fixed effects)
  mdl_district_heterogeneity       spec B premia re-estimated per district
  mdl_avm_metrics                  temporal + spatial validation of all models
  mdl_avm_importance               permutation importance (gbm_nbhd, temporal test)
  mdl_avm_errors                   test error by segment and district (gbm_nbhd)
"""

from __future__ import annotations

import json

import h3
import numpy as np
import pandas as pd

from real_estate.config import PROCESSED_DIR, REPORTS_TABLES_DIR
from real_estate.models import avm, data, hedonic_ols as ho, spatial

CALCULATOR_PATH = PROCESSED_DIR / "calculator_model.json"
# The calculator predicts current prices, so it uses the last 8 quarters and per-segment
# quarter effects (developer deed prices and resale prices diverged in 2025-26).
CALC_WINDOW = (pd.Period("2024Q2", "Q"), pd.Period("2026Q1", "Q"))
T = REPORTS_TABLES_DIR


def hedonic_outputs(rec: pd.DataFrame) -> dict:
    ra, pa = ho.fit_spec_a(rec)
    rb, pb, parts = ho.fit_spec_b(rec)
    pa.to_csv(T / "mdl_premia_spec_a.csv", index=False)
    pb.to_csv(T / "mdl_premia_spec_b.csv", index=False)
    mor = spatial.residual_autocorrelation(rec, ra, parts)
    stats = pd.DataFrame([
        {"model": "Spec A (location variables)", "n": int(ra.nobs), "r2": ra.rsquared},
        {"model": "Spec B (H3-cell fixed effects)", "n": int(rb.nobs), "r2": parts["r2_within"]},
    ]).merge(mor[["model", "morans_i", "p_value"]], on="model")
    stats.to_csv(T / "mdl_fit_stats.csv", index=False)
    lv = ho.location_value(parts, rec)
    lv.to_csv(T / "mdl_location_value_h3.csv", index=False)
    spatial.district_heterogeneity(rec).to_csv(T / "mdl_district_heterogeneity.csv", index=False)
    calc = rec[rec["q"].between(*CALC_WINDOW)]
    _, _, calc_parts = ho.fit_spec_b(calc, segment_time=True)
    write_calculator(calc_parts, calc)
    return {"stats": stats, "premia_b": pb, "premia_a": pa}


def write_calculator(parts: dict, rec: pd.DataFrame, min_n: int = 10) -> None:
    """Spec B coefficients + cell effects -> JSON for the dashboard's price calculator.

    log(price/m2) = cell_fe + delta_last[segment] + x'beta ;  price/m2 = exp(.) * smearing
    (x includes the developer dummy = developer/resale gap in the window's first quarter)
    """
    beta = parts["beta"]
    last_q = rec["q"].max()
    unit = {k: float(v) for k, v in beta.items() if not k.startswith("q_")}
    delta_last = {s: float(beta.get(f"q_{s}|{last_q}", 0.0)) for s in rec["segment"].unique()}
    cells = []
    fe, n = parts["cell_fe"], parts["cell_n"]
    district = rec.groupby("h3_r8")["district"].agg(lambda s: s.mode().iat[0])
    metro = rec.groupby("h3_r8")["dist_metro_km"].median()
    for c in fe.index[n.reindex(fe.index) >= min_n]:
        cells.append({"h3": c, "fe": round(float(fe[c]), 5), "n": int(n[c]), "district": district[c],
                      "metro_km": round(float(metro[c]), 2),
                      "boundary": [[round(lng, 5), round(lat, 5)] for lat, lng in h3.cell_to_boundary(c)]})
    fe_ref = float(np.average(fe, weights=n.reindex(fe.index)))  # same reference as location_value()
    out = {"quarter": str(last_q), "delta_last": delta_last, "beta": unit, "smearing": parts["smearing"],
           "fe_ref": fe_ref, "window": [str(q) for q in CALC_WINDOW], "cells": cells}
    CALCULATOR_PATH.write_text(json.dumps(out))


def avm_outputs(full: pd.DataFrame) -> dict:
    met_t, models, test = avm.temporal_validation(full)
    met_s = avm.spatial_validation(full)
    metrics = pd.concat([met_t, met_s], ignore_index=True)
    metrics.to_csv(T / "mdl_avm_metrics.csv", index=False)

    imp = avm.importance(models["gbm_nbhd"], avm.add_nbhd_rel(test), with_nbhd=True)
    imp.to_csv(T / "mdl_avm_importance.csv", index=False)

    ape = np.abs(np.exp(test["pred_gbm_nbhd"] - test["log_price_m2"]) - 1) * 100
    err = (test.assign(ape=ape).groupby(["segment", "district"])["ape"]
           .agg(n="size", median_ape_pct="median", within_10_pct=lambda s: (s <= 10).mean() * 100).reset_index())
    err.to_csv(T / "mdl_avm_errors.csv", index=False)
    return {"metrics": metrics, "importance": imp}


def main() -> None:
    T.mkdir(parents=True, exist_ok=True)
    rec = data.load_frame(*ho.WINDOW)
    h = hedonic_outputs(rec)
    print(h["stats"].round(3).to_string(index=False))
    a = avm_outputs(data.load_frame())
    print(a["metrics"].round(2).to_string(index=False))
    print(a["importance"].head(10).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
