"""Hedonic spec B must recover known unit effects under location confounding,
and Moran's I must separate clustered from random residuals."""

import numpy as np
import pandas as pd

from real_estate.models import hedonic_ols as ho
from real_estate.models.spatial import morans_i


def _sales(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    cells = rng.integers(0, 60, n)
    cell_eff = rng.normal(0, 0.3, 60)
    # confounding: expensive cells have more ground-floor flats
    ground = rng.random(n) < 0.2 + 0.3 * (cell_eff[cells] > 0)
    top = rng.random(n) < 0.15
    df = pd.DataFrame({
        "h3_r8": [f"c{c}" for c in cells], "building_key": [f"b{c}_{i % 20}" for i, c in enumerate(cells)],
        "q": pd.Period("2024Q1", "Q"), "segment": "secondary", "district": "X",
        "area_m2": rng.uniform(30, 90, n), "rooms": 2.0, "floor": np.where(ground, 1.0, 3.0),
        "is_top_floor": top, "height_class": "mid_5-10", "vintage": "older_or_unknown",
        "age_at_sale": np.nan, "has_ancillary_area": False,
    })
    df["log_price_m2"] = 9.6 + cell_eff[cells] - 0.05 * ground - 0.02 * top + rng.normal(0, 0.05, n)
    return df


def test_spec_b_recovers_unit_effects_despite_location_confounding():
    res, prem, parts = ho.fit_spec_b(_sales())
    assert abs(res.params["floor_ground"] - (-0.05)) < 0.005
    assert abs(res.params["top_floor"] - (-0.02)) < 0.005
    assert parts["smearing"] > 1.0


def test_morans_i_detects_clustering():
    rng = np.random.default_rng(1)
    xy = rng.uniform(0, 1000, (2000, 2))
    smooth = np.sin(xy[:, 0] / 150) + np.cos(xy[:, 1] / 150)
    assert morans_i(xy, smooth, n_perm=49)["morans_i"] > 0.5
    assert abs(morans_i(xy, rng.normal(size=2000), n_perm=49)["morans_i"]) < 0.05
