"""Each index method must recover a known price path from synthetic data -
including when the mix of units sold shifts over time."""

import numpy as np
import pandas as pd
import pytest

from real_estate.indices.hedonic import rolling_time_dummy
from real_estate.indices.repeat_sales import build_pairs, case_shiller
from real_estate.indices.simple import median_index, stratified_index

TRUE_LOG = np.array([0.0, 0.03, 0.05, 0.02, 0.08, 0.12, 0.15, 0.14])
PERIODS = list(pd.period_range("2020Q1", periods=len(TRUE_LOG), freq="Q"))


def _synthetic(n_per_period=1500, seed=0):
    """Log price/m2 = true time effect + cell effect + size effect + noise.
    Later periods sell more of the expensive cells and more small flats (mix shift)."""
    rng = np.random.default_rng(seed)
    cells = np.array([f"c{i}" for i in range(40)])
    cell_eff = dict(zip(cells, rng.normal(0, 0.3, len(cells))))
    rows = []
    for t, (p, g) in enumerate(zip(PERIODS, TRUE_LOG)):
        tilt = np.linspace(-1, 1, len(cells)) * t * 0.08
        prob = np.exp(tilt) / np.exp(tilt).sum()
        c = rng.choice(cells, n_per_period, p=prob)
        area = np.clip(rng.normal(55 - 2 * t, 15, n_per_period), 20, 150)
        y = 9.5 + g + np.array([cell_eff[x] for x in c]) - 0.1 * (np.log(area) - np.log(50)) \
            + rng.normal(0, 0.1, n_per_period)
        rows.append(pd.DataFrame({"q": p, "h3_r8": c, "district": c, "area_m2": area, "log_price_m2": y}))
    df = pd.concat(rows, ignore_index=True)
    df["price_m2"] = np.exp(df["log_price_m2"])
    df["year"] = 2020
    for col, val in {"rooms": 2.0, "floor": 3.0, "is_top_floor": False, "height_class": "mid_5-10",
                     "vintage": "older_or_unknown", "age_at_sale": np.nan, "has_ancillary_area": False,
                     "dist_metro_km": 1.0}.items():
        df[col] = val
    return df


def _normalised(log_index):
    s = pd.Series(log_index).sort_index()
    return (s - s.iloc[0]).to_numpy()


def test_hedonic_recovers_path_under_mix_shift():
    df = _synthetic()
    est = rolling_time_dummy(df, window=3)
    assert np.abs(_normalised(est["log_index"]) - TRUE_LOG).max() < 0.01


def test_median_is_biased_by_mix_shift_but_stratified_is_not():
    df = _synthetic()
    med = _normalised(median_index(df)["log_index"])
    strat = _normalised(stratified_index(df, weight_years=(2020, 2020))["log_index"])
    med_err, strat_err = np.abs(med - TRUE_LOG).max(), np.abs(strat - TRUE_LOG).max()
    assert med_err > 0.03                                # composition drift of the raw median
    assert strat_err < 0.01 and strat_err < med_err / 3


def test_repeat_sales_recovers_path():
    rng = np.random.default_rng(1)
    n_units = 3000
    quality = rng.normal(0, 0.4, n_units)
    rows = []
    for u in range(n_units):
        ts = np.sort(rng.choice(len(PERIODS), 2, replace=False))
        for t in ts:
            rows.append({"street": "ulica test", "house_no": str(u), "area_m2": 50.0, "floor": 2.0, "rooms": 2.0,
                         "date": PERIODS[t].start_time + pd.Timedelta(days=int(rng.integers(0, 80))),
                         "q": PERIODS[t], "segment": "secondary",
                         "price": np.exp(12 + quality[u] + TRUE_LOG[t] + rng.normal(0, 0.05)) })
    pairs = build_pairs(pd.DataFrame(rows), min_gap_days=0)
    est, info = case_shiller(pairs, PERIODS)
    assert info["pairs_used"] > 0.95 * n_units
    assert est["log_index"].to_numpy() == pytest.approx(TRUE_LOG, abs=0.01)
