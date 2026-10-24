import numpy as np
import pandas as pd
import pytest

from real_estate.forecast import models as fm
from real_estate.macro.rates import annuity_factor, local_projections


def test_annuity_factor_and_rate_sensitivity():
    assert annuity_factor(6.0) == pytest.approx(155.21, abs=0.05)
    # +1 pp at ~6% cuts what a fixed payment can borrow by roughly 9%
    assert annuity_factor(7.0) / annuity_factor(6.0) - 1 == pytest.approx(-0.088, abs=0.003)


def test_local_projection_recovers_delayed_response():
    rng = np.random.default_rng(0)
    n = 600
    shock = pd.Series(rng.normal(0, 1, n))
    dy = 0.01 * shock.shift(2).fillna(0) + rng.normal(0, 0.001, n)   # effect arrives after 2 periods
    y = dy.cumsum()
    irf = local_projections(y, shock, controls=[], horizons=5).set_index("h")["beta"]
    assert abs(irf[0]) < 0.05 and abs(irf[1]) < 0.05
    assert irf[2] == pytest.approx(1.0, abs=0.1) and irf[5] == pytest.approx(1.0, abs=0.1)


def _toy_frame(n=60, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.period_range("2008Q1", periods=n, freq="Q")
    g = 0.01 + rng.normal(0, 0.01, n)
    return pd.DataFrame({"y": np.cumsum(g), "g": g, "real_rate": rng.normal(2, 1, n),
                         "wage_growth": rng.normal(6, 1, n)}, index=idx)


def test_backtest_has_no_look_ahead():
    d = _toy_frame()
    d2 = d.copy()
    cut = pd.Period("2016Q4", "Q")
    later = d2.index > cut
    d2.loc[later, ["g", "real_rate", "wage_growth"]] += 0.05         # change everything after the cut
    d2["y"] = d2["g"].cumsum()
    bt1, bt2 = fm.backtest(d, "2012Q1"), fm.backtest(d2, "2012Q1")
    # implied forecast = actual x (1 + error); identical for origins up to the cut
    for bt, frame in [(bt1, d), (bt2, d2)]:
        bt["forecast"] = np.exp(frame.loc[pd.PeriodIndex(bt["target"], freq="Q"), "y"].to_numpy()) * (1 + bt["error_pct"] / 100)
    key = ["origin", "h", "model"]
    a = bt1[pd.PeriodIndex(bt1["origin"], freq="Q") <= cut].set_index(key)["forecast"]
    b = bt2[pd.PeriodIndex(bt2["origin"], freq="Q") <= cut].set_index(key)["forecast"]
    assert np.allclose(a.sort_index(), b.sort_index())
