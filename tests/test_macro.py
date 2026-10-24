import numpy as np
import pandas as pd
import pytest

from real_estate.macro.analysis import _quarterly_wage, annuity


def test_annuity_matches_textbook_value():
    # 400k PLN, 7% p.a., 25 years -> 2,827.13 PLN a month
    assert annuity(400_000, 7.0) == pytest.approx(2827.13, abs=0.05)


def test_quarterly_wage_interpolates_and_extrapolates():
    annual = pd.Series({2023: 1000.0, 2024: 1100.0})
    q = pd.period_range("2023Q1", "2025Q4", freq="Q")
    w = _quarterly_wage(annual, q)
    # the mean of the four quarters of 2024 is close to the annual 2024 value
    assert w.loc["2024Q1":"2024Q4"].mean() == pytest.approx(1100, rel=0.02)
    # 2025 continues at 10% a year
    assert w.loc["2025Q3"] / w.loc["2024Q3"] == pytest.approx(1.1, rel=1e-6)
    assert np.all(np.diff(w.to_numpy()) > 0)
