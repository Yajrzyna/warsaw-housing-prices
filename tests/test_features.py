import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from real_estate.features.build_features import building_vintage, metro_features, neighbourhood_price


def test_metro_is_time_aware():
    stations = gpd.GeoDataFrame(
        {"station": ["Old", "New"], "line": ["M1", "M2"],
         "opened": pd.to_datetime(["2000-01-01", "2020-01-01"]), "status": ["open", "open"]},
        geometry=[Point(0, 0), Point(1000, 0)], crs=2180)
    df = pd.DataFrame({"x_2180": [1000.0, 1000.0], "y_2180": [0.0, 0.0],
                       "date": pd.to_datetime(["2019-06-01", "2021-06-01"])})
    out = metro_features(df, stations)
    assert out["dist_metro_km"].tolist() == [1.0, 0.0]
    assert out["nearest_metro"].tolist() == ["Old", "New"]
    assert out["dist_metro_2026_km"].tolist() == [0.0, 0.0]
    assert out["metro_opened_later_within_1km"].tolist() == [False, False]  # 1.0 km counts as within


def test_neighbourhood_price_uses_only_past_quarters():
    n = 12
    dates = pd.to_datetime(["2020-02-01"] * n + ["2020-05-01"] * n)
    df = pd.DataFrame({
        "x_2180": np.tile(np.arange(n) * 10.0, 2), "y_2180": 0.0, "date": dates,
        "quarter": pd.PeriodIndex(dates, freq="Q").astype(str),
        "log_price_m2": [9.0] * n + [10.0] * n,
    })
    out = neighbourhood_price(df)
    assert out.loc[:n - 1, "nbhd_log_price_m2"].isna().all()   # nothing before 2020Q1
    assert (out.loc[n:, "nbhd_log_price_m2"] == 9.0).all()     # Q2 sees only Q1, never itself


def test_vintage_requires_first_sale_to_be_developer():
    df = pd.DataFrame({
        "bdot_building_id": ["new", "new", "old", "old"],
        "street": "s", "house_no": "1",
        "date": pd.to_datetime(["2015-01-01", "2016-01-01", "2010-01-01", "2015-01-01"]),
        "year": [2015, 2016, 2010, 2015],
        "segment": ["primary_developer", "primary_developer", "secondary", "primary_developer"],
    })
    out = building_vintage(df).set_index("bdot_building_id")
    assert (out.loc["new", "vintage"] == "new_2007plus").all()
    assert (out.loc["new", "build_year_proxy"] == 2015).all()
    assert (out.loc["old", "vintage"] == "older_or_unknown").all()
