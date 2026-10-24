"""Exploratory analysis: writes figures to reports/figures/eda_*.png and table
twins to reports/tables/eda_*.csv."""

from real_estate.viz import eda_figures as eda
from real_estate.viz import style

if __name__ == "__main__":
    style.apply()
    df = eda.load()
    for fn in [eda.volume_by_segment, eda.price_trend, eda.district_level, eda.district_growth,
               eda.map_price_level, eda.map_relative_growth, eda.price_gradients, eda.composition,
               eda.primary_secondary_gap]:
        path, _ = fn(df)
        print("wrote", path.name, flush=True)
