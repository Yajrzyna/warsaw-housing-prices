"""Price indices: median, mix-adjusted, hedonic (rolling time dummy), repeat sales;
real (HICP-deflated) version; annual district indices; figures.

    data/processed/rcn_apartments_features.parquet
      -> data/processed/price_indices_quarterly.parquet
      -> reports/tables/price_indices_*.csv, price_index_diagnostics.csv
      -> reports/figures/idx_*.png
"""

from real_estate.indices import build
from real_estate.viz import index_figures as fig
from real_estate.viz import style

if __name__ == "__main__":
    build.main()
    style.apply()
    long = fig.load_quarterly()
    for f in fig.ALL:
        print("wrote", f(long)[0].name, flush=True)
    print("wrote", fig.district_small_multiples()[0].name)
