"""Hedonic & spatial models and the automated valuation model (AVM).

    data/processed/rcn_apartments_features.parquet + price indices
      -> reports/tables/mdl_*.csv, data/processed/calculator_model.json
      -> reports/figures/mdl_*.png
"""

import sys

from real_estate.models import run
from real_estate.viz import model_figures, style

if __name__ == "__main__":
    if "--figures-only" not in sys.argv:
        run.main()
    style.apply()
    for f in model_figures.ALL:
        print("wrote", f().name, flush=True)
