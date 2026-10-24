"""Macro drivers & affordability (Phase 7).

    NBP rates, Eurostat 3M rate & HICP, GUS BDL (wages, supply, population) + price indices
      -> data/processed/macro_{quarterly,annual}.parquet
      -> reports/tables/mac_*.csv, reports/figures/mac_*.png

Use --download to refresh the macro sources first.
"""

import sys

from real_estate.ingest import macro
from real_estate.macro import run
from real_estate.viz import macro_figures, style

if __name__ == "__main__":
    if "--download" in sys.argv:
        macro.download_all()
    run.main()
    style.apply()
    for f in macro_figures.ALL:
        print("wrote", f().name, flush=True)
