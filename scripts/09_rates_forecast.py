"""Price-rate dependence and short-term forecasts (Phase 8).

    macro data + price indices (run scripts/05 and 08 first)
      -> reports/tables/rt_*.csv, fc_*.csv
      -> reports/figures/rt_*.png, fc_*.png
"""

from real_estate.forecast import run
from real_estate.viz import rates_figures, style

if __name__ == "__main__":
    run.main()
    style.apply()
    for f in rates_figures.ALL:
        print("wrote", f().name, flush=True)
