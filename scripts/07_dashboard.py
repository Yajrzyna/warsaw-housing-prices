"""Build the interactive dashboard from the pipeline outputs.

    reports/tables/*.csv + data/processed/{price_indices_quarterly.parquet, calculator_model.json}
      + reports/dashboard/template.html
      -> reports/dashboard/warsaw_flat_prices.html (self-contained)
"""

from real_estate.viz import dashboard

if __name__ == "__main__":
    dashboard.main()
