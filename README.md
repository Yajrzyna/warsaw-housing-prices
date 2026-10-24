# Warsaw Apartment Prices

Analysis of 366k recorded apartment sales in Warsaw (2006–2026) from the city's official transaction register (RCN).

## What it does
- **Cleaning:** flags non-market sales, bundles, partial shares, duplicates and outliers; benchmarked against the NBP index.
- **Price indices:** median, mix-adjusted, hedonic, repeat-sales, real and per-district.
- **Hedonic models:** value of floor, rooms, building age and location.
- **Valuation model:** gradient boosting, **8.4% median error** on future sales.
- **Macro:** interest rates and affordability, with forecasts to 2028 checked by a real-time backtest.
- **Dashboard:** `reports/dashboard/warsaw_flat_prices.html`

Results are summarised in `reports/*_summary.md`, with figures and tables in `reports/`.

## Run
```bash
pip install -r requirements.txt && pip install -e .
python scripts/01_collect_rcn.py      # downloads the RCN data (~550 MB)
python scripts/02_clean_rcn.py        # then 03–06, 08, 09, and 07 (dashboard) last
python -m pytest tests                # 19 tests
```

Raw data and trained models are not included; the scripts download or rebuild them.

MIT licence.
