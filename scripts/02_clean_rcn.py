"""Clean the RCN apartment table and benchmark it against NBP.

    data/interim/rcn_apartments.parquet
      -> data/interim/rcn_apartments_flagged.parquet   (all rows, excl_* flags)
      -> data/processed/rcn_apartments_clean.parquet   (analysis sample)
      -> reports/tables/rcn_cleaning_log.csv, rcn_vs_nbp_quarterly.csv
"""

from real_estate.cleaning import rcn_clean, rcn_validate

if __name__ == "__main__":
    rcn_clean.main()
    print("\nbenchmark vs NBP BaRN (Warsaw transaction prices):")
    print(rcn_validate.compare()[1].round(3).to_string(index=False))
