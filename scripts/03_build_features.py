"""Build the feature table for the clean RCN apartment sample.

    data/processed/rcn_apartments_clean.parquet
      + BDOT10k (buildings, transport, parks, forests, Vistula) + metro opening dates
      -> data/processed/rcn_apartments_features.parquet

Feature definitions: docs/data_dictionaries/features.md
"""

from real_estate.features import build_features

if __name__ == "__main__":
    build_features.main()
