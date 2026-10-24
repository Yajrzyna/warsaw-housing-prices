"""Collect Warsaw RCN transaction data end-to-end.

    download GML zips  ->  parse to parquet per feature  ->  district boundaries
    ->  unit-level table (data/interim/rcn_units.parquet, rcn_apartments.parquet)

Re-running is cheap: already downloaded / parsed archives are skipped.
Use --force to re-download and re-parse everything (e.g. after the city
publishes a new snapshot of the current year - its filename changes, so
normally a plain re-run picks it up automatically).
"""

import argparse

import pandas as pd

from real_estate.config import REPORTS_TABLES_DIR
from real_estate.geo.districts import DISTRICTS_PATH, download_districts
from real_estate.ingest import rcn_build, rcn_download, rcn_parse


def main(force: bool = False) -> None:
    print("1/4 downloading RCN archives")
    rcn_download.download_all(force=force)

    print("2/4 parsing GML")
    rcn_parse.parse_all(force=force)

    print("3/4 district boundaries")
    if force or not DISTRICTS_PATH.exists():
        download_districts()

    print("4/4 building unit-level tables")
    rcn_build.main()

    apts = pd.read_parquet(rcn_build.APARTMENTS_PATH)
    summary = pd.crosstab(apts["year"], apts["market_label"].fillna("unknown"), margins=True)
    REPORTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(REPORTS_TABLES_DIR / "rcn_apartments_by_year_market.csv")
    print(summary.tail(12))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true")
    main(ap.parse_args().force)
