"""Clean the RCN apartment table into an analysis-ready market-transaction sample.

Input : data/interim/rcn_apartments.parquet           (from ingest.rcn_build)
Output: data/interim/rcn_apartments_flagged.parquet   every row + one boolean column per rule
        data/processed/rcn_apartments_clean.parquet   rows passing all rules
        reports/tables/rcn_cleaning_log.csv           rows removed per rule (sequential and total)

Rules are evaluated independently (flag columns `excl_*`), so sensitivity
analyses can re-include any of them. The sequential log applies them in the
order of RULES and counts rows removed by each rule not already removed by an
earlier one.

Findings behind the rules (docs/data_dictionaries/rcn.md has details):
- In multi-unit deeds price_transaction is the deed total and price_unit the
  part allocated to the unit; allocations sum to the total in ~94% of deeds.
  Where price_unit is missing the apartment price cannot be separated.
- Partial-share sales are priced at ~1/3 of full value; enforcement and
  discounted (bonifikata) sales far below market.
- Low-price outliers look like related-party sales; high-price outliers are
  mostly genuine luxury units, so the outlier rule is asymmetric.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from real_estate.config import INTERIM_DIR, PROCESSED_DIR, REPORTS_TABLES_DIR

FLAGGED_PATH = INTERIM_DIR / "rcn_apartments_flagged.parquet"
CLEAN_PATH = PROCESSED_DIR / "rcn_apartments_clean.parquet"
LOG_PATH = REPORTS_TABLES_DIR / "rcn_cleaning_log.csv"

FIRST_YEAR = 2006
AREA_RANGE = (15.0, 300.0)       # m2, usable floor area
PRICE_M2_RANGE = (1_000, 150_000)  # PLN/m2 nominal, hard sanity bounds
OUTLIER_Z_LOW = -4.0             # robust z of log price/m2 within peer group
OUTLIER_Z_HIGH = 6.0
MIN_PEER_GROUP = 30              # district x year x market; smaller -> year x market

MARKET_TRANSACTION_TYPES = {1}   # wolny rynek (NA before 2016 is kept, see _rule_non_market)
PUBLIC_PARTIES = {1, 2}          # Skarb Panstwa, JST
OWNERSHIP_OF_UNIT = 3            # rodzajPrawa: wlasnosc lokalu

RULES = {
    "excl_invalid_date": "deed date missing, before 2006 or after snapshot",
    "excl_duplicate": "same address, area, floor, date and price as another record",
    "excl_non_market_type": "transaction type other than free market (tender, enforcement, bonifikata, ...)",
    "excl_public_party": "State Treasury or local government as seller or buyer",
    "excl_partial_share": "only a fraction (udzial < 1) of the unit sold",
    "excl_not_unit_ownership": "right sold is not ownership of a unit",
    "excl_bundle_price": "several units in deed and no price allocated to this apartment",
    "excl_multi_apartment_deed": "deed sells more than one apartment (portfolio / investor deal)",
    "excl_area": f"usable area missing or outside {AREA_RANGE} m2",
    "excl_no_district": "location / district unknown",
    "excl_price_m2_bounds": f"price/m2 outside {PRICE_M2_RANGE} PLN",
    "excl_outlier_low": f"price/m2 robust z < {OUTLIER_Z_LOW} vs district-year-market peers",
    "excl_outlier_high": f"price/m2 robust z > {OUTLIER_Z_HIGH} vs district-year-market peers",
}


def _normalise_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # nullable Int64 codes -> float so comparisons give plain booleans (NA -> False)
    int_cols = df.select_dtypes("Int64").columns
    df[int_cols] = df[int_cols].astype("float64")
    # Apartment price: allocated unit price; transaction price only if the deed has a single unit.
    single = df["deed_n_units"] == 1
    df["price"] = df["price_unit"].where(df["price_unit"].notna(), df["price_transaction"].where(single))
    df["price_m2"] = df["price"] / df["area_m2"].where(df["area_m2"] > 0)
    df["log_price_m2"] = np.log(df["price_m2"].where(df["price_m2"] > 0))
    # 0 rooms means "not recorded"; implausible floors -> missing
    df["rooms"] = df["rooms"].where(df["rooms"].between(1, 15))
    df["floor"] = df["floor"].where(df["floor"].between(-2, 60))
    df["has_ancillary_area"] = df["ancillary_area_m2"].fillna(0) > 0
    # Primary sales by natural persons are not developer sales (e.g. first sale after
    # a cooperative right was converted to ownership) - keep them apart.
    df["segment"] = np.select(
        [df["market"].eq(1) & df["seller_type"].eq(4), df["market"].eq(1), df["market"].eq(2)],
        ["primary_developer", "primary_other", "secondary"],
        default="unknown",
    )
    return df


def _peer_robust_z(df: pd.DataFrame, eligible: pd.Series) -> pd.Series:
    """Robust z-score of log price/m2 against district x year x market peers.

    Median / MAD are estimated on `eligible` rows only (already plausible
    market sales), so non-market rows do not distort the peer statistics.
    """
    base = df[eligible]

    def stats(keys):
        g = base.groupby(keys)["log_price_m2"]
        med = g.median().rename("med")
        mad = g.apply(lambda s: (s - s.median()).abs().median() * 1.4826).rename("mad")
        n = g.size().rename("n")
        return pd.concat([med, mad, n], axis=1)

    fine = df[["district", "year", "market"]].join(stats(["district", "year", "market"]), on=["district", "year", "market"])
    coarse = df[["year", "market"]].join(stats(["year", "market"]), on=["year", "market"])
    use_fine = fine["n"] >= MIN_PEER_GROUP
    med = fine["med"].where(use_fine, coarse["med"])
    mad = fine["mad"].where(use_fine, coarse["mad"])
    return (df["log_price_m2"] - med) / mad


def flag(df: pd.DataFrame) -> pd.DataFrame:
    df = _normalise_fields(df)
    snapshot = df["date"].max()

    df["excl_invalid_date"] = df["date"].isna() | (df["year"] < FIRST_YEAR) | (df["date"] > snapshot)

    dup_key = ["street", "house_no", "area_m2", "floor", "date", "price"]
    df["excl_duplicate"] = df["street"].notna() & df.duplicated(dup_key, keep="first")

    # Transaction type is not recorded before 2016 -> NA is treated as market.
    df["excl_non_market_type"] = df["transaction_type"].notna() & ~df["transaction_type"].isin(MARKET_TRANSACTION_TYPES)
    df["excl_public_party"] = df["seller_type"].isin(PUBLIC_PARTIES) | df["buyer_type"].isin(PUBLIC_PARTIES)
    df["excl_partial_share"] = df["share"].notna() & (df["share"] < 0.999)
    df["excl_not_unit_ownership"] = df["right_type"].ne(OWNERSHIP_OF_UNIT)
    df["excl_bundle_price"] = df["price"].isna()
    df["excl_multi_apartment_deed"] = df["deed_n_residential"] > 1
    df["excl_area"] = ~df["area_m2"].between(*AREA_RANGE)
    df["excl_no_district"] = df["district"].isna()
    df["excl_price_m2_bounds"] = ~df["price_m2"].between(*PRICE_M2_RANGE)

    pre = [c for c in RULES if c not in ("excl_outlier_low", "excl_outlier_high")]
    eligible = ~df[pre].any(axis=1)
    df["price_m2_z"] = _peer_robust_z(df, eligible)
    df["excl_outlier_low"] = df["price_m2_z"] < OUTLIER_Z_LOW
    df["excl_outlier_high"] = df["price_m2_z"] > OUTLIER_Z_HIGH

    df["excluded"] = df[list(RULES)].any(axis=1)
    return df


def cleaning_log(df: pd.DataFrame) -> pd.DataFrame:
    rows, remaining = [], pd.Series(True, index=df.index)
    for rule, desc in RULES.items():
        hit = df[rule]
        rows.append({
            "rule": rule, "description": desc,
            "flagged_total": int(hit.sum()),
            "removed_sequential": int((hit & remaining).sum()),
        })
        remaining &= ~hit
    log = pd.DataFrame(rows)
    log["remaining_after"] = len(df) - log["removed_sequential"].cumsum()
    return log


CLEAN_COLUMNS = [
    "transaction_id", "transaction_no", "doc_ref", "date", "year", "quarter",
    "segment", "market", "market_label", "transaction_type_label", "seller_type_label", "buyer_type_label",
    "price", "price_m2", "log_price_m2", "price_m2_z", "price_source", "price_transaction", "vat_unit",
    "area_m2", "ancillary_area_m2", "has_ancillary_area", "rooms", "floor",
    "deed_n_units", "deed_n_garage", "deed_n_other",
    "street", "house_no", "district_code", "district", "x_2180", "y_2180", "lon", "lat", "coord_source",
    "unit_id", "source_file",
]


def main() -> None:
    raw = pd.read_parquet(INTERIM_DIR / "rcn_apartments.parquet")
    flagged = flag(raw)
    flagged.to_parquet(FLAGGED_PATH, index=False)

    clean = flagged.loc[~flagged["excluded"], CLEAN_COLUMNS].reset_index(drop=True)
    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(CLEAN_PATH, index=False)

    log = cleaning_log(flagged)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(LOG_PATH, index=False)
    summary = clean.pivot_table(index="year", columns="segment", values="price_m2",
                                aggfunc=["size", "median"]).round(0)
    summary.to_csv(REPORTS_TABLES_DIR / "rcn_clean_by_year_segment.csv")
    print(log[["rule", "flagged_total", "removed_sequential", "remaining_after"]].to_string(index=False))
    print(f"\nclean sample: {len(clean):,} of {len(raw):,} apartments ({len(clean) / len(raw):.1%})")


if __name__ == "__main__":
    main()
