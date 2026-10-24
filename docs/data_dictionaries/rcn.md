# RCN – Rejestr Cen Nieruchomości (Warsaw)

## Source

- Publisher: Biuro Geodezji i Katastru m.st. Warszawy
- Page: https://architektura.um.warszawa.pl/udostepniane-dane-rcn1
- Legal basis: open data since the amendment of *Prawo geodezyjne i kartograficzne*
  of 26.09.2025 (Dz. U. 2025 poz. 1542).
- Format: GML 3.2, schema RCN 1.4 (conceptual model: zał. nr 9 do rozporządzenia
  w sprawie EGiB z 27.07.2021 → `docs/references/RCN_model_pojeciowy_zalacznik9.pdf`).
- Files: 2006–2015, 2016–2020, then yearly; current year as dated snapshot
  (`zbior_danych_rciwn_2026_YYYYMMDD.zip`). Downloaded by
  `src/real_estate/ingest/rcn_download.py`, file hashes in
  `data/raw/rcn_transactions/manifest.json`.
- Source CRS: EPSG:2178 (PL-2000 zone 7, northing first) → converted to EPSG:2180.

## Object model

```
RCN_Transakcja ──podstawaPrawna──► RCN_Dokument      (notarial deed: number, date)
      │
      └─nieruchomosc──► RCN_Nieruchomosc ──lokal───► RCN_Lokal   ──adres──► RCN_Adres
                                         ──budynek─► RCN_Budynek ──adres──► RCN_Adres
                                         ──dzialka─► RCN_Dzialka ──adres──► RCN_Adres
```

In the Warsaw data every transaction has exactly one property and at most one unit.
**One deed can hold several transactions** (e.g. flat + garage + storage),
linked by the same `doc_ref`.

## Output tables

| File | Grain |
|---|---|
| `data/interim/rcn/<feature>/<archive>.parquet` | one row per GML feature (raw, typed) |
| `data/interim/rcn_units.parquet` | one row per unit (lokal) sold, all functions |
| `data/interim/rcn_apartments.parquet` | same, `unit_function == 1` (mieszkalna) |

## Columns of `rcn_units` / `rcn_apartments`

| Column | Meaning |
|---|---|
| transaction_id | RCN `lokalnyId` of the transaction (unique) |
| transaction_no | `oznaczenieTransakcji` – register number |
| source_file | archive the row came from |
| doc_ref, doc_number, date | notarial deed reference, repertorium number, deed date |
| transaction_type(_label) | 1 free market, 2 non-tender, 3 tender, 4 enforcement, 5 public purpose, 6 with discount (bonifikata) |
| market(_label) | 1 primary (pierwotny), 2 secondary (wtórny) |
| seller_type / buyer_type (_label) | 1 State Treasury, 2 local government, 3 natural person, 4 legal entity |
| price_transaction, vat_transaction | gross price of the whole transaction, VAT amount |
| property_type(_label), right_type(_label) | kind of property / right (3 = ownership of a unit) |
| share_raw, share | share in the right sold (`"1/2"` → 0.5); NaN = not given |
| price_property | gross price of the property |
| unit_ref, unit_id | GML id and EGiB id of the unit (`146503_8.0702.399_BUD.101_LOK`) |
| building_ref | GML id of the linked building |
| unit_function(_label) | 1 residential, 2 retail/services, 3 office, 4 production, 5 garage, 6 other |
| rooms, floor | number of rooms (izby; 0 often = not given), floor number |
| area_m2, ancillary_area_m2 | usable floor area; area of ancillary rooms (cellar, storage) |
| price_unit, vat_unit | gross price attributed to this unit |
| unit_info | free-text remarks |
| street, house_no | building address |
| x_2180, y_2180, lon, lat | location (EPSG:2180 / WGS84) |
| coord_source | `unit_point` → `building_centroid` → `address_lookup` (median point of same address) |
| district_code, district, district_source | dzielnica (TERYT 146502…146519) from coordinates, fallback EGiB id prefix |
| deed_n_transactions, deed_n_units, deed_n_residential, deed_n_garage, deed_n_other | composition of the deed |
| deed_price_sum_units, deed_n_distinct_tx_prices | helpers to detect bundled prices |
| year, quarter | from `date` |
| price, price_source | `price_unit`, else `price_transaction` |
| price_m2 | `price / area_m2` (not yet cleaned!) |

## Known data-quality issues (handled in cleaning – see below)

1. **Coverage**: ~6–14k apartment sales/yr in 2006–2014 vs ~24–32k/yr from 2016
   → early years likely incomplete; treat 2006–2015 with care.
2. **Transaction type missing before 2016** (legacy records) → cannot separate
   free-market sales by code; use seller/buyer type, price outliers instead.
3. **Coordinates**: unit points absent in 2017–2020 → building centroid used
   (99%+). Buildings/parcels have no geometry and no EGiB ids in 2006–2015.
4. **Invalid dates**: ~15 records with typo years (e.g. 201, 219) – drop or fix.
5. **Bundles**: ~14% of apartments share a deed with other units (mostly garages
   / storage). `price_unit` usually splits the price, but when it equals
   `price_transaction` for several units the price is a bundle total.
6. **Partial shares**: ~0.5% of rows sell a fraction (1/2, 1/4 …) of a flat.
7. **Implausible values**: area < 15 m² or > 300 m², price/m² < ~2k or > ~70k PLN,
   rooms = 0 (missing), extreme floors.
8. **Current-year file** is a snapshot (deeds registered with a lag – latest
   months are incomplete).
9. Deeds contain the notary's name (`tworcaDokumentu`, interim `dokumenty`
   table only) – not carried into the unit tables.

## Cleaning (`src/real_estate/cleaning/rcn_clean.py`, `scripts/02_clean_rcn.py`)

Every rule is a boolean `excl_*` column in `data/interim/rcn_apartments_flagged.parquet`,
so any rule can be re-included for sensitivity checks. Rows passing all rules are in
`data/processed/rcn_apartments_clean.parquet`. Counts: `reports/tables/rcn_cleaning_log.csv`.

| Rule | Removed (sequential) | Why |
|---|---:|---|
| invalid_date | 17 | typo years (e.g. 201, 219) |
| duplicate | 1,637 | same address/area/floor/date/price registered twice (mostly 2006–08, deed no. written differently) |
| non_market_type | 599 | tender, enforcement, bonifikata etc. – bonifikata/enforcement sell at ~0.1–0.5× market |
| public_party | 313 | State Treasury / city as seller or buyer |
| partial_share | 2,498 | fraction of a flat sold – priced at ~0.34× peers |
| not_unit_ownership | 241 | right sold is not ownership of the unit |
| bundle_price | 2,908 | multi-unit deed without a price allocated to the apartment |
| multi_apartment_deed | 2,827 | several apartments in one deed (investor/portfolio deals) |
| area | 519 | usable area outside 15–300 m² |
| no_district | 728 | no location |
| price_m2_bounds | 529 | outside 1,000–150,000 PLN/m² |
| outlier_low | 5,352 | robust z < −4 vs district×year×market peers (related-party sales) |
| outlier_high | 299 | robust z > 6 (high side kept lenient: it is mostly genuine luxury) |
| **Clean sample** | **366,428 of 384,895 (95.2%)** | |

Derived fields in the clean table:
- `price` = allocated unit price (`price_unit`); transaction price only for single-unit deeds.
- `rooms` 0 → missing; floors outside −2…60 → missing; `has_ancillary_area` (cellar/storage
  included in price but not in `area_m2`).
- `segment`: `primary_developer` (primary market, legal-entity seller), `primary_other`
  (primary market, natural-person seller – e.g. first sale after conversion of a cooperative
  right or small partnerships), `secondary`.
- `price_m2_z`: robust z-score vs peers (kept for later sensitivity analysis).

## Validation vs NBP BaRN (`reports/tables/rcn_vs_nbp_quarterly.csv`)

| Segment | Level ratio (RCN mean / NBP mean) | corr q/q | corr y/y |
|---|---:|---:|---:|
| secondary | 1.04 | 0.89 | 0.975 |
| primary_developer | 0.98 | 0.04 | 0.45 |

**Secondary market**: excellent agreement – the cleaned register reproduces NBP's series.

**Primary market**: similar level but different timing. RCN records the date of the *final*
notarial deed (transfer of ownership after completion), while NBP records developer contract
prices when they are agreed. The best fit is with NBP leading by ~6 quarters (y/y corr 0.54).
Consequences: the primary RCN series is a lagged "completion-date" price; the 2025 surge in
primary deeds (H2 2025) is most likely completions of flats bought in the 2023 boom
("Bezpieczny Kredyt 2%"). Use secondary-market data for timely price dynamics, or model the lag.

## Remaining caveats
- Coverage effectively starts in **April 2006** (only 27 deeds in 2006Q1) – start quarterly series at 2006Q2.
- 2006–2014 has fewer records (~6–13k/yr in the clean sample vs 21–30k from 2016), so
  the register is probably less complete there. Level comparisons still match NBP.
- Transaction type is unknown before 2016; non-market sales there are caught only by the
  party, share and outlier rules.
- 2026 is a snapshot (deeds up to mid-August; the last months are incomplete).
- Prices are nominal, gross (incl. VAT on the primary market).
