# Feature table – `data/processed/rcn_apartments_features.parquet`

Built by `scripts/03_build_features.py` (`src/real_estate/features/build_features.py`) from the
clean RCN sample (366,428 apartment sales). All columns of the clean table are kept; the
features below are added. Coordinates are EPSG:2180 (metres).

## Sources

| Source | Used for | Notes |
|---|---|---|
| RCN (clean sample) | unit attributes, vintage proxy, local price level | |
| BDOT10k Warsaw (GUGiK, GPKG, state 2025/26) – `data/raw/geo/bdot10k/` | buildings (storeys, function), metro entrances, rail stations, bus/tram stops, parks, forests, Vistula | current state applied to all years |
| `data/external/metro_stations.csv` | metro opening dates | from Wikipedia (section dates); Lazurowa = planned (autumn 2026) |

## Unit

| Column | Definition |
|---|---|
| log_area | ln(area_m2) |
| area_per_room | area_m2 / rooms (NaN if rooms unknown) |
| size_band | <30, 30-40, 40-50, 50-60, 60-80, 80-100, 100+ m² |
| is_ground_floor | floor == 1 (RCN numbering: 1 = parter) |
| is_top_floor | floor == storeys (storeys > 1) |
| floor_rel | (floor−1)/(storeys−1), 0 = ground, 1 = top; NaN if BDOT storeys < floor (5% of rows) |
| month, t_quarter | calendar month; quarters since 2006Q1 (0 = 2006Q1) |

## Building (BDOT10k match)

Sale point → BDOT building containing it (80%), else nearest building within 40 m (19%);
garages, sheds, substations etc. are never matched. 0.4% unmatched.

| Column | Definition |
|---|---|
| bdot_building_id, building_function | BDOT id and predominant function (94% `budynek wielorodzinny`) |
| storeys | number of storeys (LICZBAKONDYGNACJI) |
| height_class | low 1–4 / mid 5–10 / high 11+ storeys |
| footprint_m2 | building footprint area |
| building_match, building_match_dist_m | `within` / `nearest`, distance of the match |

## Building vintage (proxy – no public building-age source was reachable)

| Column | Definition |
|---|---|
| building_key | BDOT building id, else `addr:<street>|<number>` |
| build_year_proxy | year of the building's first recorded sale **if** that sale is a developer primary sale, the building has ≥2 developer sales and the year is ≥ 2007 |
| vintage | `new_2007plus` (proxy available) / `older_or_unknown` |
| age_at_sale | year − build_year_proxy (≥0) |
| bld_first_sale_year, bld_n_sales | first sale year and number of sales in the building |

Check: 92% of developer sales fall in `new_2007plus`; 92% of "primary_other" sales (natural-person
sellers) in `older_or_unknown`, which fits conversions of old cooperative flats. For pre-2007
buildings age is **unknown**. EGiB construction years exist but Warsaw's WFS was not reachable,
and OSM has construction years for only ~430 buildings.

## Location

| Column | Definition |
|---|---|
| dist_centre_km | to Palace of Culture (PKiN) |
| right_bank | district east of the Vistula |
| dist_vistula_km | to the Vistula water polygon |
| dist_park_km | to nearest park / square / botanical / zoo garden |
| dist_forest_km | to nearest forest ≥ 2 ha |
| dist_rail_km | to nearest rail station / stop (static, current network) |
| n_stops_500m | bus + tram stops within 500 m (static, current network) |
| h3_r8, h3_r9 | H3 cells (~0.7 km² / ~0.1 km²) for spatial fixed effects / aggregation |

## Metro (time-aware)

| Column | Definition |
|---|---|
| dist_metro_km | nearest station **open on the deed date** |
| nearest_metro, nearest_metro_line | that station |
| dist_metro_2026_km | nearest station in today's network (ignores sale date) |
| dist_metro_incl_planned_km | incl. stations under construction (Lazurowa) |
| metro_within_1km | dist_metro_km ≤ 1 |
| metro_opened_later_within_1km | a station within 1 km opened *after* the sale (25k sales) – treatment group for M2-opening analyses |

## Neighbourhood price level (leakage-safe)

For each sale: the 20 nearest sales (≤1.5 km) with deed dates in the 365 days **before the start
of the sale's quarter**. It never uses the sale itself or any sale from the same or a later quarter.

| Column | Definition |
|---|---|
| nbhd_log_price_m2 | median log price/m² of those neighbours (NaN if < 5) |
| nbhd_n | neighbours used |
| nbhd_median_dist_m | their median distance |

Correlation with log price/m²: 0.87 (distance to metro −0.41, to centre −0.31).

## Caveats
- BDOT, stops, rail and parks reflect today's city; only the metro is time-aware.
- Building age is known only for buildings first sold new from 2007 on.
- Runtime: ~1 min of CPU. On a machine low on RAM, swapping can stretch it to ~10 min of wall-clock time.
