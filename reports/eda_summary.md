# Warsaw apartment transactions – exploratory analysis

Data: RCN Warszawa, 366,428 clean apartment sales (April 2006 – August 2026), features from
`data/processed/rcn_apartments_features.parquet`. Prices are nominal PLN per m², gross.
Figures: `reports/figures/eda_*.png`. Every figure has a table twin in `reports/tables/eda_*.csv`.
Reproduce: `python scripts/04_eda.py`.

## Key findings

1. **Prices more than doubled since 2016, after a decade of stagnation.** Secondary-market
   median: 5,150 PLN/m² (2006Q2) → 8,240 (2008Q3 peak) → 6,980 (2013Q2 trough) → flat to 2016
   (~7,300) → 9,650 (2019Q4) → 11,900 (2021Q4) → 14,350 (2023Q4) → **17,050 peak (2024Q3)** →
   16,460 (2025Q4). Since the 2024 peak, prices have plateaued or dipped slightly in nominal terms.
   The RCN series tracks NBP's closely throughout (`eda_02`).
2. **Every district rose 106–149% (2016 → 2025, secondary medians; Warsaw overall +121%).**
   Cheaper districts caught up: Praga-Północ +149%, Targówek +145%, Włochy +142% (Targówek
   and Bródno got M2 stations in 2019/2022). The slowest were Ochota +106%, Śródmieście +108%
   and Ursynów +110%. The spread between the dearest and cheapest district narrowed from 1.81× to
   1.66× (`eda_04`, `eda_06`).
3. **2025 price level by district:** Żoliborz 21,010 and Śródmieście 20,920 PLN/m² at the top;
   Wesoła 12,690 and Rembertów 12,650 at the bottom (`eda_03`, map `eda_05`: a high-price core
   along the left bank from Żoliborz through Śródmieście to upper Mokotów, plus Wilanów).
4. **Raw price gradients (2023–25, secondary, relative to the quarter's median; not yet controlled
   for other attributes – that's Phase 6):**
   - distance to centre: +31% within 1 km, ≈0 at 5–6 km, −15% at 10–12 km, −26% beyond 14 km;
   - distance to metro: +4–8% within 2 km, −8% at 3–4 km, −21% at 6–10 km (confounded with
     distance to centre);
   - floor: ground floor −5%; floors 2–4 ≈ typical; 16+ (towers) +8 to +25%;
   - size: **U-shape** – studios under 25 m² +19% per m², 50–60 m² −6%, 100 m²+ +10%.
5. **New-builds are shrinking.** Among developer sales, the share under 40 m² went from 10% (2007)
   → 17% (2016) → 25% (2024) → **33% (2025)**, and the median area from 66 to 46 m². The
   secondary market is stable at ~48 m². Since small flats cost more per m², a raw median price
   per m² for the primary market is pushed up by the mix, so Phase 5 needs mix-adjusted and
   hedonic indices (`eda_08`).
6. **Developer prices sit below secondary prices at the deed date** – 80–87% of the secondary
   median in 2019–2024. Reasons: outer locations, and prices fixed at contract 1–3 years before the
   deed. The ratio rose to ~91% in 2025 and ~102% in 2026Q1 as flats contracted at 2023–24 prices
   were completed (`eda_09`).
7. **Volumes:** secondary sales rose from ~10k/yr (2016) to ~14k/yr (2021–25). Developer deeds
   doubled from 7.4k (2024) to **15.1k (2025)**, concentrated in H2 2025 – completions of the
   2023 "Bezpieczny Kredyt 2%" boom. Sales by natural-person "primary" sellers (mostly conversions)
   peaked in 2016–17 (~6k/yr) and are now ~1.7k (`eda_01`).

## Data notes discovered here
- The register effectively starts in **April 2006** (27 deeds in 2006Q1), so all quarterly series
  start at 2006Q2.
- 2026 deeds are still being registered: volumes stop at 2025Q4, prices run to **2026Q1**.
  2026Q2 has only ~36 registered resales so far (vs ~3,400 per quarter), so it is excluded.
  An earlier draft reported a 2026Q2 median of 15,940 – it came from those 36 sales and was
  withdrawn.

## Figures
| File | Content |
|---|---|
| eda_01_volume_by_segment | sales per quarter by segment |
| eda_02_price_trend | median price/m² – secondary vs developer, NBP benchmark |
| eda_03_district_price_level | 2025 median by district |
| eda_04_district_growth | 2016 vs 2025 by district (dumbbell) |
| eda_05_map_price_level | H3 hexagon map of 2024–25 price level |
| eda_06_map_relative_growth | hexagon growth 2016–17 → 2024–25 relative to Warsaw |
| eda_07_price_gradients | centre, metro, floor, size gradients |
| eda_08_composition | median area, share of small flats |
| eda_09_primary_secondary_gap | developer ÷ secondary median |
