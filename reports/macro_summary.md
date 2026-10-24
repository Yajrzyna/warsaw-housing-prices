# Warsaw flats – macro drivers and affordability

Reproduce: `python scripts/08_macro.py` (add `--download` to refresh sources; ~15 s).
Data: `data/processed/macro_{quarterly,annual}.parquet`; tables `reports/tables/mac_*.csv`; figures
`reports/figures/mac_*.png`.

## Sources
| Series | Source | Notes |
|---|---|---|
| NBP reference rate | NBP decision archive (`static.nbp.pl`) | full history to 2026-03-05 (3.75%) |
| 3-month money-market rate (≈ WIBOR 3M) | Eurostat `irt_st_m` | monthly to 2026-08 |
| Mortgage rate | **proxy** = 3M rate + 2.0 pp | NBP's statistics on rates for new loans are behind bot protection; most Polish mortgages are WIBOR + margin |
| Average gross wage, Warsaw | GUS BDL var 64428 | annual; interpolated log-linearly to quarters, 2026 extrapolated at 2025 growth |
| Dwellings completed / started | GUS BDL 748585 / 747729 | annual, Warsaw |
| Population | GUS BDL 72305 | level break in 2020 (census rebasing) |
| CPI | Eurostat HICP (`prc_hicp_minr`) | |

## 1. Affordability (typical flat = 50 m² at the median resale price per m²)

| | 2007Q3 (peak) | 2015Q4 – 2016Q1 | 2024Q2 | **2026Q1** |
|---|---:|---:|---:|---:|
| Years of gross wages | 8.6 | 5.3 – 5.4 | 6.6 | **5.6** |
| m² per monthly gross wage | 0.49 | 0.78 – 0.79 | 0.63 | **0.75** |
| Mortgage payment / gross wage (80% loan, 25 yrs) | 59% (2008Q2) | 26% | 48% | **34%** |

- **Prices outran wages in 2016–2024, and wages have caught up since.** From 2015 to 2025, Warsaw
  wages rose 109% nominal (+36% real) and resale prices rose 132% (+51% real). But after the 2024Q3
  price peak, wages kept rising 9–11% a year while prices were flat. In 2026Q1 the price-to-wage
  ratio is back near its 2016–19 level.
- **Mortgage burden is driven by rates.** The payment share doubled from 25% (2020Q4, record-low
  rates) to about 50% (2022Q3) without any fall in price-to-wage. With cuts since mid-2025 it is
  down to 34%, still above the 26–29% of 2014–2021.
- **Districts (2025):** a 50 m² resale flat costs 4.5 years of the Warsaw average gross wage in
  Rembertów and Wesoła (payment 31%) and 7.5 years in Śródmieście and Żoliborz (51%).
- Caveats: gross (not net) wages of one earner. In 2006–08 many mortgages were in Swiss francs at
  far lower rates, so the PLN-based payment share overstates the burden in those years.

## 2. Drivers of price growth (quarterly, 2008–2026)

Nominal year-on-year hedonic price growth on the year-on-year change in the mortgage rate (lagged),
wage growth and a programme dummy. Newey–West standard errors (4 lags), ~74 quarters.

| Spec | Mortgage rate change, lagged | Wage growth | Sales growth (lag 2) | Programme dummy | R² |
|---|---:|---:|---:|---:|---:|
| A (rate lag 2) | −1.96*** | +2.05*** | – | +2.5 (n.s.) | 0.53 |
| B (rate lag 3) | −2.15*** | +1.81*** | – | +8.3*** | 0.60 |
| C (rate lag 2 + sales) | −1.62** | +2.03*** | +0.08** | +2.0 (n.s.) | 0.57 |

- **A 1 pp rise in mortgage rates lowers annual price growth by ~2 pp, 2–3 quarters later.**
  **1 pp of extra wage growth adds ~2 pp**, which partly also captures inflation and the cycle.
- The raw correlation with rate changes is weak in nominal terms, because the 2022 hikes coincided
  with 15%+ inflation. The rate effect appears once wage growth is controlled for.
- Sales volumes lead prices by ~2 quarters (correlation +0.24 at lag 2).
- I analysed nominal rather than real series on purpose: deflating prices and wages by the same
  CPI would make an inflation shock move both "real" series together and manufacture a correlation.
- The programme dummy is not robust (n.s. in A and C). With ~74 quarters and several regimes these
  are descriptive associations, not a structural model.

## 3. Events

- **COVID-19 (2020Q2):** resales fell 35% in one quarter (3,200 → 2,100) and recovered within two
  quarters. Prices did not fall (nominal growth stayed ~+10% y/y).
- **Rate hikes (Oct 2021 – Sep 2022, reference rate 0.1% → 6.75%):** resales fell from 3,900 to
  2,700 a quarter by 2022Q4. Nominal growth slowed from +15% to +4% y/y (2023Q1), and real prices
  fell ~10% y/y with inflation at 15–18%.
- **"Bezpieczny Kredyt 2%" (applications Jul 2023 – early 2024):** subsidised 2% fixed-rate loans
  for first-time buyers up to 45, capped at PLN 500k (single) / 600k (couple).
  - Resale volumes +29% (2023Q2 → 2023Q4); prices +25% in a year (the 2024Q2 y/y peak).
  - **Size-band hedonic indices** (resale, H3 fixed effects):

    | | before 2022Q2→2023Q2 | during 2023Q2→2024Q2 | after 2024Q2→2025Q2 |
    |---|---:|---:|---:|
    | small < 45 m² | +7.4% | +25.8% | −3.0% |
    | mid 45–65 m² | +4.7% | +25.9% | −0.1% |
    | large ≥ 65 m² | +6.5% | +19.8% | +3.8% |
    | **small − large** | +0.9 pp | **+6.0 pp** | **−6.8 pp** |

    Loan-capped buyers compete for cheaper, smaller flats, which rose ~6 pp more. The gap fully
    reversed within a year of the programme's end. The share of resales priced ≤ PLN 700k (loan +
    typical deposit) fell from 61% to 37% in a year.
  - Developer deeds lag contracts, so the programme shows up in **developer deeds in 2025**
    (15,086, double 2024) as those contracts completed. Starts fell in 2025 (14,399).
  - The time pattern is consistent with the programme but not proof of cause: the rate cuts of late
    2023 and the general boom affected all sizes.

## 4. Housing supply (GUS, Warsaw)
Completions averaged ~21k/yr in 2016–2020, then 15–16k/yr in 2022–2024, and rose to 19.8k in
2025. Starts peaked at 24k (2018), fell to 13.5k after the 2022 rate shock, and were 14.4k in 2025.
That points to a thinner completion pipeline for 2026–27.
