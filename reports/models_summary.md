# Warsaw apartment prices – hedonic, spatial and valuation models

Reproduce: `python scripts/06_models.py` (~4 min). Tables: `reports/tables/mdl_*.csv`, figures
`reports/figures/mdl_*.png`. Sample: 76,733 resale and developer sales, 2023Q1–2026Q1
(valuation model: 219,698 sales 2016–2026Q1).

## 1. What a flat's features are worth (spec B: same ~0.7 km² H3 cell)

Log price/m² on flat and building attributes, quarter effects and an **H3-cell fixed effect**
(location fully controlled). 95% CIs are clustered by building. Within-cell R² = 0.31.

| Feature | Effect on price/m² | 95% CI |
|---|---:|---|
| 1 room (vs 2, same area) | +4.3% | +2.7 to +5.9 |
| 3 / 4 / 5+ rooms (vs 2) | −6.5% / −11.6% / −14.8% | |
| Ground floor (vs 1st–3rd) | −1.8% | −2.2 to −1.4 |
| 4th–6th / 7th–9th / 10th–14th floor | +2.4% / +4.1% / +5.5% | |
| 15th floor and up (towers) | +24% | +10 to +40 |
| Top floor | −0.8% | ≈ 0 |
| Building 11+ storeys (vs 1–4) | −4.8% | mostly 1970s high-rise blocks |
| New building (first sold 2007+), 0–1 / 2–4 / 5–9 / 10+ yrs | +12.8% / +16.8% / +14.6% / +6.6% | vs older/unknown stock |
| Cellar / storage room | −1.3% | small, cheaper-building proxy |
| **Developer sale vs resale** | **−12.7%** | same area and features |

Floor-area elasticity is small once rooms are controlled (the U-shape of price/m² over size runs
through the rooms terms). The developer discount mainly reflects **timing**: the deed price was
agreed 1–3 years earlier, and new flats are often sold without interior finish.

## 2. What location is worth (spec A: explicit location variables, R² = 0.53)

| Location | Effect | | Location | Effect |
|---|---:|---|---|---:|
| Centre 0–2 km (vs 6–8 km) | **+32.8%** | | Metro < 500 m (vs > 2 km) | **+10.8%** |
| Centre 2–4 km | +14.7% | | Metro 500 m–1 km | +8.0% |
| Centre 4–6 km | +9.7% | | Metro 1–2 km | +5.3% |
| Centre 8–10 km | −3.5% | | Vistula within 1 km | +5.3% |
| Centre 10–12 km | −5.9% | | Park / forest within 300 m | +1.4% (n.s. / borderline) |
| Centre 12 km+ | −10.4% | | **Right bank of the Vistula** | **−10.9%** |

The location variables overlap (central areas are near metro and the river), so read them
together. They are associations, not causal effects.

**Location value surface** (spec B cell effects, 398 cells with ≥ 20 sales,
`mdl_03_location_value_map.png`). It runs from −29% (5th percentile) to +37% (95th) against the
Warsaw average. Median by district: Śródmieście +38%, Żoliborz +26%, Wola +13%, Ochota +11%,
Mokotów +9% … Białołęka, Wawer, Wesoła −20%, Rembertów −26%.

## 3. Spatial diagnostics

| Model | Moran's I of residuals (k = 10) |
|---|---:|
| Spec A (location variables) | 0.37 |
| Spec B (H3-cell fixed effects) | 0.20 |

Cell fixed effects remove about half of the spatial autocorrelation. The rest is mostly
**within-building** similarity (flats in one building share unobserved quality), all significant
at p < 0.01.

**District heterogeneity** (spec B per district, `mdl_district_heterogeneity.csv`): premia vary
across the city. The ground-floor discount runs from ≈0 (Białołęka, Wilanów) to −5% (Śródmieście).
The high-rise discount is largest in Wilanów (−14%), Żoliborz (−11%) and Praga-Północ (−11%).
The new-build premium (2–4 yrs) is largest where the old stock is oldest: Śródmieście and Ochota
+44%, Żoliborz +34%, Praga-Północ +32%.

## 4. Automated valuation model (AVM)

Target: log price/m² net of the segment's price index. Predictions use the index **published one
quarter earlier**, so the test never sees the quarter it predicts.

| Model | Future sales (train ≤ 2024Q4, test 2025Q1–2026Q1) | | Unseen areas (5-fold, ~36 km² blocks) | |
|---|---:|---:|---:|---:|
| | median error | within ±10% | median error | within ±10% |
| Hedonic, location variables | 11.4% | 44% | 13.0% | 40% |
| Hedonic, H3 fixed effects | 10.3% | 49% | – | – |
| Gradient boosting | 9.0% | 55% | 11.8% | 44% |
| **Gradient boosting + local prices** | **8.4%** | **58%** (87% within ±20%) | – | – |

- The biggest gain comes from the leakage-safe **local price level** (median of the 20 nearest
  sales in the previous year). Next come district, building age, distance to centre and floor area.
- Accuracy is similar for resales (8.7%) and developer sales (8.5%). It's best in homogeneous
  areas (Białołęka, Wilanów, Bemowo: 7.0–7.2%) and worst in Śródmieście (11.8%), Wawer and
  Rembertów (~11%), where building quality varies most.
- Limits: the register has no information on condition, finish, view or orientation, so a
  ~8% median error is close to what transaction-only data allow.
- LightGBM needs `libomp` (not installed), so scikit-learn's HistGradientBoosting is used.

## 5. Price calculator (dashboard)

The dashboard's calculator uses spec B refitted on **2024Q2–2026Q1** with **per-segment quarter
effects**: developer deed prices rose 16% over those quarters while resale prices were flat, and a
shared time effect over-valued resales by ~5%. On 2026Q1 sales the calculator's median
predicted/actual ratio is 1.01 for both segments. The district medians range from −4% to +5%.
Its out-of-sample accuracy is quoted from the temporal test of the same model type (median error
10.3%, 49% within ±10%, 81% within ±20%).
