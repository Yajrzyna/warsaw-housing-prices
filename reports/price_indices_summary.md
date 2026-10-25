# Warsaw apartment price indices

Reproduce: `python scripts/05_price_indices.py` (~40 s). Data: clean RCN sample, 2006Q2–2026Q1
(2026Q2 excluded – only ~36 resales registered so far). All indices: 2015 average = 100, nominal
unless marked *real*.

| Output | Content |
|---|---|
| `data/processed/price_indices_quarterly.parquet` | long table: segment × method × quarter (log index, index, n, SE / coverage) |
| `reports/tables/price_indices_quarterly.csv` | wide, one column per `segment__method` |
| `reports/tables/price_indices_district_annual.csv` | annual hedonic index per district (secondary) |
| `reports/tables/price_index_diagnostics.csv` | growth, peaks, volatility, agreement with NBP |
| `reports/figures/idx_01 … idx_05` | figures (+ CSV twins in `reports/tables/`) |

## Methods

| Method | How | Controls for |
|---|---|---|
| **Median** | median price/m² per quarter | nothing |
| **Mix-adjusted** | medians in 90 strata (district × 5 size bands), fixed 2016–25 weights, chained geometric | location and size mix |
| **Hedonic (headline)** | rolling 5-quarter time-dummy regression of log price/m² on size (log + square), rooms, floor bands, top floor, building height, new-build age, cellar, log distance to metro, **H3-cell fixed effects**; chained with a movement splice (the rolling-time-dummy method statistical offices use) | location (~0.7 km² cells) and unit/building attributes; published quarters are never revised |
| Hedonic, adjacent period | the same with a 2-quarter window | sensitivity |
| **Repeat sales** | Case–Shiller 3-stage weighted regression on 33,198 resale→resale pairs (same flat = street + number + area + floor + rooms, ≥ 6 months apart; 284 pairs with robust \|z\| > 4 dropped) | the flat itself (depreciation and renovation stay in) |
| Real | hedonic ÷ HICP Poland (Eurostat `prc_hicp_minr`, 2015 = 100) | inflation |

Synthetic tests (`tests/test_indices.py`) confirm that hedonic, mix-adjusted and repeat-sales
recover a known price path when the mix of units sold shifts, and that the plain median does not.

## Results – secondary market

| Method | 2007 → 2015 | 2015 → 2025 | per year 2015–25 | Peak | Peak → 2026Q1 |
|---|---:|---:|---:|---|---:|
| **Hedonic** | −12.4% | **+131.5%** | 8.8% | 2024Q3 | −1.3% |
| Hedonic, adjacent | −10.3% | +134.5% | 8.9% | 2024Q3 | −1.4% |
| Mix-adjusted | −11.8% | +130.1% | 8.7% | 2024Q3 | −3.0% |
| Median | −12.0% | +126.7% | 8.5% | 2024Q3 | −3.4% |
| Repeat sales | −10.6% | +126.3% | 8.5% | 2024Q4 | −0.4% |
| *Hedonic, real* | *−27.0%* | ***+50.6%*** | *4.2%* | 2024Q3 | *−5.7%* |
| NBP hedonic (benchmark) | −10.1% | +96.2% | 7.0% | 2026Q1 | 0.0% |

Growth uses annual averages (2015 vs 2025). Quarterly correlation with NBP: 0.77–0.80 (q/q),
0.94–0.95 (y/y).

### Findings
1. **Four independent RCN methods agree: +126% to +135% in 2015–2025** (about 8.5–9% a year). The
   hedonic index is the headline. Quality adjustment matters little on the secondary market in this
   period – median and hedonic differ by only 5 points over 10 years.
2. **Two regimes.** After the 2006–07 boom (y/y up to +59% in 2007Q2), prices peaked in **2007Q4**,
   fell 20% to a trough in **2013Q1** and stayed flat until 2016. Growth then accelerated to
   ~10%/yr (2018–21) and to **+24% y/y in 2024Q2**, with the peak in **2024Q3**. Since then the index
   has been flat (y/y +0.2% in 2026Q1).
3. **In real terms the boom is much smaller:** +50.6% for 2015–2025. Real prices are only ~11% above
   the 2007Q4 real peak at their 2024Q3 high, and ~5% above it in 2026Q1 (real index 142 → 157 → 148).
   They have fallen 5.7% since the peak.
4. **NBP's hedonic index rises much less (+96%).** Our median, mix-adjusted, hedonic and
   repeat-sales indices – and **NBP's own average transaction price (+123%)** – all show more. The
   repeat-sales index compares the same flats and needs no hedonic model, so the gap most likely
   comes from NBP's sample (agent-reported BaRN data) and its quality adjustment, not from RCN.
   The two series still move together (y/y correlation 0.95).
5. **Developer (primary) market:** hedonic +111% for 2015–2025 (median +106%, mix-adjusted +117%),
   i.e. **less than resale prices**. RCN records the price at the *final* deed, usually 1–3 years
   after the contract price was agreed. The raw developer median is noisy (q/q std 5.4 pp vs 3.2 pp
   hedonic): in 2025Q4 a wave of small-flat completions lifted it 18% in one quarter while the
   hedonic index rose 4% (`idx_04`).
6. **Districts (2015 → 2025, hedonic, secondary):** Wilanów +156%, Praga-Północ +152%, Żoliborz
   +140%, Praga-Południe and Wesoła +138% lead. Ursus +109%, Bemowo +118% and Ursynów +119% lag
   (Warsaw +131%). Quality adjustment changes the ranking compared with raw medians (EDA):
   Włochy's +142% median growth becomes +123% hedonic – its resale mix shifted toward newer flats.

## Robustness notes
- **Window length:** 2- vs 5-quarter windows differ by 3 points over 10 years.
- **Building fixed effects instead of H3 cells** give +163% (2015Q4–2025Q4). The gap persists in old
  buildings only (bldg FE +170% vs repeat sales +123% vs H3 +128%), so it isn't driven by new-build
  fit-out. It isn't age–period collinearity either (dropping age controls changes nothing).
  Within-building comparisons seem to pick up a systematic drift (e.g. better-kept or renovated
  flats being resold later) that compounds when chained over 80 windows. **Not adopted** – worth a
  closer look in the modelling phase.
- **Repeat-sales key:** the EGiB unit id changes when buildings are renumbered, and it is missing
  before 2016, so pairs use an address-based key. Twin flats (same building, floor, area, rooms) can
  be pooled; they are near-perfect quality matches, so this adds noise, not bias.
- **Error variance** does not grow with holding period in the repeat-sales data (stage-2 slope
  ≈ 0), so the Case–Shiller weights are close to uniform.
- **Uncertainty:** hedonic quarter-to-quarter step SE ≈ 0.5 pp (median over quarters); repeat-sales
  level SE ≈ 1.1% around 2015.
