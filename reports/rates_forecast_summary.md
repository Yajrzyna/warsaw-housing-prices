# Warsaw flat prices and interest rates · short-term outlook

Reproduce: `python scripts/09_rates_forecast.py` (after scripts 05 and 08; ~10 s).
Tables `reports/tables/rt_*.csv`, `fc_*.csv`; figures `reports/figures/rt_*.png`, `fc_*.png`.
Data: resale hedonic index (2006Q2–2026Q1); mortgage rate proxy = 3M money-market rate + 2 pp;
real rate = mortgage rate − y/y HICP inflation; Warsaw average gross wage (GUS).

## The mechanism
A mortgage is a long, large loan, so its payment is very sensitive to the rate. At ~6% over
25 years, **+1 pp cuts the loan a fixed payment can carry by ~9%** (annuity factor 155 → 142 per PLN
of monthly payment). If buyers borrowed to their limit and prices followed borrowing capacity, a
1 pp rise would take ~9% off prices. The data show how far that holds.

## 1. What the data show

**Borrowing capacity swings far more than prices** (`rt_02`). The max loan at a 40% payment-to-wage
ratio rose to 165 (2015 = 100) in 2021Q3 at record-low rates, then **fell 43% to 95 by 2022Q3**
as rates went from 2% to 9%. Prices kept rising through it. The price of a median 50 m² flat
relative to that capacity went:

| 2008Q2 | 2016–2021 | 2022–2024 | 2026Q1 |
|---:|---:|---:|---:|
| 1.49 | 0.62–0.72 | up to 1.25 | 0.85 |

So Warsaw prices do not track mortgage capacity one-for-one. The likely dampeners: many buyers pay
cash (investors, households with savings), sellers withdraw rather than cut prices (volumes
adjust), and in 2022 inflation made property a hedge.

**Real rates matter, nominal rates mislead** (`rt_01`). Annual price growth vs the rate a year
earlier (2008–2026):

| Rate a year earlier | Correlation with price growth |
|---|---:|
| Nominal mortgage rate | −0.27 |
| **Real mortgage rate** (rate − inflation) | **−0.65** |

In 2022–23 the nominal rate hit 9% but inflation was 15–18%. The real rate reached −7.1% (2023Q1),
and prices then rose 20–25% y/y in 2024. With real rates back at +3–5% since 2024, prices stalled.

**Timing: volumes first, prices later** (`rt_03`, local projections with inflation, wage-growth and
own-lag controls, 90% CI):

| After +1 pp in… | outcome | 1 year | 2 years | 3 years |
|---|---|---:|---:|---:|
| nominal mortgage rate | price (nominal) | −0.4% [−2.2, +1.4] | +2.1% [−1.8, +5.9] | +2.1% [−3.7, +7.8] |
| real mortgage rate | price (nominal) | +0.5% [−1.2, +2.2] | **−2.0%** [−3.4, −0.6] | **−6.7%** [−10.2, −3.2] |
| nominal mortgage rate | resale volume | **−9.0%** [−14.5, −3.5] | −3.3% [−6.4, −0.2] | −5.2% [−12.0, +1.7] |

- A real-rate rise lowers prices with a 2–3 year lag; the effect is significant at 2 and 3 years.
- Nominal rate rises hit **transactions** within a year (−9%) but not prices: sellers wait.
- A higher real rate also comes with *more* resales. Real rates rise when inflation falls, which is
  when activity recovers. Shocks are not clean here: the central bank sets rates in response to
  inflation and the economy.

**Direction** (Granger, quarterly changes): nominal rate changes help predict price changes
(p = 0.016 with 2 lags, 0.048 with 4); prices do not predict rates (p = 0.15 / 0.62).

**The link is unstable** (`rt_04`): the 5-year rolling correlation of price growth with rate
changes swings between −0.75 and +0.75 across regimes (2013–17 negative; 2018–20 positive when
both were flat or rising; negative again 2023–25).

**Long run** (Engle–Granger, 2007–2026): log real price ~ log real wage + nominal rate gives a
*positive* rate coefficient (+3.3% per pp; cointegration p = 0.03). That's a sign of the inflation
confound, not evidence that higher rates raise prices. With the real rate the coefficient is
−0.4% per pp and not cointegrated. With 20 years of data and three regimes, a stable long-run
rate elasticity isn't identified.

**Bottom line:** rates matter in Warsaw, but through **real** rates, with **2–3 year lags**, and
mainly through **transaction volumes** in the short run. The "1 pp ≈ −9%" mortgage math is an
upper bound that the market has not followed. The co-dependence you'd expect exists, but it's
much looser and slower than the mortgage arithmetic.

## 2. Forecast (Phase 8)

**Models** (target: log resale hedonic index; each re-estimated at every origin with data
available then): no change, trend (last 5 years), AR(2) on quarterly growth, ARX (AR + real
mortgage rate 3 quarters earlier + wage growth), and the average of AR and ARX.

**Backtest 2012–2026** (RMSE of the index level; `fc_01`):

| Model | 1 quarter | 4 quarters | 8 quarters |
|---|---:|---:|---:|
| No change | 2.4% | 8.3% | 14.9% |
| Trend | 1.9% | 6.5% | **9.9%** |
| AR | 1.5% | 6.5% | 12.6% |
| AR + rates & wages | 1.6% | 6.9% | 13.1% |
| **Average (used)** | **1.5%** | **6.3%** | 11.8% |

- Adding rates and wages did **not** improve forecasts out of sample (drivers held at their last
  value, as a forecaster at the time would have to).
- All models were biased low (−9% at 8 quarters for the average model), because 2012–2025 was
  mostly a rising market. The trend model wins at long horizons for the same reason; it would
  fail at a turning point.

**Outlook from 2026Q1** (`fc_02`; resale hedonic index, 2015 = 100, last = 233):

| Scenario (real mortgage rate) | 2026Q4 | 2027Q4 | 2028Q1 |
|---|---:|---:|---:|
| Rates hold (real ~3%) | +0.8% | +1.8% | **+2.0%** |
| 1 pp lower | +0.8% | +2.4% | +2.9% |
| 1 pp higher | +0.8% | +1.2% | +1.1% |

The 80% range at 2028Q1 is **217–265** (about −7% to +14%), taken from the spread of past errors
with their bias removed. The uncertainty dwarfs the scenario differences, so the forecast is
essentially **flat nominal prices, falling slightly in real terms** at ~3% inflation.
Cross-check: NBP's hedonic index fell 1.3% q/q in 2026Q2 against our +0.4%, which is inside the
one-quarter range.

## Caveats
- The mortgage rate is a proxy (3M + 2 pp); NBP statistics on rates for new mortgages were not
  reachable. Many 2006–08 loans were in CHF.
- About 75 quarters and several regime changes (GFC, zero rates, the 2022 inflation shock, a
  subsidy programme). All estimates are associations; standard errors are HAC but the sample is
  short.
- The forecast ignores supply (starts fell to 14k in 2025, pointing to fewer completions in
  2026–27) and policy (a new subsidy programme would add upside).
