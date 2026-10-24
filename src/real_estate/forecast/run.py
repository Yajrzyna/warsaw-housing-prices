"""Phase 8 outputs: price-rate dependence and short-term forecasts.

  reports/tables/rt_*.csv   correlations, rolling correlation, Granger, long-run, impulse responses, capacity
  reports/tables/fc_*.csv   backtest errors, accuracy, forecasts per scenario
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from real_estate.config import REPORTS_TABLES_DIR as T
from real_estate.forecast import models as fm
from real_estate.ingest import macro
from real_estate.ingest.eurostat import hicp_quarterly
from real_estate.macro import rates as rt

FORECAST_MODEL = "combo"
SCENARIOS = {"rates hold": 0.0, "real rate −1 pp": -1.0, "real rate +1 pp": 1.0}


def known_after_index(q: pd.DataFrame) -> pd.DataFrame:
    """Driver values already observed for quarters after the last index quarter (rates & CPI publish
    faster than deeds are registered)."""
    m3 = macro.interbank_3m()
    m3q = m3.groupby(m3.index.asfreq("Q")).agg(["mean", "size"])
    hicp = hicp_quarterly()
    rows = {}
    for qq in pd.period_range(q.index[-1] + 1, periods=fm.H, freq="Q"):
        if qq in m3q.index and m3q.loc[qq, "size"] == 3 and qq in hicp.index and (qq - 4) in hicp.index:
            infl = (hicp[qq] / hicp[qq - 4] - 1) * 100
            rows[qq] = {"real_rate": m3q.loc[qq, "mean"] + macro.MORTGAGE_MARGIN_PP - infl}
    return pd.DataFrame.from_dict(rows, orient="index")


def main() -> None:
    q = rt.rate_frame()
    cap = q[["mortgage_rate", "inflation", "real_rate", "wage", "max_loan", "capacity_index", "hedonic",
             "median_price_m2", "capacity_price_m2", "price_to_capacity"]].round(3)
    cap.index = cap.index.astype(str)
    cap.to_csv(T / "rt_quarterly.csv")
    rt.correlations(q).round(3).to_csv(T / "rt_correlations.csv", index=False)
    rc = rt.rolling_correlation(q).round(3)
    rc.index = rc.index.astype(str)
    rc.to_csv(T / "rt_rolling_correlation.csv")
    rt.granger(q).round(4).to_csv(T / "rt_granger.csv", index=False)
    rt.long_run(q).round(4).to_csv(T / "rt_long_run.csv", index=False)
    irf = rt.impulse_responses(q)
    irf.round(3).to_csv(T / "rt_impulse_responses.csv", index=False)

    d = fm.prepare(q)
    bt = fm.backtest(d)
    bt.round(3).to_csv(T / "fc_backtest_errors.csv", index=False)
    acc = fm.accuracy(bt)
    acc.round(3).to_csv(T / "fc_accuracy.csv", index=False)
    known = known_after_index(q)
    fc = fm.forecast(d, bt, FORECAST_MODEL, SCENARIOS, known=known)
    fc.round(3).to_csv(T / "fc_forecast.csv", index=False)

    print(rt.granger(q).round(3).to_string(index=False))
    print(irf[irf["h"].isin([4, 8, 12])].round(2).to_string(index=False))
    print(acc.pivot(index="h", columns="model", values="rmse").round(2))
    print("known after index:", known.round(2).to_dict())
    print(fc.pivot(index="q", columns="scenario", values="change_vs_last_pct").round(2))


if __name__ == "__main__":
    main()
