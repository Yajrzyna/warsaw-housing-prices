"""Short-term forecasts of the Warsaw resale price index (hedonic, nominal).

Target: y = log(index). Models forecast y_{T+h}, h = 1..8 quarters, using only data up to the
forecast origin T.

  naive   no change: y_T
  drift   y_T + h x mean quarterly growth of the previous 20 quarters
  ar      AR(2) on quarterly growth, iterated
  arx     AR(2) on quarterly growth + real mortgage rate 3 quarters earlier + wage growth
          (y/y, 1 quarter earlier). Future values of the drivers come from a scenario;
          in the backtest the scenario is "unchanged since the origin", so there is no look-ahead.
  combo   average of ar and arx

backtest()  rolling origin 2012Q1..(2026Q1 - h): errors in % of the index level
forecast()  from the last observed quarter, per rate scenario, with empirical 80% intervals
            from the spread of the backtest errors of the same model and horizon (bias removed)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

H = 8
MODELS = ("naive", "drift", "ar", "arx", "combo")


def _design(d: pd.DataFrame, exog: bool) -> pd.DataFrame:
    X = pd.DataFrame({"const": 1.0, "g1": d["g"].shift(1), "g2": d["g"].shift(2)}, index=d.index)
    if exog:
        X["rr3"] = d["real_rate"].shift(3)
        X["wg1"] = d["wage_growth"].shift(1)
    return X


def _fit(d: pd.DataFrame, exog: bool) -> pd.Series:
    X = _design(d, exog)
    D = pd.concat([d["g"], X], axis=1).dropna()
    beta, *_ = np.linalg.lstsq(D.drop(columns="g").to_numpy(), D["g"].to_numpy(), rcond=None)
    return pd.Series(beta, index=X.columns)


def _iterate(hist: pd.DataFrame, future: pd.DataFrame, beta: pd.Series, exog: bool) -> np.ndarray:
    """Iterate growth forecasts; `future` carries scenario drivers for the forecast quarters."""
    d = pd.concat([hist[["g", "real_rate", "wage_growth"]], future[["real_rate", "wage_growth"]].assign(g=np.nan)])
    for t in future.index:
        x = _design(d.loc[:t], exog).iloc[-1]
        d.loc[t, "g"] = float(x[beta.index] @ beta)
    return d.loc[future.index, "g"].to_numpy()


def predict(hist: pd.DataFrame, future: pd.DataFrame) -> dict[str, np.ndarray]:
    """Log-level forecasts for the quarters in `future` from each model."""
    yT = hist["y"].iloc[-1]
    h = np.arange(1, len(future) + 1)
    out = {"naive": np.full(len(h), yT),
           "drift": yT + h * hist["g"].iloc[-20:].mean()}
    for name, ex in [("ar", False), ("arx", True)]:
        g = _iterate(hist, future, _fit(hist, ex), ex)
        out[name] = yT + np.cumsum(g)
    out["combo"] = (out["ar"] + out["arx"]) / 2
    return out


def prepare(q: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"y": np.log(q["hedonic"]), "g": np.log(q["hedonic"]).diff(),
                         "real_rate": q["real_rate"], "wage_growth": np.log(q["wage"]).diff(4) * 100}, index=q.index)


def _hold_future(hist: pd.DataFrame, quarters: pd.PeriodIndex) -> pd.DataFrame:
    """Scenario 'no change': drivers stay at their last observed values."""
    return pd.DataFrame({"real_rate": hist["real_rate"].iloc[-1], "wage_growth": hist["wage_growth"].iloc[-1]},
                        index=quarters)


def backtest(d: pd.DataFrame, first_origin: str = "2012Q1") -> pd.DataFrame:
    rows = []
    last = d.index[-1]
    for T in d.loc[first_origin:].index:
        hist = d.loc[:T]
        quarters = pd.period_range(T + 1, periods=H, freq="Q")
        preds = predict(hist, _hold_future(hist, quarters))
        for i, qq in enumerate(quarters):
            if qq > last:
                break
            actual = d.loc[qq, "y"]
            for m in MODELS:
                rows.append({"origin": str(T), "h": i + 1, "target": str(qq), "model": m,
                             "error_pct": (np.exp(preds[m][i] - actual) - 1) * 100})
    return pd.DataFrame(rows)


def accuracy(bt: pd.DataFrame) -> pd.DataFrame:
    g = bt.groupby(["model", "h"])["error_pct"]
    acc = pd.DataFrame({"rmse": g.apply(lambda e: np.sqrt(np.mean(e ** 2))), "mae": g.apply(lambda e: e.abs().mean()),
                        "bias": g.mean(), "n": g.size()}).reset_index()
    naive = acc[acc["model"] == "naive"].set_index("h")["rmse"]
    acc["rmse_vs_naive"] = acc["rmse"] / acc["h"].map(naive)
    return acc


def forecast(d: pd.DataFrame, bt: pd.DataFrame, model: str, scenarios: dict[str, float],
             known: pd.DataFrame | None = None) -> pd.DataFrame:
    """Index forecasts (last quarter = observed level) per scenario of the real mortgage rate.

    scenarios: name -> change (pp) of the real mortgage rate vs the last observed value, applied from
    the first quarter not yet observed in `known` (driver values observed after the index ends).
    """
    T = d.index[-1]
    quarters = pd.period_range(T + 1, periods=H, freq="Q")
    base = _hold_future(d, quarters)
    if known is not None:
        k = known.reindex(quarters).dropna(how="all")
        base.loc[k.index, k.columns] = k
    level_T = np.exp(d["y"].iloc[-1])
    rows = []
    # Centre past errors before taking quantiles: 2012-2025 was mostly a rising market, so forecasts
    # were systematically too low, and raw quantiles would give an almost upside-only band.
    err = bt[bt["model"] == model].groupby("h")["error_pct"]
    centred = bt[bt["model"] == model]["error_pct"] - err.transform("mean")
    by_h = centred.groupby(bt.loc[bt["model"] == model, "h"])
    q10, q90 = by_h.quantile(0.1), by_h.quantile(0.9)
    for name, shift in scenarios.items():
        fut = base.copy()
        unknown = fut.index if known is None else fut.index.difference(known.dropna(how="all").index)
        fut.loc[unknown, "real_rate"] = fut.loc[unknown, "real_rate"] + shift
        p = predict(d, fut)[model]
        for i, qq in enumerate(quarters):
            lvl = np.exp(p[i])
            # error = forecast/actual - 1  ->  actual = forecast / (1 + error)
            rows.append({"scenario": name, "q": str(qq), "h": i + 1, "index": lvl,
                         "lo80": lvl / (1 + q90[i + 1] / 100), "hi80": lvl / (1 + q10[i + 1] / 100),
                         "change_vs_last_pct": (lvl / level_T - 1) * 100})
    return pd.DataFrame(rows)
