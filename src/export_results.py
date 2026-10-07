"""
Backtest results to Excel and CSV.

Called by src/backtest.py (run that). Writes
  data/outputs/backtest_results.xlsx  sheets Summary, Yearly, Daily_NAV, Daily_Weights_D, Turnover_Costs,
                                      Constituents, Definitions
  data/outputs/backtest_summary.csv   the Summary sheet
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest as bt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
XLSX_PATH = PROJECT_ROOT / "data" / "outputs" / "backtest_results.xlsx"
SUMMARY_CSV = PROJECT_ROOT / "data" / "outputs" / "backtest_summary.csv"
COMPARE_TO = "A-capped"
OEX = "^OEX (price only)"

DEFINITIONS = [
    ("Period", "2016-01-04 to 2020-06-30, daily closes. NAV starts at 100."),
    ("Timing", "News is date-only. A headline dated d is first used at the close of the next trading day (d+1); "
               "weights set at that close earn the return from d+1 to d+2. This is the only lag."),
    ("Total return %", "NAV(last) / NAV(first) - 1, after costs; adjusted closes, so dividends are included."),
    ("CAGR %", "(NAV(last) / NAV(first)) ^ (252 / number of daily returns) - 1."),
    ("Volatility %", "Standard deviation of daily returns x sqrt(252)."),
    ("Sharpe", "Mean daily excess return over the 13-week T-bill (^IRX) / std of daily returns x sqrt(252)."),
    ("Sortino", "Mean daily excess return / root mean square of negative excess returns x sqrt(252)."),
    ("Max drawdown %", "Largest fall of NAV from its previous peak."),
    ("Tracking error % vs A-capped", "Std of daily (strategy - A-capped) returns x sqrt(252)."),
    ("Information ratio vs A-capped", "Annualised mean active return vs A-capped / tracking error."),
    ("Alpha % vs A-capped", "CAPM alpha, annual: mean excess return - beta x mean A-capped excess return, x 252."),
    ("Turnover % per year (one-way)", "Half the sum of |weight changes| over all trades, per year."),
    ("Total costs %", "10 bp x the full sum of |weight changes| over the whole period (not halved)."),
    ("Constituent changes", "Number of additions over the 17 quarterly reviews (each addition has one deletion)."),
    ("Avg |tilt| %", "Mean over days of sum |w - FFMC base weight| / 2. For A and A-capped this is drift and caps."),
    ("Max single weight %", "Largest weight any stock had on any day (after trading)."),
    ("Top sector weight %", "Average over days of the largest GICS sector's weight."),
    ("Yearly", "Calendar-year return from NAV; 2016 starts 2016-01-04, 2020 is to 30 June. Mar-2020 = March only."),
    ("Signals", "Event = one company on one date; S = weighted mean sentiment (half-life 3 days, 15-day window); "
                "C = model confidence x independence x agreement; |S| < 0.15 -> 0. News-type weights not used."),
    ("Strategies", "A: FFMC Top 20, reviewed quarterly, weights drift. A-capped: A with 12% cap at reviews. "
                   "B: tilt with C = 1. C: tilt with C. D: C + band +-25%, 12% cap, 1 pp daily limit. "
                   "D-tone: D with finbert-tone scores. Alpha = 0.30."),
    ("Assumption: free float", "Free-float factors are today's values, held constant. BRK.B 0.85 is our estimate."),
    ("Assumption: universe", "Current S&P 100 members used for 2016-2020: survivorship bias flatters all strategies."),
    ("Assumption: ^OEX", "^OEX is a price index (no dividends), about 2% a year easier to beat."),
    ("Assumption: costs", "10 bp per unit of weight traded; no market impact; initial build is free."),
    ("Assumption: news", "News ends 2020-06-11, so tilts fade out over the last trading days."),
]


def avg_abs_tilt(weights: dict, base_table: pd.DataFrame) -> float:
    """Mean over days of sum |w - base| / 2."""
    w = pd.DataFrame(weights).T.fillna(0.0)
    base = base_table.reindex(index=w.index, columns=w.columns.union(base_table.columns)).fillna(0.0)
    return float((w.reindex(columns=base.columns, fill_value=0.0) - base).abs().sum(axis=1).mean() / 2)


def summary_table(navs, results, rf, base_table, sectors, reviews) -> pd.DataFrame:
    compare = navs[COMPARE_TO].pct_change()
    rows = []
    for name in navs.columns:
        perf = bt.performance(navs[name], rf, compare)
        row = {"strategy": name, "total return %": perf["total return %"], "CAGR %": perf["CAGR %"],
               "volatility %": perf["volatility %"], "Sharpe": perf["Sharpe"], "Sortino": perf["Sortino"],
               "max drawdown %": perf["max drawdown %"],
               "tracking error % vs A-capped": perf["tracking error %"],
               "information ratio vs A-capped": perf["information ratio"],
               "alpha % vs A-capped": perf["alpha % (annual)"]}
        if name in results:
            stats = bt.portfolio_stats(results[name], sectors, reviews)
            row.update({"turnover % per year (one-way)": stats["turnover % per year (one-way)"],
                        "total costs %": 100 * results[name]["daily"]["cost"].sum(),
                        "constituent changes": stats["constituent changes"],
                        "avg |tilt| %": 100 * avg_abs_tilt(results[name]["weights"], base_table),
                        "max single weight %": stats["max single weight %"],
                        "top sector weight %": stats["avg largest sector weight %"]})
        rows.append(row)
    return pd.DataFrame(rows).set_index("strategy")


def yearly_returns(navs: pd.DataFrame) -> pd.DataFrame:
    """Return per calendar year (first year from the first NAV), plus March 2020."""
    year_end = navs.groupby(navs.index.year).last()
    start = pd.concat([navs.iloc[[0]].set_axis([year_end.index[0]]), year_end.iloc[:-1].set_axis(year_end.index[1:])])
    out = 100 * (year_end / start - 1)
    out.index = out.index.astype(str)
    feb, mar = navs.loc[:"2020-02-29"].iloc[-1], navs.loc[:"2020-03-31"].iloc[-1]
    out.loc["Mar-2020"] = 100 * (mar / feb - 1)
    return out


def constituents_sheet(reviews: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"review date": reviews["review"].dt.date, "effective date": reviews["effective"].dt.date,
                         "members": reviews["members"].map(", ".join),
                         "entries": reviews["added"].map(", ".join), "exits": reviews["removed"].map(", ".join)})


def export_all(navs, results, rf, base_table, sectors, reviews) -> pd.DataFrame:
    summary = summary_table(navs, results, rf, base_table, sectors, reviews)
    d_weights = pd.DataFrame(results["D"]["weights"]).T.fillna(0.0)
    base = base_table.reindex(d_weights.index).fillna(0.0)
    weights_sheet = pd.concat([d_weights.add_prefix("D "), base.add_prefix("base ")], axis=1)
    costs = {}
    for name, res in results.items():
        costs[f"{name} turnover (one-way)"] = res["daily"]["traded"] / 2
        costs[f"{name} cost"] = res["daily"]["cost"]
    sheets = {"Summary": summary,
              "Yearly": yearly_returns(navs),
              "Daily_NAV": 100 * navs / navs.iloc[0],
              "Daily_Weights_D": weights_sheet,
              "Turnover_Costs": pd.DataFrame(costs),
              "Constituents": constituents_sheet(reviews),
              "Definitions": pd.DataFrame(DEFINITIONS, columns=["metric", "definition and assumptions"])}
    XLSX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(XLSX_PATH) as writer:
        for name, df in sheets.items():
            keep_index = name not in ("Constituents", "Definitions")
            if keep_index and isinstance(df.index, pd.DatetimeIndex):
                df = df.copy()
                df.index = pd.Index(df.index.date, name="date")
            df.to_excel(writer, sheet_name=name, index=keep_index)
    summary.to_csv(SUMMARY_CSV, float_format="%.4f")
    return summary
