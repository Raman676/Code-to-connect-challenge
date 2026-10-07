"""
Data for the Streamlit dashboard (app.py). Reads the CSV and report files only; no models run here.

All files are made by earlier steps:
  data/outputs/backtest_nav.csv, backtest_summary.csv, backtest_weights.csv,
  signals_daily.csv, backtest_sensitivity.csv       python src/backtest.py
  data/outputs/engine_signals.csv                    python src/build_engine_output.py
  docs/severity_report.txt                           python src/severity.py
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "data" / "outputs"
SEVERITY_REPORT = PROJECT_ROOT / "docs" / "severity_report.txt"

# Headline engine numbers, copied from docs/engine_report.txt (news test 2019-11 to 2020-06 and walk-forward).
ENGINE_NUMBERS = [
    ("Same-day direction, per headline (test)", "60.9%", "vs ~51% coin flip"),
    ("Same-day direction, per company-day (test)", "69.9%", "931 company-days"),
    ("Same-day direction, walk-forward 2016-2020", "57.2% / 60.4%", "per headline / per company-day"),
    ("Next-day direction (tradable)", "49.6%", "headline is priced in on the same day"),
    ("Rank correlation with next-day return", "0.013", "walk-forward, all years"),
    ("Severity: Spearman with actual |return|", "0.443", "walk-forward news"),
]


def load_nav() -> pd.DataFrame:
    nav = pd.read_csv(OUTPUTS / "backtest_nav.csv", index_col=0, parse_dates=True)
    return 100 * nav / nav.iloc[0]                                    # index level, base 100


def load_summary() -> pd.DataFrame:
    return pd.read_csv(OUTPUTS / "backtest_summary.csv", index_col="strategy")


def load_weights() -> pd.DataFrame:
    return pd.read_csv(OUTPUTS / "backtest_weights.csv", parse_dates=["date"])


def load_daily_signals() -> pd.DataFrame:
    return pd.read_csv(OUTPUTS / "signals_daily.csv", parse_dates=["date"])


def load_engine_signals() -> pd.DataFrame:
    return pd.read_csv(OUTPUTS / "engine_signals.csv", parse_dates=["date"])


def load_sensitivity() -> pd.DataFrame:
    return pd.read_csv(OUTPUTS / "backtest_sensitivity.csv", index_col="alpha")


def weights_wide(weights: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """Dates x tickers weight table for one strategy (0 = not held)."""
    one = weights[weights["strategy"] == strategy]
    return one.pivot(index="date", columns="ticker", values="weight").fillna(0.0)


def weights_on_date(weights: pd.DataFrame, strategy: str, date) -> pd.DataFrame:
    """Base vs final weight and the tilt (final - base), in %, on the last trading day <= date."""
    one = weights[weights["strategy"] == strategy]
    day = one.loc[one["date"] <= pd.Timestamp(date), "date"].max()
    table = one[one["date"] == day].set_index("ticker")[["base_weight", "weight"]] * 100
    table.columns = ["base %", "final %"]
    table["tilt (pp)"] = table["final %"] - table["base %"]
    return table.sort_values("final %", ascending=False)


def filter_signals(engine: pd.DataFrame, ticker: str | None, start, end, news_types=None) -> pd.DataFrame:
    """Engine output rows for the chosen ticker ('All' or None = every ticker), dates and news types."""
    rows = engine[(engine["date"] >= pd.Timestamp(start)) & (engine["date"] <= pd.Timestamp(end))]
    if ticker and ticker != "All":
        rows = rows[rows["ticker"] == ticker]
    if news_types:
        rows = rows[rows["news_type"].isin(news_types)]
    return rows


def severity_deciles(text: str, block: str = "NEWS - full") -> pd.DataFrame:
    """
    Mean actual |1-day return| per predicted-severity decile, read from the severity report's table
    for one model (the 'pred decile' columns: test predictions split into ten equal groups).
    """
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(block))
    rows = []
    for line in lines[start + 2:]:
        parts = line.split()
        if not parts or not re.fullmatch(r"\d+", parts[0]):
            break
        rows.append({"decile": int(parts[0]), "mean |r1| %": float(parts[5])})
    return pd.DataFrame(rows).set_index("decile")
