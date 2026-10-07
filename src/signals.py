"""
Security-level sentiment S and confidence C for the index (methodology sections 9-13).

Input   data/processed/signals_walkforward.csv (one row per headline, out-of-sample scores).

Steps
  1. Event      all headlines for one ticker on one date (DECISIONS.md: planned simplification).
                S = mean sentiment, importance I = max severity / 10,
                model confidence = mean |P(pos) - P(neg)|.
  2. Timing     THE ONLY LAG in the pipeline: a headline dated d is first used at the close of the
                first trading day AFTER d (call it T). The backtest uses the panel row for T as is,
                with no extra shift: weights set at T's close earn the return from T to T+1.
  3. Decay      F = exp(-lambda x dt), lambda = ln 2 / 3 (half-life 3 trading days), dt = T - availability
                day in trading days; only events with 0 <= dt < 15 count (15-day window).
  4. Sentiment  S_i,T = sum(S x I x F) / sum(I x F)                                 (section 10)
  5. Confidence C = model confidence x independence x agreement x source quality     (section 12)
                  model confidence = w-weighted mean, w = I x F
                  independence     = 1 - exp(-n / 2), n = sum(F)
                  agreement        = |sum(w x sign(S))| / sum(w)
                  source quality   = 1.0 (news only)
  6. Neutral    S_eff = 0 if |S| < 0.15                                               (section 13)
News-type weights for importance are left out on purpose (all types count equally); see STATUS.md.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HEADLINES_CSV = PROJECT_ROOT / "data" / "processed" / "signals_walkforward.csv"
HALF_LIFE = 3          # trading days
WINDOW = 15            # trading days
NEUTRAL_ZONE = 0.15
SOURCE_QUALITY = 1.0
LAMBDA = np.log(2) / HALF_LIFE


def load_headlines(path: Path = HEADLINES_CSV) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


def add_scores(headlines: pd.DataFrame, variant: str = "main") -> pd.DataFrame:
    """Add 'score' and 'model_conf' columns. variant 'main' = our sentiment model, 'tone' = finbert-tone."""
    out = headlines.copy()
    if variant == "main":
        p_pos = out["p_10"] + out["p_6"] + out["p_4"]
        p_neg = out["p_7"] + out["p_5"] + out["p_1"]
        out["score"], out["model_conf"] = out["sentiment_score"], (p_pos - p_neg).abs()
    elif variant == "tone":
        out["score"], out["model_conf"] = out["tone_score"], (out["tone_pos"] - out["tone_neg"]).abs()
    else:
        raise ValueError(variant)
    return out


def availability_position(dates: pd.Series, calendar: pd.DatetimeIndex) -> np.ndarray:
    """Position in the calendar of the first trading day strictly after each date (the one lag)."""
    return calendar.searchsorted(pd.DatetimeIndex(dates), side="right")


def build_events(scored: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """One event per ticker and date: S, I, model confidence and the availability position."""
    events = (scored.groupby(["ticker", "date"])
              .agg(S=("score", "mean"), I=("severity", "max"), model_conf=("model_conf", "mean"))
              .reset_index())
    events["I"] = events["I"] / 10
    events["avail"] = availability_position(events["date"], calendar)
    return events[events["avail"] < len(calendar)]          # news after the last trading day is never used


def security_signals(events: pd.DataFrame, calendar: pd.DatetimeIndex, tickers) -> dict[str, pd.DataFrame]:
    """
    Daily panels (calendar x tickers): S, C, S_eff and n_events.
    Each event is copied to the 15 trading days it is in the window, with its decay F.
    """
    lags = np.arange(WINDOW)
    rows = events.loc[events.index.repeat(WINDOW)].copy()
    rows["dt"] = np.tile(lags, len(events))
    rows["pos"] = rows["avail"] + rows["dt"]
    rows = rows[rows["pos"] < len(calendar)]
    rows["F"] = np.exp(-LAMBDA * rows["dt"])
    rows["w"] = rows["I"] * rows["F"]
    rows["wS"] = rows["w"] * rows["S"]
    rows["w_sign"] = rows["w"] * np.sign(rows["S"])
    rows["w_conf"] = rows["w"] * rows["model_conf"]
    g = rows.groupby(["pos", "ticker"])
    sums = g[["w", "wS", "w_sign", "w_conf", "F"]].sum()
    sums["n_events"] = g.size()

    S = sums["wS"] / sums["w"]
    model_conf = sums["w_conf"] / sums["w"]
    independence = 1 - np.exp(-sums["F"] / 2)
    agreement = sums["w_sign"].abs() / sums["w"]
    C = model_conf * independence * agreement * SOURCE_QUALITY

    def panel(values: pd.Series) -> pd.DataFrame:
        wide = values.unstack("ticker").reindex(index=range(len(calendar)), columns=list(tickers))
        wide.index = calendar
        return wide.fillna(0.0)

    s_panel = panel(S)
    return {"S": s_panel, "C": panel(C), "S_eff": s_panel.where(s_panel.abs() >= NEUTRAL_ZONE, 0.0),
            "n_events": panel(sums["n_events"]).astype(int)}


def build_signals(calendar: pd.DatetimeIndex, tickers, variant: str = "main",
                  headlines: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """Everything in one call: headlines -> events -> daily S, C, S_eff panels."""
    headlines = load_headlines() if headlines is None else headlines
    return security_signals(build_events(add_scores(headlines, variant), calendar), calendar, tickers)
