"""
Engine output: one table of structured risk signals for downstream modules.

Run:  python src/build_engine_output.py
      -> data/outputs/engine_signals.csv (+ a 1,000-row sample in data/samples/engine_signals_sample.csv)

Rows
  news    all walk-forward headlines, Jan 2016 - Jun 2020 (data/processed/signals_walkforward.csv):
          every S&P 100 headline scored by models trained only on earlier years
  social  the tweet test rows of data/outputs/sentiment_output_test.csv (Sep 27 - Oct 30, 2018)
Columns
  date            item date (YYYY-MM-DD)
  company_name    from data/universe/sp100_constituents.csv (tweets about companies outside today's
                  S&P 100, e.g. Nike, keep the name from the tweet data)
  ticker, source (news / social), text
  sentiment_score -1..1 = P(positive-first-day classes) - P(negative-first-day classes)
  impact          predicted severity 1-10 (decile of the predicted |1-day return| among the model's
                  training predictions)
  news_type       rule-based category (src/news_type.py)
  confidence      |sentiment_score|
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import news_type as nt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIGNALS_CSV = PROJECT_ROOT / "data" / "processed" / "signals_walkforward.csv"
SENTIMENT_OUTPUT = PROJECT_ROOT / "data" / "outputs" / "sentiment_output_test.csv"
UNIVERSE_CSV = PROJECT_ROOT / "data" / "universe" / "sp100_constituents.csv"
ENGINE_CSV = PROJECT_ROOT / "data" / "outputs" / "engine_signals.csv"
SAMPLE_CSV = PROJECT_ROOT / "data" / "samples" / "engine_signals_sample.csv"
SAMPLE_ROWS, SEED = 1000, 42
COLUMNS = ["date", "company_name", "ticker", "source", "text", "sentiment_score", "impact", "news_type", "confidence"]


def company_names() -> dict:
    universe = pd.read_csv(UNIVERSE_CSV)
    return dict(zip(universe["ticker"], universe["company"]))


def news_rows(names: dict) -> pd.DataFrame:
    df = pd.read_csv(SIGNALS_CSV)
    return pd.DataFrame({"date": df["date"], "company_name": df["ticker"].map(names), "ticker": df["ticker"],
                         "source": "news", "text": df["text"], "sentiment_score": df["sentiment_score"],
                         "impact": df["severity"]})


def social_rows(names: dict) -> pd.DataFrame:
    df = pd.read_csv(SENTIMENT_OUTPUT)
    df = df[df["source"] == "tweets"]
    return pd.DataFrame({"date": df["date"], "company_name": df["ticker"].map(names).fillna(df["company_name"]),
                         "ticker": df["ticker"], "source": "social", "text": df["text"],
                         "sentiment_score": df["sentiment_score"], "impact": df["impact"]})


def build() -> pd.DataFrame:
    names = company_names()
    out = pd.concat([news_rows(names), social_rows(names)], ignore_index=True)
    out = nt.add_news_type(out, "text")
    out["confidence"] = out["sentiment_score"].abs()
    out["impact"] = out["impact"].astype(int)
    return out.sort_values(["date", "source", "ticker"], kind="stable")[COLUMNS].reset_index(drop=True)


def main() -> int:
    out = build()
    ENGINE_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(ENGINE_CSV, index=False, float_format="%.4f", encoding="utf-8")
    SAMPLE_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.sample(SAMPLE_ROWS, random_state=SEED).sort_values(["date", "ticker"]).to_csv(
        SAMPLE_CSV, index=False, float_format="%.4f", encoding="utf-8")
    size_mb = ENGINE_CSV.stat().st_size / 1e6
    print(f"{ENGINE_CSV.relative_to(PROJECT_ROOT)}: {len(out):,} rows, {size_mb:.1f} MB")
    print(f"  by source: {out['source'].value_counts().to_dict()}")
    print(f"  dates: {out['date'].min()} -> {out['date'].max()}; tickers: {out['ticker'].nunique()}; "
          f"missing company names: {int(out['company_name'].isna().sum())}")
    print(f"{SAMPLE_CSV.relative_to(PROJECT_ROOT)}: {SAMPLE_ROWS} rows (seed {SEED})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
