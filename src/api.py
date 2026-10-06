"""
REST API serving the engine's risk signals (data/outputs/engine_signals.csv, read once at startup).

Run:  uvicorn src.api:app --reload        (from the project root; docs at http://127.0.0.1:8000/docs)

Endpoints
  GET /health                     status and number of rows loaded
  GET /signals                    rows, filtered by ticker, start, end (YYYY-MM-DD, inclusive),
                                  source (news/social), news_type; newest first, at most `limit`
  GET /signals/summary?ticker=    count, mean sentiment, mean impact and news-type counts
                                  (all tickers if none is given)
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = PROJECT_ROOT / "data" / "outputs" / "engine_signals.csv"
MAX_LIMIT = 5000


def load_signals(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"date": str})
    df["ticker"] = df["ticker"].str.upper()
    return df


def to_records(df: pd.DataFrame) -> list[dict]:
    """JSON-safe rows (missing values become null)."""
    return df.astype(object).where(df.notna(), None).to_dict(orient="records")


def create_app(csv_path: Path | str | None = None) -> FastAPI:
    path = Path(csv_path or os.environ.get("ENGINE_SIGNALS_CSV", DEFAULT_CSV))
    signals = load_signals(path)
    app = FastAPI(title="AI/NLP Risk Engine",
                  description="Sentiment, impact (severity 1-10) and news type for S&P 100 news and tweets.")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "rows": len(signals), "file": path.name}

    @app.get("/signals")
    def get_signals(ticker: str | None = None, start: str | None = None, end: str | None = None,
                    source: str | None = Query(None, pattern="^(news|social)$"),
                    news_type: str | None = None,
                    limit: int = Query(100, ge=1, le=MAX_LIMIT)) -> list[dict]:
        rows = signals
        if ticker:
            rows = rows[rows["ticker"] == ticker.upper()]
        if start:
            rows = rows[rows["date"] >= start]
        if end:
            rows = rows[rows["date"] <= end]
        if source:
            rows = rows[rows["source"] == source]
        if news_type:
            rows = rows[rows["news_type"].str.lower() == news_type.lower()]
        return to_records(rows.sort_values("date", ascending=False, kind="stable").head(limit))

    @app.get("/signals/summary")
    def summary(ticker: str | None = None) -> dict:
        rows = signals[signals["ticker"] == ticker.upper()] if ticker else signals
        if rows.empty:
            raise HTTPException(status_code=404, detail=f"no signals for ticker {ticker}")
        return {"ticker": ticker.upper() if ticker else "ALL", "count": int(len(rows)),
                "first_date": rows["date"].min(), "last_date": rows["date"].max(),
                "mean_sentiment": round(float(rows["sentiment_score"].mean()), 4),
                "mean_impact": round(float(rows["impact"].mean()), 2),
                "news_type_counts": {k: int(v) for k, v in rows["news_type"].value_counts().items()},
                "source_counts": {k: int(v) for k, v in rows["source"].value_counts().items()}}

    return app


app = create_app()
