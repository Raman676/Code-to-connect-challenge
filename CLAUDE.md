# Project: AI/NLP Financial Risk Engine (S&P Global & Crisil Hackathon 2026)

Start of every session: read DECISIONS.md, docs/STATUS.md and handoff/ first.

## Goal
Ingest financial text from 2 sources (tweets + news headlines), output structured
risk signals, and use them to rebalance a mock stock index (Module A).
Deadline: Oct 10. Decisions and reasons are logged in DECISIONS.md. Read it first.

## Engine output (one row per item)
date, ticker, company_name, source, text, sentiment_score (-1 to 1),
impact_score (1-10, 1 = bearish, 10 = bullish), news_type

## Data
- data/raw/tweets/  Kaggle "Tweet Sentiment's Impact on Stock Returns"
- data/raw/news/    Kaggle "Daily Financial News for 6000+ Stocks" (very large: read in chunks)
- data/phrasebank/: benchmark only (src/sentiment_data.py)
- Prices: yfinance, cached in data/prices/
- data/raw/ is gitignored. Never commit it.

## How to work with me (important)
- One step at a time. After each step, stop and wait for my approval.
- Before writing code, explain in plain language what you will do and why.
- After each step, summarize what changed and how to run or test it.
- Never install a package or download data without asking me first.
- If you need to deviate from an approved plan, ask first.
- Keep code simple, commented and readable. I must be able to explain every line
  to a jury.
- Every new module in src/ gets a test in tests/.
- Keep requirements.txt updated.
- Windows machine: use PYTHONIOENCODING=utf-8 when running scripts.
