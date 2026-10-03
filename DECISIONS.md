# Decisions Log

## 2026-10-03

**Downstream module: Module A (index rebalancing)**
Why: mentor's direction; fits the one-week timeline. Design: top 20 S&P 500 by
free-float market cap, every stock starts with tilt = 1, tilts change with news.

**Data sources**
- Social: Kaggle "Tweet Sentiment's Impact on Stock Returns"
- News: Kaggle "Daily Financial News for 6000+ Stocks" (headline, date, ticker)
- Benchmark only: Financial PhraseBank (no dates or companies, so it can't be a live source)
- Prices: yfinance (daily prices for every ticker and date in the data)
Why: the problem statement requires 2+ text sources, and the labelling method
needs dates, companies and prices.

**Labelling method: market reaction (mentor's method)**
Each item is labelled by what the stock did after it, not by human judgment.
- Volatility filter: keep only items where 10-day realized volatility after the
  item is higher than in the 10 days before. Others are dropped. (My interpretation.)
- Sentiment = sign of the 1-day return. (My choice: most direct reaction.)
- Impact (1 = bearish, 10 = bullish) from the signs of 1d/2d/3d/7d returns:
  ++++ = 10, +++- = 6, ++-- = 4, ---- = 1 (mentor);
  ---+ = 5, --++ = 7 (my mirror extension); all other patterns dropped.
- Impact is directional (bear/bull), not severity. Deliberate choice:
  sentiment = immediate reaction, impact = how the move evolves over a week.

**Tooling: Claude Code with approval mode**
I review every change before it is applied.
