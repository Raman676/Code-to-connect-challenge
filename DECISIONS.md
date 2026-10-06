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
the tweet return columns are unreliable, so returns and volatility come from yfinance
the full tweet file, deduplicated, US companies only
the news file choice
the top-20 question, pending your mentor
**Tweets: full file, deduplicated, pool companies only**
- Use full_dataset-release.csv (862k rows), not the reduced file (85k rows,
  only tweets that @mention the company's own account).
- Split rows on \n only: stray \r characters inside tweet text broke normal CSV
  parsing (it produced 143k misaligned rows with fake "stocks").
- Keep only companies in our 47-company pool (32 matched), with names mapped to
  tickers (Facebook -> META, Google -> GOOGL).
- Drop exact duplicate text + ticker, keeping the earliest.
Why: more data than the reduced file; non-US companies trade in other currencies
and aren't S&P members; ~37% of tweets were repeats, which would leak between
train and test and overweight repeated text.

**News: analyst_ratings_processed.csv**
- Headline + date + ticker, pool tickers only (FB -> META, GOOG -> GOOGL).
- Duplicate headline + ticker dropped (4.7% of pairs).
Why: the only file of the three with a time of day; the other two are date-only.
Same duplicate reasoning as tweets.

**News timestamps: date-only (pending daylight-saving check)**
- Rule fixed before checking: trust the times only if at least 95% agree within
  1 minute with the raw file's real times.
- Result: 74.5% agreed; ~25% were exactly one hour off. So news is date-only.
- Pending: the raw file labels every time as US summer time (-04:00), even in winter.
  Test whether the one-hour gaps are exactly the winter rows. If so, correct the raw
  times and re-run the same rule.
- Result: [fill in]

**Prices, anchors and horizons**
- yfinance daily adjusted closes, 2009-2020, for the pool. Cached in
  data/prices/adj_close.csv (~1.5 MB, committed).
- Trading calendar = the dates in the price data (handles exchange holidays).
- Date-only items anchor to the last close strictly BEFORE the item's date
  (weekend items -> Friday's close).
- All horizons (1/2/3/7 days) are trading days, so "7 days" is about a week and a half.
- Volatility = std of daily log returns: 10 trading days after the anchor vs the
  10 days up to and including it.
- Items with a zero return or not enough price history are dropped.
Why: markets don't trade on weekends, so trading days keep the horizons consistent.
Limitation: tweets have no time of day, so a tweet posted after the close is
partly labelled by a move that happened before it.

**Labelling results**
- 125,245 tweets and 21,733 news headlines labelled.
- Sentiment is balanced: tweets 52% positive / 48% negative, news 51% / 49%.
- The tweet dataset's own return columns are backward-looking with the sign flipped
  (-0.91 correlation with the price change on the tweet day itself). Used only as a
  cross-check, never for labels.

**Key finding: tweet labels cluster by company-day**
- Every tweet about one company on one day gets the same label, so the 125k tweets
  are only 989 independent company-days. Nike's Kaepernick-ad weekend alone is
  7,678 tweets with one label.
- Consequences: split by date, never randomly (no company-day in two splits); mask
  company identity in the text; cap at 20 tweets per company-day.
- News is the stronger training source: ~8,000 company-days from 22k rows.

**Git: label files not committed**
- data/processed/labels_*.csv are gitignored (38 MB + 6 MB, reproducible from the
  scripts). Fixed-seed 1,000-row samples go in data/samples/.
- For the "data in /data" requirement: commit samples, the price cache, and README
  instructions to re-download the original Kaggle files.
Why: GitHub's file limits, and the guidelines ask to avoid large data dumps.

**Index universe (pending mentor approval)**
- Take S&P 100 members, keep those with enough coverage in the data period, then
  pick the top 20 by free-float market cap. Weights = ffmcap x tilt, tilt starts at 1.
Why: many of today's largest companies (NVDA, MSFT, TSLA) have little or no data
in 2017-2020. The problem statement specifies the S&P 100. S&P's own index
methodologies also use eligibility rules before selection and weighting.

## 2026-10-04

**Model target: one 6-class model gives both sentiment and impact**
- Train on the 6 impact classes (10, 6, 4, 7, 5, 1).
- sentiment_score = P(10) + P(6) + P(4) - P(7) - P(5) - P(1), which lands between -1 and 1.
- Predicted impact = the most likely class.
Why: sentiment is fully determined by the impact pattern (10/6/4 all have a positive
1-day return; 7/5/1 a negative one). On live news future returns are unknown, so
impact must be predicted from the text too.

**Text preparation**
- URLs -> [URL]. Company names, tickers, cashtags, @mentions, CEO names (Zuckerberg,
  Bezos, Musk, Cook, Nadella...) and subsidiaries (Instagram, YouTube, WhatsApp)
  -> [COMPANY].
- Ambiguous names (Apple, Visa, Oracle) are masked only when capitalised. Single-letter
  tickers (T, V, C) only as cashtags or in "(T)" / "NYSE: T" form.
Why: Nike tweets were 86% negative and Apple 83% positive; without masking, the model
learns company identity instead of reading the text.
Limitation: product names (iPhone, Kaepernick) still reveal the company.

**Splits: chronological, by anchor date**
- Tweets (after the cap): train 2017-01-30 -> 2018-09-10 (684 company-days),
  validation -> 2018-09-26 (150), test 2018-09-27 -> 2018-10-30 (155).
- News: train 2009-2018 (5,796 company-days), validation Nov 2018 - Nov 2019 (1,138),
  test Nov 2019 - Jun 2020 (931), with 10-trading-day gaps between splits.
- The final tweet model trains on train only (not train + validation), because
  validation labels overlap the first test days' price moves. News refits on
  train + validation.
Why: labels use prices up to 10 trading days ahead, so adjacent periods share moves.
Note: the tweet test period is the October 2018 sell-off (impact-1 share jumps from
23% to 39%), a deliberately honest out-of-time test.

**Models compared**
- Majority-class baseline; TF-IDF + logistic regression (class-weighted); FinBERT
  embeddings + logistic regression (CPU, no fine-tuning).
- One model per source, plus one combined model as a comparison.
- Regularisation tuned on validation by impact macro-F1; the test set is scored once.
Why: separate models because tweets and news differ in style and period; the
combined model tests whether more data helps the tweet model.

**FinBERT variant: yiyanghkust/finbert-tone**
Why: ProsusAI/finbert was trained on Financial PhraseBank, which would invalidate our
PhraseBank check. finbert-tone is trained on analyst-report sentences.

**Evaluation**
- Per source: sentiment accuracy and macro-F1, impact accuracy and macro-F1, each vs
  the baseline. Predicted vs actual class distributions.
- Company-day level: average score per company-day vs that day's label (the honest
  unit for tweets, and how the index will use signals).
- PhraseBank check with the best news model: positive and negative sentences only
  (our model has no neutral class); neutral sentences are only described.

**News-type categories (10)**
Earnings, Analyst Rating, Merger/Acquisition, Product Launch, Legal/Regulatory,
Management Change, Macroeconomic, Geopolitical, Credit Event, Other.
Why: includes all five example categories from the problem statement, plus the
types that dominate company-level news. Credit Event is kept even though it will be
rare for large companies.