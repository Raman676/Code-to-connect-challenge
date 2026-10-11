# Decisions Log

## 2026-10-03

**Downstream module: Module A (index rebalancing)**
Why: fits the one-week timeline. Design: top 20 S&P 500 by
free-float market cap, every stock starts with tilt = 1, tilts change with news.

**Data sources**
- Social: Kaggle "Tweet Sentiment's Impact on Stock Returns"
- News: Kaggle "Daily Financial News for 6000+ Stocks" (headline, date, ticker)
- Benchmark only: Financial PhraseBank (no dates or companies, so it can't be a live source)
- Prices: yfinance (daily prices for every ticker and date in the data)
Why: the problem statement requires 2+ text sources, and the labelling method
needs dates, companies and prices.

**Labelling method: market reaction**
Each item is labelled by what the stock did after it, not by human judgment.
- Volatility filter: keep only items where 10-day realized volatility after the
  item is higher than in the 10 days before. Others are dropped. (My interpretation.)
- Sentiment = sign of the 1-day return. (My choice: most direct reaction.)
- Impact (1 = bearish, 10 = bullish) from the signs of 1d/2d/3d/7d returns:
  ++++ = 10, +++- = 6, ++-- = 4, ---- = 1 (base table);
  ---+ = 5, --++ = 7 (my mirror extension); all other patterns dropped.
- Impact is directional (bear/bull), not severity. Deliberate choice:
  sentiment = immediate reaction, impact = how the move evolves over a week.

**Workflow: every change reviewed**
I review every change before it is applied.
the tweet return columns are unreliable, so returns and volatility come from yfinance
the full tweet file, deduplicated, US companies only
the news file choice
the top-20 question
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
- Result: not done (time). News stays date-only, which is the safe choice.

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

**Index universe**
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

## 2026-10-06

**Old SanDisk headlines dropped (ticker reuse)**
- The news file files 99 headlines from 2016 under SNDK. They are about SanDisk, which
  Western Digital acquired in 2016; today's SNDK (Sandisk) was spun off in 2025.
- Dropped from data/processed/signals_walkforward.csv with
  walk_forward.drop_reused_ticker_headlines (64,127 -> 64,028 rows). The walk-forward was
  not re-run, so these headlines were still part of the training data for that run;
  future runs drop them before training.
Why: a ticker must always refer to the same company.
**News type: keyword rule classifier (10 categories)**
- Categories: Credit Event, Analyst Rating, Earnings, Merger/Acquisition, Legal/Regulatory,
  Management Change, Product Launch, Macroeconomic, Geopolitical, Other.
- First match wins, in that order. Credit Event goes first so "debt downgrade" isn't
  caught as an analyst downgrade. Case-insensitive, whole words only.
- Distribution on 64,028 walk-forward headlines: Earnings 19.6%, Analyst Rating 14.8%,
  M&A 2.7%, Geopolitical 2.3%, Management Change 1.8%, Legal 1.1%, Product Launch 1.0%,
  Macro 1.0%, Credit Event 0.2%, Other 55.5%.
Why: no labelled news-type data exists to train on; analyst headlines are formulaic,
so rules are fast, transparent and need no GPU. The rules favour precision: anything
unclear goes to Other.
Limitation: 55.5% Other (mostly market roundups); rules can misfire (e.g. "Apple Beats"
headphones -> Earnings, coronavirus sell-off -> Earnings via "outlook").
Next step: a learned classifier.

**News-type hand check**
- 100 random headlines (seed 42) in docs/news_type_check.csv, marked Y/N by me.
- Method: read the headline with the prediction hidden, decide the main topic first,
  then compare. Y if the predicted type is a reasonable main topic; "Other" is correct
  only if none of the 9 categories fit. Judge the headline's category, not whether the
  ticker is right. For each N, record the correct type in a true_type column.
- Accuracy: not done (time); the file is ready for it (python src/news_type.py --accuracy).

**Engine output: engine_signals.csv + FastAPI**
- 66,158 rows: 64,028 news (walk-forward, out-of-sample, 2016 to Jun 2020) + 2,130
  tweets (test period Sep-Oct 2018). 90 tickers, ~10 MB, committed.
- Columns: date, company_name, ticker, source (news/social), text, sentiment_score
  (-1..1), impact (severity 1-10), news_type, confidence.
- confidence = |sentiment_score| for now (a proper confidence is computed per stock
  in the index's signal step). Not a calibrated probability.
- 1,000-row sample in data/samples/engine_signals_sample.csv.
- API (src/api.py): GET /health, /signals (filters: ticker, dates, source, news_type,
  limit), /signals/summary?ticker=. Run: uvicorn src.api:app --reload.
Why: the problem statement asks for output via a simple API or a file; we provide both.

**Tweet impact on the same 1-10 scale as news**
- Tweet severity now also uses the decile-of-training-predictions rule (it was squeezed
  into 4-7). Ranking unchanged (Spearman 0.114).
Why: one consistent impact scale across sources.

**Engine summary report**
- docs/engine_report.txt: sources, models and why, key results, news-type distribution,
  output schema, run commands, limitations. The single reference for the README and deck.

**Meta and Booking share counts back-filled**
- Early missing share counts filled with the first known count, split-adjusted.
- LIN, RTX and DELL stay unranked before their share data starts.
Why: Meta and Booking were large S&P 100 members for the whole window and must be
eligible for the top 20. Back-filling a short gap introduces only a small error.

**Planned simplification for the index signal step**
- An "event" = all headlines for one company on one date (mean sentiment, max severity),
  instead of clustering similar headlines by text.
Why: exact duplicates are already removed; company-day events are simple to build and
explain, and match how the labels were evaluated.

## 2026-10-11

**TE/IR/alpha now vs parent index ^OEX (primary) and Strategy A; previously vs A-capped, which was wrong**
- Every strategy (A, A-capped, B, C, D, D-tone) now reports active return, tracking error,
  information ratio, beta and alpha against ^OEX, the parent index (primary benchmark), and
  against Strategy A, plain FFMC Top 20 (secondary: shows the effect of the news tilt alone).
- Same formulas for both: TE = std(daily strategy - benchmark) x sqrt(252); active return =
  annualised mean of that difference; IR = active return / TE; beta and alpha from a CAPM
  regression of daily excess returns (over the T-bill) on the benchmark's excess returns.
- All "vs A-capped" metrics removed (Summary, Sensitivity, reports, dashboard, README).
- ^OEX is price-only while our strategies include dividends: active return and alpha vs
  ^OEX are overstated by roughly the dividend yield (~2%/yr); TE is barely affected. Stated
  in the Definitions sheet and the reports.
Why: A-capped is one of our own strategies, not a benchmark. Tracking error is measured
against the index a fund is benchmarked to, here the parent index. A stays as a second
reference because it isolates the sentiment tilt.
