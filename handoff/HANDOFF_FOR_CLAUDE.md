# HANDOFF: S&P Global & Crisil Hackathon 2026: AI/NLP Risk Engine + Sentiment-Tilted Index

You are taking over an in-progress project. Read this whole file before answering.

## 0. How to work with this user (important)

- **The user works on Windows 11 in VS Code**, project folder `C:\Users\raman\Desktop\ML\snp`, Python 3.11.3.
- **You run on their phone.** You cannot see or run their code. They copy your code into VS Code, run it, and paste back results or errors.
- **Before changing any file, ask them to paste its current contents.** Never guess what's in a file. Function names and signatures below are from progress reports; the exact code has not been seen.
- **Give complete file replacements, or very clear "replace this block with this block" edits**, always with the file path at the top. Changes must be small and testable.
- **Every change comes with the exact command to run it and the test command.** Set the encoding first:
  `$env:PYTHONIOENCODING='utf-8'; python src/<file>.py`
  Tests are plain scripts: `python tests/test_<name>.py`, and each prints PASS/FAIL lines.
- **Commit after every working step:** `git add <files>` then `git commit -m "<message>"`. Never commit `data/raw/`, `data/interim/` or `docs/index construction.docx`.
- **The user wants decisions explained simply, then wants to make them.** For each decision, recommend one option and give the reason.
- **Mentor:** Rajesh. He gave the labelling rules and the index methodology. Keep him updated.
- **Deadline:** submit by **Fri Oct 9, 2026** (official deadline Oct 10). The user is at "Mon Oct 5 evening" when this file is written.

## 1. The assignment

**Problem statement:** build an AI/NLP Risk Engine that ingests text from at least 2 sources and outputs, per item:
- a sentiment score (-1 to 1)
- an event or news type
- an impact score (1-10)

It must expose the output through an API or a file. Then build at least one downstream module. Ours is **Module A: a sentiment-driven index rebalancer, with a dashboard of weights over time.**

**Submission requirements:**
- **Public GitHub repo** named `<college>-<candidate-name>-hackathon`, containing:
  - `README.md` in the mandatory template (section 9)
  - `LICENSE` (MIT, mandatory)
  - `requirements.txt`
  - `src/`, `data/`, `docs/presentation.pdf` (5-7 slides), `docs/architecture.png`
- **Demo video** on YouTube (unlisted), linked in the README. Aim for about 5 minutes.
- **Repo hygiene:** no files over 100 MB, no large data dumps, incremental commits.
- **AI use is allowed**, but the README must say so honestly.
- **Tie-breakers:** first Domain Understanding, then Presentation.

## 2. Data sources (all public)

| Source | File (gitignored, in `data/raw/`) | Use |
|---|---|---|
| Kaggle "Tweet Sentiment's Impact on Stock Returns" | `tweets/full_dataset-release.csv` (862k rows, 2017-01 to 2018-10) | Social source. Has no direction signal; used for attention/volume only |
| Kaggle "Daily Financial News for 6000+ Stocks" | `news/analyst_ratings_processed.csv` (1.4M headlines, 2009 to 2020-06-11) | Main news source |
| Financial PhraseBank | `data/phrasebank/` (committed, CC BY-NC-SA, keep License.txt) | Human-tone benchmark only |
| yfinance | cached in `data/prices/` | Prices, shares, splits, info, ^OEX benchmark, ^IRX T-bill rate |
| Wikipedia S&P 100 list | `data/universe/sp100_constituents.csv` | Universe + GICS sectors |

**Data gotchas:**
- **The tweet CSV contains stray `\r` characters inside tweet text.** Split rows on `\n` only. The helper `read_chunks` in `src/inspect_data.py` does this.
- **Tweet dates are DD/MM/YYYY.** Tweets identify companies by name ("Apple"), not by ticker.
- **The tweet file's own return columns are WRONG.** They're backward-looking with the sign flipped. Never use them.
- **News timestamps failed validation** (74.5% agreement vs a pre-set 95% threshold), so news is treated as **date-only**.
- **Ticker remaps:** FB→META, GOOG+GOOGL→one company traded as GOOGL, PCLN→BKNG, BK→BNY, FPL→NEE. Pre-2025 "SNDK" headlines are the OLD SanDisk: drop them (a known bug, not yet fixed).
- **Missing news coverage.** No news at all for MSFT, BRK.B, CVX, COST, GS, GE, ABT, ISRG, LIN, PM, RTX, DELL, UBER, PLTR, GEV. Only about 10 headlines for JPM, V, WMT, INTC, DIS, BA, T, GM, SCHW. These stocks get no tilt.

## 3. Repo layout (`C:\Users\raman\Desktop\ML\snp`)

```
CLAUDE.md                 project brief + working rules (was for Claude Code)
DECISIONS.md              dated log of every decision + reason (jury prep). Keep updating.
.gitignore                data/raw/, data/interim/, *.zip, organiser PDFs,
                          financial_transactions.csv, labels_*.csv, docs/index construction.docx
requirements.txt          pandas 3.0.6, numpy 2.4.6, yfinance 1.7.0, scikit-learn 1.9.1,
                          scipy 1.17.1, torch 2.14.1+cpu (CPU index note in file), transformers 5.18.0,
                          huggingface_hub 1.33.0, matplotlib (version pinned in file)
data/
  raw/                    (gitignored) Kaggle downloads
  phrasebank/             PhraseBank files + all-data.csv
  processed/              fpb_sentiment.csv; labels_tweets.csv + labels_news.csv (gitignored);
                          signals_walkforward.csv (14.6 MB, NOT YET COMMITTED; commit it)
  samples/                engine_input_sample_200.csv + 1,000-row label samples
  prices/                 adj_close.csv (47-pool), sp100_adj_close.csv, unadjusted closes,
                          sp100_splits.csv, sp100_shares_clean.csv, info, ^OEX, ^IRX (~7 MB)
  universe/               sp100_constituents.csv (tickers + GICS sectors)
  interim/                (gitignored) FinBERT embedding cache
  outputs/                sentiment_predictions_test.csv, sentiment_output_test.csv
docs/
  data_inspection.txt, labels_report.txt, model_report.txt, model_checks.txt,
  severity_report.txt, index_data_checks.txt, walkforward_report.txt (uncommitted)
  index construction.docx (mentor's methodology, gitignored)
src/
  sentiment_data.py   PhraseBank loader (mojibake fix, dedup, agreement levels, 70/15/15 split)
  inspect_data.py     data inspection; POOL (47 tickers); read_chunks (\r-safe CSV reader)
  prices.py           yfinance download + cache + trading calendar
  labels.py           market-reaction labelling (anchor, returns, vol filter, sign pattern, impact)
  text_prep.py        masking: company names, tickers, cashtags, @mentions, CEOs, subsidiaries ->
                      [COMPANY]; URLs -> [URL]; 20-tweets-per-company-day cap. Covers S&P 100.
  modeling.py         TF-IDF + logistic regression (shared settings), baselines
  run_model.py        trains sentiment models; writes sentiment_predictions_test.csv
  model_checks.py     tokenizer sanity check, top terms, random baselines, next-day check
  severity.py         severity model; writes final sentiment_output_test.csv
  index_data.py       S&P 100 prices/shares/splits/float download + cleaning
                      (MANUAL_DROPS, KNOWN_EVENTS)
  walk_forward.py     yearly retrain 2016-2020, scores every S&P 100 headline
  universe.py         IN PROGRESS, uncommitted: FFMC ranking, 20/25 buffer, quarterly reviews
  weights.py          IN PROGRESS, uncommitted: base weights, caps, daily change limit, normalise
  (to build) signals.py, backtest.py, news_type.py, dashboard (app.py)
tests/  test_sentiment_data, test_inspect_data, test_prices, test_labels, test_text_prep,
        test_modeling, test_severity, test_index_data, test_walk_forward (+ model checks)
```

Run order of the existing pipeline: `labels.py` → `run_model.py` → `severity.py`. Separately: `index_data.py` → `walk_forward.py`. The walk-forward takes about 46 minutes on CPU.

## 4. Labelling rules (mentor's method plus our choices)

- **Anchor:** the last trading-day close strictly BEFORE the item's date. Weekend items anchor to Friday.
- **Returns:** r1, r2, r3, r7, measured in **trading days** from the anchor.
- **Volatility filter:** std of daily log returns over the 10 trading days after the anchor > the 10 days up to and including the anchor. Items that fail are dropped.
- **Sentiment label:** the sign of r1. A zero return is dropped.
- **Impact pattern** (signs of r1/r2/r3/r7): ++++=10, +++-=6, ++--=4, ----=1 (mentor); ---+=5, --++=7 (our mirror). Every other pattern is dropped.
- **Results:** 125,245 tweets and 21,733 news headlines labelled (47-company pool), sentiment about 50/50.
- **Key finding:** all tweets about one company on one day share a label, so the 125k tweets are only 989 independent company-days. Consequences:
  - split by date, never randomly
  - mask company identity in the text
  - cap at 20 tweets per company-day

## 5. Models and results

**Sentiment model**
- TF-IDF + logistic regression (class-weighted), trained on the 6 impact classes.
- sentiment_score = P(10)+P(6)+P(4) − P(7)−P(5)−P(1).
- Splits are chronological, with 10-trading-day gaps for news.

| Result (single test split) | News (Nov 2019 – Jun 2020) | Tweets (Sep–Oct 2018) |
|---|---|---|
| Sentiment accuracy, per item | 0.609 (baseline 0.511) | 0.507 (≈ chance) |
| Sentiment accuracy, per company-day | 0.70 (931 days) | 0.56 (155 days) |
| 6-class impact | ≈ random | ≈ random |

**Other sentiment findings:**
- **FinBERT embeddings (`yiyanghkust/finbert-tone`) did NOT beat TF-IDF.** The tokenizer was verified: finbert-tone scores 0.759 on PhraseBank. We deliberately avoided ProsusAI/finbert because it was trained on PhraseBank.
- **Our score vs human tone (PhraseBank):** Spearman 0.04. Market reaction ≠ human tone.
- **Next-day (tradable) accuracy: 0.496.** The same-day signal is mostly headlines describing the day's move.

**Impact, redefined as SEVERITY (decision made):**
- Target: |r1|, the size of the move. Direction lives in the sentiment score.
- Ridge regression on TF-IDF + log trailing 10-day volatility + the company's news count on the previous trading day.
- Score = the decile of the prediction among training predictions, giving 1-10.
- Tweets use the volatility-only model. Every variant tied on validation, so the simplest-model tie-break rule picked it.
- News test: Spearman 0.509 per row, 0.603 per company-day. Mean |r1| rises from 1.0% (severity 1) to 6.3% (severity 10).
- Most of the power comes from volatility clustering; text adds a little. Scores are flattered by COVID.

**Walk-forward:**
- Retrain yearly; test years 2016 to 2020-06. Every S&P 100 headline is scored out-of-sample: 64,127 headlines, 85 companies.
- Output: `signals_walkforward.csv`, with columns item_id, date, ticker, headline, test year, the 6 class probabilities, sentiment_score, predicted log|r1|, severity 1-10, tone_pos, tone_neg, tone_neutral, tone_score (finbert-tone).

| Year | Same-day acc (ours/tone) | Next-day corr (ours/tone) | Top−bottom quintile next day |
|---|---|---|---|
| 2016 | 0.537 / 0.555 | 0.005 / 0.024 | −2.8 / +2.9 bp |
| 2017 | 0.542 / 0.536 | 0.011 / 0.004 | +4.1 / +2.5 bp |
| 2018 | 0.538 / 0.561 | 0.004 / 0.009 | +2.1 / +3.0 bp |
| 2019 | 0.589 / 0.613 | −0.006 / −0.015 | −3.4 / −7.0 bp |
| 2020 H1 | 0.657 / 0.640 | 0.058 / 0.049 | +88 / +80 bp |

- **Conclusion:** there's no reliable next-day edge in 2016–2019. A small signal appears only in the 2020 COVID regime.
- **Walk-forward severity works:** Spearman 0.35–0.44 every year, mean |r1| rising from 0.8% to 4.3%. It skews toward 10 because the test years were more volatile than the training years.
- **Expect strategies B–D to roughly match strategy A.** Report this honestly: daily news gets priced in within the same day.

## 6. Index methodology (mentor's document, adapted to the S&P 100)

**Universe and selection**
- Universe: **current S&P 100** (the mentor's document says Nifty 50; adapted because the data is US). Benchmark: ^OEX, which is price-only (no dividends), so strategy A is the fair benchmark.
- FFMC = unadjusted price × historical shares (split-adjusted) × free-float factor.
- The free-float factor is today's ratio, held constant. BRK.B uses 0.85 (our estimate). Alphabet uses 0.89 across all share classes.
- **Selection:** top 20 by FFMC with a 20/25 buffer, reviewed quarterly (last trading day of Mar/Jun/Sep/Dec, effective the next day). Exactly 20 names: keep members ranked up to 25, fill to 20 from the top, and if over 20, drop the lowest-ranked holdovers. The mentor's §7 example ends with 21 names; note this deviation.

**Weights**
- Base weight = FFMC / sum of FFMC across the 20.

**Signals** (per stock, per day)
- Cluster near-duplicates: same ticker, TF-IDF cosine > 0.8 within 2 trading days → one event, with S = the mean of its items.
- Aggregate: S_i = Σ S·I·F / Σ I·F over events in the last 15 trading days.
  - I = severity / 10 (× a news-type weight later)
  - F = e^(−λΔt), with λ = ln2 / 3 trading days
- **Confidence C_i** = the product of four parts:
  1. model confidence (mean |P(pos) − P(neg)|)
  2. independence, 1 − e^(−n/2), where n = independent events
  3. agreement, |Σ w·sign(S)| / Σ w
  4. source quality (news 1.0, tweets 0.25)
- Neutral zone: |S| < 0.15 → 0.

**Tilt**
- w_raw = w_FFMC × e^(α·S·C), with α = 0.30.
- Then apply the caps (max weight 12%, max tilt ±25% of w_FFMC), then the daily change limit (1 percentage point; exempt on review days), then normalise. Repeat until all limits hold.

**Trading and costs**
- Trade timing: signals from date t trade at the close of t+1, earning returns from t+2.
- Cost: 10 bp × Σ|Δw|.

**Backtest window:** Jan 2016 – Jun 2020. Share-count history only starts in late 2015.

**Strategies:**

| Strategy | Definition |
|---|---|
| A | FFMC top 20, weights drift with prices between reviews |
| A-capped | A + 12% cap. **Compare D against this** |
| B | A + tilt with C = 1, neutral zone, no risk controls |
| C | B + confidence |
| D | C + caps, tilt limit, daily limit |
| D-tone | D using finbert-tone's tone_score instead of our model |

**Metrics:**
- return, CAGR, volatility, Sharpe and Sortino (risk-free rate from ^IRX), max drawdown
- information ratio, tracking error, alpha vs A-capped and vs ^OEX
- turnover, costs, number of constituent changes
- average tilt, max weight, sector concentration (GICS)
- performance during the top 1% of severity-weighted event days, and in March 2020
- **chart: the number of index members with an active signal per day**

**Key test (mentor's §25):** does sentiment add value beyond FFMC? Answer it with:
- the daily rank correlation between S and next-day returns
- a regression of next-day returns on S plus the past 1-day and 5-day returns

**Share-data cleaning (done, committed):**
- 438 splits were applied.
- Spikes were removed automatically (ANET 2019-11, DELL 2019-06 in the window).
- MANUAL_DROPS: ACN 2015-12-17, ISRG 2017-08-28, NVDA 2020-03-17.
- KNOWN_EVENTS: AVGO 2016-03, BMY–Celgene 2019-11, CVS–Aetna 2018-12, TMUS–Sprint 2020-04.

**Biases to state:**
- survivorship bias (today's S&P 100 list used for 2016–2020)
- a constant free-float factor
- ^OEX excludes dividends

## 7. Open decisions (recommended answers)

1. **Meta and Booking share counts start late.** Back-fill each with its first known count, split-adjusted: YES. Leave LIN, RTX and DELL unranked before their data starts: YES. Strategies A and A-capped are provisional until this is done.
2. **SNDK:** drop the pre-2025 headlines. Do NOT re-run the 46-minute walk-forward for this; just filter them out of `signals_walkforward.csv` and note it in DECISIONS.md.
3. **Commit** `signals_walkforward.csv` and `walkforward_report.txt`. They're needed to run the prototype.

## 8. Remaining work, in order

1. **Finish `universe.py` and `weights.py`, then run strategies A and A-capped.** Sanity check: the constituents should look like AAPL, MSFT, AMZN, GOOGL, META, BRK.B, JNJ, JPM, V, PG, XOM and so on.
2. **`signals.py`:** event clustering, aggregation, confidence, neutral zone (all per section 6).
3. **`backtest.py`:** strategies A, A-capped, B, C, D, D-tone; the metrics; matplotlib charts saved to `docs/`:
   - index levels
   - weights over time (stacked)
   - active-signal count
   - drawdowns
4. **`news_type.py`:** 10 categories: Earnings, Analyst Rating, Merger/Acquisition, Product Launch, Legal/Regulatory, Management Change, Macroeconomic, Geopolitical, Credit Event, Other.
   - **Use a keyword/regex rule classifier.** It's fast, transparent and needs no GPU. Analyst headlines are formulaic: upgrade/downgrade/price target/initiates → Analyst Rating; EPS/Q1-Q4/revenue/beats/misses/guidance → Earnings; acquire/merger/deal/buyout → M&A; launch/unveil → Product Launch; lawsuit/SEC/FTC/probe/fine → Legal/Regulatory; CEO/CFO/resigns/appoints → Management Change; Fed/rates/inflation/GDP/jobs → Macroeconomic; tariff/China/trade war/sanctions → Geopolitical; default/downgrade of debt/credit rating/bankruptcy → Credit Event.
   - Hand-check 100 random headlines and report the accuracy.
   - Optional: a news-type importance weight.
5. **Final engine output** `data/outputs/engine_signals.csv`, with columns: date, company_name, ticker, source, text, sentiment_score, impact (severity 1-10), news_type, confidence. Optional: a tiny FastAPI app serving `/signals?ticker=`. The problem statement allows file output.
6. **Dashboard:** `app.py` using Streamlit + Plotly, reading the CSVs, showing:
   - weights over time
   - strategy D vs A-capped vs ^OEX
   - the latest signals table, filterable by ticker
   - the active-signal count
   - a strategy picker

   Run with `streamlit run app.py`. Needs `pip install streamlit plotly`; pin them in requirements.txt.
7. **Docs and submission:**
   - README.md in the template below
   - LICENSE (MIT)
   - `docs/architecture.png`
   - `docs/presentation.pdf` (7 slides: Title, Problem & Approach, System Design, Implementation, Key Results, Domain Impact, Limitations & Next Steps)
   - a ~5-minute video
   - create the public repo, push, and test every link in an incognito window

## 9. README template (mandatory structure)

```
# [Project Title] - S&P Global & Crisil Campus Hackathon
**Candidate Name:** ... **College Email ID:** ... **College / Campus:** ...
**Demo Video Link:** ... **Slide Deck Link (if hosted externally):**
## 1. Project Overview / Problem Statement & Approach
## 2. Architecture & Tech Stack
## 3. Dataset Used   (sources, Kaggle links, re-download steps, assumptions)
## 4. Quickstart & Installation   (Python 3.11 on Windows 11; exact commands)
## 5. Key Results & Domain Impact
```

Also add an "AI usage" note: AI assistants were used for code generation; all code was reviewed and is understood by the author.

## 10. Honest story for the deck (what the jury should hear)

- **We validated the given data before trusting it.** The tweet return columns turned out to be backward-looking, and the news timestamps failed a pre-registered check.
- **Market-reaction labels (mentor's method)** gave us leakage-safe evaluation: date splits, gaps, masked company names, company-day units.
- **News sentiment captures the same-day reaction** (70% per company-day), but it's priced in by the next day. That fits market efficiency; exploiting it would need intraday timestamps.
- **Impact redefined as severity works:** Spearman 0.35–0.60, a monotonic decile chart. Volatility clustering plus news intensity.
- **Tweets carry no direction signal** (customer chatter, about 1k independent days), so they're used as attention with low source quality.
- **The index follows a real index methodology:** eligibility, selection buffer, FFMC weights, tilt, risk controls. A-D ablations measure what each layer adds.
- **Limitations:** news coverage gaps for MSFT, BRK.B, JPM and others; survivorship bias; constant free float; date-only news.
