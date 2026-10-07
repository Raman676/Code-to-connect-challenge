# IMPLEMENTATION PLAN: finish by Fri Oct 9

**For phone Claude:** read HANDOFF_FOR_CLAUDE.md first. Work through these steps in order. Each step has a spec and a "done when" check. Ask the user to paste any file before editing it.

**Priority labels:**
- **MUST:** required for submission
- **SHOULD:** do if on schedule
- **CUT:** only if there's spare time

## Step 0: Get the real file schemas (MUST, 10 min)

The user creates `src/show_schemas.py`, runs it, and pastes the output to Claude:

```python
import glob, os
import pandas as pd

for p in sorted(glob.glob("data/**/*.csv", recursive=True)):
    if "raw" in p or "interim" in p:
        continue
    try:
        df = pd.read_csv(p, nrows=3)
        print(f"\n== {p} ({os.path.getsize(p)/1e6:.1f} MB) shape-cols={df.shape[1]}")
        print("columns:", list(df.columns)[:30])
        print(df.head(2).to_string()[:600])
    except Exception as e:
        print(p, "ERROR", e)
```

Also paste the current `src/universe.py` and `src/weights.py` (they may be partial), plus `git status`.

**Done when:** Claude knows the exact columns of `signals_walkforward.csv`, the price, share and float files, and `sp100_constituents.csv`.

## Step 1: Housekeeping (MUST, 15 min)

- Drop pre-2025 SNDK rows from `signals_walkforward.csv` (a small script, or a filter in `signals.py`).
- **Meta and Booking:** back-fill missing early share counts with the first known count (split-adjusted). LIN, RTX and DELL stay unranked before their data starts.
- **Commit:** `signals_walkforward.csv`, `docs/walkforward_report.txt`, plus any finished universe and weights work.

## Step 2: `src/news_type.py` (MUST, 1 h)

A keyword rule classifier. The first matching rule wins, in this order:

| # | Category | Keywords (case-insensitive, word boundaries) |
|---|---|---|
| 1 | Credit Event | default, bankruptcy, chapter 11, credit rating, debt downgrade, junk, moody's, s&p downgrade, fitch |
| 2 | Analyst Rating | upgrade(s/d), downgrade(s/d), price target, initiates, reiterates, maintains, overweight, underweight, outperform, underperform, buy rating, sell rating, neutral rating |
| 3 | Earnings | eps, earnings, q1, q2, q3, q4, quarter, revenue, sales, guidance, beats, misses, estimates, outlook, profit |
| 4 | Merger/Acquisition | acquire(s/d), acquisition, merger, merge, buyout, takeover, deal to buy, stake |
| 5 | Legal/Regulatory | lawsuit, sue(d/s), sec, ftc, doj, antitrust, probe, investigation, fine(d), settlement, regulator, recall |
| 6 | Management Change | ceo, cfo, chairman, resign(s), steps down, appoint(s/ed), names .* (ceo\|cfo\|president), hires |
| 7 | Product Launch | launch(es), unveil(s), introduce(s), release(s), new product, rollout, announces new |
| 8 | Macroeconomic | fed, interest rate, inflation, gdp, jobs report, unemployment, recession, treasury yield |
| 9 | Geopolitical | tariff, trade war, china, sanction, brexit, war, election |
| 10 | Other | (no match) |

- Credit Event goes first so "debt downgrade" isn't caught by the Analyst Rating rule.
- **Functions:** `classify(text) -> str`, and `add_news_type(df, text_col) -> df`.
- **Test:** about 10 hand-written headlines, one per category.
- **Hand-check:** write 100 random rows (fixed seed) to `docs/news_type_check.csv` with an empty `correct` column. The user marks each Y or N. A short function then reports accuracy, which goes in the README.

**Done when:** the category counts are printed and the hand-check accuracy is reported. Expect Analyst Rating to dominate.

## Step 3: `src/build_engine_output.py` (MUST, 45 min)

**Input:**
- `signals_walkforward.csv` (news, 2016 to 2020-06)
- the tweet rows of `data/outputs/sentiment_output_test.csv`

**Output:** `data/outputs/engine_signals.csv`

| Column | Source |
|---|---|
| date | item date |
| company_name | from sp100_constituents.csv |
| ticker | |
| source | news / social |
| text | headline or tweet |
| sentiment_score | sentiment_score |
| impact | severity 1–10 |
| news_type | Step 2 |
| confidence | \|P(10)+P(6)+P(4) − P(7)−P(5)−P(1)\| = \|sentiment_score\| |

Also write a 1,000-row sample to `data/samples/engine_signals_sample.csv`.

**CUT:** a FastAPI `/signals?ticker=` endpoint. The brief allows file output.

**Done when:** the file exists with all of the mentor's columns. Send the sample to Rajesh.

## Step 4: `src/universe.py` (MUST, 1.5 h; finish what exists)

- **`ffmc_panel()`** → a DataFrame [date × ticker] = unadjusted close × cleaned shares (as-of, forward-filled) × float factor (BRK.B 0.85, Alphabet 0.89 over all classes as GOOGL).
- **`review_dates(calendar)`** → the last trading day of Mar/Jun/Sep/Dec, from 2015-12-31 to 2020-03-31.
- **`select_constituents(ranks, current, n=20, entry=20, exit=25)`**:
  1. keep current members ranked up to 25
  2. fill to 20 from the best-ranked non-members
  3. if over 20 (because new names ranked 20 or better are forced in), drop the worst-ranked holdovers
  4. return exactly 20
- **`membership()`** → a dict {review_date: list of 20}.
- **Tests:** the mentor's §7 example (S at 21 stays unless forced out, T at 27 exits, U at 18 and V at 20 enter, always exactly 20), and a case where a stock with no data is excluded.

**Done when:** the constituents printed at each review look sensible (AAPL, MSFT, AMZN, GOOGL, META, BRK.B, JNJ, JPM, V, PG, XOM, ...).

## Step 5: `src/signals.py` (MUST, 1.5 h; simplified)

**Simplification:** event = company-day. All headlines for one ticker on one date form one event. This replaces cosine clustering; exact duplicates were already removed upstream.
- event sentiment = mean sentiment_score
- event importance I = max(severity) / 10
- event model confidence m = mean |score|

**Daily security signal**, for each trading day t and ticker:
- use events with date ≤ t, within the last 15 trading days
- Δt = trading days between the event date and t
- F = exp(−ln2/3 × Δt)
- S = Σ(S_e · I_e · F) / Σ(I_e · F), or 0 if there are no events
- **Confidence C** = product of:
  - mean m
  - (1 − exp(−n/2)), where n = number of events in the window
  - agreement = |Σ w·sign(S_e)| / Σ w, with w = I·F
  - source quality = 1.0 (news only)
- **Neutral zone:** if |S| < 0.15, then S_eff = 0.

**Output:** `data/processed/daily_signals.csv` with columns date, ticker, S, S_eff, C, n_events.

**Variant:** the same pipeline, but using tone_score instead of sentiment_score → `daily_signals_tone.csv`.

**Done when:** a few tickers show sensible values on known news days (e.g. a big earnings day).

## Step 6: `src/weights.py` (MUST, 1 h; finish what exists)

- `base_weights(ffmc_row, members)` = ffmc / sum over the members.
- `tilt(w_base, S_eff, C, alpha=0.30)` = w_base · exp(alpha · S_eff · C).
- `normalize(w)`.
- `apply_caps(w, w_base, max_w=0.12, max_tilt=0.25)`:
  1. clip each weight to [w_base·(1−0.25), min(w_base·1.25, 0.12)]
  2. redistribute the excess proportionally across the unclipped names
  3. repeat until stable (at most 50 iterations)
- `limit_change(w, w_prev, max_d=0.01)`: clip |w − w_prev| ≤ 0.01, then renormalise. Skip this on review days.
- **Tests:** weights sum to 1; caps hold; a positive S raises a weight; S = 0 returns w_base.

## Step 7: `src/backtest.py` (MUST, 2.5 h)

**Period:** trading days from 2016-01-04 to 2020-06-10. Daily returns come from adjusted closes.

**Timing:** weights set at the close of day t use signals dated ≤ t−1. They earn the return from t to t+1. So news dated d trades at the close of d+1 and earns from d+2, with no look-ahead.

**Loop for each strategy:**

```
members = membership at the latest review <= t-1 (effective the day after the review)
w_base  = base_weights(ffmc[t], members)      # FFMC moves with price, so this acts as drift
A        : w = w_base
A-capped : w = apply_caps(w_base, w_base, max_tilt=inf)   # 12% cap only
B        : w = normalize(tilt(w_base, S_eff, C=1))
C        : w = normalize(tilt(w_base, S_eff, C))
D        : w = limit_change(apply_caps(normalize(tilt(...))), w_prev_drifted)
D-tone   : D using daily_signals_tone
w_prev_drifted = w_prev * (1+r_t) / sum(w_prev * (1+r_t))
cost_t   = 0.0010 * sum(|w - w_prev_drifted|)
ret_{t+1} = sum(w * r_{t+1}) - cost_t
```

**Metrics**, one table, one row per strategy:
- total return, CAGR, annualised volatility, Sharpe (risk-free from ^IRX ÷ 100 ÷ 252, or 0), Sortino, max drawdown
- tracking error and information ratio vs A-capped
- average annual turnover, total cost, number of constituent changes
- average |tilt| = mean |w − w_base|, max single weight
- top-sector weight (GICS)
- **SHOULD:** a March 2020 return row, and returns on the top 1% of severity-event days
- **CUT:** a regression of next-day returns on S

**Charts** (matplotlib) saved to `docs/`:
- `index_levels.png`: all strategies + ^OEX, rebased to 100
- `weights_D.png`: stacked area of strategy D's weights
- `active_signals.png`: the number of members with S_eff ≠ 0 per day
- `drawdowns.png`

**Outputs:** `docs/backtest_report.txt`, `data/outputs/backtest_daily.csv` (date, strategy, index level, weights as JSON or wide).

**Expected result:** B–D ≈ A-capped (no next-day edge). Report it honestly. The interesting comparisons are March 2020 and the turnover vs cost trade-off.

## Step 8: `app.py` Streamlit dashboard (MUST, 2 h)

Install: `pip install streamlit plotly`, and add both to requirements.txt pinned. Run with `streamlit run app.py`.

**Sidebar:** pick a strategy, a date range and a ticker.

**Tabs:**
1. **Index:** line chart of the strategies + ^OEX, plus the metrics table.
2. **Weights:** stacked area of weights over time for the chosen strategy, plus a table of the weights on a chosen date (base vs final vs tilt).
3. **Signals feed:** a filterable table from `engine_signals.csv` (date, company, text, sentiment, impact, news type), plus a chart of the daily S for the chosen ticker.
4. **Engine results:** key numbers (70% same-day, ~50% next-day, the severity decile chart from the reports).

Read only the CSVs; no model runs inside the app.

## Step 9: Docs and submission (MUST, Thu–Fri)

1. **`README.md`** in the template from HANDOFF section 9. Quickstart = clone, `pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu`, download the Kaggle files into `data/raw/...`, then the run order:
   ```
   labels.py → run_model.py → severity.py → index_data.py → walk_forward.py →
   news_type.py → build_engine_output.py → signals.py → backtest.py → streamlit run app.py
   ```
   Also note which steps can be skipped because their outputs are committed. Add the AI-usage note.
2. **`LICENSE`:** MIT, with the user's name and 2026.
3. **`docs/architecture.png`:** write it as Mermaid, render at mermaid.live, export a PNG. The flow:
   - Kaggle news + tweets + yfinance
   - → cleaning/validation → market-reaction labels
   - → masking → TF-IDF+LR sentiment / ridge severity / rule news type (walk-forward)
   - → engine_signals.csv
   - → daily signals (freshness, confidence, neutral zone)
   - → index (FFMC top 20, buffer, tilt, caps)
   - → backtest + Streamlit dashboard
4. **`docs/presentation.pdf`**, 7 slides:
   1. Title: name, college
   2. Problem & approach: market-labelled NLP engine + mentor's index methodology
   3. System design: the architecture diagram
   4. Implementation highlights: data validation finds, leakage controls, TF-IDF vs FinBERT, walk-forward
   5. Key results: 70% same-day / ~50% next-day; the severity decile chart; the backtest table
   6. Domain impact: severity as a risk signal; priced-in-same-day insight; real index rules (buffer, caps, turnover)
   7. Limitations & next steps: news coverage gaps, survivorship bias, date-only news; next would be intraday news, fuller feeds, an LLM news-type classifier
5. **Video (~5 min):**
   - 30 s problem
   - 30 s run commands
   - 2.5 min dashboard walkthrough (signals feed → weights → index)
   - 1 min results + limitations

   Upload to YouTube as **unlisted** and test in an incognito window.
6. **GitHub:** create the PUBLIC repo `<college>-<name>-hackathon`, then `git remote add origin <url>` and `git push -u origin main`. Check that no file is over 100 MB, and test every link in incognito.

## Schedule

| Day | Steps |
|---|---|
| Tue Oct 6 | 0, 1, 2, 3 (send Rajesh the engine sample), 4 |
| Wed Oct 7 | 5, 6, 7 |
| Thu Oct 8 | 8, then README + LICENSE + architecture. Feature freeze at night |
| Fri Oct 9 | Deck, video, push, submit |
| Sat Oct 10 | Buffer only |

**If behind schedule, cut in this order:**
1. D-tone
2. the event-day metrics
3. strategies B and C (keep A, A-capped, D)
4. the dashboard's engine-results tab

Never cut the README, LICENSE, deck or video.
