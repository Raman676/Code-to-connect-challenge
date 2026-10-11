# Project status — AI/NLP Risk Engine + news-sentiment-tilted index

*Last updated 2026-10-07. For a developer taking over this repository. The hackathon deadline
is 2026-10-10 (target: submit Fri 2026-10-09).*

## 0. Current state at a glance (2026-10-07)

- **2026-10-11: benchmark fix.** Tracking error, information ratio, beta and alpha are now measured vs `^OEX`, the
  parent index (primary), and vs Strategy A (secondary: the news tilt alone). Before they were measured vs A-capped,
  which was wrong (A-capped is one of our strategies). D vs ^OEX: TE 3.67%, IR 1.33, beta 1.030; D vs A: TE 0.31%,
  IR -0.91 (active return -0.28%/yr). `^OEX` is price-only, so active return and alpha vs `^OEX` are overstated by
  roughly the dividend yield (~2%/yr); TE is barely affected. 114 tests pass.

- **Last code commit:** `0762630` (universe, weights, backtest engine, Strategies A and A-capped). No code
  has changed since; all 85 tests passed at that commit.
- **Working tree:** clean after the 2026-10-07 handover commit, which added:
  - `DECISIONS.md` entries: news-type classifier and hand-check method, engine output + API,
    tweet impact scale, engine report, META/BKNG back-fill,
    and the **planned company-day event simplification** (see step 1).
- **Done 2026-10-07 (later commit):** steps 1, 2 and part of 4. `src/signals.py` (company-day events; the only
  lag: news dated d -> weights at d+1's close), Strategies B, C, D, D-tone in `backtest.py`,
  `src/export_results.py` (`data/outputs/backtest_results.xlsx` with 7 sheets + `backtest_summary.csv`) and
  `src/charts.py` (`docs/figures/`: index levels, drawdowns, D weights, active signals). 99 tests pass.
  Result: no sentiment strategy beats Strategy A meaningfully (D vs A: active return -0.28%/yr, TE 0.31%, IR -0.91;
  see the 2026-10-11 note above) - as the next-day check predicted. Run: `python src/backtest.py` (~3 min).
- **Done (dashboard):** `app.py` (Streamlit, 5 tabs: Index, Weights, Signals feed, Engine results,
  Sensitivity), helpers in `src/dashboard_data.py`. Run: `streamlit run app.py` (after `python src/backtest.py`).
  Strategy D alpha sensitivity (0.3/1.0/3.0) is in the Excel file and `backtest_sensitivity.csv`. 106 tests pass.
- **Done (step 3):** `src/analysis.py` -> `docs/analysis_report.txt` (run after the backtest, ~1 min).
  Daily IC mean 0.0099 (t 1.40, hit rate 52.5%); Fama-MacBeth S_eff coef 0.00104 (t 0.99); neither
  significant. Top-1% news days (all Feb-Mar 2020): D -14.7 bp vs A in total; COVID window: all within 0.1 pp of A.
  `signals.py` now also returns an importance panel I (used to rank news-heavy days).
- **Done (submission):** README (template, quickstart, testing), LICENSE, `docs/architecture.png`,
  `docs/presentation.pdf`, demo video link, `run_tests.py`. Submitted 2026-10-11.
- **Not done (possible next steps):** news-type hand check (`docs/news_type_check.csv`), news-type importance
  weights (all types weighted equally), daylight-saving check on news timestamps.

The project has two parts:

1. **The AI/NLP Risk Engine (done).** It turns S&P 100 news headlines and tweets into structured
   signals: sentiment (−1..1), impact (severity 1–10) and news type. The signals are served as a CSV and a REST API.
2. **Module A: a Top-20 S&P 100 index (done).** The index is weighted by free-float market cap (FFMC) and
   tilted by the engine's news sentiment, following the index methodology summarised below.
   Membership, base weights, constraints, all six strategies (A, A-capped, B, C, D, D-tone), the charts, the
   dashboard and the submission documents are done.

Read these first: `DECISIONS.md` (every design decision and its reason), `docs/engine_report.txt`
(the finished engine on one page) and `docs/backtest_report.txt` (current index results).

---

## 1. Setup

- **Runtime:** Python 3.11.3 on Windows 11.
  - On Windows, set `PYTHONIOENCODING=utf-8` before running scripts (in PowerShell: `$env:PYTHONIOENCODING='utf-8'`).
  - Windows Smart App Control blocked a pandas DLL on this machine. It was switched off to continue.
- **Install** (the CPU build of torch comes from PyTorch's index):
  ```
  pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
  ```
- **Raw data is not in the repo** (`data/raw/` is gitignored because the files are too large for GitHub). Download them from Kaggle into:
  - `data/raw/tweets/`: "Tweet Sentiment's Impact on Stock Returns" (`full_dataset-release.csv`, `reduced_dataset-release.csv`)
  - `data/raw/news/`: "Daily Financial News for 6000+ Stocks" (`analyst_ratings_processed.csv`,
    `raw_analyst_ratings.csv`, `raw_partner_headlines.csv`)
- **Already in the repo:**
  - Financial PhraseBank: `data/phrasebank/`
  - S&P 100 list: `data/universe/sp100_constituents.csv`, from Wikipedia as of 2026-09-21
  - yfinance downloads: `data/prices/`
  - all processed label and signal files: `data/processed/`, `data/outputs/`
- **Models:** the FinBERT model (`yiyanghkust/finbert-tone`, about 440 MB) is downloaded from Hugging Face on first use.

## 2. Run commands (in pipeline order, from the project root)

| Step | Command | Output | Time (CPU) |
|---|---|---|---|
| Raw data report | `python src/inspect_data.py` | `docs/data_inspection.txt` | ~3 min |
| Market-reaction labels (47-company pool) | `python src/labels.py` | `data/processed/labels_*.csv`, `docs/labels_report.txt` | ~1 min |
| Sentiment models | `python src/run_model.py` | `data/outputs/sentiment_predictions_test.csv`, `docs/model_report.txt` | ~20 min |
| Model checks | `python src/model_checks.py` | `docs/model_checks.txt` | ~11 min |
| Severity model | `python src/severity.py` | `data/outputs/sentiment_output_test.csv`, `docs/severity_report.txt` | ~2 min |
| S&P 100 market data + checks | `python src/index_data.py` (`--refresh` re-downloads) | `data/prices/sp100_*`, `docs/index_data_checks.txt` | ~2 min |
| Walk-forward signals 2016–2020 | `python src/walk_forward.py` | `data/processed/signals_walkforward.csv`, `docs/walkforward_report.txt` | ~45 min |
| News type | `python src/news_type.py` (`--accuracy` after the hand check) | counts + `docs/news_type_check.csv` | <1 min |
| Engine output | `python src/build_engine_output.py` | `data/outputs/engine_signals.csv` (+ sample) | <1 min |
| API | `uvicorn src.api:app --reload` | http://127.0.0.1:8000/docs | — |
| Index backtest (all strategies) | `python src/backtest.py` | `docs/backtest_report.txt`, `data/outputs/backtest_*.csv/.xlsx`, `docs/figures/` | ~4 min |
| Predictive tests (IC, Fama-MacBeth, events) | `python src/analysis.py` | `docs/analysis_report.txt` | ~1 min |
| Dashboard | `streamlit run app.py` | http://localhost:8501 | — |

**Tests:** one plain script per module (pytest isn't installed; the files also work under pytest).
```
python tests/test_<module>.py          # e.g. python tests/test_weights.py
```
There are 16 test files and 85 tests, and all pass as of commit `0762630`. Every new module in `src/` must get a test (project rule).

---

## 3. What's done

### 3.1 Engine (complete; see `docs/engine_report.txt`)
| Module | What it does |
|---|---|
| `sentiment_data.py` | Financial PhraseBank loader (encoding repair, agreement levels, split), the benchmark only |
| `inspect_data.py` | Raw-data inspection; `read_chunks` (CSV reader that handles stray `\r`), `POOL` (47 companies) |
| `prices.py` | Adjusted-close cache for the 47-company pool |
| `labels.py` | Market-reaction labels: anchors, returns, volatility filter, sign-pattern impact table |
| `text_prep.py` | Masks company names, tickers, CEOs, subsidiaries, URLs and @mentions (S&P 100 coverage) |
| `modeling.py` | Sentiment score from class probabilities, metrics, TF-IDF + LR, FinBERT loader, embedding cache, baselines |
| `run_model.py` / `model_checks.py` | Model comparison, random baselines, tokenizer check, next-day check |
| `severity.py` | Ridge severity model; impact = decile of the prediction among training predictions |
| `walk_forward.py` | Yearly retrain (10-trading-day gap) and out-of-sample scoring of every S&P 100 headline, 2016–Jun 2020 |
| `news_type.py` | Keyword rules, 10 categories, first match wins |
| `build_engine_output.py` | `data/outputs/engine_signals.csv`: 66,158 rows (news + social) |
| `api.py` | FastAPI service: `/health`, `/signals`, `/signals/summary` |

The most important finding is the next-day check. Sentiment matches the **same-day** move (walk-forward accuracy 0.572; company-day accuracy 0.604), but it barely predicts the **next** day. The rank correlation with the tradable next-day return is 0.013 overall: about 0 in 2016–2019 and 0.058 in H1 2020. So the index overlay should not be expected to beat Strategy A.

### 3.2 Index (built and tested; Strategy A and A-capped run)

**Methodology in brief** (applied to the current S&P 100 with `^OEX` as the benchmark):
- Rank the universe by FFMC and select the top 20 with a 20/25 buffer, reviewed quarterly.
- Base weights are FFMC weights.
- The sentiment tilt is w = base × e^(α·S·C), with α = 0.30 and a neutral zone of |S| < 0.15.
- Risk controls (Strategy D): 12% max weight, ±25% max tilt around the base weight, and a 1 percentage point max daily weight change (not applied on review days).
- Costs are 10 bps per unit of weight traded.
- Weights decided at day t's close earn returns from day t+1.

**`src/index_data.py`**: market data and cleaning.
- **Downloads:**
  - `ensure_downloads`, `download_closes`, `download_shares_history`, `download_info`, `download_splits`, `download_benchmarks`
  - outputs: `data/prices/sp100_close.csv`, `sp100_adj_close.csv`, `sp100_shares_history.csv`, `sp100_info.csv`, `sp100_splits.csv`, `benchmarks.csv`
- **`split_adjust_shares`**: puts share counts on today's split basis, like yfinance closes.
- **`clean_shares`**: applies the manual drops, removes spikes (a jump over 25% that reverts within 30 days), keeps the known merger events, and flags the remaining jumps.
  - Output: `sp100_shares_clean.csv`.
  - `MANUAL_DROPS`: ACN 2015-12-17, ISRG 2017-08-28, NVDA 2020-03-17.
  - `KNOWN_EVENTS`: AVGO 2016, BMY 2019, CVS 2018, TMUS 2020.
- **`backfill_shares`**: META and BKNG get their first known count before their share history starts. yfinance's share history only covers the current ticker, and Meta was FB until 2022. LIN, RTX and DELL are deliberately not back-filled.
- **`float_factors`**: floatShares ÷ sharesOutstanding, using today's value held constant.
  - Alphabet: computed over all share classes (0.89).
  - BRK.B: **0.85, our estimate** (yfinance's float figure is broken).
- **Other helpers:** `shares_asof`, `quarter_end_reviews`, `news_counts_by_year`, `load_universe`.
- **Tested:** universe file, as-of lookup, float ratio and factors, quarter-end dates, split adjustment, spike/known/flagged cleaning, manual drops, back-fill.

**`src/universe.py`**: FFMC, ranking, selection, reviews.
- **`load_market_data()`**: returns a `MarketData` object with closes, adjusted closes, the dates × tickers share table, float factors, sectors and company names. Columns are index tickers; GOOG is dropped, so Alphabet counts once as GOOGL.
- **`ffmc(data)`** = close × split-adjusted shares × float factor.
- **`rank_universe`**: rank 1 = largest FFMC.
- **`select_constituents`**: the exact-20 rule.
  - keep members ranked ≤ 25
  - add non-members ranked ≤ 20
  - if over 20, drop the lowest-ranked holdovers; if under 20, fill from the best-ranked non-members
  - The methodology's §7 example ends with 21 names; this rule resolves it to 20.
- **`review_dates`**: the last trading day of each quarter, from 2015-12-31.
- **`effective_date`**: the next trading day, when the new membership is traded.
- **`run_reviews`**: members, ranks, additions and deletions per review.
- **Tested:** first selection, the §7 example, buffer at 25 vs 26, fill when short, unranked members, dates, as-of shares, FFMC.

**`src/weights.py`**: weights and constraints.
- `base_weights`, `tilt(base, S, C, alpha)`, `normalise`.
- **`fit_to_bounds(raw, lower, upper)`**: weights = clip(k × raw, lower, upper), summing to 1, with the single scale k found by bisection. This one solver handles every constraint.
- **`cap_weights`**: the 12% cap, for A-capped.
- **`constrained_weights(raw, base, previous, review_day)`**: Strategy D's bounds.
  - the tilt band ±25%
  - the 12% cap, which wins over the tilt band
  - the 1 pp daily change limit, which wins over the cap and band between reviews and doesn't apply on review days
  - It returns a `relaxed` flag if a day was infeasible.
- **Tested:** the §15 tilt example, the bound solver (including infeasible bounds), capping and pro-rata redistribution, the review-day exemption, and that the 1 pp limit wins.

**`src/backtest.py`**: engine and metrics.
- **`simulate(adj_close, days, target_fn)`**: the daily loop. Weights drift with prices; at each close, trade to the target and pay 10 bps × Σ|Δw|; earn the next day's return. `target_fn` returns `None` for "hold".
- **`review_target_fn`**: Strategies A and A-capped.
- **`performance`**:
  - total return, CAGR, volatility, Sharpe and Sortino (risk-free rate from `^IRX`), max drawdown
  - tracking error, information ratio, beta and alpha vs a benchmark
- **`portfolio_stats`**: turnover, costs, constituent changes, max weight, sector concentration (largest-sector weight and HHI, using GICS sectors from the universe file).
- **Tested:** buy-and-hold drift, rebalancing costs, metric formulas, risk-free conversion.

### 3.3 Current backtest numbers (`docs/backtest_report.txt`, 2016-01-04 → 2020-06-30)

| Metric | A | A-capped | ^OEX (price only) |
|---|---|---|---|
| Total return | 98.7% | 97.6% | 59.2% |
| CAGR | 16.55% | 16.41% | 10.93% |
| Volatility | 20.35% | 20.28% | 19.35% |
| Sharpe / Sortino | 0.795 / 0.736 | 0.791 / 0.729 | 0.571 / 0.526 |
| Max drawdown | −29.7% | −29.7% | −31.5% |
| Tracking error / IR vs ^OEX | 3.76% / 1.37 | 3.68% / 1.36 | — |
| Beta / alpha vs ^OEX | 1.035 / 4.77% a year | 1.031 / 4.66% a year | — |
| Turnover (one-way) / cost | 19.0% / 3.8 bps a year | 20.1% / 4.0 bps a year | — |
| Max single weight | 16.2% (AAPL) | 14.4% (AAPL) | — |
| Constituent changes | 29 additions, 29 deletions over 17 reviews | same | — |
| Largest sector weight (average) / sector HHI | 25.9% / 0.183 | 25.3% / 0.182 | — |

Vs Strategy A: A-capped TE 0.23% / IR −0.61; B 0.10% / −0.98; C 0.06% / 0.13; D 0.31% / −0.91; D-tone 0.32% / −1.09.
`^OEX` is price-only, so active return and alpha vs `^OEX` are overstated by roughly the dividend yield (~2%/yr).

**How to read these numbers:**
- **Most of the alpha vs `^OEX` isn't skill.** About 2% a year is dividends that `^OEX` excludes; most of the rest is **survivorship bias** (today's S&P 100 members applied to 2016–2020) and concentration in the mega caps.
- **Strategy A is the fair benchmark** for the sentiment strategies.
- **A-capped can exceed 12% between reviews** (up to 14.4%), because the cap applies only at reviews. Strategy D will hold the cap daily.

**Membership:**
- First review (2015-12-31): AAPL, GOOGL, MSFT, XOM, GE, AMZN, JNJ, META, WFC, BRK.B, JPM, PG, T, PFE, V, VZ, BAC, DIS, KO, HD.
- Last review (2020-03-31): MSFT, AAPL, AMZN, GOOGL, META, BRK.B, V, JNJ, JPM, PG, MA, UNH, VZ, T, INTC, HD, WMT, MRK, BAC, PFE.

---

## 4. What's left, in order

### Step 1: `src/signals.py`, security-level sentiment and confidence (methodology §9–13)
- **Input:** `data/processed/signals_walkforward.csv`, which has per headline: `date, ticker, text, p_1..p_10, sentiment_score, severity, tone_*`. Add `news_type` with `news_type.add_news_type`.
- **Availability:** a headline dated d can first be used at the close of the **first trading day after d** (call it T). Weights set at T's close earn the return from T to T+1. The trading calendar is `MarketData.close.index`.
- **Events (§11), simplified per `DECISIONS.md` ("Planned simplification for the index signal step"):**
  an event = all headlines for one ticker on one date (exact duplicates are already removed). This replaces
  the earlier plan of TF-IDF cosine ≥ 0.8 clustering with union-find. Per event:
  - S = mean `sentiment_score`
  - importance I = **max** `severity` ÷ 10
  - model confidence = mean |P(pos) − P(neg)|, where P(pos) = p_10 + p_6 + p_4 and P(neg) = p_7 + p_5 + p_1
  - availability day = the earliest headline's availability day
- **Security sentiment (§10):** for ticker i on day T, over events available in (T−15, T] trading days:
  - S_i,T = Σ S·I·F ÷ Σ I·F, with F = e^(−λΔt), λ = ln 2 ÷ 3 (half-life 3 trading days) and Δt in trading days
  - **Open decision:** importance may later be multiplied by a news-type weight (e.g. Earnings 1.0, Analyst Rating 0.8, Other 0.5). These weights are not decided; ask the owner.
- **Confidence (§12, approved formula):** C = model confidence × independence × agreement × source quality.
  - model confidence: the w-weighted mean over events
  - independence = 1 − e^(−n/2), with n = Σ F (freshness-weighted event count)
  - agreement = |Σ w·sign(S)| ÷ Σ w, with w = I·F
  - source quality = 1.0 (news only; tweets are excluded from the backtest)
- **Neutral zone (§13):** S_eff = 0 if |S| < 0.15.
- **Output:** a daily panel (date, ticker, S, C, S_eff, n_events, active = S_eff ≠ 0) for every S&P 100 ticker, 2016-01-04 → 2020-06-30.
  - Note: news ends 2020-06-11, so signals decay to zero over the last ~13 trading days.
- **Variant (D-tone):** the same pipeline using `tone_score` instead of `sentiment_score`, and |tone_pos − tone_neg| as model confidence.
- **Tests:** decay, the 15-day window, events (same ticker + same date merge, different dates don't), the confidence formula, the neutral zone, the availability lag (a weekend item becomes usable on Monday).

### Step 2: Strategies B, C, D, D-tone (§14–18, 22, 24) in `backtest.py`
- **Membership:** from `run_reviews`, as in A. **Base weights** = FFMC weights of the current members, recomputed **daily** (§23).
- **Daily targets:**
  - **B:** `normalise(tilt(base, S_eff, C=1))`, no risk controls.
  - **C:** `normalise(tilt(base, S_eff, C))`, no risk controls.
  - **D:** `constrained_weights(tilt(base, S_eff, C), base, previous=drifted weights, review_day=(day is an effective date))`. Count the `relaxed` days.
  - **D-tone:** D with the finbert-tone signals.
- **Comparisons (changed 2026-10-11):** every strategy's TE, IR, beta and alpha vs `^OEX` (parent index, primary) and vs Strategy A (secondary). No metric is measured vs A-capped.
- **Also save** `data/outputs/backtest_weights.csv` (date, strategy, ticker, weight, base_weight, S_eff, C): about 140k rows, needed for the charts and dashboard.

### Step 3: remaining §25 metrics
- **Average sentiment tilt** = mean over days of Σ|w − base| ÷ 2. Also the **number of active members per day** (members with S_eff ≠ 0); the owner asked for a chart of it.
- **Performance during major news events:**
  - active return (strategy − A) on the top 1% of days by Σ|S·I| across members
  - over 2020-02-19 → 2020-04-30 (COVID)
- **Key test, "does sentiment predict beyond FFMC and price?":**
  - **Daily IC:** each day, the Spearman correlation across members between S_eff and the next day's return. Report the mean, t-stat and hit rate.
  - **Fama–MacBeth:** each day, regress next-day returns on S_eff, the past 1-day return and the past 5-day return; average the coefficients and t-stats. numpy only.
- **Market impact:** not modelled; state this in the report.

### Step 4: charts (matplotlib, saved to `docs/figures/` as PNG, 150 dpi)
1. NAV of A, A-capped, B, C, D, D-tone and `^OEX` (log scale).
2. Drawdowns.
3. **Active-signal members per day** (required by the owner).
4. Strategy D weights over time for the top names, plus the base vs tilted weight for 2–3 example stocks.
5. Sector weights over time (Strategy D).
6. Daily IC (rolling 60-day mean).

### Step 5: Streamlit dashboard (Module A requires a dashboard of changing weights over time)
- **Install:** `streamlit` isn't installed yet. Ask the owner before installing (project rule), then pin it in `requirements.txt`.
- **File:** `src/dashboard.py`, run with `streamlit run src/dashboard.py`.
- **Inputs:** `backtest_nav.csv`, `backtest_weights.csv`, `engine_signals.csv`, `docs/backtest_report.txt`.
- **Views:**
  - strategy selector
  - NAV and drawdown
  - weights over time (stacked area, with a date slider)
  - per-ticker view: weight vs base weight, S_eff and C, plus that ticker's headlines with sentiment, impact and news type
  - constituents and changes per review
  - metrics table

### Step 6: submission documents (from the submission guidelines)
- **`README.md`** (mandatory template). Sections:
  - title, candidate name, college email ID, college, demo video link, slide-deck link
  - 1 Overview / problem and approach
  - 2 Architecture and tech stack (embed `docs/architecture.png`)
  - 3 Dataset used: sources (the Kaggle datasets, yfinance, Financial PhraseBank, the Wikipedia S&P 100 list as of 2026-09-21) and assumptions; say `data/raw` isn't committed and how to download it
  - 4 Quickstart: runtime and the commands from section 2
  - 5 Key results and domain impact
- **`LICENSE`:** MIT is recommended; the repo must be public. It needs the owner's name and year.
- **`docs/architecture.png`:** a high-resolution diagram of the data flow:
  - data sources → ingestion and cleaning → masking → models (sentiment, severity, news type)
  - → walk-forward signals → `engine_signals.csv` and the API
  - → signal aggregation (events, S, C) → index (universe → base weights → tilt → constraints) → backtest → dashboard
- **Owner's tasks:** `docs/presentation.pdf` (5–7 slides) and the demo video (YouTube, unlisted, linked from the README).
- **Repo:** public, named `<college>-<candidate-name>-hackathon`, with all data used under `/data` (already true, apart from the gitignored raw downloads).

---

## 5. Known issues and open decisions
- **Not yet decided:**
  - **News-type weights for importance** (see step 1).
  - **Tweet severity model.** The full model (Spearman 0.274) is in use, although trailing volatility alone does better (0.408).
  - **News-type accuracy is unknown** until the owner fills in `docs/news_type_check.csv`; 55.5% of headlines are "Other".
- **Committed results that are slightly stale:**
  - **Walk-forward training data.** The run's training data still included 99 old SanDisk headlines. They were removed from the output afterwards, and future runs drop them before training. A re-run is optional (~45 min).
  - **Masking changes.** The masking lists were extended after `docs/model_report.txt` was produced: ambiguous tickers such as COST, LOW and NOW are now masked only as "(TICKER)". Re-running `run_model.py` gives slightly different numbers than the committed report.
  - **`DECISIONS.md` is inaccurate in one place.** Its entry "Git: label files not committed" doesn't match the repo: `data/processed/labels_*.csv` *are* committed (36 MB + 6 MB). The owner should correct the entry or untrack the files.
- **Data assumptions to state in the report:**
  - **Timestamps:** news timestamps failed the reliability check, so all news is date-only. The theory that a winter-time labelling bug causes the one-hour gaps is untested.
  - **Coverage:** no headlines exist for MSFT, BRK.B, CVX, COST and others; JPM, V, WMT, INTC, DIS, T and BA have ≤ 10 each. Many index members will never tilt.
  - **Free float and universe:**
    - free-float factors are today's values held constant
    - BRK.B 0.85 is our estimate
    - current S&P 100 members are used historically (survivorship bias)
    - `^OEX` is price-only
- **Project conventions:**
  - one step at a time, with the owner's approval
  - ask before installing packages or downloading data
  - ask before deviating from an approved plan
  - simple, commented code
  - a test for every module
  - pinned `requirements.txt`
  - no Co-Authored-By lines in commits
