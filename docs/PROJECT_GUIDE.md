# Project Guide: what every file does

This guide walks through the repository folder by folder. It explains what each file does, why it exists and
how the pieces connect. Read it top to bottom once and you can explain the whole project.

---

## 1. The big picture in one minute

The project has **two halves**:

1. **The Risk Engine** reads financial text (news headlines and tweets) and outputs, for each item:
   - a **sentiment score** from -1 (bearish) to +1 (bullish)
   - an **impact score** from 1 to 10 (how big the price move is likely to be)
   - a **news type** (Earnings, Analyst Rating, M&A, ...)
2. **Module A, the index rebalancer**, builds a Top-20 S&P 100 index weighted by free-float market cap (FFMC),
   tilts the weights towards stocks with good news and away from stocks with bad news, and backtests it from
   2016 to mid-2020.

The data flows like this (see `docs/architecture.png`):

```
raw text -> check & clean -> market-reaction labels -> mask company names -> train models (walk-forward)
        -> engine output (CSV + API) -> daily stock signals -> index weights -> backtest -> analysis + dashboard
```

**Three words you will hear everywhere:**

- **Label:** the "right answer" a model learns from. Here it comes from what the stock price actually did
  after the news, not from a human's opinion.
- **Walk-forward:** to score the headlines of year Y, the models are trained only on data before year Y. Every
  score is out of sample, as it would have been in real life.
- **Lag:** news dated day d is first used at the close of the next trading day (d+1), and those weights earn the
  return from d+1 to d+2. This is the only delay in the system, and it prevents look-ahead bias.

---

## 2. Top-level files

| File | What it is |
|---|---|
| `README.md` | The submission front page (mandatory template): overview, architecture, data, quickstart, results. |
| `LICENSE` | MIT open-source licence. |
| `requirements.txt` | Every Python package with its exact version. Install with `pip install -r requirements.txt`. |
| `DECISIONS.md` | The decision log: every important design choice, **with the reason**. Use it to answer "why did you do X?". |
| `app.py` | The **Streamlit dashboard** (5 tabs). Run `streamlit run app.py`. |
| `.gitignore` | Files git must not upload: raw Kaggle data (too large), caches, the mentor's private document, personal notes. |

---

## 3. `src/`: the code, in pipeline order

### 3.1 Stage 1: raw data and labels

| File | What it does | Key ideas to explain |
|---|---|---|
| `sentiment_data.py` | Loads the **Financial PhraseBank** (sentences labelled by humans as positive, neutral or negative). Used **only as a benchmark**. | Repairs text encoding, handles annotator agreement levels (50%, 66%, 75%, 100%), splits the data. |
| `inspect_data.py` | **Data validation** of the raw Kaggle files. Writes `docs/data_inspection.txt`. | Found broken rows (line breaks inside tweets), 37% duplicate tweets, unreliable news timestamps (so news is **date-only**). Defines `POOL`, the 47 companies used for labelling, and a safe CSV reader. Also holds small report helpers used by other scripts. |
| `prices.py` | Downloads and caches **adjusted closing prices** from yfinance for the 47 pool companies. | Adjusted prices include dividends and splits, so returns are correct. |
| `labels.py` | Builds the **market-reaction labels**. Writes `data/processed/labels_*.csv` and `docs/labels_report.txt`. | Anchor = the last close before the news. **Sentiment label** = sign of the 1-day return. Items are kept only if volatility rose afterwards. **Impact** comes from the 1/2/3/7-day return pattern (for example all four up = 10, all four down = 1). |

### 3.2 Stage 2: text models

| File | What it does | Key ideas to explain |
|---|---|---|
| `text_prep.py` | **Masks** company names, tickers, CEO names, subsidiaries, URLs and @mentions, replacing them with `[COMPANY]` etc. | Without masking, the model would learn "Apple = up" instead of reading the news. |
| `modeling.py` | The model toolbox: TF-IDF + logistic regression, FinBERT loading, the sentiment score formula, evaluation metrics, baselines. | **sentiment_score = P(bullish classes) - P(bearish classes)**, a number between -1 and +1. |
| `run_model.py` | Trains and **compares** the sentiment models. Writes `docs/model_report.txt` and test predictions. | TF-IDF + logistic regression beat FinBERT embeddings, and is faster and easier to explain. |
| `model_checks.py` | Sanity checks: random baselines, FinBERT tokenizer check on PhraseBank, top words, the **next-day check**. Writes `docs/model_checks.txt`. | The next-day check is the key finding: same-day accuracy is good, next-day accuracy is about 50% (the news is already priced in). |
| `severity.py` | The **impact (severity) model**: ridge regression that predicts the size of the move. Writes `docs/severity_report.txt`. | Inputs: the text, recent volatility, yesterday's news count. Output: the decile of the prediction, giving the 1-10 impact. |
| `news_type.py` | Assigns one of **10 news types** with keyword rules (first match wins). `--accuracy` scores the hand check. | No labelled news-type data exists, so rules are transparent and need no training. 55.5% of headlines end up as "Other". |
| `walk_forward.py` | Re-trains the models **every year** and scores every S&P 100 headline from 2016 to June 2020 out of sample. Writes `data/processed/signals_walkforward.csv`. | A 10-trading-day gap between training and scoring prevents leakage. Takes about 45 minutes. |

### 3.3 Stage 3: engine output

| File | What it does | Key ideas to explain |
|---|---|---|
| `build_engine_output.py` | Combines scored news and tweets into **one table**: `data/outputs/engine_signals.csv` (66,158 rows). | Columns: date, company, ticker, source, text, sentiment_score, impact, news_type, confidence. |
| `api.py` | A **REST API** (FastAPI) serving the engine output. Run `uvicorn src.api:app --reload`; docs at `http://127.0.0.1:8000/docs`. | Endpoints: `/health`, `/signals` (filters: ticker, dates, source, news type, limit), `/signals/summary`. |

### 3.4 Stage 4: the index (Module A)

| File | What it does | Key ideas to explain |
|---|---|---|
| `index_data.py` | Downloads and **cleans** S&P 100 market data: prices, share counts, splits, free float, `^OEX` and the T-bill rate. Writes `docs/index_data_checks.txt`. | Share counts are put on today's split basis, spikes are removed, and META/BKNG are back-filled. Free float = floatShares / sharesOutstanding (BRK.B 0.85 is my estimate). |
| `universe.py` | **Who is in the index**: FFMC = price x shares x float factor. Ranks the 100 companies and picks the Top 20 each quarter. | **Buffer rule:** a member stays while ranked in the top 25, and a newcomer must reach the top 20, which reduces turnover. Membership changes at the close of the next trading day after a review. |
| `signals.py` | Turns headline scores into a **daily sentiment S and confidence C per stock**. | Event = one company on one date. Newer news counts more (half-life 3 trading days, 15-day window). C = model confidence x independence (how many items) x agreement (do they point the same way). The neutral zone sets small signals (abs(S) < 0.15) to 0. **This is where the one-day lag is applied.** |
| `weights.py` | **Weight maths**: FFMC base weights, the tilt `base x exp(0.30 x S x C)`, and the constraints. | One solver (`fit_to_bounds`) handles every rule: 12% cap, +-25% tilt band, at most 1 percentage point of change per day between reviews. |

### 3.5 Stage 5: backtest, results and analysis

| File | What it does | Key ideas to explain |
|---|---|---|
| `backtest.py` | **Runs the strategies** day by day, 2016-01-04 to 2020-06-30. Writes `docs/backtest_report.txt` and calls the export and chart modules. Run `python src/backtest.py` (~4 min). | Each day at the close: the weights have drifted with prices, trade to the target, pay 10 bp per unit traded, earn the next day's return. Strategies: **A** (FFMC Top 20), **A-capped** (12% cap at reviews), **B** (tilt, no confidence), **C** (tilt x confidence), **D** (C + all risk controls), **D-tone** (D using FinBERT-tone). Also runs D with alpha 1.0 and 3.0 for the sensitivity test. |
| `export_results.py` | Writes `data/outputs/backtest_results.xlsx` (8 sheets) and the CSV files the dashboard reads. | Sheets: Summary, Yearly, Daily_NAV, Daily_Weights_D, Turnover_Costs, Constituents, Sensitivity, Definitions (every formula and assumption). |
| `charts.py` | Saves 4 PNG charts to `docs/figures/`. | Index levels, drawdowns, D weights, number of stocks with an active signal. |
| `analysis.py` | **Does sentiment predict returns?** Writes `docs/analysis_report.txt`. Run `python src/analysis.py`. | **Daily IC:** each day, the rank correlation between S and the next day's return (mean 0.0099, t = 1.40). **Fama-MacBeth:** each day, regress next-day returns on S plus past returns, then average (S: t = 0.99). **News events:** the top 1% news days and the COVID window. Conclusion: no significant predictive power. |
| `architecture.py` | Draws `docs/architecture.png` with matplotlib. | Boxes and arrows of the whole pipeline. |
| `dashboard_data.py` | Loads the files for the dashboard and prepares tables (weights on a date, filtered signals, severity deciles). | Kept separate from `app.py` so it can be tested. |

---

## 4. `tests/`: one test file per module

Every file in `src/` has a matching `tests/test_<name>.py` (113 tests in total). Each is a plain script:

```powershell
$env:PYTHONIOENCODING='utf-8'
python tests/test_weights.py      # prints PASS/FAIL per test and a total
```

Tests worth mentioning to a jury:

- `test_backtest.py`: **`test_one_lag_single_fake_headline`** puts one fake headline through the whole
  pipeline. It proves the headline changes the weights at d+1's close and earns the return from d+1 to d+2, with
  exactly one lag.
- `test_signals.py`: decay, the 15-day window, weekend news usable on Monday, the confidence formula, the
  neutral zone.
- `test_weights.py`: the tilt example from the methodology, the caps, the daily change limit.
- `test_universe.py`: the buffer rule (rank 25 stays, rank 26 leaves), always exactly 20 members.
- `test_app.py`: opens the dashboard without a browser and checks every tab loads.

---

## 5. `data/`: inputs and outputs

| Folder / file | Contents |
|---|---|
| `data/raw/` | **Not in git** (too large). The original Kaggle downloads: `news/` and `tweets/`. See README section 3 to re-download. |
| `data/phrasebank/` | Financial PhraseBank (human-labelled sentences + licence). Benchmark only. |
| `data/universe/sp100_constituents.csv` | The S&P 100 list (Wikipedia, 2026-09-21): ticker, company, sector, yfinance symbol. |
| `data/prices/adj_close.csv` | Adjusted closes for the 47-company labelling pool (`prices.py`). |
| `data/prices/sp100_*.csv` | S&P 100 closes, adjusted closes, share history (raw and cleaned), splits, company info (`index_data.py`). |
| `data/prices/benchmarks.csv` | `^OEX` (S&P 100 price index) and `^IRX` (13-week T-bill, the risk-free rate). |
| `data/processed/labels_news.csv`, `labels_tweets.csv` | Market-reaction labels (`labels.py`). |
| `data/processed/fpb_sentiment.csv` | PhraseBank in the project's format (`sentiment_data.py`). |
| `data/processed/signals_walkforward.csv` | **Every S&P 100 headline 2016-2020 with out-of-sample scores** (`walk_forward.py`). The input to the index. |
| `data/outputs/engine_signals.csv` | **The engine's final output** (66,158 rows), served by the API. |
| `data/outputs/sentiment_predictions_test.csv`, `sentiment_output_test.csv` | Test-period predictions from `run_model.py` and `severity.py`. |
| `data/outputs/backtest_nav.csv` | Daily value of every strategy and `^OEX`. |
| `data/outputs/backtest_results.xlsx` | **The full results workbook** (8 sheets). |
| `data/outputs/backtest_summary.csv`, `backtest_sensitivity.csv` | The Summary and Sensitivity sheets as CSV. |
| `data/outputs/backtest_weights.csv` | Daily weights of every strategy with base weights (for the dashboard). |
| `data/outputs/signals_daily.csv` | Daily S, C, S_eff and event count per stock (for the dashboard). |
| `data/samples/` | Small samples: 1,000 rows of the engine output and 200 example input rows. |

---

## 6. `docs/`: reports and figures

| File | What it contains |
|---|---|
| `architecture.png` | The pipeline diagram (used in the README). |
| `PROJECT_GUIDE.md` | This file. |
| `STATUS.md` | Detailed project status: setup, run commands, every module, results, known issues. |
| `engine_report.txt` | **One-page summary of the engine**: sources, models, key results, limitations. |
| `data_inspection.txt` | Raw-data checks (`inspect_data.py`). |
| `labels_report.txt` | How many items were labelled and how (`labels.py`). |
| `model_report.txt` | Sentiment model comparison (`run_model.py`). |
| `model_checks.txt` | Baselines, tokenizer check, next-day check (`model_checks.py`). |
| `severity_report.txt` | Impact model results and decile tables (`severity.py`). |
| `walkforward_report.txt` | Year-by-year out-of-sample results (`walk_forward.py`). |
| `news_type_check.csv` | 100 random headlines for the manual news-type accuracy check. |
| `index_data_checks.txt` | Market-data cleaning checks (`index_data.py`). |
| `backtest_report.txt` | Members at each review, all metrics, sensitivity (`backtest.py`). |
| `analysis_report.txt` | Daily IC, Fama-MacBeth, news-event and COVID results (`analysis.py`). |
| `figures/*.png` | Index levels, drawdowns, Strategy D weights, active signals. |

---

## 7. How to run everything

```powershell
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
$env:PYTHONIOENCODING='utf-8'

# Results from the committed data (no raw data needed)
python src/backtest.py          # strategies, Excel, charts            ~4 min
python src/analysis.py          # predictive tests                      ~1 min
uvicorn src.api:app --reload    # API at http://127.0.0.1:8000/docs
streamlit run app.py            # dashboard at http://localhost:8501
```

The full pipeline from the raw Kaggle files is the 12-step table in README section 4.

---

## 8. The story in five sentences (for the jury)

1. I turned 1.4 million news headlines and 862k tweets into structured signals, labelling them by **what the
   market actually did**, not by human opinion.
2. The engine reads news well: **70% accuracy per company-day** on the same-day move, and its impact score ranks
   move size correctly (0.8% average move at impact 1, 4.3% at impact 10).
3. But the move is already in the price by the next close: **next-day accuracy is about 50%**, and the daily IC
   and Fama-MacBeth tests are not significant.
4. So the sentiment-tilted index performs almost exactly like the plain FFMC index, and a stronger tilt only adds
   costs.
5. The value of the engine is **risk monitoring**: flagging which index members have heavy or negative news.
   Using it as a return signal would need faster, intraday data.
