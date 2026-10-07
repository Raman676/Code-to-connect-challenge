# My Project Prep: Risk Engine + Sentiment-Tilted Index

## 1. The project in 30 seconds

The engine reads financial text (news headlines + tweets) and outputs, per item:
- **sentiment** (-1 to 1): which way the stock moves
- **impact** (1-10): how big the move will be
- **news type**

Those signals tilt the weights of a 20-stock index built with my mentor's methodology. Labels come from what the **market actually did**, not from human opinion.

## 2. What I did, step by step

**1. Data validation**
- Tweets: 862k (2017-18). News: 1.4M headlines (2009-20). Prices: yfinance.
- **Fixes:** broken line breaks inside tweets (143k corrupted rows), and 37% duplicate tweets.
- **Errors found in the data:**
  - The tweet file's return columns were backward-looking with the sign flipped, so I never used them.
  - The news timestamps failed a 95% check that I fixed *before* looking (only 74.5% agreed), so news is date-only.

**2. Labels (Rajesh's rules)**
- Anchor = the close before the news date.
- **Volatility filter:** keep an item only if the 10-day volatility after it is higher than before.
- **Sentiment** = the sign of the 1-day return.
- **Impact** = the pattern of 1/2/3/7-day returns: ++++=10, +++-=6, ++--=4, ----=1, plus my mirrors ---+=5, --++=7.
- Result: 125k tweets and 21.7k headlines labelled, roughly 50/50 positive and negative.

**3. The big discovery**
All tweets about one company on one day share a label, so 125k tweets = only 989 independent cases. That's why I:
- split by date, never randomly
- hid company names in the text (`[COMPANY]`)
- capped each company-day at 20 tweets

**4. Sentiment model**
- TF-IDF + logistic regression beat FinBERT.
- **News:** 61% per headline vs 51% baseline; **70% per company-day** (931 test days).
- **Tweets:** about 51%, no signal.
- **The catch:** on the *next* day's return (when we could actually trade) it's about 50%. The headline is priced in on the same day.

**5. Impact → severity (size of the move)**
- Rajesh's pattern-based impact came out near random: days 2-7 depend on later news that no text can see.
- So impact became severity, and it works: Spearman 0.60 per company-day, with the average move rising from 1.0% (impact 1) to 6.3% (impact 10).
- Most of the power comes from **volatility clustering** (jumpy stocks stay jumpy); text adds a little.

**6. Walk-forward (2016 to June 2020)**
- Retrained every year on past data only, then scored 64,127 headlines out-of-sample.
- **Sentiment:** no reliable next-day edge in 2016-19; a small edge in COVID 2020.
- **Severity:** works every year (Spearman 0.35-0.44).
- finbert-tone (no training) roughly matched my model.

**7. Index (Rajesh's methodology, adapted to the S&P 100)**
- Top 20 by free-float market cap, with a 20/25 buffer and quarterly reviews.
- New weight = base weight × e^(0.3 × sentiment × confidence).
- A neutral zone of ±0.15, a 12% max weight, a ±25% max tilt and a 1-percentage-point daily limit.
- Backtest strategies: A, A-capped, B, C, D, and D-tone.

**Data cleaning for the index:**
- 438 stock splits applied
- data spikes removed
- 3 manual fixes: ACN, ISRG, NVDA
- 4 real mergers kept: AVGO, BMY, CVS, TMUS

**Still to do:**
- finish the index modules and backtest
- news-type model (10 categories, keyword rules)
- final output file + dashboard
- README, deck, video, submission

## 3. Numbers to remember

| What | Number |
|---|---|
| News sentiment, same day, per company-day | 70% |
| Next-day (tradable) accuracy | ~50% |
| Severity Spearman (single test / walk-forward) | 0.60 / 0.35-0.44 |
| Severity deciles | 1.0% → 6.3% average move |
| Tweets: rows → independent cases | 125k → 989 |
| finbert-tone on PhraseBank (tokenizer check) | 76% |
| My score vs human tone | correlation 0.04 |

## 4. Jury and mentor Q&A

**Why not the dataset's return columns?** They don't match its own prices (1.5% agreement). They're backward-looking with the sign flipped. I rebuilt everything from yfinance.

**Why does TF-IDF beat FinBERT?** Price reactions are driven by event words ("downgrade", "beats"), not general tone. I verified FinBERT was working first (76% on PhraseBank).

**Why do tweets fail?** They're mostly customer chatter, there are only about 1k independent days, and the test period was a market-wide sell-off. They're still useful as an attention signal.

**Why change impact?** The week-long pattern depends on future news, and I measured it as random. Severity is predictable from information available at the time, and it matches the brief's definition of impact.

**Is 70% real?** Yes, for the same-day reaction. It's not tradable next day, which is consistent with efficient markets. The index applies signals the next day, so the backtest can't cheat.

**How do you prevent leakage?**
- date splits with 10-day gaps
- masked company names
- yearly walk-forward
- the test set scored once
- previous-day counts only (same-day counts would include future items)

**Limitations:**
- no news for MSFT, BRK.B and others, and only ~10 headlines for JPM, V, WMT and others, so those stocks never tilt
- survivorship bias (today's S&P 100 list)
- a constant free-float factor
- date-only news

## 5. Message for Rajesh (send now)

```
Rajesh, big update:
1. Walk-forward done: models retrained every year (2016–2020), so all 64k S&P 100
   headlines are scored out-of-sample.
2. Severity (impact) works every year: Spearman 0.35–0.44, and on the 2019–20 test
   the avg move rises from 1.0% (impact 1) to 6.3% (impact 10).
3. Sentiment: same-day accuracy up to 66%, but next-day (tradable) correlation ~0
   in 2016–19, a small positive edge only in COVID 2020. So headlines are priced in
   the same day. Off-the-shelf FinBERT-tone matches our model.
4. Index per your doc, on the S&P 100: FFMC from cleaned share data (438 splits,
   spikes removed, mergers kept), top 20 with the 20/25 buffer, quarterly reviews.
   Backtest window 2016–Jun 2020 (share history starts late 2015).
   Strategies A, A-capped, B, C, D + a FinBERT-tone variant.
5. Limitation: no news in the dataset for MSFT, BRK.B, CVX and only ~10 headlines for
   JPM, V, WMT, so roughly a third to half of the top 20 can't tilt.
Questions: (a) is 0.85 OK as BRK.B's free-float factor? (b) Given no next-day edge,
should I focus the story on risk (severity + drawdowns, e.g. March 2020) rather than
returns?
```

## 6. Switching from Claude Code to manual coding (phone Claude → WhatsApp → VS Code)

**Setup once:**
- Give the new Claude `HANDOFF_FOR_CLAUDE.md` first, and say: "Read this fully. Don't write code until I paste the file you need."
- In VS Code, open the `snp` folder. Open the terminal with Ctrl+`.

**The loop, for every change:**
1. Ask for ONE small step, e.g. "write `universe.py` select_constituents".
2. Paste the current file to Claude first, so it never guesses.
3. Ask for the **complete new file**, or an exact "replace X with Y" edit, with the file path.
4. In VS Code: open the file, select all (Ctrl+A), paste the new version, save (Ctrl+S).
5. Run it: `$env:PYTHONIOENCODING='utf-8'; python src/<file>.py`
6. Run the tests: `python tests/test_<name>.py`
7. **On an error:** copy the WHOLE error text and send it back. Don't summarise it.
8. **When it works:** `git add <files>` → `git commit -m "what changed"`
9. Add any decision to `DECISIONS.md`.

**Rules for yourself:**
- **Never paste code you don't understand.** Ask: "explain this function line by line."
- **Before a big change, save your state:** `git commit`. If something breaks: `git restore <file>`.
- **Long runs** (the walk-forward takes ~46 min): don't close VS Code or let the laptop sleep.
- **Check `git status` before committing.** Nothing from `data/raw/` should appear.
- **WhatsApp can mangle code** (smart quotes, broken indentation). Ask Claude to send code in code blocks, and if you get `IndentationError` or `SyntaxError`, check the quotes and spaces first.

**Useful commands:**
- `git status`, `git log --oneline -5`, `git diff <file>`
- `pip install -r requirements.txt`
- `streamlit run app.py` (dashboard, once built)

## 7. Remaining schedule

| Day | Work |
|---|---|
| Tue Oct 6 | Index modules + backtest A–D |
| Wed Oct 7 | News-type rules + final output file + dashboard |
| Thu Oct 8 | README, LICENSE, architecture diagram, results; freeze |
| Fri Oct 9 | Deck (7 slides), video (~5 min), push public repo, submit |
| Sat Oct 10 | Buffer only |
