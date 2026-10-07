"""
Draws the project's architecture diagram (boxes and arrows) with matplotlib.

Run:  python src/architecture.py        -> docs/architecture.png (200 dpi)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")                     # write the file only, no window
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "docs" / "architecture.png"

ENGINE_COLOR, INDEX_COLOR, DATA_COLOR = "#dbe9f6", "#e3f1df", "#f3f3f3"

# (title, details, column, row, colour): the pipeline in reading order, two columns of five steps
STEPS = [
    ("1. Data sources", "Kaggle news headlines (1.4M)\nKaggle tweets (862k)\nyfinance prices, shares, ^OEX",
     0, 0, DATA_COLOR),
    ("2. Validation & cleaning", "broken rows, duplicates, ticker renames\ntimestamp check -> date-only news",
     0, 1, ENGINE_COLOR),
    ("3. Market-reaction labels", "sign of 1-day return, volatility filter\n1/2/3/7-day sign pattern -> impact",
     0, 2, ENGINE_COLOR),
    ("4. Masking", "company names, tickers, CEOs,\nsubsidiaries, URLs -> [COMPANY]", 0, 3, ENGINE_COLOR),
    ("5. Models (walk-forward 2016-2020)", "sentiment: TF-IDF + logistic regression\nseverity: ridge -> decile 1-10\n"
     "news type: keyword rules (10 types)", 0, 4, ENGINE_COLOR),
    ("6. Engine output", "engine_signals.csv (66k rows)\nFastAPI: /signals, /signals/summary", 1, 4, ENGINE_COLOR),
    ("7. Daily stock signals", "company-day events, 3-day half-life\nconfidence = model x independence x agreement\n"
     "neutral zone |S| < 0.15", 1, 3, INDEX_COLOR),
    ("8. Index construction", "FFMC Top 20, 20/25 buffer, quarterly\ntilt = base x exp(0.3 x S x C)\n"
     "12% cap, +-25% band, 1 pp/day", 1, 2, INDEX_COLOR),
    ("9. Backtest", "Strategies A, A-capped, B, C, D, D-tone\n10 bp costs, 2016 - Jun 2020", 1, 1, INDEX_COLOR),
    ("10. Analysis & dashboard", "daily IC, Fama-MacBeth, news events\nExcel results, charts, Streamlit app",
     1, 0, INDEX_COLOR),
]
BOX_W, BOX_H, COL_X, ROW_GAP = 4.2, 1.35, (0.4, 5.9), 1.75


def box_xy(col: int, row: int) -> tuple[float, float]:
    """Bottom-left corner of a box; row 0 is at the top."""
    return COL_X[col], 8.6 - row * ROW_GAP


def draw(path: Path = OUTPUT) -> Path:
    fig, ax = plt.subplots(figsize=(11, 10))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 10.3)
    ax.axis("off")
    ax.text(5.25, 10.1, "AI/NLP Risk Engine -> sentiment-tilted Top-20 index", ha="center", fontsize=15,
            weight="bold")
    centres = []
    for title, details, col, row, color in STEPS:
        x, y = box_xy(col, row)
        ax.add_patch(FancyBboxPatch((x, y), BOX_W, BOX_H, boxstyle="round,pad=0.05", fc=color, ec="#555555"))
        ax.text(x + BOX_W / 2, y + BOX_H - 0.25, title, ha="center", va="center", fontsize=10.5, weight="bold")
        ax.text(x + BOX_W / 2, y + 0.5, details, ha="center", va="center", fontsize=8.3, linespacing=1.3)
        centres.append((x, y))
    for (x1, y1), (x2, y2) in zip(centres, centres[1:]):
        if x1 == x2:                                   # same column: vertical arrow
            down = y2 < y1
            start = (x1 + BOX_W / 2, y1 if down else y1 + BOX_H)
            end = (x2 + BOX_W / 2, y2 + BOX_H if down else y2)
        else:                                          # across: horizontal arrow
            start, end = (x1 + BOX_W, y1 + BOX_H / 2), (x2, y2 + BOX_H / 2)
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=18, lw=1.5, color="#333333"))
    ax.text(0.4 + BOX_W / 2, 1.15, "Risk engine", ha="center", fontsize=10, color="#2b5d8a", weight="bold")
    ax.text(5.9 + BOX_W / 2, 1.15, "Module A: index rebalancer", ha="center", fontsize=10, color="#3b7a2f",
            weight="bold")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    print(f"Saved {draw().relative_to(PROJECT_ROOT)}")
