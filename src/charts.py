"""
Backtest charts (PNG, 150 dpi) in docs/figures/. Called by src/backtest.py.

  index_levels.png    NAV of every strategy and ^OEX, base 100, log scale
  drawdowns.png       fall from the previous peak, per strategy
  weights_D.png       Strategy D weights over time (top names, the rest grouped)
  active_signals.png  number of index members with a non-zero sentiment signal each day
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")                     # write files only, no window
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = PROJECT_ROOT / "docs" / "figures"
TOP_NAMES = 10


def active_signal_count(base_table: pd.DataFrame, s_eff: pd.DataFrame) -> pd.Series:
    """Members (base weight present) whose S_eff is not zero, per day."""
    s = s_eff.reindex(index=base_table.index, columns=base_table.columns).fillna(0.0)
    return ((s != 0) & base_table.notna()).sum(axis=1)


def _save(fig, name: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def make_charts(navs: pd.DataFrame, d_result: dict, base_table: pd.DataFrame, s_eff: pd.DataFrame,
                out_dir: Path = FIGURE_DIR) -> list[Path]:
    levels = 100 * navs / navs.iloc[0]
    paths = []

    fig, ax = plt.subplots(figsize=(10, 5))
    levels.plot(ax=ax, logy=True, linewidth=1)
    ax.set_title("Index level (base 100, log scale)")
    ax.set_xlabel("")
    paths.append(_save(fig, "index_levels.png", out_dir))

    fig, ax = plt.subplots(figsize=(10, 4))
    (100 * (navs / navs.cummax() - 1)).plot(ax=ax, linewidth=1)
    ax.set_title("Drawdown from previous peak (%)")
    ax.set_xlabel("")
    paths.append(_save(fig, "drawdowns.png", out_dir))

    w = pd.DataFrame(d_result["weights"]).T.fillna(0.0)
    top = w.mean().nlargest(TOP_NAMES).index
    stacked = pd.concat([w[top], w.drop(columns=top).sum(axis=1).rename("Others")], axis=1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.stackplot(stacked.index, (100 * stacked).T.values, labels=stacked.columns)
    ax.set_title("Strategy D weights (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)
    paths.append(_save(fig, "weights_D.png", out_dir))

    fig, ax = plt.subplots(figsize=(10, 4))
    active_signal_count(base_table, s_eff).plot(ax=ax, linewidth=0.8)
    ax.set_title("Index members with an active sentiment signal (|S| >= 0.15), per day")
    ax.set_ylim(0, 20)
    ax.set_xlabel("")
    paths.append(_save(fig, "active_signals.png", out_dir))
    return paths
