"""
Index weights: FFMC base weights, sentiment tilt, caps, daily change limit and normalisation
(methodology sections 8 and 14-18, in the order of section 22).

All constraints are solved together: every weight is k x (its raw weight), clipped to its own
[lower, upper] bounds, with the single scale k found by bisection so the weights sum to 100%.
Bounds (Strategy D):
  tilt band     base x (1 - 25%) <= w <= base x (1 + 25%)
  cap           w <= 12%                     (wins over the tilt band if they conflict)
  daily change  |w - previous w| <= 1 pp     (not on review days; between reviews it wins over
                                              the cap/tilt band, which are then reached gradually)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MAX_WEIGHT = 0.12
MAX_TILT = 0.25
MAX_DAILY_CHANGE = 0.01
ALPHA = 0.30


def base_weights(ffmc_members: pd.Series) -> pd.Series:
    """FFMC weights of the current members (section 8)."""
    return ffmc_members / ffmc_members.sum()


def tilt(base: pd.Series, sentiment: pd.Series, confidence: pd.Series | float = 1.0, alpha: float = ALPHA) -> pd.Series:
    """Raw sentiment-tilted weights w_FFMC x exp(alpha x S x C), not yet normalised (section 14)."""
    s = sentiment.reindex(base.index).fillna(0.0)
    c = confidence.reindex(base.index).fillna(0.0) if isinstance(confidence, pd.Series) else confidence
    return base * np.exp(alpha * s * c)


def normalise(w: pd.Series) -> pd.Series:
    return w / w.sum()


def fit_to_bounds(raw: pd.Series, lower: pd.Series, upper: pd.Series, tol: float = 1e-12) -> pd.Series:
    """Weights clip(k x raw, lower, upper) summing to 1. Raises ValueError if the bounds make that impossible."""
    raw, lower, upper = raw.astype(float), lower.reindex(raw.index).astype(float), upper.reindex(raw.index).astype(float)
    if lower.sum() > 1 + 1e-9 or upper.sum() < 1 - 1e-9:
        raise ValueError(f"infeasible bounds: sum(lower)={lower.sum():.4f}, sum(upper)={upper.sum():.4f}")
    weights = lambda k: (k * raw).clip(lower, upper)  # noqa: E731
    lo, hi = 0.0, 1.0
    while weights(hi).sum() < 1:                      # find a k large enough
        hi *= 2
        if hi > 1e12:
            break
    for _ in range(200):                              # bisection on k
        mid = (lo + hi) / 2
        if weights(mid).sum() < 1:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    w = weights(hi)
    return w / w.sum()                                # remove the last rounding error


def cap_weights(base: pd.Series, max_weight: float = MAX_WEIGHT) -> pd.Series:
    """Cap every weight at max_weight and spread the excess pro rata over the others (A-capped)."""
    return fit_to_bounds(base, pd.Series(0.0, index=base.index), pd.Series(max_weight, index=base.index))


def constrained_weights(raw: pd.Series, base: pd.Series, previous: pd.Series | None, review_day: bool,
                        max_weight: float = MAX_WEIGHT, max_tilt: float = MAX_TILT,
                        max_change: float = MAX_DAILY_CHANGE) -> tuple[pd.Series, bool]:
    """
    Strategy D weights: tilt band, cap and daily change limit (see module docstring).
    Returns (weights, relaxed) where relaxed=True means the change limit had to be dropped for the day
    because no weights could satisfy all bounds at once.
    """
    lower = (base * (1 - max_tilt)).clip(upper=max_weight)    # the cap wins over the tilt band
    upper = (base * (1 + max_tilt)).clip(upper=max_weight)
    if review_day or previous is None:
        return fit_to_bounds(raw, lower, upper), False
    prev = previous.reindex(base.index).fillna(0.0)
    c_lo, c_hi = (prev - max_change).clip(lower=0.0), prev + max_change
    # intersect with the change band; if they don't overlap, the change limit wins (move as far as allowed)
    new_lo = np.maximum(lower, c_lo)
    new_hi = np.minimum(upper, c_hi)
    no_overlap = new_lo > new_hi
    new_lo[no_overlap] = np.where(upper[no_overlap] < c_lo[no_overlap], c_lo[no_overlap], c_hi[no_overlap])
    new_hi[no_overlap] = new_lo[no_overlap]
    try:
        return fit_to_bounds(raw, new_lo, new_hi), False
    except ValueError:
        return fit_to_bounds(raw, lower, upper), True
