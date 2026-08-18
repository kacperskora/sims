"""
Kelly Criterion utilities.

For a repeated bet with:
    p = probability of winning
    b = net odds received on a win (i.e. win amount = b * stake, lose amount = stake)

the Kelly fraction f* that maximizes the expected long-run logarithmic growth
rate of capital is:

    f* = (b * p - (1 - p)) / b = p - (1 - p) / b

This module provides the core formulas plus helpers for evaluating growth
rate and building fraction functions usable by CoinFlipGame simulations.
"""

from __future__ import annotations

import numpy as np


def kelly_fraction(p: float, b: float) -> float:
    """
    Compute the optimal Kelly betting fraction.

    Parameters
    ----------
    p : probability of winning (0 < p < 1)
    b : net odds received on a win (b > 0). E.g. b=1.0 means a 1:1 payout
        (win = stake, lose = stake).

    Returns
    -------
    Optimal fraction of capital to bet. Can be negative (meaning "don't take
    this bet" / bet the other side) or > 1 if edge is very large relative to
    odds -- callers should clip as appropriate for their use case.
    """
    if not (0.0 < p < 1.0):
        raise ValueError("p must be strictly between 0 and 1")
    if b <= 0:
        raise ValueError("b must be positive")

    q = 1.0 - p
    return p - q / b


def expected_log_growth(p: float, b: float, f: float) -> float:
    """
    Expected per-bet logarithmic growth rate for betting fraction f, given
    win probability p and net odds b.

        E[log growth] = p * log(1 + f*b) + (1 - p) * log(1 - f)

    Only valid for -1/b < f < 1 (otherwise a single loss can wipe out or
    invert capital). Returns -inf outside that range.
    """
    if f <= -1.0 / b or f >= 1.0:
        return -np.inf

    q = 1.0 - p
    return p * np.log1p(f * b) + q * np.log1p(-f)


def growth_rate_curve(p: float, b: float, f_values: np.ndarray) -> np.ndarray:
    """Vectorized expected_log_growth over an array of fractions."""
    return np.array([expected_log_growth(p, b, f) for f in f_values])


def has_positive_edge(p: float, b: float) -> bool:
    """True if the bet has positive expectation (Kelly fraction > 0)."""
    return kelly_fraction(p, b) > 0


# ---------------------------------------------------------------------------
# Fraction functions: callables usable as `bet_fraction_fn` in CoinFlipGame.
# Signature: fn(capital: float, history: list[bool]) -> float (0 to 1)
# ---------------------------------------------------------------------------


def fixed_fraction(f: float):
    """Always bet a fixed fraction f of current capital."""

    def strategy(capital: float, history: list) -> float:
        return f

    strategy.__name__ = f"fixed_fraction_{f}"
    return strategy


def kelly_strategy(p: float, b: float, multiplier: float = 1.0):
    """
    Bet multiplier * f_kelly of capital every round.
    multiplier=1.0 -> full Kelly, 0.5 -> half Kelly, etc.
    Negative or zero optimal fractions are clipped to 0 (don't bet).
    """
    f_star = max(kelly_fraction(p, b), 0.0) * multiplier

    def strategy(capital: float, history: list) -> float:
        return f_star

    strategy.__name__ = f"kelly_x{multiplier}"
    return strategy


def martingale(base_fraction: float, max_fraction: float = 1.0):
    """
    Double the bet fraction after each loss, reset to base after a win.
    Classic (and classically dangerous) strategy -- capped at max_fraction
    to avoid betting more than 100% of capital.
    """

    def strategy(capital: float, history: list) -> float:
        if not history or history[-1]:  # empty history or last flip was a win
            return base_fraction
        # count consecutive losses at the end of history
        streak = 0
        for outcome in reversed(history):
            if outcome:
                break
            streak += 1
        return min(base_fraction * (2 ** streak), max_fraction)

    strategy.__name__ = f"martingale_{base_fraction}"
    return strategy


def all_in():
    """Bet 100% of capital every round. Ruin is a matter of time."""

    def strategy(capital: float, history: list) -> float:
        return 1.0

    strategy.__name__ = "all_in"
    return strategy
