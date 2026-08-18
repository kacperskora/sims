"""
Coin flip betting simulation.

A CoinFlipGame models repeated bets on a (possibly biased) coin. Each round
a betting strategy decides what fraction of current capital to stake; a win
pays out `payout_ratio` times the stake, a loss forfeits the stake.

Strategies are plain callables with signature:
    strategy(capital: float, history: list[bool]) -> float
See `quant_sims.betting.kelly` for ready-made strategies (fixed_fraction,
kelly_strategy, martingale, all_in).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SimulationResult:
    """Container for a single simulated capital path."""

    capital_path: np.ndarray          # length n_flips + 1 (includes starting capital)
    outcomes: np.ndarray              # length n_flips, bool (True = win)
    ruined: bool                      # True if capital hit 0 (or below) at any point

    @property
    def final_capital(self) -> float:
        return float(self.capital_path[-1])

    @property
    def total_return(self) -> float:
        return self.final_capital / self.capital_path[0] - 1.0


class CoinFlipGame:
    """
    A repeated coin-flip betting game.

    Parameters
    ----------
    p_win : probability the coin lands in the bettor's favor (0 < p_win < 1)
    payout_ratio : net odds b -- a win of stake s returns s + b*s (i.e. profit
        of b*s); a loss forfeits the full stake s. b=1.0 is a fair-odds coin
        (even money).
    seed : optional RNG seed for reproducibility
    """

    def __init__(self, p_win: float = 0.5, payout_ratio: float = 1.0, seed: int | None = None):
        if not (0.0 < p_win < 1.0):
            raise ValueError("p_win must be strictly between 0 and 1")
        if payout_ratio <= 0:
            raise ValueError("payout_ratio must be positive")

        self.p_win = p_win
        self.payout_ratio = payout_ratio
        self.rng = np.random.default_rng(seed)

    def flip(self) -> bool:
        """Simulate a single coin flip. Returns True on a win."""
        return bool(self.rng.random() < self.p_win)

    def simulate_series(
        self,
        strategy_fn,
        n_flips: int,
        initial_capital: float = 100.0,
        ruin_threshold: float = 1e-6,
    ) -> SimulationResult:
        """
        Simulate one full series of n_flips bets using the given strategy.

        The strategy is queried before each flip for the fraction of current
        capital to stake. Simulation stops early (path is padded flat) if
        capital falls to or below `ruin_threshold`.
        """
        capital_path = np.empty(n_flips + 1, dtype=float)
        outcomes = np.empty(n_flips, dtype=bool)

        capital_path[0] = initial_capital
        capital = initial_capital
        history: list[bool] = []
        ruined = False

        for i in range(n_flips):
            if ruined:
                capital_path[i + 1] = capital
                outcomes[i] = False
                continue

            fraction = float(np.clip(strategy_fn(capital, history), 0.0, 1.0))
            stake = fraction * capital
            win = self.flip()
            outcomes[i] = win
            history.append(win)

            if win:
                capital += stake * self.payout_ratio
            else:
                capital -= stake

            if capital <= ruin_threshold:
                capital = 0.0
                ruined = True

            capital_path[i + 1] = capital

        return SimulationResult(capital_path=capital_path, outcomes=outcomes, ruined=ruined)

    def monte_carlo(
        self,
        strategy_fn,
        n_flips: int,
        n_simulations: int,
        initial_capital: float = 100.0,
    ) -> np.ndarray:
        """
        Run n_simulations independent series and return a matrix of shape
        (n_simulations, n_flips + 1) of capital paths.
        """
        paths = np.empty((n_simulations, n_flips + 1), dtype=float)
        for i in range(n_simulations):
            result = self.simulate_series(strategy_fn, n_flips, initial_capital)
            paths[i] = result.capital_path
        return paths


def ruin_probability_analytic(p: float, b: float, f: float, initial_capital: float, ruin_level: float) -> float:
    """
    Placeholder for an analytic gambler's-ruin-style approximation.

    Closed-form gambler's ruin formulas assume fixed unit bets, not a fixed
    *fraction* of capital (which can never technically reach exactly zero
    under continuous compounding). For fixed-fraction betting we instead
    estimate ruin probability empirically via Monte Carlo -- see
    `ruin_probability_montecarlo` below. This function is kept as a documented
    no-op to make that distinction explicit in the codebase.
    """
    raise NotImplementedError(
        "Fixed-fraction betting has no simple closed-form ruin probability; "
        "use ruin_probability_montecarlo instead."
    )


def ruin_probability_montecarlo(
    game: CoinFlipGame,
    strategy_fn,
    n_flips: int,
    n_simulations: int,
    initial_capital: float = 100.0,
    ruin_level_fraction: float = 0.05,
) -> float:
    """
    Estimate probability that capital ever drops to/below
    `ruin_level_fraction * initial_capital` within n_flips, across
    n_simulations independent runs.
    """
    ruin_level = ruin_level_fraction * initial_capital
    ruin_count = 0
    for _ in range(n_simulations):
        result = game.simulate_series(strategy_fn, n_flips, initial_capital)
        if np.any(result.capital_path <= ruin_level):
            ruin_count += 1
    return ruin_count / n_simulations
