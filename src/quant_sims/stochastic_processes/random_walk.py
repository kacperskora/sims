"""
Simple random walk.

S_n = X_1 + X_2 + ... + X_n,  X_i in {-1, +1} (or a general step size)

For an unbiased walk (p=0.5), E[S_n] = 0 and Var(S_n) = n * step_size^2.
This module also demonstrates the scaling limit that connects a discrete
random walk to Brownian motion (Donsker's theorem): normalizing
S_floor(n*t) by sqrt(n) and letting n -> infinity converges to a standard
Wiener process.
"""

from __future__ import annotations

import numpy as np


class RandomWalk:
    """
    A simple (possibly biased) random walk with +/- step_size increments.

    Parameters
    ----------
    p_up : probability of a +step_size move (0 < p_up < 1)
    step_size : magnitude of each step
    seed : optional RNG seed
    """

    def __init__(self, p_up: float = 0.5, step_size: float = 1.0, seed: int | None = None):
        if not (0.0 < p_up < 1.0):
            raise ValueError("p_up must be strictly between 0 and 1")
        if step_size <= 0:
            raise ValueError("step_size must be positive")

        self.p_up = p_up
        self.step_size = step_size
        self.rng = np.random.default_rng(seed)

    def simulate(self, n_steps: int) -> np.ndarray:
        """
        Simulate a single walk. Returns array of length n_steps + 1
        (including S_0 = 0).
        """
        moves = np.where(self.rng.random(n_steps) < self.p_up, 1.0, -1.0) * self.step_size
        path = np.empty(n_steps + 1)
        path[0] = 0.0
        path[1:] = np.cumsum(moves)
        return path

    def simulate_paths(self, n_steps: int, n_paths: int) -> np.ndarray:
        """
        Simulate n_paths independent walks. Returns array of shape
        (n_paths, n_steps + 1).
        """
        moves = np.where(self.rng.random((n_paths, n_steps)) < self.p_up, 1.0, -1.0) * self.step_size
        paths = np.empty((n_paths, n_steps + 1))
        paths[:, 0] = 0.0
        paths[:, 1:] = np.cumsum(moves, axis=1)
        return paths

    def scaling_limit_paths(self, n_steps: int, n_paths: int, T: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate the *rescaled* walk W_n(t) = S_floor(n*t) / sqrt(n), which by
        Donsker's theorem converges in distribution to a standard Wiener
        process as n_steps -> infinity. Returns (time_grid, paths) where
        paths has shape (n_paths, n_steps + 1) and time_grid has length
        n_steps + 1, spanning [0, T].

        This uses unbiased (p_up=0.5), unit step_size increments regardless
        of the instance's configured step_size/p_up, since the scaling limit
        result specifically requires mean-zero, finite-variance increments
        normalized by sqrt(n) to converge to a standard (not just any) Wiener
        process.
        """
        moves = np.where(self.rng.random((n_paths, n_steps)) < 0.5, 1.0, -1.0)
        raw_paths = np.empty((n_paths, n_steps + 1))
        raw_paths[:, 0] = 0.0
        raw_paths[:, 1:] = np.cumsum(moves, axis=1)

        rescaled_paths = raw_paths / np.sqrt(n_steps / T)
        time_grid = np.linspace(0.0, T, n_steps + 1)
        return time_grid, rescaled_paths
