"""
Standard and generalized Brownian motion (Wiener process) simulation.

Standard Wiener process W(t) satisfies:
    W(0) = 0
    independent increments
    W(t) - W(s) ~ N(0, t - s) for t > s
    continuous (but nowhere differentiable) paths

Discretizing on a grid with step dt, increments are simulated exactly
(no numerical approximation error) as:
    W(t_{k+1}) = W(t_k) + sqrt(dt) * Z_k,   Z_k ~ N(0, 1) i.i.d.

Generalized (arithmetic) Brownian motion with drift mu and volatility sigma:
    X(t) = X(0) + mu*t + sigma*W(t)
"""

from __future__ import annotations

import numpy as np


class BrownianMotion:
    """
    Simulate paths of a (drifted) Brownian motion on a regular time grid.

    Parameters
    ----------
    mu : drift term (0.0 for standard Brownian motion)
    sigma : volatility / diffusion coefficient (1.0 for standard Brownian motion)
    seed : optional RNG seed
    """

    def __init__(self, mu: float = 0.0, sigma: float = 1.0, seed: int | None = None):
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self.mu = mu
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)

    def simulate_path(self, T: float, n_steps: int, X0: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate a single path over [0, T] with n_steps increments.
        Returns (time_grid, path), both of length n_steps + 1.
        """
        dt = T / n_steps
        increments = self.rng.normal(loc=0.0, scale=np.sqrt(dt), size=n_steps)

        path = np.empty(n_steps + 1)
        path[0] = X0
        # X(t_{k+1}) = X(t_k) + mu*dt + sigma*dW_k
        path[1:] = X0 + np.cumsum(self.mu * dt + self.sigma * increments)

        time_grid = np.linspace(0.0, T, n_steps + 1)
        return time_grid, path

    def simulate_paths(self, T: float, n_steps: int, n_paths: int, X0: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate n_paths independent paths. Returns (time_grid, paths) where
        paths has shape (n_paths, n_steps + 1).
        """
        dt = T / n_steps
        increments = self.rng.normal(loc=0.0, scale=np.sqrt(dt), size=(n_paths, n_steps))

        paths = np.empty((n_paths, n_steps + 1))
        paths[:, 0] = X0
        paths[:, 1:] = X0 + np.cumsum(self.mu * dt + self.sigma * increments, axis=1)

        time_grid = np.linspace(0.0, T, n_steps + 1)
        return time_grid, paths

    @staticmethod
    def theoretical_mean(t: np.ndarray, mu: float, X0: float = 0.0) -> np.ndarray:
        """E[X(t)] = X0 + mu*t"""
        return X0 + mu * t

    @staticmethod
    def theoretical_std(t: np.ndarray, sigma: float) -> np.ndarray:
        """Std(X(t)) = sigma * sqrt(t)"""
        return sigma * np.sqrt(t)
