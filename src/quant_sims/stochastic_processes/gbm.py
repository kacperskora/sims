"""
Geometric Brownian Motion (GBM) -- the classic stock price model underlying
Black-Scholes.

SDE:
    dS_t = mu * S_t dt + sigma * S_t dW_t

Applying Ito's lemma to f(S) = log(S) gives an SDE with constant
coefficients for log(S), which integrates to a closed-form solution (no
discretization error needed to simulate at any set of time points):

    S_t = S_0 * exp[(mu - sigma^2 / 2) * t + sigma * W_t]

Consequently S_t is lognormally distributed:
    log(S_t / S_0) ~ N((mu - sigma^2/2) * t, sigma^2 * t)
"""

from __future__ import annotations

import numpy as np


class GeometricBrownianMotion:
    """
    Simulate GBM paths using the exact closed-form solution (not Euler-Maruyama
    discretization -- for GBM the closed-form is exact at any time grid).

    Parameters
    ----------
    mu : drift (expected return, annualized if t is in years)
    sigma : volatility (annualized if t is in years)
    seed : optional RNG seed
    """

    def __init__(self, mu: float, sigma: float, seed: int | None = None):
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self.mu = mu
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)

    def simulate_paths(self, S0: float, T: float, n_steps: int, n_paths: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate n_paths independent GBM paths over [0, T] on a grid of
        n_steps increments, using the exact closed-form solution.

        Returns (time_grid, paths) where paths has shape (n_paths, n_steps + 1).
        """
        if S0 <= 0:
            raise ValueError("S0 must be positive")

        dt = T / n_steps
        # standard Brownian increments
        dW = self.rng.normal(loc=0.0, scale=np.sqrt(dt), size=(n_paths, n_steps))
        W = np.cumsum(dW, axis=1)
        W = np.hstack([np.zeros((n_paths, 1)), W])  # prepend W(0) = 0

        time_grid = np.linspace(0.0, T, n_steps + 1)
        drift_term = (self.mu - 0.5 * self.sigma ** 2) * time_grid  # shape (n_steps+1,)
        log_paths = np.log(S0) + drift_term[np.newaxis, :] + self.sigma * W

        paths = np.exp(log_paths)
        return time_grid, paths

    def theoretical_mean(self, S0: float, t: np.ndarray) -> np.ndarray:
        """E[S_t] = S0 * exp(mu * t)"""
        return S0 * np.exp(self.mu * t)

    def theoretical_variance(self, S0: float, t: np.ndarray) -> np.ndarray:
        """Var[S_t] = S0^2 * exp(2*mu*t) * (exp(sigma^2 * t) - 1)"""
        return (S0 ** 2) * np.exp(2 * self.mu * t) * (np.exp(self.sigma ** 2 * t) - 1.0)

    @staticmethod
    def log_returns(paths: np.ndarray) -> np.ndarray:
        """
        Compute log-returns log(S_{t+1}/S_t) along the time axis (axis=1) for
        a batch of paths of shape (n_paths, n_steps + 1). Returns shape
        (n_paths, n_steps).
        """
        return np.diff(np.log(paths), axis=1)

    @staticmethod
    def fit_mu_sigma(prices: np.ndarray, dt: float) -> tuple[float, float]:
        """
        Estimate (mu, sigma) from a single observed price series via
        log-return moment matching:
            sigma_hat = std(log_returns) / sqrt(dt)
            mu_hat = mean(log_returns) / dt + sigma_hat^2 / 2

        Useful for calibrating a GBM to a real price series (e.g. before
        comparing simulated paths to actual market behavior).
        """
        log_ret = np.diff(np.log(prices))
        sigma_hat = np.std(log_ret, ddof=1) / np.sqrt(dt)
        mu_hat = np.mean(log_ret) / dt + 0.5 * sigma_hat ** 2
        return float(mu_hat), float(sigma_hat)
