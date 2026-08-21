"""
Value at Risk (VaR) and Expected Shortfall (ES / CVaR).

Convention used throughout this module: we work with a *loss* random
variable L = -R, where R is a (simple) return over the risk horizon. VaR
and ES are then reported as positive numbers representing the size of the
loss:

    VaR_alpha : the loss level such that P(L > VaR_alpha) = 1 - alpha
                (e.g. alpha=0.95 -> "95% VaR", a loss exceeded only 5% of
                the time)
    ES_alpha  : the expected loss *given* that the loss exceeds VaR_alpha
                (also called Conditional VaR / CVaR) -- unlike VaR, ES
                actually describes how bad the tail is, not just where it
                starts, which is why regulators increasingly prefer it.

Three estimation approaches are provided:
    1. Parametric (assumes returns are normally distributed)
    2. Historical (empirical quantile of realized/simulated returns --
       makes no distributional assumption)
    3. Monte Carlo (simulate a multi-asset portfolio under correlated GBM,
       then apply the historical approach to the simulated portfolio
       returns)
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Parametric (normal) VaR / ES
# ---------------------------------------------------------------------------


def parametric_var(mu: float, sigma: float, alpha: float = 0.95, horizon: float = 1.0) -> float:
    """
    Parametric (Gaussian) VaR, assuming returns over one base period are
    N(mu, sigma^2) and scaling to the given horizon via mu*horizon,
    sigma*sqrt(horizon) (the standard iid/no-autocorrelation scaling
    assumption).

    Returns a positive number: the loss at the alpha confidence level.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be strictly between 0 and 1")

    mu_h = mu * horizon
    sigma_h = sigma * np.sqrt(horizon)
    z_alpha = norm.ppf(alpha)
    return -mu_h + sigma_h * z_alpha


def parametric_es(mu: float, sigma: float, alpha: float = 0.95, horizon: float = 1.0) -> float:
    """
    Parametric (Gaussian) Expected Shortfall. For a normal distribution
    this has a closed form:

        ES_alpha = -mu + sigma * phi(Phi^-1(alpha)) / (1 - alpha)

    where phi is the standard normal pdf and Phi^-1 the standard normal
    quantile function.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be strictly between 0 and 1")

    mu_h = mu * horizon
    sigma_h = sigma * np.sqrt(horizon)
    z_alpha = norm.ppf(alpha)
    return -mu_h + sigma_h * norm.pdf(z_alpha) / (1.0 - alpha)


# ---------------------------------------------------------------------------
# Historical (empirical) VaR / ES -- works on any sample of returns,
# whether realized market data or simulated Monte Carlo returns.
# ---------------------------------------------------------------------------


def historical_var(returns: np.ndarray, alpha: float = 0.95) -> float:
    """
    Empirical VaR: the alpha-quantile of the loss distribution, estimated
    directly from a sample of returns (no distributional assumption).
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be strictly between 0 and 1")

    losses = -np.asarray(returns)
    return float(np.quantile(losses, alpha))


def historical_es(returns: np.ndarray, alpha: float = 0.95) -> float:
    """
    Empirical Expected Shortfall: the mean loss among the worst (1-alpha)
    fraction of observations (i.e. the tail average beyond the VaR
    threshold).
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be strictly between 0 and 1")

    losses = -np.asarray(returns)
    var = np.quantile(losses, alpha)
    tail = losses[losses >= var]
    if len(tail) == 0:
        return float(var)
    return float(np.mean(tail))


# ---------------------------------------------------------------------------
# Monte Carlo VaR / ES for a multi-asset portfolio under correlated GBM
# ---------------------------------------------------------------------------


def simulate_correlated_returns(
    mus: np.ndarray,
    sigmas: np.ndarray,
    corr: np.ndarray,
    T: float,
    n_paths: int,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simulate n_paths draws of simple returns for a set of assets under
    correlated GBM over horizon T, using the exact terminal-value
    distribution (no path simulation needed -- only the horizon-T return
    matters here, same "n_steps=1 shortcut" idea as European option pricing
    in Module 3).

    Parameters
    ----------
    mus, sigmas : arrays of shape (n_assets,) -- annualized drift/vol per asset
    corr : (n_assets, n_assets) correlation matrix (symmetric, PSD, unit diagonal)
    T : horizon in years
    n_paths : number of Monte Carlo draws

    Returns
    -------
    array of shape (n_paths, n_assets) of simple returns over horizon T.
    """
    mus = np.asarray(mus, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)
    corr = np.asarray(corr, dtype=float)
    n_assets = len(mus)

    if corr.shape != (n_assets, n_assets):
        raise ValueError("corr must be an (n_assets, n_assets) matrix matching mus/sigmas")

    L = np.linalg.cholesky(corr)  # raises LinAlgError if corr isn't PSD
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n_paths, n_assets)) @ L.T  # correlated standard normals

    log_returns = (mus - 0.5 * sigmas ** 2) * T + sigmas * np.sqrt(T) * Z
    simple_returns = np.exp(log_returns) - 1.0
    return simple_returns


def monte_carlo_portfolio_var_es(
    weights: np.ndarray,
    mus: np.ndarray,
    sigmas: np.ndarray,
    corr: np.ndarray,
    T: float,
    alpha: float = 0.95,
    n_paths: int = 100_000,
    seed: int | None = None,
) -> dict:
    """
    Simulate a multi-asset portfolio under correlated GBM and estimate VaR
    and ES of the portfolio return over horizon T via the historical
    (empirical quantile) method applied to the simulated sample.

    weights : portfolio weights, shape (n_assets,), should sum to ~1
              (not enforced, to allow e.g. leveraged/short portfolios)

    Returns a dict with var, es, and the simulated portfolio_returns array
    (useful for plotting the simulated distribution).
    """
    weights = np.asarray(weights, dtype=float)
    asset_returns = simulate_correlated_returns(mus, sigmas, corr, T, n_paths, seed)
    portfolio_returns = asset_returns @ weights

    return {
        "var": historical_var(portfolio_returns, alpha),
        "es": historical_es(portfolio_returns, alpha),
        "portfolio_returns": portfolio_returns,
    }
