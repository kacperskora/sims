"""
Monte Carlo pricing for European options.

Simulates the underlying under the risk-neutral measure (GBM with drift r
instead of the real-world mu -- see black_scholes.py for the reasoning),
computes the discounted expected payoff, and compares convergence to the
closed-form Black-Scholes price.
"""

from __future__ import annotations

import numpy as np

from quant_sims.stochastic_processes.gbm import GeometricBrownianMotion
from .black_scholes import BlackScholesParams, price as bs_price


def _payoff(S_T: np.ndarray, K: float, option_type: str) -> np.ndarray:
    if option_type == "call":
        return np.maximum(S_T - K, 0.0)
    elif option_type == "put":
        return np.maximum(K - S_T, 0.0)
    raise ValueError("option_type must be 'call' or 'put'")


def monte_carlo_price(
    p: BlackScholesParams,
    option_type: str = "call",
    n_simulations: int = 100_000,
    n_steps: int = 1,
    seed: int | None = None,
    return_std_error: bool = False,
):
    """
    Price a European option via Monte Carlo simulation under the
    risk-neutral measure (GBM with drift = r).

    Since European options only depend on the terminal price S_T (no
    path-dependence), n_steps=1 is sufficient and fastest -- we only need to
    sample S_T directly, not the whole path. n_steps > 1 is supported for
    consistency/testing against the full path-simulation machinery.

    Returns the estimated price, or (price, standard_error) if
    return_std_error=True. Standard error is the Monte Carlo standard error
    of the mean discounted payoff, useful for constructing confidence
    intervals and understanding convergence (~ 1/sqrt(n_simulations)).
    """
    gbm = GeometricBrownianMotion(mu=p.r, sigma=p.sigma, seed=seed)
    _, paths = gbm.simulate_paths(S0=p.S, T=p.T, n_steps=n_steps, n_paths=n_simulations)
    S_T = paths[:, -1]

    payoffs = _payoff(S_T, p.K, option_type)
    discounted = np.exp(-p.r * p.T) * payoffs

    estimated_price = float(np.mean(discounted))

    if return_std_error:
        std_error = float(np.std(discounted, ddof=1) / np.sqrt(n_simulations))
        return estimated_price, std_error
    return estimated_price


def convergence_study(
    p: BlackScholesParams,
    option_type: str = "call",
    sample_sizes: np.ndarray | None = None,
    seed: int | None = None,
) -> dict:
    """
    Run Monte Carlo pricing at increasing sample sizes and compare against
    the closed-form Black-Scholes price, to visualize/quantify convergence.

    Returns a dict with:
        sample_sizes : array of N values used
        mc_prices : MC price estimate at each N
        std_errors : MC standard error at each N
        bs_price : the closed-form reference price
        abs_errors : |mc_price - bs_price| at each N
    """
    if sample_sizes is None:
        sample_sizes = np.array([100, 500, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000])

    reference_price = bs_price(p, option_type)

    mc_prices = np.empty(len(sample_sizes))
    std_errors = np.empty(len(sample_sizes))

    rng = np.random.default_rng(seed)
    for i, n in enumerate(sample_sizes):
        run_seed = int(rng.integers(0, 2**31 - 1))
        est, se = monte_carlo_price(p, option_type, n_simulations=int(n), seed=run_seed, return_std_error=True)
        mc_prices[i] = est
        std_errors[i] = se

    return {
        "sample_sizes": sample_sizes,
        "mc_prices": mc_prices,
        "std_errors": std_errors,
        "bs_price": reference_price,
        "abs_errors": np.abs(mc_prices - reference_price),
    }
