import numpy as np
import pytest

from quant_sims.risk.var_es import (
    parametric_var,
    parametric_es,
    historical_var,
    historical_es,
    simulate_correlated_returns,
    monte_carlo_portfolio_var_es,
)


def test_parametric_var_invalid_alpha_raises():
    with pytest.raises(ValueError):
        parametric_var(0.0, 0.1, alpha=0.0)
    with pytest.raises(ValueError):
        parametric_var(0.0, 0.1, alpha=1.0)


def test_parametric_var_zero_mean_known_value():
    # mu=0, sigma=1, alpha=0.95 -> VaR = z_0.95 = 1.6449 (standard normal quantile)
    var = parametric_var(mu=0.0, sigma=1.0, alpha=0.95)
    assert var == pytest.approx(1.6449, abs=0.001)


def test_parametric_var_increases_with_alpha():
    var_95 = parametric_var(mu=0.0, sigma=0.02, alpha=0.95)
    var_99 = parametric_var(mu=0.0, sigma=0.02, alpha=0.99)
    assert var_99 > var_95


def test_parametric_es_exceeds_var_at_same_alpha():
    # ES should always be >= VaR at the same confidence level, since ES
    # averages over the tail beyond VaR
    mu, sigma, alpha = 0.001, 0.02, 0.95
    var = parametric_var(mu, sigma, alpha)
    es = parametric_es(mu, sigma, alpha)
    assert es > var


def test_historical_var_matches_empirical_quantile():
    rng = np.random.default_rng(1)
    returns = rng.normal(0.0, 0.02, size=100_000)
    var = historical_var(returns, alpha=0.95)
    expected = -np.quantile(returns, 0.05)  # equivalent formulation
    assert var == pytest.approx(expected, rel=0.01)


def test_historical_var_close_to_parametric_for_normal_data():
    rng = np.random.default_rng(2)
    mu, sigma = 0.0005, 0.015
    returns = rng.normal(mu, sigma, size=500_000)
    hist_var = historical_var(returns, alpha=0.95)
    param_var = parametric_var(mu, sigma, alpha=0.95)
    assert hist_var == pytest.approx(param_var, rel=0.05)


def test_historical_es_at_least_as_large_as_var():
    rng = np.random.default_rng(3)
    returns = rng.normal(0.0, 0.02, size=10_000)
    var = historical_var(returns, alpha=0.95)
    es = historical_es(returns, alpha=0.95)
    assert es >= var


def test_historical_var_invalid_alpha_raises():
    with pytest.raises(ValueError):
        historical_var(np.array([0.1, -0.1]), alpha=1.5)


def test_simulate_correlated_returns_shape():
    mus = np.array([0.08, 0.05])
    sigmas = np.array([0.2, 0.15])
    corr = np.array([[1.0, 0.3], [0.3, 1.0]])
    returns = simulate_correlated_returns(mus, sigmas, corr, T=1.0, n_paths=1000, seed=42)
    assert returns.shape == (1000, 2)


def test_simulate_correlated_returns_correlation_recovered():
    mus = np.array([0.05, 0.05])
    sigmas = np.array([0.2, 0.2])
    true_corr = 0.6
    corr = np.array([[1.0, true_corr], [true_corr, 1.0]])
    returns = simulate_correlated_returns(mus, sigmas, corr, T=1.0, n_paths=200_000, seed=7)

    log_returns = np.log(1 + returns)  # correlation is exact in log-return space
    empirical_corr = np.corrcoef(log_returns[:, 0], log_returns[:, 1])[0, 1]
    assert empirical_corr == pytest.approx(true_corr, abs=0.02)


def test_simulate_correlated_returns_invalid_corr_shape_raises():
    mus = np.array([0.05, 0.05, 0.05])
    sigmas = np.array([0.2, 0.2, 0.2])
    bad_corr = np.array([[1.0, 0.3], [0.3, 1.0]])  # wrong shape for 3 assets
    with pytest.raises(ValueError):
        simulate_correlated_returns(mus, sigmas, bad_corr, T=1.0, n_paths=100, seed=1)


def test_monte_carlo_portfolio_var_es_output_structure():
    weights = np.array([0.6, 0.4])
    mus = np.array([0.08, 0.05])
    sigmas = np.array([0.2, 0.1])
    corr = np.array([[1.0, 0.2], [0.2, 1.0]])

    result = monte_carlo_portfolio_var_es(weights, mus, sigmas, corr, T=1.0, alpha=0.95, n_paths=50_000, seed=1)

    assert set(result.keys()) == {"var", "es", "portfolio_returns"}
    assert result["es"] >= result["var"]
    assert len(result["portfolio_returns"]) == 50_000


def test_monte_carlo_var_higher_for_riskier_portfolio():
    mus = np.array([0.05, 0.05])
    corr = np.array([[1.0, 0.0], [0.0, 1.0]])

    low_vol = monte_carlo_portfolio_var_es(
        weights=np.array([0.5, 0.5]), mus=mus, sigmas=np.array([0.1, 0.1]),
        corr=corr, T=1.0, alpha=0.95, n_paths=100_000, seed=5,
    )
    high_vol = monte_carlo_portfolio_var_es(
        weights=np.array([0.5, 0.5]), mus=mus, sigmas=np.array([0.4, 0.4]),
        corr=corr, T=1.0, alpha=0.95, n_paths=100_000, seed=5,
    )
    assert high_vol["var"] > low_vol["var"]
