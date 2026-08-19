import numpy as np
import pytest

from quant_sims.stochastic_processes.gbm import GeometricBrownianMotion


def test_invalid_sigma_raises():
    with pytest.raises(ValueError):
        GeometricBrownianMotion(mu=0.05, sigma=0.0)


def test_invalid_S0_raises():
    gbm = GeometricBrownianMotion(mu=0.05, sigma=0.2, seed=1)
    with pytest.raises(ValueError):
        gbm.simulate_paths(S0=0.0, T=1.0, n_steps=100, n_paths=10)


def test_simulate_paths_shape_and_start():
    gbm = GeometricBrownianMotion(mu=0.05, sigma=0.2, seed=1)
    S0 = 100.0
    t, paths = gbm.simulate_paths(S0=S0, T=1.0, n_steps=252, n_paths=500)
    assert paths.shape == (500, 253)
    assert t.shape == (253,)
    np.testing.assert_allclose(paths[:, 0], S0)


def test_paths_always_positive():
    # GBM paths should never go negative or zero, unlike arithmetic Brownian motion
    gbm = GeometricBrownianMotion(mu=0.05, sigma=0.6, seed=3)  # high vol
    t, paths = gbm.simulate_paths(S0=50.0, T=2.0, n_steps=500, n_paths=2000)
    assert np.all(paths > 0.0)


def test_mean_matches_theory():
    mu, sigma = 0.08, 0.25
    S0, T = 100.0, 1.0
    gbm = GeometricBrownianMotion(mu=mu, sigma=sigma, seed=42)
    t, paths = gbm.simulate_paths(S0=S0, T=T, n_steps=252, n_paths=8000)
    final_values = paths[:, -1]

    theoretical = gbm.theoretical_mean(S0, np.array([T]))[0]
    assert np.mean(final_values) == pytest.approx(theoretical, rel=0.05)


def test_log_returns_are_approximately_normal_with_correct_moments():
    mu, sigma = 0.05, 0.2
    S0, T, n_steps = 100.0, 1.0, 252
    dt = T / n_steps
    gbm = GeometricBrownianMotion(mu=mu, sigma=sigma, seed=9)
    t, paths = gbm.simulate_paths(S0=S0, T=T, n_steps=n_steps, n_paths=3000)

    log_ret = gbm.log_returns(paths)
    assert log_ret.shape == (3000, n_steps)

    expected_mean_per_step = (mu - 0.5 * sigma ** 2) * dt
    expected_std_per_step = sigma * np.sqrt(dt)

    assert np.mean(log_ret) == pytest.approx(expected_mean_per_step, abs=1e-3)
    assert np.std(log_ret) == pytest.approx(expected_std_per_step, rel=0.05)


def test_fit_mu_sigma_recovers_known_parameters():
    # Simulate a single long path with known (mu, sigma), then re-estimate
    # them from the resulting price series and check we recover values
    # reasonably close to the truth.
    true_mu, true_sigma = 0.07, 0.22
    S0, T, n_steps = 100.0, 5.0, 1260  # ~5 years of daily data
    dt = T / n_steps

    gbm = GeometricBrownianMotion(mu=true_mu, sigma=true_sigma, seed=123)
    t, paths = gbm.simulate_paths(S0=S0, T=T, n_steps=n_steps, n_paths=1)
    prices = paths[0]

    mu_hat, sigma_hat = GeometricBrownianMotion.fit_mu_sigma(prices, dt=dt)

    assert sigma_hat == pytest.approx(true_sigma, rel=0.15)
    # mu is much harder to estimate precisely from a single path (high variance
    # estimator even over years of data), so allow a wide absolute tolerance
    assert mu_hat == pytest.approx(true_mu, abs=0.15)
