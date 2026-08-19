import numpy as np
import pytest

from quant_sims.stochastic_processes.brownian_motion import BrownianMotion


def test_invalid_sigma_raises():
    with pytest.raises(ValueError):
        BrownianMotion(sigma=0.0)


def test_simulate_path_starts_at_X0():
    bm = BrownianMotion(mu=0.0, sigma=1.0, seed=1)
    t, path = bm.simulate_path(T=1.0, n_steps=100, X0=5.0)
    assert path[0] == 5.0
    assert path.shape == (101,)
    assert t.shape == (101,)
    assert t[0] == 0.0
    assert t[-1] == pytest.approx(1.0)


def test_simulate_paths_shape():
    bm = BrownianMotion(seed=1)
    t, paths = bm.simulate_paths(T=1.0, n_steps=50, n_paths=200)
    assert paths.shape == (200, 51)
    assert np.all(paths[:, 0] == 0.0)


def test_standard_brownian_motion_mean_zero():
    bm = BrownianMotion(mu=0.0, sigma=1.0, seed=42)
    T = 2.0
    t, paths = bm.simulate_paths(T=T, n_steps=200, n_paths=5000)
    final_values = paths[:, -1]
    assert np.mean(final_values) == pytest.approx(0.0, abs=0.15)


def test_standard_brownian_motion_variance_equals_T():
    bm = BrownianMotion(mu=0.0, sigma=1.0, seed=7)
    T = 2.0
    t, paths = bm.simulate_paths(T=T, n_steps=200, n_paths=5000)
    final_values = paths[:, -1]
    # Var(W(T)) = T for standard Brownian motion
    assert np.var(final_values) == pytest.approx(T, rel=0.1)


def test_drifted_brownian_motion_mean_matches_theory():
    mu, sigma = 0.5, 0.3
    bm = BrownianMotion(mu=mu, sigma=sigma, seed=13)
    T = 1.5
    X0 = 10.0
    t, paths = bm.simulate_paths(T=T, n_steps=150, n_paths=5000, X0=X0)
    final_values = paths[:, -1]
    expected_mean = X0 + mu * T
    assert np.mean(final_values) == pytest.approx(expected_mean, rel=0.05)


def test_theoretical_mean_and_std_helpers():
    t = np.array([0.0, 1.0, 2.0, 4.0])
    mean = BrownianMotion.theoretical_mean(t, mu=0.5, X0=1.0)
    np.testing.assert_allclose(mean, np.array([1.0, 1.5, 2.0, 3.0]))

    std = BrownianMotion.theoretical_std(t, sigma=2.0)
    np.testing.assert_allclose(std, 2.0 * np.sqrt(t))


def test_paths_are_continuous_no_large_jumps():
    # Sanity check: increments should be roughly N(0, dt), so no single-step
    # jump should be wildly larger than ~10 standard deviations under normal
    # circumstances across a reasonably sized sample.
    bm = BrownianMotion(mu=0.0, sigma=1.0, seed=5)
    T, n_steps = 1.0, 500
    dt = T / n_steps
    t, path = bm.simulate_path(T=T, n_steps=n_steps)
    increments = np.diff(path)
    assert np.max(np.abs(increments)) < 10 * np.sqrt(dt)
