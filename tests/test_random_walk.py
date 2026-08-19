import numpy as np
import pytest

from quant_sims.stochastic_processes.random_walk import RandomWalk


def test_invalid_p_up_raises():
    with pytest.raises(ValueError):
        RandomWalk(p_up=0.0)
    with pytest.raises(ValueError):
        RandomWalk(p_up=1.0)


def test_invalid_step_size_raises():
    with pytest.raises(ValueError):
        RandomWalk(step_size=0.0)


def test_simulate_starts_at_zero():
    walk = RandomWalk(seed=1)
    path = walk.simulate(n_steps=50)
    assert path[0] == 0.0
    assert path.shape == (51,)


def test_simulate_paths_shape():
    walk = RandomWalk(seed=1)
    paths = walk.simulate_paths(n_steps=30, n_paths=100)
    assert paths.shape == (100, 31)
    assert np.all(paths[:, 0] == 0.0)


def test_unbiased_walk_mean_close_to_zero():
    walk = RandomWalk(p_up=0.5, seed=42)
    paths = walk.simulate_paths(n_steps=200, n_paths=5000)
    final_values = paths[:, -1]
    # E[S_n] = 0 for unbiased walk; check empirically within a tolerance
    assert abs(np.mean(final_values)) < 3.0  # generous tolerance for stochastic noise


def test_unbiased_walk_variance_matches_n_steps():
    walk = RandomWalk(p_up=0.5, step_size=1.0, seed=7)
    n_steps = 100
    paths = walk.simulate_paths(n_steps=n_steps, n_paths=5000)
    final_values = paths[:, -1]
    # Var(S_n) = n * step_size^2 for unbiased unit-step walk
    empirical_var = np.var(final_values)
    assert empirical_var == pytest.approx(n_steps, rel=0.15)


def test_biased_walk_mean_matches_theory():
    p_up = 0.7
    walk = RandomWalk(p_up=p_up, step_size=1.0, seed=3)
    n_steps = 200
    paths = walk.simulate_paths(n_steps=n_steps, n_paths=5000)
    final_values = paths[:, -1]
    # E[step] = p*1 + (1-p)*(-1) = 2p - 1; E[S_n] = n * (2p - 1)
    expected_mean = n_steps * (2 * p_up - 1)
    assert np.mean(final_values) == pytest.approx(expected_mean, rel=0.05)


def test_scaling_limit_paths_shape_and_variance():
    walk = RandomWalk(seed=11)
    T = 1.0
    n_steps = 1000
    n_paths = 2000
    time_grid, paths = walk.scaling_limit_paths(n_steps=n_steps, n_paths=n_paths, T=T)

    assert time_grid.shape == (n_steps + 1,)
    assert paths.shape == (n_paths, n_steps + 1)
    assert time_grid[0] == 0.0
    assert time_grid[-1] == pytest.approx(T)

    # By Donsker's theorem, rescaled walk at t=T should behave like N(0, T)
    final_values = paths[:, -1]
    assert np.mean(final_values) == pytest.approx(0.0, abs=0.2)
    assert np.var(final_values) == pytest.approx(T, rel=0.15)
