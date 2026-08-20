import numpy as np
import pytest

from quant_sims.options.black_scholes import BlackScholesParams, call_price, put_price
from quant_sims.options.monte_carlo import monte_carlo_price, convergence_study


@pytest.fixture
def standard_params():
    return BlackScholesParams(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)


def test_mc_call_price_close_to_black_scholes(standard_params):
    bs = call_price(standard_params)
    mc = monte_carlo_price(standard_params, option_type="call", n_simulations=200_000, seed=42)
    assert mc == pytest.approx(bs, abs=0.1)


def test_mc_put_price_close_to_black_scholes(standard_params):
    bs = put_price(standard_params)
    mc = monte_carlo_price(standard_params, option_type="put", n_simulations=200_000, seed=42)
    assert mc == pytest.approx(bs, abs=0.1)


def test_mc_price_reproducible_with_seed(standard_params):
    mc1 = monte_carlo_price(standard_params, n_simulations=10_000, seed=7)
    mc2 = monte_carlo_price(standard_params, n_simulations=10_000, seed=7)
    assert mc1 == mc2


def test_mc_price_returns_std_error_when_requested(standard_params):
    price, std_error = monte_carlo_price(standard_params, n_simulations=10_000, seed=1, return_std_error=True)
    assert isinstance(price, float)
    assert std_error > 0.0


def test_mc_std_error_shrinks_with_more_simulations(standard_params):
    _, se_small = monte_carlo_price(standard_params, n_simulations=1_000, seed=1, return_std_error=True)
    _, se_large = monte_carlo_price(standard_params, n_simulations=100_000, seed=1, return_std_error=True)
    assert se_large < se_small


def test_convergence_study_output_structure(standard_params):
    result = convergence_study(
        standard_params,
        option_type="call",
        sample_sizes=np.array([100, 1_000, 10_000]),
        seed=0,
    )
    assert set(result.keys()) == {"sample_sizes", "mc_prices", "std_errors", "bs_price", "abs_errors"}
    assert len(result["mc_prices"]) == 3
    assert len(result["std_errors"]) == 3
    assert result["bs_price"] == pytest.approx(call_price(standard_params))


def test_convergence_study_errors_trend_downward(standard_params):
    result = convergence_study(
        standard_params,
        option_type="call",
        sample_sizes=np.array([100, 10_000, 500_000]),
        seed=123,
    )
    # std_errors should shrink monotonically with sample size (theoretical
    # guarantee); abs_errors are noisier but should trend down too
    assert result["std_errors"][0] > result["std_errors"][1] > result["std_errors"][2]
