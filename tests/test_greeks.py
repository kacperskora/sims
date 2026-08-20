import pytest

from quant_sims.options.black_scholes import (
    BlackScholesParams,
    delta,
    gamma,
    vega,
    theta,
    rho,
)
from quant_sims.options.greeks import (
    fd_delta,
    fd_gamma,
    fd_vega,
    fd_theta,
    fd_rho,
    all_fd_greeks,
)


@pytest.fixture
def standard_params():
    return BlackScholesParams(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)


def test_fd_delta_matches_analytical_call(standard_params):
    assert fd_delta(standard_params, "call") == pytest.approx(delta(standard_params, "call"), abs=1e-4)


def test_fd_delta_matches_analytical_put(standard_params):
    assert fd_delta(standard_params, "put") == pytest.approx(delta(standard_params, "put"), abs=1e-4)


def test_fd_gamma_matches_analytical(standard_params):
    assert fd_gamma(standard_params, "call") == pytest.approx(gamma(standard_params), rel=1e-2)


def test_fd_vega_matches_analytical(standard_params):
    assert fd_vega(standard_params, "call") == pytest.approx(vega(standard_params), abs=1e-3)


def test_fd_theta_matches_analytical_call(standard_params):
    assert fd_theta(standard_params, "call") == pytest.approx(theta(standard_params, "call"), abs=1e-2)


def test_fd_theta_matches_analytical_put(standard_params):
    assert fd_theta(standard_params, "put") == pytest.approx(theta(standard_params, "put"), abs=1e-2)


def test_fd_rho_matches_analytical_call(standard_params):
    assert fd_rho(standard_params, "call") == pytest.approx(rho(standard_params, "call"), abs=1e-3)


def test_fd_theta_raises_if_h_too_large(standard_params):
    tiny_T_params = BlackScholesParams(S=100.0, K=100.0, T=1e-6, r=0.05, sigma=0.2)
    with pytest.raises(ValueError):
        fd_theta(tiny_T_params, "call", h=1e-5)


def test_all_fd_greeks_returns_five_keys(standard_params):
    g = all_fd_greeks(standard_params, "call")
    assert set(g.keys()) == {"delta", "gamma", "vega", "theta", "rho"}
