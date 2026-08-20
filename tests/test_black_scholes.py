import numpy as np
import pytest

from quant_sims.options.black_scholes import (
    BlackScholesParams,
    call_price,
    put_price,
    price,
    put_call_parity_check,
    delta,
    gamma,
    vega,
    theta,
    rho,
    all_greeks,
)


@pytest.fixture
def standard_params():
    # A common textbook example: S=K=100, T=1yr, r=5%, sigma=20%
    return BlackScholesParams(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)


def test_invalid_params_raise():
    with pytest.raises(ValueError):
        BlackScholesParams(S=0.0, K=100.0, T=1.0, r=0.05, sigma=0.2)
    with pytest.raises(ValueError):
        BlackScholesParams(S=100.0, K=0.0, T=1.0, r=0.05, sigma=0.2)
    with pytest.raises(ValueError):
        BlackScholesParams(S=100.0, K=100.0, T=0.0, r=0.05, sigma=0.2)
    with pytest.raises(ValueError):
        BlackScholesParams(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.0)


def test_call_price_known_value(standard_params):
    # Known reference value for S=K=100, T=1, r=5%, sigma=20%: approx 10.4506
    c = call_price(standard_params)
    assert c == pytest.approx(10.4506, abs=0.001)


def test_put_price_known_value(standard_params):
    # Known reference value: approx 5.5735
    p = put_price(standard_params)
    assert p == pytest.approx(5.5735, abs=0.001)


def test_price_dispatch(standard_params):
    assert price(standard_params, "call") == call_price(standard_params)
    assert price(standard_params, "put") == put_price(standard_params)
    with pytest.raises(ValueError):
        price(standard_params, "invalid")


def test_put_call_parity_holds(standard_params):
    assert put_call_parity_check(standard_params) is True


def test_put_call_parity_holds_across_random_params():
    rng = np.random.default_rng(0)
    for _ in range(20):
        p = BlackScholesParams(
            S=float(rng.uniform(10, 500)),
            K=float(rng.uniform(10, 500)),
            T=float(rng.uniform(0.01, 5)),
            r=float(rng.uniform(-0.02, 0.15)),
            sigma=float(rng.uniform(0.05, 1.0)),
        )
        assert put_call_parity_check(p, atol=1e-6) is True


def test_deep_itm_call_converges_to_intrinsic():
    # Deep in-the-money call with tiny time to expiry and low vol should be
    # close to its intrinsic value S - K*exp(-rT)
    p = BlackScholesParams(S=200.0, K=100.0, T=0.001, r=0.01, sigma=0.01)
    c = call_price(p)
    intrinsic = p.S - p.K * np.exp(-p.r * p.T)
    assert c == pytest.approx(intrinsic, abs=0.5)


def test_deep_otm_call_is_near_zero():
    p = BlackScholesParams(S=50.0, K=200.0, T=0.1, r=0.05, sigma=0.2)
    c = call_price(p)
    assert c < 0.01


def test_call_delta_between_zero_and_one(standard_params):
    d = delta(standard_params, "call")
    assert 0.0 < d < 1.0


def test_put_delta_between_minus_one_and_zero(standard_params):
    d = delta(standard_params, "put")
    assert -1.0 < d < 0.0


def test_gamma_is_positive_and_same_for_call_and_put(standard_params):
    g = gamma(standard_params)
    assert g > 0.0
    # gamma formula doesn't depend on option_type -- verify by construction
    assert gamma(standard_params) == g


def test_vega_is_positive(standard_params):
    assert vega(standard_params) > 0.0


def test_call_and_put_vega_are_equal(standard_params):
    assert vega(standard_params) == pytest.approx(vega(standard_params))


def test_all_greeks_returns_all_five_keys(standard_params):
    g = all_greeks(standard_params, "call")
    assert set(g.keys()) == {"delta", "gamma", "vega", "theta", "rho"}


def test_invalid_option_type_raises_in_greeks(standard_params):
    with pytest.raises(ValueError):
        delta(standard_params, "invalid")
    with pytest.raises(ValueError):
        theta(standard_params, "invalid")
    with pytest.raises(ValueError):
        rho(standard_params, "invalid")
