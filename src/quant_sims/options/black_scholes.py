"""
Black-Scholes closed-form pricing for European options.

Under the risk-neutral measure Q, the underlying follows GBM with drift r
(the risk-free rate) instead of the real-world drift mu:

    dS_t = r*S_t dt + sigma*S_t dW_t^Q

This gives S_T lognormal, and the discounted expected payoff under Q has a
closed-form solution -- the Black-Scholes formula.

    d1 = [ln(S/K) + (r + sigma^2/2)*T] / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)

    Call = S*N(d1) - K*exp(-r*T)*N(d2)
    Put  = K*exp(-r*T)*N(-d2) - S*N(-d1)

Assumes a non-dividend-paying underlying, constant r and sigma, and European
exercise (no early exercise).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass
class BlackScholesParams:
    """Container for the standard Black-Scholes input parameters."""

    S: float       # current underlying price
    K: float       # strike price
    T: float       # time to expiry, in years
    r: float       # risk-free rate (annualized, continuously compounded)
    sigma: float   # volatility (annualized)

    def __post_init__(self):
        if self.S <= 0:
            raise ValueError("S must be positive")
        if self.K <= 0:
            raise ValueError("K must be positive")
        if self.T <= 0:
            raise ValueError("T must be positive")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")


def _d1_d2(p: BlackScholesParams) -> tuple[float, float]:
    d1 = (np.log(p.S / p.K) + (p.r + 0.5 * p.sigma ** 2) * p.T) / (p.sigma * np.sqrt(p.T))
    d2 = d1 - p.sigma * np.sqrt(p.T)
    return d1, d2


def call_price(p: BlackScholesParams) -> float:
    """Closed-form Black-Scholes price of a European call."""
    d1, d2 = _d1_d2(p)
    return p.S * norm.cdf(d1) - p.K * np.exp(-p.r * p.T) * norm.cdf(d2)


def put_price(p: BlackScholesParams) -> float:
    """Closed-form Black-Scholes price of a European put."""
    d1, d2 = _d1_d2(p)
    return p.K * np.exp(-p.r * p.T) * norm.cdf(-d2) - p.S * norm.cdf(-d1)


def price(p: BlackScholesParams, option_type: str = "call") -> float:
    """Dispatch to call_price or put_price based on option_type ('call' or 'put')."""
    if option_type == "call":
        return call_price(p)
    elif option_type == "put":
        return put_price(p)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def put_call_parity_check(p: BlackScholesParams, atol: float = 1e-8) -> bool:
    """
    Sanity check: Call - Put = S - K*exp(-rT) should hold exactly for
    Black-Scholes prices (put-call parity, a model-free no-arbitrage
    relationship). Returns True if the identity holds within atol.
    """
    lhs = call_price(p) - put_price(p)
    rhs = p.S - p.K * np.exp(-p.r * p.T)
    return bool(abs(lhs - rhs) < atol)


# ---------------------------------------------------------------------------
# Analytical Greeks (closed-form derivatives of the Black-Scholes formula)
# ---------------------------------------------------------------------------


def delta(p: BlackScholesParams, option_type: str = "call") -> float:
    """dPrice/dS -- sensitivity to a $1 move in the underlying."""
    d1, _ = _d1_d2(p)
    if option_type == "call":
        return norm.cdf(d1)
    elif option_type == "put":
        return norm.cdf(d1) - 1.0
    raise ValueError("option_type must be 'call' or 'put'")


def gamma(p: BlackScholesParams) -> float:
    """d^2 Price/dS^2 -- rate of change of delta. Same for calls and puts."""
    d1, _ = _d1_d2(p)
    return norm.pdf(d1) / (p.S * p.sigma * np.sqrt(p.T))


def vega(p: BlackScholesParams) -> float:
    """dPrice/dsigma -- sensitivity to a 1.0 (100 percentage point) change
    in volatility. Same for calls and puts. Conventionally reported per 1%
    vol change, i.e. divide by 100 for that convention."""
    d1, _ = _d1_d2(p)
    return p.S * norm.pdf(d1) * np.sqrt(p.T)


def theta(p: BlackScholesParams, option_type: str = "call") -> float:
    """dPrice/dt with t = calendar time (i.e. -dPrice/dT) -- time decay,
    reported per year. Divide by 365 for a per-day convention."""
    d1, d2 = _d1_d2(p)
    term1 = -(p.S * norm.pdf(d1) * p.sigma) / (2 * np.sqrt(p.T))
    if option_type == "call":
        term2 = -p.r * p.K * np.exp(-p.r * p.T) * norm.cdf(d2)
        return term1 + term2
    elif option_type == "put":
        term2 = p.r * p.K * np.exp(-p.r * p.T) * norm.cdf(-d2)
        return term1 + term2
    raise ValueError("option_type must be 'call' or 'put'")


def rho(p: BlackScholesParams, option_type: str = "call") -> float:
    """dPrice/dr -- sensitivity to a 1.0 (100 percentage point) change in
    the risk-free rate. Conventionally reported per 1%, i.e. divide by 100."""
    _, d2 = _d1_d2(p)
    if option_type == "call":
        return p.K * p.T * np.exp(-p.r * p.T) * norm.cdf(d2)
    elif option_type == "put":
        return -p.K * p.T * np.exp(-p.r * p.T) * norm.cdf(-d2)
    raise ValueError("option_type must be 'call' or 'put'")


def all_greeks(p: BlackScholesParams, option_type: str = "call") -> dict:
    """Convenience wrapper returning all five Greeks in a dict."""
    return {
        "delta": delta(p, option_type),
        "gamma": gamma(p),
        "vega": vega(p),
        "theta": theta(p, option_type),
        "rho": rho(p, option_type),
    }
