"""
Finite-difference (numerical) Greeks -- an alternative to the analytical
Black-Scholes derivatives in black_scholes.py, computed by perturbing each
input parameter by a small amount and re-pricing.

This is the general-purpose approach used when a closed-form price isn't
available (e.g. for Monte Carlo pricing of path-dependent or American
options, where derivatives can't be written down analytically). Comparing
finite-difference Greeks against the analytical ones here validates that
the numerical approach is correct on a case where we do have ground truth.
"""

from __future__ import annotations

from dataclasses import replace

from .black_scholes import BlackScholesParams, price


def _bump(p: BlackScholesParams, **kwargs) -> BlackScholesParams:
    """Return a copy of p with the given fields perturbed."""
    return replace(p, **kwargs)


def fd_delta(p: BlackScholesParams, option_type: str = "call", h: float = 1e-4) -> float:
    """Central difference: (Price(S+h) - Price(S-h)) / (2h)."""
    dS = h * p.S
    up = price(_bump(p, S=p.S + dS), option_type)
    down = price(_bump(p, S=p.S - dS), option_type)
    return (up - down) / (2 * dS)


def fd_gamma(p: BlackScholesParams, option_type: str = "call", h: float = 1e-4) -> float:
    """Central second difference: (Price(S+h) - 2*Price(S) + Price(S-h)) / h^2."""
    dS = h * p.S
    up = price(_bump(p, S=p.S + dS), option_type)
    mid = price(p, option_type)
    down = price(_bump(p, S=p.S - dS), option_type)
    return (up - 2 * mid + down) / (dS ** 2)


def fd_vega(p: BlackScholesParams, option_type: str = "call", h: float = 1e-4) -> float:
    """Central difference w.r.t. sigma."""
    d_sigma = h
    up = price(_bump(p, sigma=p.sigma + d_sigma), option_type)
    down = price(_bump(p, sigma=p.sigma - d_sigma), option_type)
    return (up - down) / (2 * d_sigma)


def fd_theta(p: BlackScholesParams, option_type: str = "call", h: float = 1e-5) -> float:
    """
    Forward difference w.r.t. calendar time (i.e. dPrice/dt, t = calendar
    time). Since T = time-to-expiry decreases as calendar time advances,
    "one step forward in calendar time" corresponds to T -> T - h:

        theta = dPrice/dt ~= (Price(T - h) - Price(T)) / h

    matching the sign convention of black_scholes.theta (negative for a
    typical long option position -- value decays as expiry approaches).
    """
    dT = h
    if p.T - dT <= 0:
        raise ValueError("h too large relative to T; would make T non-positive")
    price_now = price(p, option_type)
    price_later = price(_bump(p, T=p.T - dT), option_type)
    return (price_later - price_now) / dT


def fd_rho(p: BlackScholesParams, option_type: str = "call", h: float = 1e-4) -> float:
    """Central difference w.r.t. r."""
    dr = h
    up = price(_bump(p, r=p.r + dr), option_type)
    down = price(_bump(p, r=p.r - dr), option_type)
    return (up - down) / (2 * dr)


def all_fd_greeks(p: BlackScholesParams, option_type: str = "call") -> dict:
    """Convenience wrapper returning all five finite-difference Greeks."""
    return {
        "delta": fd_delta(p, option_type),
        "gamma": fd_gamma(p, option_type),
        "vega": fd_vega(p, option_type),
        "theta": fd_theta(p, option_type),
        "rho": fd_rho(p, option_type),
    }
