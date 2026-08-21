"""
Extreme Value Theory (EVT) for tail risk estimation.

Two classical approaches, both fit to *losses* (positive = bad), reflecting
the same loss convention as var_es.py:

1. Block Maxima -> Generalized Extreme Value (GEV) distribution.
   Split the loss series into blocks (e.g. monthly), take the max loss in
   each block, and fit a GEV distribution to those block maxima. Useful for
   "what's the worst loss we'd expect over the next N blocks" (return level
   questions).

2. Peaks-Over-Threshold (POT) -> Generalized Pareto Distribution (GPD).
   Fit a GPD to the exceedances over a high threshold u (i.e. losses - u,
   for losses > u). Uses more of the data than block maxima (every
   exceedance, not just one per block) and is the more commonly used
   approach in practice for tail VaR/ES estimation.

Both rely on limit theorems (Fisher-Tippett-Gnedenko for GEV, Pickands-
Balkema-de Haan for GPD) that justify these specific parametric families
for tail behavior, regardless of the underlying full distribution -- this
is precisely the point: we don't need to correctly model the *entire*
return distribution, just its tail.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import genextreme, genpareto


@dataclass
class GEVFit:
    shape: float   # xi: shape parameter (xi > 0 => heavy (Frechet) tail, xi=0 => Gumbel, xi<0 => bounded (Weibull))
    loc: float     # mu: location parameter
    scale: float   # sigma: scale parameter


@dataclass
class GPDFit:
    shape: float   # xi: shape parameter, same interpretation as GEV's xi
    scale: float   # beta: scale parameter
    threshold: float
    n_total: int         # total number of observations the exceedances were drawn from
    n_exceedances: int    # number of observations exceeding the threshold


def block_maxima(losses: np.ndarray, block_size: int) -> np.ndarray:
    """
    Split a loss series into non-overlapping blocks of block_size
    observations and return the maximum loss in each complete block
    (a trailing incomplete block, if any, is discarded).
    """
    losses = np.asarray(losses)
    n_blocks = len(losses) // block_size
    if n_blocks < 1:
        raise ValueError("Not enough observations for even one full block")
    trimmed = losses[: n_blocks * block_size]
    return trimmed.reshape(n_blocks, block_size).max(axis=1)


def fit_gev(maxima: np.ndarray) -> GEVFit:
    """
    Fit a Generalized Extreme Value distribution to a series of block
    maxima via maximum likelihood (scipy's genextreme.fit).

    Note: scipy's genextreme parameterizes the shape as `c`, related to the
    conventional EVT shape xi by xi = -c. We convert here so `shape`
    follows the standard EVT convention (xi > 0 = heavy tail).
    """
    c, loc, scale = genextreme.fit(maxima)
    return GEVFit(shape=-c, loc=loc, scale=scale)


def gev_return_level(fit: GEVFit, return_period_blocks: float) -> float:
    """
    Compute the return level associated with a given return period
    (expressed in number of blocks), i.e. the loss level expected to be
    exceeded on average once every `return_period_blocks` blocks.

        z_p = mu + (sigma/xi) * [(-log(1 - 1/T))^(-xi) - 1]     if xi != 0
        z_p = mu - sigma * log(-log(1 - 1/T))                     if xi == 0

    where T = return_period_blocks.
    """
    T = return_period_blocks
    p = 1.0 - 1.0 / T
    xi, mu, sigma = fit.shape, fit.loc, fit.scale

    if abs(xi) < 1e-8:
        return mu - sigma * np.log(-np.log(p))
    return mu + (sigma / xi) * ((-np.log(p)) ** (-xi) - 1.0)


def fit_gpd(losses: np.ndarray, threshold: float) -> GPDFit:
    """
    Fit a Generalized Pareto Distribution to the exceedances over a
    threshold u via maximum likelihood, using the Peaks-Over-Threshold
    (POT) method. The GPD is fit to (exceedance - u) with location fixed
    at 0 (floc=0), as is standard for POT.
    """
    losses = np.asarray(losses)
    exceedances = losses[losses > threshold] - threshold
    if len(exceedances) < 10:
        raise ValueError(
            f"Only {len(exceedances)} exceedances above threshold={threshold}; "
            "need at least ~10 for a stable GPD fit. Try a lower threshold."
        )

    shape, _, scale = genpareto.fit(exceedances, floc=0)
    return GPDFit(
        shape=shape,
        scale=scale,
        threshold=threshold,
        n_total=len(losses),
        n_exceedances=len(exceedances),
    )


def pot_var_es(fit: GPDFit, alpha: float = 0.99) -> tuple[float, float]:
    """
    Compute VaR and ES at confidence level alpha using the POT/GPD tail fit
    (McNeil & Frey formulas). Only valid for alpha above the fraction of
    data below the threshold (i.e. alpha must be in the tail actually
    modeled by the GPD fit).

        VaR_alpha = u + (beta/xi) * [(n/Nu * (1-alpha))^(-xi) - 1]
        ES_alpha  = (VaR_alpha + beta - xi*u) / (1 - xi)        for xi < 1

    where u = threshold, beta = GPD scale, xi = GPD shape, n = total sample
    size, Nu = number of exceedances.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be strictly between 0 and 1")

    xi, beta, u = fit.shape, fit.scale, fit.threshold
    n, Nu = fit.n_total, fit.n_exceedances

    prob_tail = Nu / n
    if (1.0 - alpha) >= prob_tail:
        raise ValueError(
            f"alpha={alpha} is not deep enough in the tail modeled by this fit "
            f"(threshold captures top {prob_tail:.4f} of losses); choose a higher "
            f"alpha or a lower threshold."
        )

    if abs(xi) < 1e-8:
        var = u - beta * np.log((1.0 - alpha) / prob_tail)
    else:
        var = u + (beta / xi) * ((n / Nu * (1.0 - alpha)) ** (-xi) - 1.0)

    if xi >= 1.0:
        es = np.inf  # ES is undefined (infinite) for xi >= 1
    else:
        es = (var + beta - xi * u) / (1.0 - xi)

    return float(var), float(es)


def mean_residual_life(losses: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """
    Compute the mean excess (mean of losses - u, among losses > u) at each
    candidate threshold u. Used to build a mean residual life plot for
    visually selecting a stable threshold for POT: the GPD is a reasonable
    fit above thresholds where this curve becomes roughly linear.

    Returns an array of mean excess values (NaN where there are too few
    exceedances to compute a meaningful mean).
    """
    losses = np.asarray(losses)
    result = np.empty(len(thresholds))
    for i, u in enumerate(thresholds):
        exceedances = losses[losses > u] - u
        result[i] = np.mean(exceedances) if len(exceedances) >= 5 else np.nan
    return result
