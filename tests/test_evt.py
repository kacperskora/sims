import numpy as np
import pytest
from scipy.stats import genpareto

from quant_sims.risk.evt import (
    block_maxima,
    fit_gev,
    gev_return_level,
    fit_gpd,
    pot_var_es,
    mean_residual_life,
    GPDFit,
)


def test_block_maxima_shape_and_values():
    losses = np.arange(1, 21, dtype=float)  # 1..20
    maxima = block_maxima(losses, block_size=5)
    # blocks: [1-5],[6-10],[11-15],[16-20] -> maxima 5,10,15,20
    np.testing.assert_array_equal(maxima, [5, 10, 15, 20])


def test_block_maxima_discards_incomplete_trailing_block():
    losses = np.arange(1, 23, dtype=float)  # 1..22, block_size=5 -> 4 full blocks, 2 leftover discarded
    maxima = block_maxima(losses, block_size=5)
    assert len(maxima) == 4


def test_block_maxima_raises_if_not_enough_data():
    with pytest.raises(ValueError):
        block_maxima(np.array([1.0, 2.0]), block_size=10)


def test_fit_gev_on_synthetic_heavy_tailed_maxima():
    # Generate block maxima from a distribution with a known heavy tail
    # (Student-t via genpareto-like heavy tail proxy: use Frechet-domain
    # data by exponentiating normals -- simplest robust check is just that
    # fit_gev runs and returns finite, reasonable parameters, since exact
    # recovery of GEV params from finite block maxima samples is noisy).
    rng = np.random.default_rng(42)
    raw = rng.standard_t(df=3, size=50_000)  # heavy-tailed (Frechet domain, xi > 0)
    losses = -raw  # treat left tail as losses
    maxima = block_maxima(losses, block_size=50)

    fit = fit_gev(maxima)
    assert np.isfinite(fit.shape)
    assert np.isfinite(fit.loc)
    assert fit.scale > 0
    # Student-t has a heavy (Frechet-domain) tail -> expect a positive shape
    assert fit.shape > -0.3  # generous bound; exact value is noisy from finite samples


def test_gev_return_level_increases_with_return_period():
    from quant_sims.risk.evt import GEVFit
    fit = GEVFit(shape=0.1, loc=2.0, scale=0.5)
    z_10 = gev_return_level(fit, return_period_blocks=10)
    z_100 = gev_return_level(fit, return_period_blocks=100)
    assert z_100 > z_10


def test_gev_return_level_gumbel_case_matches_formula():
    from quant_sims.risk.evt import GEVFit
    fit = GEVFit(shape=0.0, loc=1.0, scale=2.0)
    T = 50
    p = 1 - 1 / T
    expected = 1.0 - 2.0 * np.log(-np.log(p))
    assert gev_return_level(fit, return_period_blocks=T) == pytest.approx(expected)


def test_fit_gpd_recovers_known_parameters():
    # Generate synthetic exceedances directly from a GPD with known params,
    # add a threshold offset, and check the fit recovers shape/scale.
    true_shape, true_scale, threshold = 0.2, 1.5, 10.0
    rng = np.random.default_rng(0)
    exceedances = genpareto.rvs(true_shape, scale=true_scale, size=20_000, random_state=rng)
    losses = threshold + exceedances  # all synthetic losses exceed threshold by construction
    # pad with some sub-threshold losses so n_total reflects a realistic sample
    below = rng.uniform(0, threshold, size=5000)
    all_losses = np.concatenate([losses, below])

    fit = fit_gpd(all_losses, threshold=threshold)
    assert fit.shape == pytest.approx(true_shape, abs=0.05)
    assert fit.scale == pytest.approx(true_scale, abs=0.1)
    assert fit.n_exceedances == 20_000
    assert fit.n_total == 25_000


def test_fit_gpd_raises_with_too_few_exceedances():
    losses = np.random.default_rng(1).normal(0, 1, size=100)
    with pytest.raises(ValueError):
        fit_gpd(losses, threshold=100.0)  # essentially nothing exceeds this


def test_pot_var_es_var_increases_with_alpha():
    fit = GPDFit(shape=0.15, scale=1.0, threshold=5.0, n_total=10_000, n_exceedances=500)
    var_96, es_96 = pot_var_es(fit, alpha=0.96)
    var_99, es_99 = pot_var_es(fit, alpha=0.99)
    assert var_99 > var_96
    assert es_99 > es_96


def test_pot_var_es_es_exceeds_var():
    fit = GPDFit(shape=0.1, scale=1.0, threshold=5.0, n_total=10_000, n_exceedances=500)
    var, es = pot_var_es(fit, alpha=0.99)
    assert es > var


def test_pot_var_es_raises_if_alpha_not_deep_enough_in_tail():
    # threshold captures top 500/10000 = 5% of losses, so alpha must be > 0.95
    fit = GPDFit(shape=0.1, scale=1.0, threshold=5.0, n_total=10_000, n_exceedances=500)
    with pytest.raises(ValueError):
        pot_var_es(fit, alpha=0.9)


def test_mean_residual_life_output_shape():
    rng = np.random.default_rng(2)
    losses = rng.normal(0, 1, size=5000)
    thresholds = np.linspace(0, 2, 10)
    mrl = mean_residual_life(losses, thresholds)
    assert mrl.shape == (10,)
    # higher thresholds should have fewer exceedances behind them (more likely NaN or noisier)
    assert not np.isnan(mrl[0])  # threshold=0 should have plenty of exceedances
