import numpy as np
import pytest

from quant_sims.betting.coin_flip import CoinFlipGame, ruin_probability_montecarlo
from quant_sims.betting.kelly import fixed_fraction, all_in, kelly_strategy


def test_invalid_p_win_raises():
    with pytest.raises(ValueError):
        CoinFlipGame(p_win=0.0)
    with pytest.raises(ValueError):
        CoinFlipGame(p_win=1.0)


def test_invalid_payout_ratio_raises():
    with pytest.raises(ValueError):
        CoinFlipGame(payout_ratio=0.0)


def test_flip_is_deterministic_with_seed():
    game_a = CoinFlipGame(p_win=0.5, seed=42)
    game_b = CoinFlipGame(p_win=0.5, seed=42)
    outcomes_a = [game_a.flip() for _ in range(50)]
    outcomes_b = [game_b.flip() for _ in range(50)]
    assert outcomes_a == outcomes_b


def test_zero_bet_fraction_capital_unchanged():
    game = CoinFlipGame(p_win=0.5, seed=1)
    strat = fixed_fraction(0.0)
    result = game.simulate_series(strat, n_flips=100, initial_capital=100.0)
    assert np.all(result.capital_path == 100.0)
    assert result.ruined is False


def test_certain_win_grows_capital():
    game = CoinFlipGame(p_win=1.0 - 1e-9, payout_ratio=1.0, seed=1)
    # not literally certain (constructor forbids p=1.0), but overwhelmingly likely to win all flips
    strat = fixed_fraction(0.1)
    result = game.simulate_series(strat, n_flips=20, initial_capital=100.0)
    assert result.final_capital >= 100.0 * 0.99  # allow for the near-zero chance of a loss


def test_all_in_can_reach_ruin():
    game = CoinFlipGame(p_win=0.5, seed=7)
    strat = all_in()
    result = game.simulate_series(strat, n_flips=50, initial_capital=100.0)
    # all-in on a fair coin: a single loss wipes out capital completely
    assert result.ruined is True
    assert result.final_capital == 0.0


def test_simulate_series_output_shapes():
    game = CoinFlipGame(p_win=0.5, seed=3)
    strat = fixed_fraction(0.1)
    n_flips = 30
    result = game.simulate_series(strat, n_flips=n_flips, initial_capital=100.0)
    assert result.capital_path.shape == (n_flips + 1,)
    assert result.outcomes.shape == (n_flips,)


def test_monte_carlo_output_shape():
    game = CoinFlipGame(p_win=0.5, seed=5)
    strat = fixed_fraction(0.1)
    n_flips, n_sims = 20, 50
    paths = game.monte_carlo(strat, n_flips=n_flips, n_simulations=n_sims, initial_capital=100.0)
    assert paths.shape == (n_sims, n_flips + 1)
    assert np.all(paths[:, 0] == 100.0)


def test_kelly_strategy_outperforms_all_in_on_average_with_edge():
    # With a positive edge, full Kelly should heavily outperform all-in
    # (which almost surely goes to ruin) over many simulations, in median terms.
    p, b = 0.55, 1.0
    game_kelly = CoinFlipGame(p_win=p, payout_ratio=b, seed=123)
    game_allin = CoinFlipGame(p_win=p, payout_ratio=b, seed=123)

    kelly_strat = kelly_strategy(p, b)
    allin_strat = all_in()

    paths_kelly = game_kelly.monte_carlo(kelly_strat, n_flips=100, n_simulations=200, initial_capital=100.0)
    paths_allin = game_allin.monte_carlo(allin_strat, n_flips=100, n_simulations=200, initial_capital=100.0)

    assert np.median(paths_kelly[:, -1]) > np.median(paths_allin[:, -1])


def test_ruin_probability_montecarlo_is_high_for_all_in():
    game = CoinFlipGame(p_win=0.5, seed=9)
    strat = all_in()
    ruin_prob = ruin_probability_montecarlo(game, strat, n_flips=10, n_simulations=100, initial_capital=100.0)
    # all-in on a fair coin: ruin after the very first loss, extremely likely within 10 flips
    assert ruin_prob > 0.9


def test_ruin_probability_montecarlo_is_low_for_conservative_fixed_fraction():
    game = CoinFlipGame(p_win=0.6, seed=11)
    strat = fixed_fraction(0.05)
    ruin_prob = ruin_probability_montecarlo(game, strat, n_flips=50, n_simulations=200, initial_capital=100.0)
    assert ruin_prob < 0.2
