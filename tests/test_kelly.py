import numpy as np
import pytest

from quant_sims.betting.kelly import (
    kelly_fraction,
    expected_log_growth,
    has_positive_edge,
    fixed_fraction,
    kelly_strategy,
    martingale,
    all_in,
)


def test_kelly_fraction_fair_coin_no_edge():
    # p=0.5, b=1.0 -> no edge, optimal fraction should be 0
    assert kelly_fraction(0.5, 1.0) == pytest.approx(0.0, abs=1e-9)


def test_kelly_fraction_known_case():
    # classic textbook example: p=0.6, b=1.0 -> f* = 0.2
    assert kelly_fraction(0.6, 1.0) == pytest.approx(0.2, abs=1e-9)


def test_kelly_fraction_negative_edge():
    # p=0.4, b=1.0 -> negative edge, f* should be negative
    f_star = kelly_fraction(0.4, 1.0)
    assert f_star < 0


def test_kelly_fraction_invalid_probability_raises():
    with pytest.raises(ValueError):
        kelly_fraction(0.0, 1.0)
    with pytest.raises(ValueError):
        kelly_fraction(1.0, 1.0)


def test_kelly_fraction_invalid_odds_raises():
    with pytest.raises(ValueError):
        kelly_fraction(0.5, 0.0)


def test_has_positive_edge():
    assert has_positive_edge(0.6, 1.0) is True
    assert has_positive_edge(0.4, 1.0) is False


def test_expected_log_growth_at_kelly_is_maximum():
    p, b = 0.6, 1.0
    f_star = kelly_fraction(p, b)
    growth_at_star = expected_log_growth(p, b, f_star)

    # perturb slightly in both directions -- growth should be lower
    for delta in (-0.05, 0.05):
        f_perturbed = f_star + delta
        if -1.0 / b < f_perturbed < 1.0:
            assert expected_log_growth(p, b, f_perturbed) < growth_at_star


def test_expected_log_growth_out_of_bounds_is_negative_infinity():
    assert expected_log_growth(0.6, 1.0, 1.0) == -np.inf
    assert expected_log_growth(0.6, 1.0, -2.0) == -np.inf


def test_fixed_fraction_strategy_returns_constant():
    strat = fixed_fraction(0.1)
    assert strat(100.0, []) == 0.1
    assert strat(50.0, [True, False]) == 0.1


def test_kelly_strategy_clips_negative_edge_to_zero():
    strat = kelly_strategy(0.4, 1.0)  # negative edge
    assert strat(100.0, []) == 0.0


def test_kelly_strategy_half_kelly_is_half_of_full():
    full = kelly_strategy(0.6, 1.0, multiplier=1.0)
    half = kelly_strategy(0.6, 1.0, multiplier=0.5)
    assert half(100.0, []) == pytest.approx(full(100.0, []) / 2, abs=1e-9)


def test_martingale_doubles_after_losses():
    strat = martingale(base_fraction=0.05, max_fraction=1.0)
    assert strat(100.0, []) == 0.05                       # no history -> base
    assert strat(100.0, [True]) == 0.05                    # last was win -> base
    assert strat(100.0, [True, False]) == 0.10              # one loss -> 2x
    assert strat(100.0, [True, False, False]) == 0.20       # two losses -> 4x


def test_martingale_caps_at_max_fraction():
    strat = martingale(base_fraction=0.5, max_fraction=1.0)
    long_loss_streak = [False] * 10
    assert strat(100.0, long_loss_streak) == 1.0


def test_all_in_always_bets_full_capital():
    strat = all_in()
    assert strat(100.0, []) == 1.0
    assert strat(1.0, [False, False]) == 1.0
