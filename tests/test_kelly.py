import pytest

from kelly import (
    apply_stake_cap,
    calculate_final_stake_fraction,
    calculate_fractional_kelly,
    calculate_full_kelly,
    calculate_suggested_stake,
)


def test_positive_kelly():
    assert calculate_full_kelly(0.55, 2.0) == pytest.approx(0.10)


def test_negative_kelly_becomes_zero():
    assert calculate_full_kelly(0.40, 2.0) == 0


def test_fractional_kelly():
    assert calculate_fractional_kelly(0.10, 0.25) == pytest.approx(0.025)


def test_stake_cap():
    assert apply_stake_cap(0.10, 0.025) == pytest.approx(0.025)


def test_stake_calculated_from_current_bankroll():
    assert calculate_suggested_stake(8000, 0.025) == pytest.approx(200)


def test_final_stake_fraction_includes_fractional_kelly_and_cap():
    result = calculate_final_stake_fraction(0.60, 2.2)
    assert result["fractional_kelly"] > result["final_stake_fraction"]
    assert result["final_stake_fraction"] == pytest.approx(0.025)

