import pytest

from odds_utils import (
    calculate_edge,
    decimal_odds_to_implied_probability,
    remove_overround_proportional,
)


def test_decimal_odds_to_implied_probability():
    assert decimal_odds_to_implied_probability(2.0) == pytest.approx(0.5)


def test_remove_overround_proportional_sums_to_one():
    normalized = remove_overround_proportional([0.55, 0.30, 0.25])
    assert sum(normalized) == pytest.approx(1.0)


def test_calculate_edge():
    assert calculate_edge(0.55, 2.0) == pytest.approx(0.10)

