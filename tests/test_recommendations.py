import pandas as pd
import pytest

from recommendations import recommend_for_match


def _row(**overrides):
    data = {
        "model_home_prob": 0.50,
        "model_draw_prob": 0.30,
        "model_away_prob": 0.20,
        "ds_home_odds": 2.10,
        "ds_draw_odds": 3.20,
        "ds_away_odds": 4.50,
        "best_home_odds": 2.20,
        "best_home_bookmaker": "Bet365",
        "best_draw_odds": 3.60,
        "best_draw_bookmaker": "Unibet",
        "best_away_odds": 5.50,
        "best_away_bookmaker": "Betfair",
    }
    data.update(overrides)
    return pd.Series(data)


def test_minimum_edge_threshold():
    recommendation = recommend_for_match(
        _row(ds_home_odds=2.04, best_home_odds=2.04, best_draw_odds=3.30, best_away_odds=5.00),
        current_bankroll=10000,
    )
    assert recommendation["recommended_outcome_ds"] == "No bet"
    assert recommendation["recommended_outcome_best"] == "No bet"
    assert recommendation["status"] == "No bet"


def test_minimum_stake_threshold():
    recommendation = recommend_for_match(
        _row(
            model_home_prob=0.50,
            model_draw_prob=0.30,
            model_away_prob=0.20,
            ds_home_odds=2.052,
            ds_draw_odds=3.20,
            ds_away_odds=4.50,
            best_home_odds=2.052,
            best_draw_odds=3.20,
            best_away_odds=4.50,
        ),
        current_bankroll=10000,
        staking_profile={"min_edge_threshold": 0.025, "min_stake_pct_threshold": 0.01},
    )
    assert recommendation["recommended_outcome_ds"] == "No bet"
    assert recommendation["recommended_outcome_best"] == "No bet"


def test_danske_spil_recommendation_separate_from_best_market():
    recommendation = recommend_for_match(
        _row(ds_home_odds=2.10, best_home_odds=2.40, best_home_bookmaker="Betfair"),
        current_bankroll=10000,
    )
    assert recommendation["recommended_outcome_ds"] == "Home"
    assert recommendation["recommended_odds_ds"] == pytest.approx(2.10)
    assert recommendation["recommended_outcome_best"] == "Home"
    assert recommendation["recommended_odds_best"] == pytest.approx(2.40)
    assert recommendation["recommended_bookmaker_best"] == "Betfair"
    assert recommendation["recommended_stake_ds"] == pytest.approx(113.63636363636364)
    assert recommendation["recommended_stake_best"] == pytest.approx(250)


def test_better_elsewhere_if_best_qualifies_but_danske_spil_does_not():
    recommendation = recommend_for_match(
        _row(ds_home_odds=2.00, best_home_odds=2.20, best_home_bookmaker="Unibet"),
        current_bankroll=10000,
    )
    assert recommendation["recommended_outcome_ds"] == "No bet"
    assert recommendation["recommended_outcome_best"] == "Home"
    assert recommendation["recommended_bookmaker_best"] == "Unibet"
    assert recommendation["status"] == "Better elsewhere"
