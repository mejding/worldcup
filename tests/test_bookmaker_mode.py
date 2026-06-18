import pandas as pd
import pytest

from recommendations import (
    calculate_expected_value,
    calculate_fair_odds,
    recommend_for_match,
)


def _row(**overrides):
    data = {
        "home_team": "Argentina",
        "away_team": "Austria",
        "active_home_prob": 0.648,
        "active_draw_prob": 0.231,
        "active_away_prob": 0.121,
        "model_home_prob": 0.648,
        "model_draw_prob": 0.231,
        "model_away_prob": 0.121,
        "market_home_prob": 0.597,
        "market_draw_prob": 0.243,
        "market_away_prob": 0.160,
        "ds_home_odds": 1.60,
        "ds_draw_odds": 3.90,
        "ds_away_odds": 7.00,
        "best_home_odds": 1.65,
        "best_home_bookmaker": "1xBet",
        "best_draw_odds": 4.00,
        "best_draw_bookmaker": "Bet365",
        "best_away_odds": 7.25,
        "best_away_bookmaker": "Unibet",
    }
    data.update(overrides)
    return pd.Series(data)


def test_default_bookmaker_mode_is_danske_spil():
    recommendation = recommend_for_match(_row(), current_bankroll=1000)

    assert recommendation["selected_bookmaker"] == "danske_spil"
    assert recommendation["primary_bookmaker"] == "Danske Spil"


def test_ds_playable_produces_primary_play_recommendation():
    recommendation = recommend_for_match(_row(ds_home_odds=1.60), current_bankroll=1000)

    assert recommendation["primary_status"] == "play"
    assert recommendation["primary_outcome"] == "Home"
    assert recommendation["primary_stake"] > 0


def test_ds_not_playable_but_best_market_playable_is_secondary_elsewhere():
    recommendation = recommend_for_match(_row(ds_home_odds=1.50, best_home_odds=1.65), current_bankroll=1000)

    assert recommendation["recommended_outcome_ds"] == "No bet"
    assert recommendation["recommended_outcome_best"] == "Home"
    assert recommendation["primary_status"] == "no_bet"
    assert recommendation["comparison_status"] == "better_elsewhere"
    assert recommendation["primary_stake"] == 0
    assert recommendation["comparison_stake"] > 0


def test_ds_odds_missing_but_best_market_exists_is_specific_to_ds():
    recommendation = recommend_for_match(_row(ds_home_odds=None, ds_draw_odds=None, ds_away_odds=None), current_bankroll=1000)

    assert recommendation["primary_status"] == "odds_missing"
    assert recommendation["comparison_outcome"] == "Home"
    assert recommendation["comparison_status"] == "better_elsewhere"


def test_best_market_mode_uses_best_market_as_primary():
    recommendation = recommend_for_match(_row(ds_home_odds=1.50, best_home_odds=1.65), current_bankroll=1000, preferred_bookmaker_mode="best_market")

    assert recommendation["selected_bookmaker"] == "best_market"
    assert recommendation["primary_bookmaker"] == "Best market"
    assert recommendation["primary_status"] == "play"
    assert recommendation["primary_odds"] == pytest.approx(1.65)


def test_favorite_is_separate_from_best_value():
    recommendation = recommend_for_match(
        _row(ds_home_odds=1.50, ds_draw_odds=5.00, active_draw_prob=0.231),
        current_bankroll=1000,
    )

    assert recommendation["model_favorite_label"] == "Argentina"
    assert recommendation["primary_outcome"] != recommendation["model_favorite_outcome"]


def test_fair_odds_and_expected_value():
    assert calculate_fair_odds(0.648) == pytest.approx(1.5432, rel=1e-3)
    assert calculate_expected_value(0.648, 1.60) == pytest.approx(0.0368)
