import pandas as pd

from recommendations import recommend_for_match


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


def test_ds_mode_does_not_show_best_market_stake_as_primary_stake():
    recommendation = recommend_for_match(_row(ds_home_odds=1.50, best_home_odds=1.65), current_bankroll=1000)

    assert recommendation["primary_status"] == "no_bet"
    assert recommendation["primary_stake"] == 0
    assert recommendation["comparison_stake"] > 0


def test_missing_odds_messages_are_specific():
    recommendation = recommend_for_match(_row(ds_home_odds=None, ds_draw_odds=None, ds_away_odds=None), current_bankroll=1000)

    assert recommendation["primary_status"] == "odds_missing"
    assert "selected bookmaker" in recommendation["primary_recommendation"]["reason"]


def test_no_bet_reason_is_shown():
    recommendation = recommend_for_match(_row(ds_home_odds=1.50), current_bankroll=1000)

    assert recommendation["primary_status"] == "no_bet"
    assert "too low" in recommendation["primary_reason"] or "threshold" in recommendation["primary_reason"]


def test_playable_ds_bet_exposes_odds_fair_odds_edge_kelly_and_stake():
    recommendation = recommend_for_match(_row(ds_home_odds=1.60), current_bankroll=1000)

    assert recommendation["recommended_odds_ds"] == 1.60
    assert recommendation["recommended_fair_odds_ds"] > 0
    assert recommendation["recommended_edge_ds"] > 0
    assert recommendation["recommended_fractional_kelly_ds"] > 0
    assert recommendation["recommended_stake_ds"] > 0


def test_better_elsewhere_card_is_secondary_in_ds_mode():
    recommendation = recommend_for_match(_row(ds_home_odds=1.50, best_home_odds=1.65), current_bankroll=1000)

    assert recommendation["comparison_status"] == "better_elsewhere"
    assert recommendation["comparison_bookmaker"] == "1xBet"
    assert recommendation["primary_bookmaker"] == "Danske Spil"
