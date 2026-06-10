import pytest
import pandas as pd

from fetch_odds import normalize_odds_response
from odds_mapping import (
    calculate_market_fair_probabilities_from_best_or_consensus,
    identify_best_market_odds,
    identify_preferred_bookmaker_odds,
)


def _odds_df():
    raw = [
        {
            "id": "evt1",
            "commence_time": "2026-06-11T20:00:00Z",
            "home_team": "Mexico",
            "away_team": "South Africa",
            "bookmakers": [
                {
                    "key": "danske_spil",
                    "title": "Danske Spil",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Mexico", "price": 2.0},
                                {"name": "Draw", "price": 3.2},
                                {"name": "South Africa", "price": 4.5},
                            ],
                        }
                    ],
                },
                {
                    "key": "bet365",
                    "title": "Bet365",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Mexico", "price": 2.2},
                                {"name": "Tie", "price": 3.5},
                                {"name": "South Africa", "price": 4.2},
                            ],
                        }
                    ],
                },
            ],
        }
    ]
    return normalize_odds_response(raw)


def test_preferred_bookmaker_odds_identified_correctly():
    preferred = identify_preferred_bookmaker_odds(_odds_df())

    assert preferred.iloc[0]["ds_home_odds"] == pytest.approx(2.0)
    assert preferred.iloc[0]["ds_draw_odds"] == pytest.approx(3.2)


def test_missing_danske_spil_odds_handled_gracefully():
    odds = _odds_df()
    odds = odds[odds["bookmaker_title"] != "Danske Spil"]

    preferred = identify_preferred_bookmaker_odds(odds)

    assert pd.isna(preferred.iloc[0]["ds_home_odds"])
    assert "Danske Spil odds unavailable" in preferred.iloc[0]["warning"]


def test_best_home_draw_away_odds_identified_correctly():
    best = identify_best_market_odds(_odds_df())

    assert best.iloc[0]["best_home_odds"] == pytest.approx(2.2)
    assert best.iloc[0]["best_draw_odds"] == pytest.approx(3.5)
    assert best.iloc[0]["best_away_odds"] == pytest.approx(4.5)


def test_consensus_probabilities_sum_to_one():
    probs = calculate_market_fair_probabilities_from_best_or_consensus(_odds_df(), method="consensus")

    total = probs.iloc[0][["market_home_prob", "market_draw_prob", "market_away_prob"]].sum()
    assert total == pytest.approx(1.0)


def test_consensus_method_does_not_use_best_odds_by_default():
    consensus = calculate_market_fair_probabilities_from_best_or_consensus(_odds_df(), method="consensus")
    best = calculate_market_fair_probabilities_from_best_or_consensus(_odds_df(), method="best")

    assert consensus.iloc[0]["market_home_prob"] != pytest.approx(best.iloc[0]["market_home_prob"])


def test_incomplete_1x2_odds_are_skipped_for_consensus():
    odds = _odds_df()
    odds = odds[~odds["outcome_name"].isin(["Draw", "Tie"])]

    probs = calculate_market_fair_probabilities_from_best_or_consensus(odds, method="consensus")

    assert probs.empty
