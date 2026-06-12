import pytest
import pandas as pd

from fetch_odds import normalize_odds_response
from odds_mapping import (
    build_match_odds_table,
    calculate_market_fair_probabilities_from_best_or_consensus,
    identify_best_market_odds,
    identify_preferred_bookmaker_odds,
    map_odds_to_fixtures,
    normalize_team_name,
)
from odds_normalizer import normalize_the_odds_api_response


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


def _fixtures():
    return pd.DataFrame(
        [
            {
                "match_id": "WC2026-GRA-001",
                "kickoff_utc": "2026-06-11T19:00:00Z",
                "home_team": "Mexico",
                "away_team": "South Africa",
                "group": "A",
                "stage": "Group stage",
                "matchday": 1,
            },
            {
                "match_id": "WC2026-GRB-001",
                "kickoff_utc": "2026-06-12T19:00:00Z",
                "home_team": "Canada",
                "away_team": "Bosnia and Herzegovina",
                "group": "B",
                "stage": "Group stage",
                "matchday": 1,
            },
        ]
    )


def _normalized_odds(event_id="WC2026-GRA-001", home_team="Mexico", away_team="South Africa"):
    raw = [
        {
            "id": event_id,
            "commence_time": "2026-06-11T20:00:00Z",
            "home_team": home_team,
            "away_team": away_team,
            "bookmakers": [
                {
                    "key": "book_a",
                    "title": "Book A",
                    "last_update": "2026-06-10T12:00:00Z",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": home_team, "price": 1.5},
                                {"name": "Draw", "price": 4.0},
                                {"name": away_team, "price": 7.0},
                            ],
                        }
                    ],
                },
                {
                    "key": "danske_spil",
                    "title": "Danske Spil",
                    "last_update": "2026-06-10T12:01:00Z",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": home_team, "price": 1.44},
                                {"name": "Tie", "price": 4.2},
                                {"name": away_team, "price": 7.5},
                            ],
                        }
                    ],
                },
            ],
        }
    ]
    return normalize_the_odds_api_response(raw, "2026-06-10T12:02:00Z")


def test_maps_odds_to_fixture_by_match_id():
    mapped, warnings = map_odds_to_fixtures(_fixtures(), _normalized_odds())

    assert mapped[mapped["odds_available"] == True]["match_id"].nunique() == 1
    assert "WC2026-GRA-001" in set(mapped["match_id"])
    assert any("WC2026-GRB-001" in warning for warning in warnings)


def test_maps_odds_to_fixture_by_team_names_and_kickoff_tolerance():
    odds = _normalized_odds(event_id="provider-event")

    mapped, _ = map_odds_to_fixtures(_fixtures(), odds)

    priced = mapped[mapped["odds_available"] == True]
    assert set(priced["match_id"]) == {"WC2026-GRA-001"}


def test_unmatched_fixture_remains_visible():
    mapped, _ = map_odds_to_fixtures(_fixtures(), pd.DataFrame())

    assert len(mapped) == 2
    assert mapped["odds_available"].eq(False).all()


def test_team_name_normalization_variations():
    assert normalize_team_name("USA") == normalize_team_name("United States")
    assert normalize_team_name("Bosnia-Herzegovina") == normalize_team_name("Bosnia and Herzegovina")
    assert normalize_team_name("Côte d'Ivoire") == normalize_team_name("Ivory Coast")
    assert normalize_team_name("Korea Republic") == normalize_team_name("South Korea")
    assert normalize_team_name("IR Iran") == normalize_team_name("Iran")
    assert normalize_team_name("Türkiye") == normalize_team_name("Turkey")


def test_build_match_odds_table_best_ds_and_consensus_probabilities():
    mapped, _ = map_odds_to_fixtures(_fixtures(), _normalized_odds())
    table = build_match_odds_table(mapped)
    row = table[table["match_id"] == "WC2026-GRA-001"].iloc[0]

    assert row["odds_available"] == True
    assert row["best_home_odds"] == pytest.approx(1.5)
    assert row["best_away_odds"] == pytest.approx(7.5)
    assert row["ds_home_odds"] == pytest.approx(1.44)
    assert row["bookmaker_count"] == 2
    assert row[["market_home_prob", "market_draw_prob", "market_away_prob"]].sum() == pytest.approx(1.0)


def test_danske_spil_unavailable_does_not_create_fake_odds():
    odds = _normalized_odds()
    odds = odds[odds["bookmaker_key"] != "danske_spil"]
    mapped, _ = map_odds_to_fixtures(_fixtures(), odds)
    row = build_match_odds_table(mapped)
    row = row[row["match_id"] == "WC2026-GRA-001"].iloc[0]

    assert pd.isna(row["ds_home_odds"])
    assert row["best_home_odds"] == pytest.approx(1.5)
