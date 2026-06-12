from odds_normalizer import normalize_the_odds_api_response


def test_normalizes_event_bookmaker_market_outcome_rows():
    raw = [
        {
            "id": "evt1",
            "commence_time": "2026-06-11T19:00:00Z",
            "home_team": "Mexico",
            "away_team": "South Africa",
            "bookmakers": [
                {
                    "key": "book",
                    "title": "Book",
                    "last_update": "2026-06-10T12:00:00Z",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "South Africa", "price": 7.0},
                                {"name": "Draw", "price": 4.0},
                                {"name": "Mexico", "price": 1.5},
                            ],
                        }
                    ],
                }
            ],
        }
    ]

    df = normalize_the_odds_api_response(raw, "2026-06-10T12:01:00Z")

    assert len(df) == 3
    assert set(df["outcome_type"]) == {"home", "draw", "away"}
    assert df.loc[df["outcome_name"] == "Draw", "outcome_type"].iloc[0] == "draw"
    assert df.loc[df["outcome_name"] == "Mexico", "outcome_type"].iloc[0] == "home"
    assert df.loc[df["outcome_name"] == "South Africa", "outcome_type"].iloc[0] == "away"


def test_handles_missing_bookmakers_and_markets():
    assert normalize_the_odds_api_response([], "now").empty
    assert normalize_the_odds_api_response([{"id": "evt1", "bookmakers": []}], "now").empty
    assert normalize_the_odds_api_response([{"id": "evt1", "bookmakers": [{"key": "b", "markets": []}]}], "now").empty
