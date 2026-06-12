import pandas as pd
import pytest

from fetch_odds import fetch_odds_from_api, is_draw_outcome, load_manual_odds, normalize_odds_response


def _raw_response():
    return [
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
                                {"name": "Mexico", "price": 2.1},
                                {"name": "Draw", "price": 3.4},
                                {"name": "South Africa", "price": 4.8},
                            ],
                        }
                    ],
                }
            ],
        }
    ]


def test_normalize_odds_response_handles_valid_response():
    df = normalize_odds_response(_raw_response())

    assert len(df) == 3
    assert set(df["outcome_name"]) == {"Mexico", "Draw", "South Africa"}
    assert df.iloc[0]["bookmaker_title"] == "Danske Spil"


def test_normalize_odds_response_handles_empty_response():
    df = normalize_odds_response([])

    assert df.empty


def test_normalize_odds_response_detects_draw_outcome():
    assert is_draw_outcome("Draw")
    assert is_draw_outcome("Tie")
    assert is_draw_outcome("Uafgjort")


def test_normalize_odds_response_handles_missing_bookmaker():
    raw = _raw_response()
    raw[0]["bookmakers"] = []

    df = normalize_odds_response(raw)

    assert df.empty


def test_fetch_odds_from_api_uses_mocked_http(monkeypatch):
    class Response:
        status_code = 200

        def json(self):
            return _raw_response()

    def fake_get(url, params, timeout):
        assert params["apiKey"] == "key"
        assert params["regions"] == "eu"
        return Response()

    monkeypatch.setattr("fetch_odds.requests.get", fake_get)

    df = fetch_odds_from_api("key", "soccer_fifa_world_cup")

    assert len(df) == 3


def test_fetch_odds_from_api_raises_for_invalid_key(monkeypatch):
    class Response:
        status_code = 401

        def json(self):
            return {}

    monkeypatch.setattr("fetch_odds.requests.get", lambda *args, **kwargs: Response())

    with pytest.raises(ValueError, match="API key"):
        fetch_odds_from_api("bad", "soccer_fifa_world_cup")


def test_load_manual_odds_enriches_rows_from_fixture_match_id(tmp_path):
    manual_path = tmp_path / "manual_odds.csv"
    pd.DataFrame(
        [
            {
                "match_id": "WC2026-GRA-001",
                "bookmaker_title": "Danske Spil",
                "outcome_name": "Mexico",
                "outcome_price": 1.44,
                "source": "manual check",
                "source_last_checked": "2026-06-12",
            }
        ]
    ).to_csv(manual_path, index=False)
    fixtures = pd.DataFrame(
        [
            {
                "match_id": "WC2026-GRA-001",
                "kickoff_utc": "2026-06-11T19:00:00Z",
                "home_team": "Mexico",
                "away_team": "South Africa",
            }
        ]
    )

    df = load_manual_odds(manual_path, fixtures_df=fixtures)

    assert len(df) == 1
    row = df.iloc[0]
    assert row["event_id"] == "WC2026-GRA-001"
    assert row["commence_time"] == "2026-06-11T19:00:00Z"
    assert row["home_team"] == "Mexico"
    assert row["bookmaker_key"] == "danske_spil"
    assert row["market_key"] == "h2h"
    assert row["outcome_price"] == 1.44


def test_load_manual_odds_skips_incomplete_rows(tmp_path):
    manual_path = tmp_path / "manual_odds.csv"
    pd.DataFrame(
        [
            {"match_id": "WC2026-GRA-001", "bookmaker_title": "Danske Spil", "outcome_name": "Mexico"},
            {"match_id": "", "bookmaker_title": "Danske Spil", "outcome_name": "Draw", "outcome_price": 3.2},
        ]
    ).to_csv(manual_path, index=False)

    df = load_manual_odds(manual_path)

    assert df.empty
