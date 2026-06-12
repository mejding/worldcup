import pytest

from config import REQUIRED_PREDICTION_COLUMNS
from fetch_odds import normalize_odds_response
from live_data_pipeline import build_live_predictions


def _odds_df(include_ds=True, include_draw=True):
    bookmakers = []
    if include_ds:
        outcomes = [
            {"name": "Mexico", "price": 2.0},
            {"name": "South Africa", "price": 4.5},
        ]
        if include_draw:
            outcomes.insert(1, {"name": "Draw", "price": 3.2})
        bookmakers.append(
            {
                "key": "danske_spil",
                "title": "Danske Spil",
                "markets": [{"key": "h2h", "outcomes": outcomes}],
            }
        )
    bookmakers.append(
        {
            "key": "bet365",
            "title": "Bet365",
            "markets": [
                {
                    "key": "h2h",
                    "outcomes": [
                        {"name": "Mexico", "price": 2.2},
                        *([{"name": "Tie", "price": 3.5}] if include_draw else []),
                        {"name": "South Africa", "price": 4.2},
                    ],
                }
            ],
        }
    )
    return normalize_odds_response(
        [
            {
                "id": "evt1",
                "commence_time": "2026-06-11T20:00:00Z",
                "home_team": "Mexico",
                "away_team": "South Africa",
                "bookmakers": bookmakers,
            }
        ]
    )


def test_build_live_predictions_returns_required_schema(tmp_path):
    output = tmp_path / "live_predictions.csv"

    df = build_live_predictions(_odds_df(), output_path=output)

    assert list(df[REQUIRED_PREDICTION_COLUMNS].columns) == REQUIRED_PREDICTION_COLUMNS
    assert "kickoff_utc" in df.columns
    assert "fixture_source" in df.columns
    assert output.exists()


def test_model_probabilities_equal_market_probabilities_in_sprint_5(tmp_path):
    df = build_live_predictions(_odds_df(), output_path=tmp_path / "live.csv")

    row = df.iloc[0]
    assert row["model_home_prob"] == pytest.approx(row["market_home_prob"])
    assert row["model_draw_prob"] == pytest.approx(row["market_draw_prob"])
    assert row["model_away_prob"] == pytest.approx(row["market_away_prob"])


def test_output_probabilities_sum_to_one(tmp_path):
    df = build_live_predictions(_odds_df(), output_path=tmp_path / "live.csv")

    total = df.iloc[0][["model_home_prob", "model_draw_prob", "model_away_prob"]].sum()
    assert total == pytest.approx(1.0)


def test_missing_ds_odds_does_not_drop_match(tmp_path):
    df = build_live_predictions(_odds_df(include_ds=False), output_path=tmp_path / "live.csv")

    assert len(df) == 1
    assert df.iloc[0]["best_home_bookmaker"] == "Bet365"


def test_missing_complete_1x2_odds_drops_match(tmp_path):
    df = build_live_predictions(_odds_df(include_draw=False), output_path=tmp_path / "live.csv")

    assert df.empty


def test_placeholder_draw_context_fields_are_present(tmp_path):
    df = build_live_predictions(_odds_df(), output_path=tmp_path / "live.csv")

    row = df.iloc[0]
    assert row["draw_context_score"] == 50
    assert row["draw_context_label"] == "Medium"
    assert row["one_team_must_win"] == False
