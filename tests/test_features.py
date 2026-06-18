import pandas as pd

from features import FEATURE_COLUMNS, build_training_dataset, build_upcoming_feature_dataset, categorize_tournament
from historical_data import standardize_historical_results


def _history():
    return standardize_historical_results(
        pd.DataFrame(
            [
                {"date": "2020-01-01", "home_team": "A", "away_team": "B", "home_score": 2, "away_score": 0, "tournament": "Friendly", "neutral": False},
                {"date": "2020-01-02", "home_team": "C", "away_team": "D", "home_score": 1, "away_score": 1, "tournament": "World Cup qualification", "neutral": False},
                {"date": "2020-01-03", "home_team": "A", "away_team": "C", "home_score": 0, "away_score": 1, "tournament": "World Cup", "neutral": True},
            ]
        )
    )


def test_build_training_dataset_returns_rows(tmp_path):
    df = build_training_dataset(_history(), output_path=tmp_path / "training.csv")

    assert len(df) == 3
    assert (tmp_path / "training.csv").exists()


def test_no_future_leakage_in_simple_case(tmp_path):
    df = build_training_dataset(_history(), output_path=tmp_path / "training.csv")

    first = df.iloc[0]
    assert first["home_matches_played_before"] == 0
    assert first["away_matches_played_before"] == 0


def test_required_feature_columns_exist(tmp_path):
    df = build_training_dataset(_history(), output_path=tmp_path / "training.csv")

    for column in FEATURE_COLUMNS:
        assert column in df.columns


def test_fallback_defaults_work_for_teams_with_no_history(tmp_path):
    df = build_training_dataset(_history(), output_path=tmp_path / "training.csv")

    assert df.iloc[0]["home_win_rate_before"] == 0.33
    assert df.iloc[0]["home_points_per_match_last5"] == 1.0


def test_tournament_categorization():
    assert categorize_tournament("FIFA World Cup") == "world_cup"
    assert categorize_tournament("UEFA Euro") == "euro"
    assert categorize_tournament("World Cup qualification") == "qualifier"
    assert categorize_tournament("Friendly") == "friendly"


def test_upcoming_team_aliases_match_historical_names():
    historical = standardize_historical_results(
        pd.DataFrame(
            [
                {
                    "date": "2026-01-01",
                    "home_team": "Turkey",
                    "away_team": "A",
                    "home_score": 2,
                    "away_score": 0,
                    "tournament": "Friendly",
                    "neutral": True,
                }
            ]
        )
    )
    upcoming = pd.DataFrame(
        [
            {
                "kickoff_time": "2026-06-20T04:00:00Z",
                "home_team": "Türkiye",
                "away_team": "A",
                "group": "D",
                "matchday": 2,
            }
        ]
    )

    features = build_upcoming_feature_dataset(upcoming, historical)

    assert features.iloc[0]["home_matches_played_before"] == 1
