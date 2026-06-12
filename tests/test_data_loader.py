import pandas as pd

from config import REQUIRED_PREDICTION_COLUMNS
from data_loader import load_predictions, load_predictions_by_mode, validate_predictions


def _valid_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_id": "M001",
                "kickoff_time": "2026-06-11T20:00:00Z",
                "group": "A",
                "matchday": 1,
                "home_team": "Mexico",
                "away_team": "South Africa",
                "model_home_prob": 0.50,
                "model_draw_prob": 0.30,
                "model_away_prob": 0.20,
                "market_home_prob": 0.48,
                "market_draw_prob": 0.30,
                "market_away_prob": 0.22,
                "ds_home_odds": 2.10,
                "ds_draw_odds": 3.40,
                "ds_away_odds": 4.80,
                "best_home_odds": 2.20,
                "best_home_bookmaker": "Bet365",
                "best_draw_odds": 3.60,
                "best_draw_bookmaker": "Unibet",
                "best_away_odds": 5.00,
                "best_away_bookmaker": "Betfair",
                "draw_context_score": 45,
                "draw_context_label": "Medium",
                "home_draw_utility": 0.30,
                "away_draw_utility": 0.25,
                "mutual_draw_acceptance": 0.28,
                "one_team_must_win": False,
                "both_teams_draw_satisfied": False,
            }
        ]
    )


def test_load_predictions(tmp_path):
    path = tmp_path / "sample_predictions.csv"
    _valid_predictions().to_csv(path, index=False)

    df = load_predictions(path)

    assert len(df) == 1
    assert list(df[REQUIRED_PREDICTION_COLUMNS].columns) == REQUIRED_PREDICTION_COLUMNS
    assert df.loc[0, "kickoff_time"] == "2026-06-11T20:00:00Z"
    assert df.loc[0, "kickoff_time_dk"] == "11. juni 2026, 22:00 dansk tid"


def test_validate_required_columns():
    df = _valid_predictions().drop(columns=["match_id"])

    warnings, errors = validate_predictions(df)

    assert warnings == []
    assert "Missing required columns: match_id" in errors


def test_validate_valid_predictions():
    warnings, errors = validate_predictions(_valid_predictions())

    assert warnings == []
    assert errors == []


def test_detect_invalid_probabilities():
    df = _valid_predictions()
    df["model_home_prob"] = df["model_home_prob"].astype(object)
    df.loc[0, "model_home_prob"] = "bad"

    warnings, errors = validate_predictions(df)

    assert warnings == []
    assert any("Invalid probability columns" in error for error in errors)


def test_detect_invalid_odds():
    df = _valid_predictions()
    df.loc[0, "best_home_odds"] = 1.0

    warnings, errors = validate_predictions(df)

    assert warnings == []
    assert "Best market odds values must be numeric and greater than 1.0." in errors


def test_live_mode_does_not_fall_back_to_sample_when_live_file_is_missing(tmp_path):
    sample_path = tmp_path / "sample_predictions.csv"
    _valid_predictions().to_csv(sample_path, index=False)

    df, warnings, actual_mode = load_predictions_by_mode(
        "live",
        sample_path=sample_path,
        live_path=tmp_path / "missing_live.csv",
        model_source="market_only",
    )

    assert actual_mode == "live"
    assert df.empty
    assert any("No sample fallback" in warning for warning in warnings)


def test_sample_mode_is_explicit_and_marked_as_demo(tmp_path):
    sample_path = tmp_path / "sample_predictions.csv"
    _valid_predictions().to_csv(sample_path, index=False)

    df, warnings, actual_mode = load_predictions_by_mode(
        "sample",
        sample_path=sample_path,
        model_source="market_only",
    )

    assert actual_mode == "sample"
    assert warnings == []
    assert set(df["fixture_source"]) == {"sample_demo"}
    assert set(df["kickoff_utc"]) == {"2026-06-11T20:00:00Z"}


def test_official_mode_uses_reference_fixtures_not_sample_data():
    df, warnings, actual_mode = load_predictions_by_mode("official", model_source="market_only")

    assert actual_mode == "official"
    assert any("incomplete" in warning.lower() for warning in warnings)
    assert ((df["home_team"] == "Canada") & (df["away_team"] == "Bosnia and Herzegovina")).any()
    assert not (
        (df["home_team"] == "Canada")
        & (df["away_team"] == "Switzerland")
        & df["kickoff_utc"].astype(str).str.startswith("2026-06-12")
    ).any()
