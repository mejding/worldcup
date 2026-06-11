import pandas as pd

from historical_data import standardize_historical_results, validate_historical_results


def test_standardizes_valid_historical_data():
    df = pd.DataFrame(
        [{"date": "2020-01-01", "home_team": "A", "away_team": "B", "home_score": 2, "away_score": 1, "tournament": "Friendly", "neutral": False}]
    )

    result = standardize_historical_results(df)

    assert result.iloc[0]["result"] == "H"
    assert result.iloc[0]["tournament"] == "Friendly"


def test_detects_missing_required_columns():
    warnings, errors = validate_historical_results(pd.DataFrame({"date": ["2020-01-01"]}))

    assert warnings == []
    assert any("Missing required historical columns" in error for error in errors)


def test_creates_result_h_d_a_correctly():
    df = pd.DataFrame(
        [
            {"date": "2020-01-01", "home_team": "A", "away_team": "B", "home_score": 2, "away_score": 1},
            {"date": "2020-01-02", "home_team": "A", "away_team": "B", "home_score": 1, "away_score": 1},
            {"date": "2020-01-03", "home_team": "A", "away_team": "B", "home_score": 0, "away_score": 1},
        ]
    )

    result = standardize_historical_results(df)

    assert result["result"].tolist() == ["H", "D", "A"]


def test_handles_missing_tournament_and_neutral():
    df = pd.DataFrame(
        [{"date": "2020-01-01", "home_team": "A", "away_team": "B", "home_score": 2, "away_score": 1}]
    )

    warnings, errors = validate_historical_results(df)
    result = standardize_historical_results(df)

    assert errors == []
    assert "Missing tournament column. It will be set to Unknown." in warnings
    assert "Missing neutral column. It will be set to False." in warnings
    assert result.iloc[0]["tournament"] == "Unknown"
    assert result.iloc[0]["neutral"] == False

