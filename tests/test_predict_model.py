import pandas as pd
import pytest

from features import build_training_dataset
from historical_data import standardize_historical_results
from predict_model import predict_upcoming_matches, prediction_file_uses_market_as_model
from train_model import train_historical_model


def _history():
    rows = []
    teams = ["A", "B", "C", "D", "E", "F"]
    scores = [(2, 0), (1, 1), (0, 2)]
    for i in range(45):
        hs, aw = scores[i % 3]
        rows.append(
            {
                "date": f"2020-02-{(i % 28) + 1:02d}",
                "home_team": teams[i % 6],
                "away_team": teams[(i + 1) % 6],
                "home_score": hs,
                "away_score": aw,
                "tournament": "Friendly",
                "neutral": False,
            }
        )
    return standardize_historical_results(pd.DataFrame(rows))


def _upcoming():
    return pd.DataFrame(
        [
            {
                "match_id": "M1",
                "kickoff_time": "2026-06-11T20:00:00Z",
                "group": "A",
                "matchday": 1,
                "home_team": "Unknown",
                "away_team": "A",
                "model_home_prob": 0.4,
                "model_draw_prob": 0.3,
                "model_away_prob": 0.3,
                "market_home_prob": 0.45,
                "market_draw_prob": 0.25,
                "market_away_prob": 0.30,
                "ds_home_odds": 2.0,
                "ds_draw_odds": 3.2,
                "ds_away_odds": 4.0,
                "best_home_odds": 2.1,
                "best_home_bookmaker": "Bet365",
                "best_draw_odds": 3.4,
                "best_draw_bookmaker": "Bet365",
                "best_away_odds": 4.2,
                "best_away_bookmaker": "Bet365",
                "draw_context_score": 50,
                "draw_context_label": "Medium",
                "home_draw_utility": 0.0,
                "away_draw_utility": 0.0,
                "mutual_draw_acceptance": 0.0,
                "one_team_must_win": False,
                "both_teams_draw_satisfied": False,
            }
        ]
    )


def test_model_can_predict_upcoming_matches(tmp_path, monkeypatch):
    monkeypatch.setattr("train_model.MODEL_METADATA_PATH", tmp_path / "metadata.json")
    monkeypatch.setattr("train_model.FEATURE_COLUMNS_PATH", tmp_path / "features.json")
    model_path = tmp_path / "model.pkl"
    train_historical_model(build_training_dataset(_history(), output_path=tmp_path / "training.csv"), model_output_path=model_path)

    result, warnings = predict_upcoming_matches(_upcoming(), _history(), model_path=model_path, output_path=tmp_path / "pred.csv")

    assert warnings == []
    assert result.iloc[0][["model_home_prob", "model_draw_prob", "model_away_prob"]].sum() == pytest.approx(1.0)
    assert result.iloc[0]["market_home_prob"] == pytest.approx(0.45)


def test_missing_model_falls_back_gracefully(tmp_path):
    result, warnings = predict_upcoming_matches(_upcoming(), _history(), model_path=tmp_path / "missing.pkl")

    assert warnings
    assert result.iloc[0]["model_home_prob"] == pytest.approx(result.iloc[0]["market_home_prob"])


def test_prediction_file_uses_market_as_model_detects_fallback(tmp_path):
    path = tmp_path / "predictions.csv"
    _upcoming().assign(
        model_home_prob=0.45,
        model_draw_prob=0.25,
        model_away_prob=0.30,
    ).to_csv(path, index=False)

    assert prediction_file_uses_market_as_model(path) is True


def test_prediction_file_uses_market_as_model_detects_mostly_fallback(tmp_path):
    path = tmp_path / "predictions.csv"
    rows = []
    for index in range(10):
        row = _upcoming().iloc[0].to_dict()
        row["match_id"] = f"M{index}"
        if index < 9:
            row["model_home_prob"] = row["market_home_prob"]
            row["model_draw_prob"] = row["market_draw_prob"]
            row["model_away_prob"] = row["market_away_prob"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)

    assert prediction_file_uses_market_as_model(path) is True


def test_prediction_file_uses_market_as_model_allows_distinct_model(tmp_path):
    path = tmp_path / "predictions.csv"
    _upcoming().to_csv(path, index=False)

    assert prediction_file_uses_market_as_model(path) is False
