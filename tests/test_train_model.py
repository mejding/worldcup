import pandas as pd
import pytest

from features import build_training_dataset
from historical_data import standardize_historical_results
from train_model import train_historical_model


def _synthetic_training():
    rows = []
    teams = ["A", "B", "C", "D", "E", "F"]
    results = [(2, 0), (1, 1), (0, 2)]
    for i in range(45):
        home = teams[i % len(teams)]
        away = teams[(i + 1) % len(teams)]
        hs, away_score = results[i % 3]
        rows.append(
            {
                "date": f"2020-01-{(i % 28) + 1:02d}",
                "home_team": home,
                "away_team": away,
                "home_score": hs,
                "away_score": away_score,
                "tournament": "Friendly" if i % 2 else "World Cup qualification",
                "neutral": False,
            }
        )
    return build_training_dataset(standardize_historical_results(pd.DataFrame(rows)))


def test_training_function_returns_metrics_and_saves_files(tmp_path, monkeypatch):
    monkeypatch.setattr("train_model.MODEL_METADATA_PATH", tmp_path / "model_metadata.json")
    monkeypatch.setattr("train_model.FEATURE_COLUMNS_PATH", tmp_path / "feature_columns.json")
    model_path = tmp_path / "model.pkl"

    metadata = train_historical_model(_synthetic_training(), model_output_path=model_path)

    assert "metrics" in metadata
    assert model_path.exists()
    assert (tmp_path / "model_metadata.json").exists()


def test_training_handles_too_little_data_gracefully(tmp_path):
    with pytest.raises(ValueError, match="Too little historical data"):
        train_historical_model(_synthetic_training().head(10), model_output_path=tmp_path / "model.pkl")

