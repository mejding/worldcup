import pandas as pd

from features import build_training_dataset
from fifa_rankings import load_fifa_rankings
from historical_data import standardize_historical_results
from train_model import train_historical_model


def _history():
    rows = []
    results = [(1, 0), (1, 1), (0, 2)]
    teams = ["A", "B", "C"]
    for index in range(45):
        home_score, away_score = results[index % 3]
        rows.append(
            {
                "date": f"2020-01-{(index % 28) + 1:02d}",
                "home_team": teams[index % 3],
                "away_team": teams[(index + 1) % 3],
                "home_score": home_score,
                "away_score": away_score,
                "tournament": "Friendly",
                "neutral": True,
            }
        )
    return standardize_historical_results(pd.DataFrame(rows))


def test_metadata_records_fifa_fields(tmp_path, monkeypatch):
    monkeypatch.setattr("train_model.MODEL_METADATA_PATH", tmp_path / "metadata.json")
    monkeypatch.setattr("train_model.FEATURE_COLUMNS_PATH", tmp_path / "features.json")
    rankings_path = tmp_path / "rankings.csv"
    rankings_path.write_text(
        "ranking_date,team,fifa_rank,fifa_points\n"
        "2019-01-01,A,1,1800\n"
        "2019-01-01,B,2,1700\n"
        "2019-01-01,C,3,1600\n"
    )
    rankings, _ = load_fifa_rankings(rankings_path)
    training = build_training_dataset(_history(), output_path=tmp_path / "training.csv", include_fifa_ranking_features=True, fifa_rankings_df=rankings)

    metadata = train_historical_model(
        training,
        model_output_path=tmp_path / "model.pkl",
        include_fifa_ranking_features=True,
        model_variant="elo_plus_fifa",
        selected_reason="test selection",
        fifa_ranking_metadata={"fifa_ranking_rows": len(rankings), "fifa_rank_missing_rate": 0.0},
    )

    assert metadata["includes_fifa_ranking_features"] is True
    assert metadata["model_variant"] == "elo_plus_fifa"
    assert metadata["selected_reason"] == "test selection"
    assert metadata["fifa_ranking_rows"] == 3
