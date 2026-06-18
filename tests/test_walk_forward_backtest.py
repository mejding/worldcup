import pandas as pd

from historical_data import standardize_historical_results
from walk_forward_backtest import run_full_walk_forward_backtest


def _historical(rows: int = 120, include_market: bool = True):
    teams = ["A", "B", "C", "D", "E", "F", "G", "H"]
    scores = [(2, 0), (1, 1), (0, 2)]
    data = []
    for index in range(rows):
        home_score, away_score = scores[index % 3]
        row = {
            "date": pd.Timestamp("2010-01-01") + pd.Timedelta(days=index * 20),
            "home_team": teams[index % len(teams)],
            "away_team": teams[(index + 3) % len(teams)],
            "home_score": home_score,
            "away_score": away_score,
            "tournament": "Friendly" if index % 5 else "FIFA World Cup qualification",
            "neutral": bool(index % 2),
        }
        if include_market:
            row.update({"market_home_prob": 0.45, "market_draw_prob": 0.25, "market_away_prob": 0.30})
        data.append(row)
    return standardize_historical_results(pd.DataFrame(data))


def test_full_walk_forward_folds_are_chronological(tmp_path):
    result = run_full_walk_forward_backtest(_historical(), min_train_matches=30, output_dir=tmp_path)
    by_fold = result["by_fold"]

    assert not by_fold.empty
    assert (pd.to_datetime(by_fold["train_end_date"]) < pd.to_datetime(by_fold["test_start_date"])).all()


def test_training_data_never_includes_test_period(tmp_path):
    result = run_full_walk_forward_backtest(_historical(), min_train_matches=30, output_dir=tmp_path)
    predictions = result["predictions"]

    assert not predictions.empty
    assert (pd.to_datetime(predictions["train_end_date"]) < pd.to_datetime(predictions["test_start_date"])).all()


def test_minimum_training_rows_enforced(tmp_path):
    result = run_full_walk_forward_backtest(_historical(20), min_train_matches=1000, output_dir=tmp_path)

    assert result["status"] == "validation_error"
    assert "At least 1000 historical matches" in result["error"]


def test_predictions_and_metrics_are_saved(tmp_path):
    result = run_full_walk_forward_backtest(_historical(), min_train_matches=30, output_dir=tmp_path)

    assert result["paths"]["predictions"].exists()
    assert result["paths"]["summary"].exists()
    assert result["paths"]["by_fold"].exists()
    assert result["summary"].iloc[0]["match_count"] > 0
    assert pd.notna(result["summary"].iloc[0]["log_loss"])


def test_market_comparison_saved_when_historical_market_probs_exist(tmp_path):
    result = run_full_walk_forward_backtest(_historical(include_market=True), min_train_matches=30, output_dir=tmp_path)

    assert result["paths"]["market_comparison"].exists()
    assert not result["market_comparison"].empty
    assert "market" in set(result["market_comparison"]["source"])


def test_missing_market_probs_returns_clear_partial_status(tmp_path):
    result = run_full_walk_forward_backtest(_historical(include_market=False), min_train_matches=30, output_dir=tmp_path)

    assert result["market_comparison"].empty or "market" not in set(result["market_comparison"].get("source", []))
    assert any("historical market odds" in item for item in result["best_source_validation"]["caveats"])
