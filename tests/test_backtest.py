import pandas as pd

from backtest import PREDICTION_COLUMNS, run_walk_forward_backtest, run_world_cup_backtest
from historical_data import standardize_historical_results


def _synthetic_historical(rows: int = 180):
    teams = ["A", "B", "C", "D", "E", "F", "G", "H"]
    scores = [(2, 0), (1, 1), (0, 2)]
    data = []
    for i in range(rows):
        year = 2010 + i // 24
        tournament = "FIFA World Cup" if year in {2014, 2018, 2022} and i % 24 < 4 else "Friendly"
        home_score, away_score = scores[i % 3]
        data.append(
            {
                "date": pd.Timestamp("2010-01-01") + pd.Timedelta(days=i * 20),
                "home_team": teams[i % len(teams)],
                "away_team": teams[(i + 3) % len(teams)],
                "home_score": home_score,
                "away_score": away_score,
                "tournament": tournament,
                "neutral": bool(i % 2),
            }
        )
    return standardize_historical_results(pd.DataFrame(data))


def test_walk_forward_backtest_creates_folds(tmp_path):
    result = run_walk_forward_backtest(
        _synthetic_historical(),
        initial_train_end_date="2012-01-01",
        test_window="365D",
        step_size="365D",
        min_train_matches=30,
        output_dir=tmp_path,
    )

    assert len(result["summary"]) > 0
    assert len(result["predictions"]) > 0


def test_no_leakage_train_end_before_test_start(tmp_path):
    result = run_walk_forward_backtest(
        _synthetic_historical(),
        initial_train_end_date="2012-01-01",
        min_train_matches=30,
        output_dir=tmp_path,
    )
    summary = result["summary"]

    assert (pd.to_datetime(summary["train_end_date"]) < pd.to_datetime(summary["test_start_date"])).all()


def test_prediction_output_contains_required_columns(tmp_path):
    result = run_walk_forward_backtest(_synthetic_historical(), "2012-01-01", min_train_matches=30, output_dir=tmp_path)

    assert set(PREDICTION_COLUMNS).issubset(result["predictions"].columns)


def test_summary_output_contains_required_metrics(tmp_path):
    result = run_walk_forward_backtest(_synthetic_historical(), "2012-01-01", min_train_matches=30, output_dir=tmp_path)

    assert set(["accuracy", "log_loss", "brier_score", "ece"]).issubset(result["summary"].columns)


def test_too_little_data_handled_gracefully(tmp_path):
    result = run_walk_forward_backtest(_synthetic_historical(20), "2012-01-01", min_train_matches=1000, output_dir=tmp_path)

    assert result["predictions"].empty


def test_world_cup_backtest_handles_missing_world_cup_rows_gracefully(tmp_path):
    historical = _synthetic_historical(80)
    historical["tournament"] = "Friendly"
    result = run_world_cup_backtest(historical, world_cup_years=[2014], output_dir=tmp_path)

    assert result["predictions"].empty
    assert not result["summary"].empty
