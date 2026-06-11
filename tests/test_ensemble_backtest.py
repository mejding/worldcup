import pandas as pd

from ensemble_backtest import (
    compare_market_model_ensemble,
    evaluate_probability_source,
    run_ensemble_backtest_from_saved_predictions,
    select_best_probability_source,
)


def _predictions():
    return pd.DataFrame(
        {
            "actual_result": ["H", "D", "A", "H"],
            "market_home_prob": [0.7, 0.2, 0.1, 0.6],
            "market_draw_prob": [0.2, 0.6, 0.2, 0.2],
            "market_away_prob": [0.1, 0.2, 0.7, 0.2],
            "model_home_prob": [0.6, 0.3, 0.2, 0.5],
            "model_draw_prob": [0.2, 0.5, 0.2, 0.3],
            "model_away_prob": [0.2, 0.2, 0.6, 0.2],
        }
    )


def test_evaluate_probability_source_returns_expected_metrics():
    result = evaluate_probability_source(_predictions(), "market", "market_home_prob", "market_draw_prob", "market_away_prob")

    assert result["match_count"] == 4
    assert "log_loss" in result


def test_compare_market_model_ensemble_tests_all_weights():
    result = compare_market_model_ensemble(_predictions())

    assert result["source_name"].str.startswith("ensemble").sum() == 11


def test_select_best_probability_source_prefers_lower_log_loss():
    comparison = pd.DataFrame(
        {
            "source_name": ["market", "ensemble_0.8_0.2"],
            "match_count": [1000, 1000],
            "log_loss": [0.9, 0.8],
            "brier_score": [0.5, 0.5],
            "ece": [0.1, 0.1],
            "draw_calibration_gap": [0.1, 0.1],
            "w_market": [1.0, 0.8],
            "w_model": [0.0, 0.2],
        }
    )

    assert select_best_probability_source(comparison)["recommended_source"] == "ensemble"


def test_if_log_loss_tie_brier_score_used():
    comparison = pd.DataFrame(
        {
            "source_name": ["market", "historical_model"],
            "match_count": [1000, 1000],
            "log_loss": [0.8, 0.8],
            "brier_score": [0.4, 0.3],
            "ece": [0.1, 0.1],
            "draw_calibration_gap": [0.1, 0.1],
            "w_market": [1.0, 0.0],
            "w_model": [0.0, 1.0],
        }
    )

    assert select_best_probability_source(comparison)["recommended_source"] == "historical_model"


def test_market_can_be_recommended_if_similar():
    comparison = pd.DataFrame(
        {
            "source_name": ["market", "ensemble_0.8_0.2"],
            "match_count": [1000, 1000],
            "log_loss": [0.8, 0.7995],
            "brier_score": [0.3, 0.3],
            "ece": [0.1, 0.1],
            "draw_calibration_gap": [0.1, 0.1],
            "w_market": [1.0, 0.8],
            "w_model": [0.0, 0.2],
        }
    )

    assert select_best_probability_source(comparison)["recommended_source"] == "market"


def test_missing_historical_market_probabilities_handled_gracefully(tmp_path):
    path = tmp_path / "predictions.csv"
    _predictions().drop(columns=["market_home_prob"]).to_csv(path, index=False)
    result = run_ensemble_backtest_from_saved_predictions(path, tmp_path)

    assert result["status"] == "market_probabilities_missing"
