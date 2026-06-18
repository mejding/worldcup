import json

import pandas as pd

from model_performance_summary import build_model_quality_summary, load_model_performance_summary
from validation import load_best_prediction_source_validation, save_best_prediction_source_validation


def _readiness():
    return {
        "status": "production_ready",
        "is_usable_as_best_available": True,
        "training_rows": 2000,
        "metadata": {"model_version": "v1"},
    }


def _model_status():
    return {
        "model_exists": True,
        "accuracy": 0.55,
        "log_loss": 0.94,
        "brier_score": 0.54,
        "ece": 0.03,
        "number_of_test_rows": 500,
    }


def test_full_validation_complete_status():
    comparison = pd.DataFrame(
        {
            "source": ["market", "model"],
            "source_name": ["market", "model"],
            "match_count": [1000, 1000],
            "accuracy": [0.55, 0.57],
            "log_loss": [0.93, 0.91],
            "brier_score": [0.54, 0.52],
            "ece": [0.04, 0.02],
            "draw_calibration_gap": [0.02, 0.01],
        }
    )

    summary = load_model_performance_summary(
        readiness=_readiness(),
        model_status=_model_status(),
        backtest_status={"backtest_exists": True, "prediction_count": 1000, "overall_accuracy": 0.57, "overall_log_loss": 0.91, "overall_brier_score": 0.52, "overall_ece": 0.02},
        comparison_df=comparison,
        metadata={},
    )

    assert summary["model_confidence"] == "High"
    assert summary["betting_use"] == "Ready"


def test_partial_validation_if_market_comparison_missing():
    summary = load_model_performance_summary(
        readiness=_readiness(),
        model_status=_model_status(),
        backtest_status={"backtest_exists": True, "prediction_count": 1000, "overall_accuracy": 0.57, "overall_log_loss": 0.91, "overall_brier_score": 0.52, "overall_ece": 0.02},
        comparison_df=pd.DataFrame(),
        metadata={},
    )
    quality = build_model_quality_summary({}, summary, pd.DataFrame())

    assert summary["model_confidence"] == "Medium"
    assert summary["market_comparison_available"] is False
    assert "cannot yet confirm" not in quality["headline"].lower()


def test_missing_validation_status():
    summary = load_model_performance_summary(
        readiness={"status": "missing", "metadata": {}},
        model_status={"model_exists": False},
        backtest_status={"backtest_exists": False, "prediction_count": 0},
        comparison_df=pd.DataFrame(),
        metadata={},
    )

    assert summary["model_confidence"] == "Low"
    assert summary["betting_use"] == "Market fallback"


def test_best_source_selection_file_is_read_correctly(tmp_path):
    path = tmp_path / "best_prediction_source_validation.json"
    saved = save_best_prediction_source_validation(
        {
            "selected_source": "ensemble_0.8_0.2",
            "selected_label": "Ensemble",
            "w_market": 0.8,
            "w_model": 0.2,
            "reason": "test",
            "caveats": [],
            "market_comparison_available": True,
        },
        output_path=path,
    )

    loaded = load_best_prediction_source_validation(path)

    assert loaded["selected_source"] == saved["selected_source"]
    assert json.loads(path.read_text())["selected_label"] == "Ensemble"
