import pandas as pd

from model_performance_summary import build_model_quality_summary, load_model_performance_summary


def _readiness(status="production_ready"):
    return {
        "status": status,
        "is_usable_as_best_available": status == "production_ready",
        "metadata": {"model_version": "v1"},
    }


def _model_status():
    return {
        "model_exists": True,
        "accuracy": 0.594,
        "log_loss": 0.876,
        "brier_score": 0.516,
        "ece": None,
        "number_of_test_rows": 9882,
    }


def test_production_model_with_holdout_metrics_is_medium_if_full_validation_missing():
    performance = load_model_performance_summary(
        metadata={},
        readiness=_readiness(),
        model_status=_model_status(),
        backtest_status={"backtest_exists": False, "prediction_count": 0},
        comparison_df=pd.DataFrame(),
    )
    quality = build_model_quality_summary({}, performance, pd.DataFrame())

    assert quality["quality_label"] == "Model confidence: Medium"
    assert performance["model_confidence"] == "Medium"
    assert performance["betting_use"] == "Use cautiously"
    assert "conservative" in quality["user_conclusion"]


def test_missing_calibration_is_explicit_not_dash():
    performance = load_model_performance_summary(
        metadata={},
        readiness=_readiness(),
        model_status=_model_status(),
        backtest_status={"backtest_exists": False, "prediction_count": 0},
        comparison_df=pd.DataFrame(),
    )

    assert performance["ece"] is None
    assert "Calibration not calculated" in performance["missing_items"]


def test_missing_market_comparison_is_reported():
    performance = load_model_performance_summary(
        metadata={},
        readiness=_readiness(),
        model_status=_model_status(),
        backtest_status={"backtest_exists": False, "prediction_count": 0},
        comparison_df=pd.DataFrame(),
    )

    assert performance["market_comparison_available"] is False
    assert "Market comparison not available" in performance["missing_items"]


def test_full_validation_with_market_comparison_is_high_confidence():
    comparison = pd.DataFrame(
        {
            "source": ["market", "ensemble_0.8_0.2"],
            "source_name": ["market", "ensemble_0.8_0.2"],
            "match_count": [1000, 1000],
            "accuracy": [0.55, 0.57],
            "log_loss": [0.92, 0.90],
            "brier_score": [0.54, 0.52],
            "ece": [0.04, 0.02],
            "draw_calibration_gap": [0.03, 0.01],
        }
    )
    performance = load_model_performance_summary(
        metadata={},
        readiness=_readiness(),
        model_status=_model_status(),
        backtest_status={
            "backtest_exists": True,
            "prediction_count": 1000,
            "overall_accuracy": 0.57,
            "overall_log_loss": 0.90,
            "overall_brier_score": 0.52,
            "overall_ece": 0.02,
        },
        comparison_df=comparison,
    )
    quality = build_model_quality_summary({}, performance, comparison)

    assert performance["model_confidence"] == "High"
    assert performance["betting_use"] == "Ready"
    assert quality["quality_label"] == "Model confidence: High"


def test_full_backtest_takes_priority_over_metadata():
    performance = load_model_performance_summary(
        metadata={},
        readiness=_readiness(),
        model_status=_model_status(),
        backtest_status={
            "backtest_exists": True,
            "prediction_count": 100,
            "overall_accuracy": 0.51,
            "overall_log_loss": 0.99,
            "overall_brier_score": 0.58,
            "overall_ece": 0.04,
        },
        comparison_df=pd.DataFrame(),
    )

    assert performance["metrics_source"] == "full_backtest"
    assert performance["accuracy"] == 0.51


def test_demo_model_produces_demo_warning():
    performance = load_model_performance_summary(
        metadata={},
        readiness=_readiness("demo_model"),
        model_status=_model_status(),
        backtest_status={"backtest_exists": False, "prediction_count": 0},
        comparison_df=pd.DataFrame(),
    )
    quality = build_model_quality_summary({}, performance, pd.DataFrame())

    assert quality["quality_label"] == "Model confidence: Low"
    assert "market odds as fallback" in quality["summary_text"]


def test_no_model_produces_fallback_message():
    performance = load_model_performance_summary(
        metadata={},
        readiness=_readiness("missing"),
        model_status={"model_exists": False},
        backtest_status={"backtest_exists": False, "prediction_count": 0},
        comparison_df=pd.DataFrame(),
    )
    quality = build_model_quality_summary({}, performance, pd.DataFrame())

    assert quality["quality_label"] == "Model confidence: Low"
    assert "market odds as fallback" in quality["headline"]
