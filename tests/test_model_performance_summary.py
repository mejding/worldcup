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


def test_production_model_with_holdout_metrics_is_good_baseline():
    performance = load_model_performance_summary(
        metadata={},
        readiness=_readiness(),
        model_status=_model_status(),
        backtest_status={"backtest_exists": False, "prediction_count": 0},
        comparison_df=pd.DataFrame(),
    )
    quality = build_model_quality_summary({}, performance, pd.DataFrame())

    assert quality["quality_label"] == "Good baseline model"
    assert "production-ready" in quality["user_conclusion"]


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

    assert quality["quality_label"] == "Demo model"
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

    assert quality["quality_label"] == "Model unavailable"
    assert "market odds as fallback" in quality["headline"]
