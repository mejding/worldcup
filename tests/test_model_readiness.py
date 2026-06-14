import json

from model_readiness import is_production_performance_available, validate_model_artifact
from model_registry import get_model_readiness


def _metadata(training_rows=1200, test_rows=250, feature_count=15, is_demo_model=False):
    return {
        "model_version": "prod-v1",
        "trained_at_utc": "2026-06-01T12:00:00+00:00",
        "training_rows": training_rows,
        "test_rows": test_rows,
        "feature_count": feature_count,
        "training_data_source": "historical_international_results",
        "training_data_start_date": "2000-01-01",
        "training_data_end_date": "2025-12-31",
        "training_year_span": 25.0,
        "is_demo_model": is_demo_model,
        "includes_elo_features": True,
        "includes_form_features": True,
        "includes_tournament_features": True,
        "includes_neutral_venue": True,
        "includes_qualifiers": True,
        "includes_world_cup_or_major_tournaments": True,
        "includes_schedule_features": True,
        "performance_accuracy": 0.52,
        "performance_log_loss": 1.02,
        "performance_brier_score": 0.62,
        "performance_ece": 0.04,
    }


def _write_model_artifacts(tmp_path, metadata):
    model_path = tmp_path / "model.pkl"
    metadata_path = tmp_path / "model_metadata.json"
    feature_columns_path = tmp_path / "feature_columns.json"
    model_path.write_bytes(b"model")
    metadata_path.write_text(json.dumps(metadata))
    feature_columns_path.write_text(json.dumps([f"feature_{index}" for index in range(metadata.get("feature_count", 0))]))
    return model_path, metadata_path, feature_columns_path


def test_36_training_rows_and_9_test_rows_is_demo_model():
    result = validate_model_artifact(_metadata(training_rows=36, test_rows=9))

    assert result["status"] == "demo_model"
    assert result["is_usable_as_best_available"] is False
    assert any("sample/demo" in warning for warning in result["warnings"])


def test_demo_model_flag_is_not_usable_as_best_available():
    result = validate_model_artifact(_metadata(is_demo_model=True))

    assert result["status"] == "demo_model"
    assert result["is_usable_for_predictions"] is True
    assert result["is_usable_as_best_available"] is False


def test_production_model_with_sufficient_rows_is_ready():
    result = validate_model_artifact(_metadata())

    assert result["status"] == "production_ready"
    assert result["is_usable_as_best_available"] is True
    assert result["warnings"] == []


def test_model_without_qualifier_and_tournament_coverage_is_not_production_ready():
    metadata = _metadata()
    metadata["includes_qualifiers"] = False
    metadata["includes_world_cup_or_major_tournaments"] = False

    result = validate_model_artifact(metadata)

    assert result["status"] == "demo_model"
    assert result["is_usable_as_best_available"] is False
    assert any("qualifiers" in warning for warning in result["warnings"])


def test_model_with_short_training_period_is_not_production_ready():
    metadata = _metadata()
    metadata["training_data_start_date"] = "2024-01-01"
    metadata["training_data_end_date"] = "2025-01-01"
    metadata["training_year_span"] = 1.0

    result = validate_model_artifact(metadata)

    assert result["status"] == "demo_model"
    assert result["is_usable_as_best_available"] is False
    assert any("period is too short" in warning for warning in result["warnings"])


def test_demo_backtest_is_not_reliable_production_performance():
    readiness = validate_model_artifact(_metadata(training_rows=36, test_rows=9))
    backtest_status = {"backtest_exists": True, "prediction_count": 500}

    assert is_production_performance_available(readiness, backtest_status) is False


def test_production_backtest_is_reliable_performance():
    readiness = validate_model_artifact(_metadata())
    backtest_status = {"backtest_exists": True, "prediction_count": 500}

    assert is_production_performance_available(readiness, backtest_status) is True


def test_missing_metadata_is_invalid():
    result = validate_model_artifact({})

    assert result["status"] == "missing"
    assert result["is_usable_as_best_available"] is False


def test_missing_feature_columns_is_invalid():
    metadata = _metadata()
    metadata["_feature_columns_exists"] = False

    result = validate_model_artifact(metadata)

    assert result["status"] == "invalid"
    assert result["is_usable_as_best_available"] is False


def test_demo_model_in_live_mode_falls_back_to_market(tmp_path):
    model_path, metadata_path, feature_columns_path = _write_model_artifacts(
        tmp_path,
        _metadata(training_rows=36, test_rows=9),
    )

    readiness = get_model_readiness(
        model_path=model_path,
        metadata_path=metadata_path,
        feature_columns_path=feature_columns_path,
        historical_path=tmp_path / "missing_history.csv",
    )

    assert readiness["status"] == "demo_model"
    assert readiness["fallback_to_market"] is True
    assert readiness["normal_user_message"] == (
        "Predictions are based on market odds because the available model is only a demo model."
    )


def test_normal_ui_does_not_claim_demo_model_is_production_ready(tmp_path):
    model_path, metadata_path, feature_columns_path = _write_model_artifacts(
        tmp_path,
        _metadata(training_rows=36, test_rows=9),
    )

    readiness = get_model_readiness(
        model_path=model_path,
        metadata_path=metadata_path,
        feature_columns_path=feature_columns_path,
        historical_path=tmp_path / "missing_history.csv",
    )

    assert readiness["is_usable_as_best_available"] is False
    assert "Pre-trained model loaded" not in readiness["normal_user_message"]
