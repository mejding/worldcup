import json

from model_registry import get_model_readiness


def _write_model_artifacts(tmp_path):
    model_path = tmp_path / "model.pkl"
    metadata_path = tmp_path / "model_metadata.json"
    feature_columns_path = tmp_path / "feature_columns.json"
    model_path.write_bytes(b"model")
    metadata_path.write_text(
        json.dumps(
            {
                "trained_at": "2026-06-01T12:00:00+00:00",
                "model_version": "test-model",
                "metrics": {"accuracy": 0.52},
            }
        )
    )
    feature_columns_path.write_text(json.dumps(["elo_diff"]))
    return model_path, metadata_path, feature_columns_path


def test_model_artifacts_ready_without_historical_csv_has_no_user_csv_warning(tmp_path):
    model_path, metadata_path, feature_columns_path = _write_model_artifacts(tmp_path)
    readiness = get_model_readiness(
        model_path=model_path,
        metadata_path=metadata_path,
        feature_columns_path=feature_columns_path,
        historical_path=tmp_path / "missing.csv",
    )

    assert readiness["artifacts_ready"] is True
    assert readiness["historical_csv_exists"] is False
    assert readiness["normal_user_message"] == "Pre-trained model loaded."
    assert "international_results.csv" not in readiness["normal_user_message"]
    assert readiness["admin_training_message"] == (
        "Historical training data is not available in this deployment. Retraining is disabled."
    )


def test_missing_artifacts_and_missing_historical_csv_falls_back_to_market(tmp_path):
    readiness = get_model_readiness(
        model_path=tmp_path / "missing_model.pkl",
        metadata_path=tmp_path / "missing_metadata.json",
        feature_columns_path=tmp_path / "missing_features.json",
        historical_path=tmp_path / "missing_history.csv",
    )

    assert readiness["artifacts_ready"] is False
    assert readiness["fallback_to_market"] is True
    assert readiness["normal_user_message"] == (
        "Pre-trained model unavailable. The app is using market probabilities as fallback."
    )
    assert readiness["retraining_available"] is False


def test_missing_artifacts_with_historical_csv_keeps_retraining_admin_only(tmp_path):
    historical_path = tmp_path / "international_results.csv"
    historical_path.write_text("date,home_team,away_team,home_score,away_score\n")

    readiness = get_model_readiness(
        model_path=tmp_path / "missing_model.pkl",
        metadata_path=tmp_path / "missing_metadata.json",
        feature_columns_path=tmp_path / "missing_features.json",
        historical_path=historical_path,
    )

    assert readiness["artifacts_ready"] is False
    assert readiness["historical_csv_exists"] is True
    assert readiness["retraining_available"] is True
    assert readiness["normal_user_message"] == (
        "Pre-trained model unavailable. The app is using market probabilities as fallback."
    )
    assert readiness["admin_training_message"] == (
        "Historical training data is available. Retraining can be run from developer tools."
    )


def test_model_artifacts_ready_with_predictions_missing_reports_generation(tmp_path):
    model_path, metadata_path, feature_columns_path = _write_model_artifacts(tmp_path)
    readiness = get_model_readiness(
        model_path=model_path,
        metadata_path=metadata_path,
        feature_columns_path=feature_columns_path,
        historical_path=tmp_path / "missing.csv",
        predictions_exist=False,
    )

    assert readiness["artifacts_ready"] is True
    assert readiness["normal_user_message"] == "Model loaded. Predictions are being generated for upcoming matches."
