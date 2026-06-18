import json
from pathlib import Path

import pandas as pd

from backtest_paths import BACKTEST_BY_SEGMENT_PATH, BACKTEST_PREDICTIONS_PATH, BACKTEST_SUMMARY_PATH
from backtest_paths import DRAW_FEATURE_COMPARISON_PATH
from backtest_paths import ENSEMBLE_COMPARISON_PATH
from config import FEATURE_COLUMNS_PATH, HISTORICAL_RESULTS_PATH, MODEL_METADATA_PATH, MODEL_PATH
from draw_hypothesis import recommend_draw_context_usage
from ensemble_backtest import select_best_probability_source
from model_readiness import validate_model_artifact


def model_exists(model_path: Path = None) -> bool:
    return Path(model_path or MODEL_PATH).exists()


def model_artifacts_exist(
    model_path: Path = None,
    metadata_path: Path = None,
    feature_columns_path: Path = None,
) -> bool:
    return all(
        Path(path).exists() and Path(path).stat().st_size > 0
        for path in [
            model_path or MODEL_PATH,
            metadata_path or MODEL_METADATA_PATH,
            feature_columns_path or FEATURE_COLUMNS_PATH,
        ]
    )


def load_model_metadata(path: Path = None) -> dict:
    path = Path(path or MODEL_METADATA_PATH)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _load_feature_columns(path: Path = None) -> list:
    path = Path(path or FEATURE_COLUMNS_PATH)
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        columns = json.loads(path.read_text())
        return columns if isinstance(columns, list) else []
    except Exception:
        return []


def _metadata_with_artifact_flags(
    model_path: Path = None,
    metadata_path: Path = None,
    feature_columns_path: Path = None,
) -> dict:
    model_path = Path(model_path or MODEL_PATH)
    metadata_path = Path(metadata_path or MODEL_METADATA_PATH)
    feature_columns_path = Path(feature_columns_path or FEATURE_COLUMNS_PATH)
    metadata = load_model_metadata(metadata_path)
    feature_columns = _load_feature_columns(feature_columns_path)
    if feature_columns and "feature_columns" not in metadata:
        metadata["feature_columns"] = feature_columns
    metadata["_model_file_exists"] = model_path.exists() and model_path.stat().st_size > 0
    metadata["_metadata_exists"] = metadata_path.exists() and metadata_path.stat().st_size > 0
    metadata["_feature_columns_exists"] = feature_columns_path.exists() and feature_columns_path.stat().st_size > 0
    return metadata


def get_active_model_status() -> dict:
    metadata = _metadata_with_artifact_flags()
    validation = validate_model_artifact(metadata)
    artifacts_ready = model_artifacts_exist()
    return {
        "model_exists": model_exists(),
        "artifacts_ready": artifacts_ready,
        "readiness_status": validation["status"],
        "is_usable_for_predictions": validation["is_usable_for_predictions"],
        "is_usable_as_best_available": validation["is_usable_as_best_available"],
        "readiness_warnings": validation["warnings"],
        "trained_at": validation["trained_at"],
        "model_version": validation["model_version"],
        "number_of_training_rows": validation["training_rows"],
        "number_of_test_rows": validation["test_rows"],
        "feature_count": validation["feature_count"],
        "training_data_source": validation["training_data_source"],
        "accuracy": validation["performance_accuracy"],
        "log_loss": validation["performance_log_loss"],
        "brier_score": validation["performance_brier_score"],
        "ece": validation["performance_ece"],
        "draw_rate_actual": metadata.get("metrics", {}).get("draw_rate_actual"),
        "draw_rate_predicted": metadata.get("metrics", {}).get("draw_rate_predicted"),
        "include_draw_context_features": metadata.get("include_draw_context_features", False),
    }


def get_model_readiness(
    model_path: Path = None,
    metadata_path: Path = None,
    feature_columns_path: Path = None,
    historical_path: Path = None,
    predictions_exist: bool = True,
) -> dict:
    metadata = _metadata_with_artifact_flags(model_path, metadata_path, feature_columns_path)
    validation = validate_model_artifact(metadata)
    artifacts_ready = model_artifacts_exist(model_path, metadata_path, feature_columns_path)
    historical_exists = Path(historical_path or HISTORICAL_RESULTS_PATH).exists()

    if validation["status"] == "production_ready" and predictions_exist:
        user_status = "Pre-trained model loaded."
    elif validation["status"] == "production_ready":
        user_status = "Model loaded. Predictions are being generated for upcoming matches."
    elif validation["status"] == "demo_model":
        user_status = "Predictions are based on market odds because the available model is only a demo model."
    else:
        user_status = "Pre-trained model unavailable. The app is using market probabilities as fallback."

    return {
        "artifacts_ready": artifacts_ready,
        "model_file_exists": bool(metadata.get("_model_file_exists")),
        "metadata_exists": bool(metadata.get("_metadata_exists")),
        "feature_columns_exists": bool(metadata.get("_feature_columns_exists")),
        "status": validation["status"],
        "is_usable_for_predictions": validation["is_usable_for_predictions"],
        "is_usable_as_best_available": validation["is_usable_as_best_available"],
        "warnings": validation["warnings"],
        "historical_csv_exists": historical_exists,
        "retraining_available": historical_exists,
        "normal_user_message": user_status,
        "fallback_to_market": not validation["is_usable_as_best_available"],
        "admin_training_message": (
            "Historical training data is available. Retraining can be run from developer tools."
            if historical_exists
            else "Historical training data is not available in this deployment. Retraining is disabled."
        ),
        "trained_at": validation["trained_at"],
        "model_version": validation["model_version"],
        "training_rows": validation["training_rows"],
        "test_rows": validation["test_rows"],
        "feature_count": validation["feature_count"],
        "training_data_source": validation["training_data_source"],
        "training_data_start_date": validation["training_data_start_date"],
        "training_data_end_date": validation["training_data_end_date"],
        "training_year_span": validation["training_year_span"],
        "includes_elo_features": validation["includes_elo_features"],
        "includes_form_features": validation["includes_form_features"],
        "includes_fifa_ranking_features": validation["includes_fifa_ranking_features"],
        "includes_tournament_features": validation["includes_tournament_features"],
        "includes_neutral_venue": validation["includes_neutral_venue"],
        "includes_qualifiers": validation["includes_qualifiers"],
        "includes_world_cup_or_major_tournaments": validation["includes_world_cup_or_major_tournaments"],
        "metadata": metadata,
    }


def get_latest_backtest_status() -> dict:
    predictions_path = Path(BACKTEST_PREDICTIONS_PATH)
    summary_path = Path(BACKTEST_SUMMARY_PATH)
    segment_path = Path(BACKTEST_BY_SEGMENT_PATH)
    status = {
        "backtest_exists": predictions_path.exists(),
        "last_modified": None,
        "prediction_count": 0,
        "summary_exists": summary_path.exists(),
        "overall_accuracy": None,
        "overall_log_loss": None,
        "overall_brier_score": None,
        "overall_ece": None,
        "draw_calibration_gap": None,
    }
    if predictions_path.exists():
        status["last_modified"] = predictions_path.stat().st_mtime
        try:
            status["prediction_count"] = len(pd.read_csv(predictions_path))
        except Exception:
            status["prediction_count"] = 0
    if segment_path.exists():
        try:
            segments = pd.read_csv(segment_path)
            overall = segments[
                (segments["segment_name"] == "Overall") & (segments["segment_value"] == "All")
            ].head(1)
            if not overall.empty:
                row = overall.iloc[0]
                status["overall_accuracy"] = row.get("accuracy")
                status["overall_log_loss"] = row.get("log_loss")
                status["overall_brier_score"] = row.get("brier_score")
                status["overall_ece"] = row.get("ece")
                status["draw_calibration_gap"] = row.get("draw_calibration_gap")
        except Exception:
            pass
    return status


def get_latest_draw_context_status() -> dict:
    path = Path(DRAW_FEATURE_COMPARISON_PATH)
    status = {
        "comparison_exists": path.exists(),
        "last_modified": None,
        "recommended": False,
        "reason": "No comparison results available.",
        "caveats": [],
    }
    if not path.exists() or path.stat().st_size == 0:
        return status
    status["last_modified"] = path.stat().st_mtime
    try:
        comparison = pd.read_csv(path)
        recommendation = recommend_draw_context_usage(comparison)
        status.update(recommendation)
    except Exception as exc:
        status["reason"] = f"Could not read comparison results: {exc}"
    return status


def get_latest_ensemble_status() -> dict:
    path = Path(ENSEMBLE_COMPARISON_PATH)
    status = {
        "comparison_exists": path.exists(),
        "last_modified": None,
        "recommended_source": "market",
        "w_market": 1.0,
        "w_model": 0.0,
        "reason": "No ensemble comparison results available.",
        "caveats": [],
    }
    if not path.exists() or path.stat().st_size == 0:
        return status
    status["last_modified"] = path.stat().st_mtime
    try:
        recommendation = select_best_probability_source(pd.read_csv(path))
        status.update(recommendation)
    except Exception as exc:
        status["reason"] = f"Could not read ensemble comparison: {exc}"
    return status
