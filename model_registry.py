import json
from pathlib import Path

import pandas as pd

from backtest_paths import BACKTEST_BY_SEGMENT_PATH, BACKTEST_PREDICTIONS_PATH, BACKTEST_SUMMARY_PATH
from backtest_paths import DRAW_FEATURE_COMPARISON_PATH
from backtest_paths import ENSEMBLE_COMPARISON_PATH
from config import FEATURE_COLUMNS_PATH, HISTORICAL_RESULTS_PATH, MODEL_METADATA_PATH, MODEL_PATH
from draw_hypothesis import recommend_draw_context_usage
from ensemble_backtest import select_best_probability_source


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


def get_active_model_status() -> dict:
    metadata = load_model_metadata()
    metrics = metadata.get("metrics", {})
    artifacts_ready = model_artifacts_exist()
    return {
        "model_exists": model_exists(),
        "artifacts_ready": artifacts_ready,
        "trained_at": metadata.get("trained_at"),
        "model_version": metadata.get("model_version", metadata.get("trained_at")),
        "number_of_training_rows": metadata.get("number_of_training_rows", 0),
        "number_of_test_rows": metadata.get("number_of_test_rows", 0),
        "accuracy": metrics.get("accuracy"),
        "log_loss": metrics.get("log_loss"),
        "brier_score": metrics.get("brier_score"),
        "draw_rate_actual": metrics.get("draw_rate_actual"),
        "draw_rate_predicted": metrics.get("draw_rate_predicted"),
        "include_draw_context_features": metadata.get("include_draw_context_features", False),
    }


def get_model_readiness(
    model_path: Path = None,
    metadata_path: Path = None,
    feature_columns_path: Path = None,
    historical_path: Path = None,
    predictions_exist: bool = True,
) -> dict:
    metadata = load_model_metadata(metadata_path)
    artifacts_ready = model_artifacts_exist(model_path, metadata_path, feature_columns_path)
    historical_exists = Path(historical_path or HISTORICAL_RESULTS_PATH).exists()

    if artifacts_ready and predictions_exist:
        user_status = "Pre-trained model loaded."
    elif artifacts_ready:
        user_status = "Model loaded. Predictions are being generated for upcoming matches."
    else:
        user_status = "Pre-trained model unavailable. The app is using market probabilities as fallback."

    return {
        "artifacts_ready": artifacts_ready,
        "historical_csv_exists": historical_exists,
        "retraining_available": historical_exists,
        "normal_user_message": user_status,
        "fallback_to_market": not artifacts_ready,
        "admin_training_message": (
            "Historical training data is available. Retraining can be run from developer tools."
            if historical_exists
            else "Historical training data is not available in this deployment. Retraining is disabled."
        ),
        "trained_at": metadata.get("trained_at"),
        "model_version": metadata.get("model_version", metadata.get("trained_at")),
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
