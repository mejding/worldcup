import json
from pathlib import Path

import pandas as pd

from backtest_paths import BACKTEST_BY_SEGMENT_PATH, BACKTEST_PREDICTIONS_PATH, BACKTEST_SUMMARY_PATH
from backtest_paths import DRAW_FEATURE_COMPARISON_PATH
from config import MODEL_METADATA_PATH, MODEL_PATH
from draw_hypothesis import recommend_draw_context_usage


def model_exists(model_path: Path = MODEL_PATH) -> bool:
    return Path(model_path).exists()


def load_model_metadata(path: Path = MODEL_METADATA_PATH) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def get_active_model_status() -> dict:
    metadata = load_model_metadata()
    metrics = metadata.get("metrics", {})
    return {
        "model_exists": model_exists(),
        "trained_at": metadata.get("trained_at"),
        "number_of_training_rows": metadata.get("number_of_training_rows", 0),
        "number_of_test_rows": metadata.get("number_of_test_rows", 0),
        "accuracy": metrics.get("accuracy"),
        "log_loss": metrics.get("log_loss"),
        "brier_score": metrics.get("brier_score"),
        "draw_rate_actual": metrics.get("draw_rate_actual"),
        "draw_rate_predicted": metrics.get("draw_rate_predicted"),
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
