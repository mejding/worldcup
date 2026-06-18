from __future__ import annotations

from pathlib import Path

import pandas as pd

from backtest_paths import BACKTEST_BY_SEGMENT_PATH, BACKTEST_PREDICTIONS_PATH, ENSEMBLE_COMPARISON_PATH
import config as app_config
from model_registry import get_active_model_status, get_latest_backtest_status, get_model_readiness, load_model_metadata


PROJECT_ROOT = getattr(app_config, "PROJECT_ROOT", Path(__file__).resolve().parent)
DATA_DIR = getattr(app_config, "DATA_DIR", PROJECT_ROOT / "data")
MODELS_DIR = getattr(app_config, "MODELS_DIR", DATA_DIR / "models")
MODEL_METADATA_PATH = getattr(app_config, "MODEL_METADATA_PATH", MODELS_DIR / "model_metadata.json")


def _is_missing(value) -> bool:
    return value is None or pd.isna(value)


def _clean_float(value):
    if _is_missing(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def metric_status(metric_name: str, value) -> str:
    value = _clean_float(value)
    if value is None:
        return "Not calculated"
    if metric_name == "accuracy":
        if value >= 0.55:
            return "Strong"
        if value >= 0.50:
            return "Decent"
        return "Weak"
    if metric_name == "log_loss":
        if value < 0.95:
            return "Good"
        if value <= 1.05:
            return "Acceptable"
        return "Weak"
    if metric_name == "brier_score":
        if value < 0.55:
            return "Good"
        if value <= 0.65:
            return "Acceptable"
        return "Weak"
    if metric_name == "ece":
        if value < 0.03:
            return "Very good"
        if value <= 0.06:
            return "Acceptable"
        return "Needs improvement"
    return "Available"


def display_metric_value(metric_name: str, value) -> str:
    value = _clean_float(value)
    if value is None:
        return "Not calculated yet"
    if metric_name == "accuracy":
        return f"{value:.1%}"
    if metric_name == "match_count":
        return f"{int(value):,}"
    return f"{value:.3f}"


def metric_interpretation(metric_name: str, value) -> str:
    status = metric_status(metric_name, value)
    if metric_name == "accuracy":
        return {
            "Strong": "Strong for 1X2 football prediction.",
            "Decent": "Decent signal, but not enough on its own for betting.",
            "Weak": "Weak outcome-picking performance.",
        }.get(status, "How often the model picked the correct 1X2 outcome.")
    if metric_name == "log_loss":
        return {
            "Good": "Good probability quality. Lower is better.",
            "Acceptable": "Acceptable probability quality.",
            "Weak": "Weak probability quality.",
        }.get(status, "Lower is better. Punishes confident wrong predictions.")
    if metric_name == "brier_score":
        return {
            "Good": "Good average probability error. Lower is better.",
            "Acceptable": "Acceptable average probability error.",
            "Weak": "Large probability errors.",
        }.get(status, "Lower is better. Measures probability error.")
    if metric_name == "ece":
        if status == "Not calculated":
            return "Calibration should be calculated before relying heavily on value betting."
        return "Shows whether confidence matches reality."
    if metric_name == "match_count":
        return "Historical matches used for this validation."
    return ""


def metric_tooltip(metric_name: str) -> str:
    return {
        "accuracy": "How often the highest-probability 1X2 outcome was correct.",
        "log_loss": "Probability quality metric. Lower is better and confident wrong predictions are punished.",
        "brier_score": "Squared probability error across home/draw/away. Lower is better.",
        "ece": "Expected calibration error. Lower means predicted confidence better matches actual hit rate.",
        "match_count": "Number of evaluated historical matches.",
    }.get(metric_name, "")


def _comparison_rows(comparison_df: pd.DataFrame) -> pd.DataFrame:
    if comparison_df is None or comparison_df.empty:
        return pd.DataFrame()
    df = comparison_df.copy()
    if "source_name" not in df.columns:
        return pd.DataFrame()
    return df


def _best_source_from_comparison(comparison_df: pd.DataFrame) -> dict | None:
    df = _comparison_rows(comparison_df)
    if df.empty:
        return None
    required = ["log_loss", "brier_score", "ece"]
    for column in required:
        if column not in df.columns:
            return None
        df[column] = pd.to_numeric(df[column], errors="coerce")
    candidates = df.dropna(subset=["log_loss", "brier_score", "ece"])
    if candidates.empty:
        return None
    best = candidates.sort_values(["log_loss", "brier_score", "ece"]).iloc[0]
    source_name = str(best["source_name"])
    label = "Ensemble / Best available" if source_name.startswith("ensemble") else "Market odds" if source_name == "market" else "ML model"
    return {"source_name": source_name, "label": label, "row": best.to_dict()}


def load_model_performance_summary(
    metadata: dict | None = None,
    readiness: dict | None = None,
    model_status: dict | None = None,
    backtest_status: dict | None = None,
    comparison_df: pd.DataFrame | None = None,
    metadata_path: Path = MODEL_METADATA_PATH,
) -> dict:
    metadata = metadata if metadata is not None else load_model_metadata(metadata_path)
    model_status = model_status if model_status is not None else get_active_model_status()
    readiness = readiness if readiness is not None else get_model_readiness()
    backtest_status = backtest_status if backtest_status is not None else get_latest_backtest_status()
    if comparison_df is None:
        comparison_path = Path(ENSEMBLE_COMPARISON_PATH)
        comparison_df = pd.read_csv(comparison_path) if comparison_path.exists() and comparison_path.stat().st_size > 0 else pd.DataFrame()

    full_backtest_available = bool(backtest_status.get("backtest_exists")) and int(backtest_status.get("prediction_count") or 0) > 0
    if full_backtest_available and not _is_missing(backtest_status.get("overall_accuracy")):
        metrics_source = "full_backtest"
        accuracy = backtest_status.get("overall_accuracy")
        log_loss = backtest_status.get("overall_log_loss")
        brier_score = backtest_status.get("overall_brier_score")
        ece = backtest_status.get("overall_ece")
        match_count = backtest_status.get("prediction_count")
    elif model_status.get("accuracy") is not None:
        metrics_source = "holdout_metadata"
        accuracy = model_status.get("accuracy")
        log_loss = model_status.get("log_loss")
        brier_score = model_status.get("brier_score")
        ece = model_status.get("ece")
        match_count = model_status.get("number_of_test_rows")
    else:
        metrics_source = "missing"
        accuracy = log_loss = brier_score = ece = match_count = None

    comparison = _comparison_rows(comparison_df)
    market_comparison_available = not comparison.empty and comparison["source_name"].astype(str).eq("market").any()
    calibration_available = _clean_float(ece) is not None
    missing_items = []
    if metrics_source == "missing":
        missing_items.append("Holdout metrics are missing")
    if not full_backtest_available:
        missing_items.append("Full walk-forward backtest not run")
    if not market_comparison_available:
        missing_items.append("Market comparison not available")
    if not calibration_available:
        missing_items.append("Calibration not calculated")
    if comparison.empty:
        missing_items.append("Ensemble validation not available")

    return {
        "model_status": readiness.get("status", "missing"),
        "is_production_ready": readiness.get("status") == "production_ready",
        "metrics_source": metrics_source,
        "accuracy": _clean_float(accuracy),
        "log_loss": _clean_float(log_loss),
        "brier_score": _clean_float(brier_score),
        "ece": _clean_float(ece),
        "match_count": int(match_count) if _clean_float(match_count) is not None else None,
        "market_comparison_available": market_comparison_available,
        "full_backtest_available": full_backtest_available,
        "calibration_available": calibration_available,
        "comparison_df": comparison,
        "best_comparison_source": _best_source_from_comparison(comparison),
        "missing_items": list(dict.fromkeys(missing_items)),
        "metadata": metadata,
        "readiness": readiness,
        "model_artifacts_available": bool(model_status.get("model_exists")),
    }


def build_model_quality_summary(metadata, performance, comparison) -> dict:
    status = performance.get("model_status")
    comparison_available = performance.get("market_comparison_available", False)
    missing_items = performance.get("missing_items", [])
    match_count = performance.get("match_count")
    accuracy = performance.get("accuracy")
    best = performance.get("best_comparison_source")

    if status == "production_ready" and comparison_available and best:
        return {
            "quality_label": "Validated",
            "status_color": "green",
            "headline": f"Currently using: {best['label']}",
            "summary_text": "Model has been compared against market odds.",
            "user_conclusion": "The app uses the best validated prediction source for value calculations.",
            "missing_items": missing_items,
            "recommended_next_action": "Keep validation updated when new backtests or odds data are available.",
        }
    if status == "production_ready":
        return {
            "quality_label": "Good baseline model",
            "status_color": "amber",
            "headline": f"Tested on {match_count:,} historical matches" if match_count else "Holdout metrics available",
            "summary_text": f"Prediction accuracy: {accuracy:.1%}" if accuracy is not None else "Prediction metrics are available.",
            "user_conclusion": "The model is production-ready, but full market comparison is still missing.",
            "missing_items": missing_items,
            "recommended_next_action": "Run full validation in Advanced / Admin.",
        }
    if status == "demo_model":
        return {
            "quality_label": "Demo model",
            "status_color": "red",
            "headline": "Not enough historical validation data",
            "summary_text": "Predictions use market odds as fallback.",
            "user_conclusion": "The model is not suitable for real predictions yet.",
            "missing_items": missing_items or ["Production model is missing"],
            "recommended_next_action": "Train and export a production model from historical international data.",
        }
    return {
        "quality_label": "Model unavailable",
        "status_color": "gray",
        "headline": "Predictions use market odds as fallback",
        "summary_text": "No usable model artifact is available.",
        "user_conclusion": "The app can still show market-based probabilities, but ML predictions are unavailable.",
        "missing_items": missing_items or ["Pre-trained model artifact is missing"],
        "recommended_next_action": "Add or redeploy the bundled model artifacts.",
    }


def validation_checklist(summary: dict) -> list[dict]:
    readiness = summary.get("readiness", {})
    return [
        {"Item": "Pre-trained model loaded", "Status": "Complete" if summary.get("model_artifacts_available") else "Missing"},
        {"Item": "Holdout metrics available", "Status": "Complete" if summary.get("metrics_source") in {"holdout_metadata", "full_backtest"} else "Missing"},
        {"Item": "Full walk-forward backtest", "Status": "Complete" if summary.get("full_backtest_available") else "Recommended"},
        {"Item": "Market comparison", "Status": "Complete" if summary.get("market_comparison_available") else "Recommended"},
        {"Item": "Calibration", "Status": "Complete" if summary.get("calibration_available") else "Recommended"},
        {"Item": "Production readiness", "Status": "Complete" if readiness.get("status") == "production_ready" else "Missing"},
    ]
