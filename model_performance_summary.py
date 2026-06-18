from __future__ import annotations

from pathlib import Path

import pandas as pd

import backtest_paths as backtest_path_config
import config as app_config
from model_registry import get_active_model_status, get_latest_backtest_status, get_model_readiness, load_model_metadata
from validation import load_best_prediction_source_validation


PROJECT_ROOT = getattr(app_config, "PROJECT_ROOT", Path(__file__).resolve().parent)
DATA_DIR = getattr(app_config, "DATA_DIR", PROJECT_ROOT / "data")
MODELS_DIR = getattr(app_config, "MODELS_DIR", DATA_DIR / "models")
PROCESSED_DATA_DIR = getattr(app_config, "PROCESSED_DATA_DIR", DATA_DIR / "processed")
MODEL_METADATA_PATH = getattr(app_config, "MODEL_METADATA_PATH", MODELS_DIR / "model_metadata.json")
ENSEMBLE_COMPARISON_PATH = getattr(backtest_path_config, "ENSEMBLE_COMPARISON_PATH", PROCESSED_DATA_DIR / "ensemble_comparison.csv")
FULL_BACKTEST_MARKET_COMPARISON_PATH = getattr(
    backtest_path_config,
    "FULL_BACKTEST_MARKET_COMPARISON_PATH",
    PROCESSED_DATA_DIR / "full_backtest_market_comparison.csv",
)
FULL_BACKTEST_PREDICTIONS_PATH = getattr(
    backtest_path_config,
    "FULL_BACKTEST_PREDICTIONS_PATH",
    PROCESSED_DATA_DIR / "full_backtest_predictions.csv",
)
FULL_BACKTEST_SUMMARY_PATH = getattr(
    backtest_path_config,
    "FULL_BACKTEST_SUMMARY_PATH",
    PROCESSED_DATA_DIR / "full_backtest_summary.csv",
)


def _is_missing(value) -> bool:
    return value is None or pd.isna(value)


def _clean_float(value):
    if _is_missing(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _max_numeric_column(df: pd.DataFrame, column: str) -> float:
    if df is None or df.empty or column not in df.columns:
        return 0.0
    values = pd.to_numeric(df[column], errors="coerce").fillna(0)
    return float(values.max()) if not values.empty else 0.0


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
        "accuracy": "How often the model picked the correct 1X2 outcome.",
        "log_loss": "Log loss. Lower is better and confident wrong predictions are punished.",
        "brier_score": "Brier score. Lower is better and measures average probability error.",
        "ece": "Calibration check: do 60% predictions actually win about 60% of the time?",
        "match_count": "Number of evaluated historical matches.",
    }.get(metric_name, "")


def _comparison_rows(comparison_df: pd.DataFrame) -> pd.DataFrame:
    if comparison_df is None or comparison_df.empty:
        return pd.DataFrame()
    df = comparison_df.copy()
    if "source_name" not in df.columns and "source" in df.columns:
        df["source_name"] = df["source"]
    if "source" not in df.columns and "source_name" in df.columns:
        df["source"] = df["source_name"]
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
    full_summary_path = Path(FULL_BACKTEST_SUMMARY_PATH)
    full_predictions_path = Path(FULL_BACKTEST_PREDICTIONS_PATH)
    full_summary_df = pd.read_csv(full_summary_path) if full_summary_path.exists() and full_summary_path.stat().st_size > 0 else pd.DataFrame()
    if comparison_df is None:
        full_comparison_path = Path(FULL_BACKTEST_MARKET_COMPARISON_PATH)
        comparison_path = full_comparison_path if full_comparison_path.exists() and full_comparison_path.stat().st_size > 0 else Path(ENSEMBLE_COMPARISON_PATH)
        comparison_df = pd.read_csv(comparison_path) if comparison_path.exists() and comparison_path.stat().st_size > 0 else pd.DataFrame()

    full_backtest_available = (
        (not full_summary_df.empty and int(_max_numeric_column(full_summary_df, "match_count")) > 0)
        or (bool(backtest_status.get("backtest_exists")) and int(backtest_status.get("prediction_count") or 0) > 0)
        or (full_predictions_path.exists() and full_predictions_path.stat().st_size > 0)
    )
    if not full_summary_df.empty and _clean_float(full_summary_df.iloc[0].get("accuracy")) is not None:
        row = full_summary_df.iloc[0]
        metrics_source = "full_backtest"
        accuracy = row.get("accuracy")
        log_loss = row.get("log_loss")
        brier_score = row.get("brier_score")
        ece = row.get("ece")
        match_count = row.get("match_count")
    elif full_backtest_available and not _is_missing(backtest_status.get("overall_accuracy")):
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
    best_source_validation = load_best_prediction_source_validation()
    if best_source_validation:
        market_comparison_available = bool(best_source_validation.get("market_comparison_available", market_comparison_available))
        full_backtest_available = bool(best_source_validation.get("full_walk_forward_backtest_available", full_backtest_available))
        calibration_available = bool(best_source_validation.get("calibration_available", calibration_available))
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
    if readiness.get("status") not in {"production_ready", "demo_model"}:
        model_confidence = "Low"
        betting_use = "Market fallback"
    elif readiness.get("status") == "demo_model":
        model_confidence = "Low"
        betting_use = "Market fallback"
    elif full_backtest_available and market_comparison_available and calibration_available:
        model_confidence = "High"
        betting_use = "Ready"
    else:
        model_confidence = "Medium"
        betting_use = "Use cautiously"

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
        "best_source_validation": best_source_validation,
        "missing_items": list(dict.fromkeys(missing_items)),
        "model_confidence": model_confidence,
        "betting_use": betting_use,
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
    best_validation = performance.get("best_source_validation") or {}
    confidence = performance.get("model_confidence", "Low")
    betting_use = performance.get("betting_use", "Market fallback")

    if status == "production_ready" and comparison_available and (best or best_validation):
        label = best_validation.get("selected_label") or best["label"]
        return {
            "quality_label": f"Model confidence: {confidence}",
            "status_color": "green",
            "headline": f"Best prediction source: {label}",
            "summary_text": f"The model has been tested on {match_count:,} historical matches and compared with market odds." if match_count else "The model has been compared with market odds.",
            "user_conclusion": "Conclusion: The app uses the prediction source that performed best in validation.",
            "betting_use": betting_use,
            "missing_items": missing_items,
            "recommended_next_action": "Keep validation updated when new backtests or odds data are available.",
        }
    if status == "production_ready":
        return {
            "quality_label": f"Model confidence: {confidence}",
            "status_color": "amber",
            "headline": f"Tested on {match_count:,} historical matches" if match_count else "Basic model test available",
            "summary_text": (
                "The model is trained and usable, but the most realistic historical test has not been completed yet."
                if not performance.get("full_backtest_available")
                else "The model has been tested historically, but has not yet been compared against bookmaker odds."
            ),
            "user_conclusion": "Conclusion: Predictions can be used as guidance, but betting recommendations should remain conservative.",
            "betting_use": betting_use,
            "missing_items": missing_items,
            "recommended_next_action": "Run full validation in Advanced / Admin.",
        }
    if status == "demo_model":
        return {
            "quality_label": "Model confidence: Low",
            "status_color": "red",
            "headline": "Not enough historical validation data",
            "summary_text": "Predictions use market odds as fallback.",
            "user_conclusion": "The model is not suitable for real predictions yet.",
            "betting_use": "Market fallback",
            "missing_items": missing_items or ["Production model is missing"],
            "recommended_next_action": "Train and export a production model from historical international data.",
        }
    return {
        "quality_label": "Model confidence: Low",
        "status_color": "gray",
        "headline": "Predictions use market odds as fallback",
        "summary_text": "No usable model artifact is available.",
        "user_conclusion": "The app can still show market-based probabilities, but ML predictions are unavailable.",
        "betting_use": "Market fallback",
        "missing_items": missing_items or ["Pre-trained model artifact is missing"],
        "recommended_next_action": "Add or redeploy the bundled model artifacts.",
    }


def validation_checklist(summary: dict) -> list[dict]:
    readiness = summary.get("readiness", {})
    return [
        {"Item": "Pre-trained model loaded", "Status": "Complete" if summary.get("model_artifacts_available") else "Missing"},
        {"Item": "Enough historical data used", "Status": "Complete" if readiness.get("training_rows", 0) and readiness.get("training_rows", 0) >= 1000 else "Recommended"},
        {"Item": "Basic test completed", "Status": "Complete" if summary.get("metrics_source") in {"holdout_metadata", "full_backtest"} else "Missing"},
        {"Item": "Realistic historical test completed", "Status": "Complete" if summary.get("full_backtest_available") else "Recommended", "Explanation": "The model is tested over time using only information available before each match."},
        {"Item": "Compared against bookmaker odds", "Status": "Complete" if summary.get("market_comparison_available") else "Recommended", "Explanation": "We cannot yet confirm whether the model beats bookmaker odds."},
        {"Item": "Probability realism checked", "Status": "Complete" if summary.get("calibration_available") else "Recommended", "Explanation": "Calibration checks whether predicted probabilities match reality."},
        {"Item": "Best prediction source selected", "Status": "Complete" if summary.get("best_source_validation") else "Recommended"},
    ]
