from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import backtest_paths as backtest_path_config
import config as app_config
from calibration import calculate_class_specific_calibration, calculate_expected_calibration_error
from ensemble import calculate_ensemble_probabilities


PROJECT_ROOT = getattr(app_config, "PROJECT_ROOT", Path(__file__).resolve().parent)
DATA_DIR = getattr(app_config, "DATA_DIR", PROJECT_ROOT / "data")
PROCESSED_DATA_DIR = getattr(app_config, "PROCESSED_DATA_DIR", DATA_DIR / "processed")
BEST_PREDICTION_SOURCE_VALIDATION_PATH = getattr(
    backtest_path_config,
    "BEST_PREDICTION_SOURCE_VALIDATION_PATH",
    PROCESSED_DATA_DIR / "best_prediction_source_validation.json",
)

LABELS = ["H", "D", "A"]
MODEL_COLUMNS = {"home": "pred_home_prob", "draw": "pred_draw_prob", "away": "pred_away_prob"}
MARKET_COLUMNS = {"home": "market_home_prob", "draw": "market_draw_prob", "away": "market_away_prob"}


def _empty_metric_result(reason: str | None = None) -> dict:
    return {
        "match_count": 0,
        "accuracy": None,
        "log_loss": None,
        "brier_score": None,
        "ece": None,
        "draw_calibration_gap": None,
        "home_calibration_gap": None,
        "away_calibration_gap": None,
        "avg_confidence": None,
        "reason": reason,
    }


def _probability_frame(y_true, y_pred_probs, labels=None) -> pd.DataFrame:
    labels = labels or LABELS
    if y_true is None or y_pred_probs is None:
        return pd.DataFrame(columns=labels + ["actual"])
    try:
        df = pd.DataFrame(y_pred_probs, columns=labels)
    except ValueError:
        return pd.DataFrame(columns=labels + ["actual"])
    df[labels] = df[labels].apply(pd.to_numeric, errors="coerce")
    df["actual"] = list(y_true)
    df = df[df["actual"].isin(labels)].copy()
    df = df.dropna(subset=labels)
    totals = df[labels].sum(axis=1)
    df = df[totals > 0].copy()
    if df.empty:
        return df
    df[labels] = df[labels].div(df[labels].sum(axis=1), axis=0).clip(lower=1e-15, upper=1 - 1e-15)
    df[labels] = df[labels].div(df[labels].sum(axis=1), axis=0)
    return df


def calculate_accuracy(y_true, y_pred) -> float | None:
    if y_true is None or y_pred is None or len(y_true) == 0:
        return None
    actual = pd.Series(list(y_true))
    predicted = pd.Series(list(y_pred))
    valid = actual.isin(LABELS) & predicted.isin(LABELS)
    if not valid.any():
        return None
    return float((actual[valid].reset_index(drop=True) == predicted[valid].reset_index(drop=True)).mean())


def calculate_log_loss(y_true, y_pred_probs, labels=None) -> float | None:
    labels = labels or LABELS
    df = _probability_frame(y_true, y_pred_probs, labels)
    if df.empty:
        return None
    losses = []
    for _, row in df.iterrows():
        losses.append(-float(np.log(row[row["actual"]])))
    return float(sum(losses) / len(losses)) if losses else None


def calculate_multiclass_brier(y_true, y_pred_probs, labels=None) -> float | None:
    labels = labels or LABELS
    df = _probability_frame(y_true, y_pred_probs, labels)
    if df.empty:
        return None
    total = 0.0
    for _, row in df.iterrows():
        total += sum((float(row[label]) - (1.0 if row["actual"] == label else 0.0)) ** 2 for label in labels)
    return float(total / len(df))


def calculate_class_calibration(y_true, y_pred_probs, labels=None) -> dict:
    labels = labels or LABELS
    result = calculate_class_specific_calibration(y_true, y_pred_probs, labels)
    if not result:
        return {"home_calibration_gap": None, "draw_calibration_gap": None, "away_calibration_gap": None}
    return {
        "home_calibration_gap": None if result.get("H") is None else result["H"]["calibration_gap"],
        "draw_calibration_gap": None if result.get("D") is None else result["D"]["calibration_gap"],
        "away_calibration_gap": None if result.get("A") is None else result["A"]["calibration_gap"],
    }


def calculate_draw_calibration(y_true, draw_probabilities) -> float | None:
    if y_true is None or draw_probabilities is None or len(y_true) == 0:
        return None
    df = pd.DataFrame({"actual": list(y_true), "draw_probability": list(draw_probabilities)})
    df["draw_probability"] = pd.to_numeric(df["draw_probability"], errors="coerce")
    df = df[df["actual"].isin(LABELS)].dropna(subset=["draw_probability"])
    if df.empty:
        return None
    return float(df["draw_probability"].mean() - (df["actual"] == "D").mean())


def calculate_probability_metrics(y_true, y_pred_probs, labels=None) -> dict:
    labels = labels or LABELS
    df = _probability_frame(y_true, y_pred_probs, labels)
    if df.empty:
        return _empty_metric_result("No valid predictions to evaluate.")
    predicted = df[labels].idxmax(axis=1)
    class_gaps = calculate_class_calibration(df["actual"].tolist(), df[labels].to_numpy(), labels)
    return {
        "match_count": int(len(df)),
        "accuracy": calculate_accuracy(df["actual"].tolist(), predicted.tolist()),
        "log_loss": calculate_log_loss(df["actual"].tolist(), df[labels].to_numpy(), labels),
        "brier_score": calculate_multiclass_brier(df["actual"].tolist(), df[labels].to_numpy(), labels),
        "ece": calculate_expected_calibration_error(df["actual"].tolist(), df[labels].to_numpy(), labels=labels),
        "draw_calibration_gap": class_gaps["draw_calibration_gap"],
        "home_calibration_gap": class_gaps["home_calibration_gap"],
        "away_calibration_gap": class_gaps["away_calibration_gap"],
        "avg_confidence": float(df[labels].max(axis=1).mean()),
        "reason": None,
    }


def evaluate_prediction_columns(df: pd.DataFrame, source: str, columns: dict, actual_col: str = "actual_result") -> dict:
    if df.empty or actual_col not in df.columns or any(column not in df.columns for column in columns.values()):
        row = _empty_metric_result("Required probability columns are missing.")
        row.update({"source": source})
        return row
    probabilities = df[[columns["home"], columns["draw"], columns["away"]]].apply(pd.to_numeric, errors="coerce")
    metrics = calculate_probability_metrics(df[actual_col].tolist(), probabilities.to_numpy())
    metrics.update({"source": source})
    return metrics


def compare_model_market_and_ensembles(predictions_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    has_market = all(column in predictions_df.columns for column in MARKET_COLUMNS.values())
    has_model = all(column in predictions_df.columns for column in MODEL_COLUMNS.values())
    if has_market:
        row = evaluate_prediction_columns(predictions_df, "market", MARKET_COLUMNS)
        row.update({"w_market": 1.0, "w_model": 0.0})
        rows.append(row)
    if has_model:
        row = evaluate_prediction_columns(predictions_df, "model", MODEL_COLUMNS)
        row.update({"w_market": 0.0, "w_model": 1.0})
        rows.append(row)
    if has_market and has_model:
        for w_market in [0.9, 0.8, 0.7, 0.6, 0.5]:
            combined = calculate_ensemble_probabilities(predictions_df, w_market, MODEL_COLUMNS, MARKET_COLUMNS)
            row = evaluate_prediction_columns(
                combined,
                f"ensemble_{w_market:.1f}_{1 - w_market:.1f}",
                {"home": "ensemble_home_prob", "draw": "ensemble_draw_prob", "away": "ensemble_away_prob"},
            )
            row.update({"w_market": float(w_market), "w_model": float(1 - w_market)})
            rows.append(row)
    comparison = pd.DataFrame(rows)
    if comparison.empty:
        return comparison
    recommendation = select_best_source_from_validation(comparison, market_comparison_available=has_market)
    comparison["selected"] = comparison["source"].eq(recommendation["selected_source"])
    comparison["selection_reason"] = recommendation["reason"]
    return comparison


def select_best_source_from_validation(
    comparison_df: pd.DataFrame,
    market_comparison_available: bool = True,
    negligible_log_loss: float = 0.001,
) -> dict:
    if comparison_df is None or comparison_df.empty:
        return {
            "selected_source": "model",
            "selected_label": "Model",
            "w_market": 0.0,
            "w_model": 1.0,
            "reason": "Only model validation is available. Historical market odds are missing.",
            "caveats": ["Market comparison cannot be calculated because historical market odds are not available."],
            "market_comparison_available": False,
        }
    candidates = comparison_df.copy()
    for column in ["log_loss", "brier_score", "ece", "draw_calibration_gap", "accuracy"]:
        candidates[column] = pd.to_numeric(candidates.get(column), errors="coerce")
    candidates = candidates.dropna(subset=["log_loss", "brier_score", "ece"])
    if candidates.empty:
        return {
            "selected_source": "unknown",
            "selected_label": "Unknown",
            "w_market": None,
            "w_model": None,
            "reason": "Validation metrics are incomplete.",
            "caveats": ["Run full validation again."],
            "market_comparison_available": market_comparison_available,
        }
    candidates["_abs_draw_gap"] = candidates["draw_calibration_gap"].abs()
    candidates["_accuracy_sort"] = -candidates["accuracy"].fillna(0)
    best = candidates.sort_values(["log_loss", "brier_score", "ece", "_abs_draw_gap", "_accuracy_sort"]).iloc[0]
    caveats = []
    if market_comparison_available:
        market = candidates[candidates["source"] == "market"].head(1)
        if not market.empty and best["source"] != "market":
            market_row = market.iloc[0]
            improvement = float(market_row["log_loss"] - best["log_loss"])
            materially_better = (
                improvement > negligible_log_loss
                or float(best["brier_score"]) < float(market_row["brier_score"]) - 0.001
                or float(best["ece"]) < float(market_row["ece"]) - 0.001
            )
            if not materially_better:
                best = market_row
                caveats.append("A more complex source did not clearly improve on market-only probabilities.")
    else:
        caveats.append("Market comparison cannot be calculated because historical market odds are not available.")
    source = str(best["source"])
    if source == "market":
        label = "Market"
    elif source == "model":
        label = "Model"
    elif source.startswith("ensemble"):
        label = "Ensemble"
    else:
        label = source.replace("_", " ").title()
    if source != "market" and float(best.get("ece") or 0) > 0.06:
        caveats.append("The selected source has weak probability realism, so betting edges should be treated cautiously.")
    return {
        "selected_source": source,
        "selected_label": label,
        "w_market": None if pd.isna(best.get("w_market")) else float(best.get("w_market")),
        "w_model": None if pd.isna(best.get("w_model")) else float(best.get("w_model")),
        "reason": f"Selected by probability quality: log loss first, then Brier score, probability realism and draw calibration.",
        "caveats": caveats,
        "market_comparison_available": market_comparison_available,
    }


def save_best_prediction_source_validation(
    selection: dict,
    output_path: Path = BEST_PREDICTION_SOURCE_VALIDATION_PATH,
    metrics_source: str = "full_backtest",
    full_walk_forward_backtest_available: bool = True,
    calibration_available: bool = True,
) -> dict:
    def json_safe(value):
        if isinstance(value, (np.bool_,)):
            return bool(value)
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return None if np.isnan(value) else float(value)
        if isinstance(value, list):
            return [json_safe(item) for item in value]
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        return value

    result = {
        "selected_source": selection.get("selected_source"),
        "selected_label": selection.get("selected_label"),
        "w_market": selection.get("w_market"),
        "w_model": selection.get("w_model"),
        "reason": selection.get("reason"),
        "caveats": selection.get("caveats", []),
        "validation_completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "metrics_source": metrics_source,
        "market_comparison_available": bool(selection.get("market_comparison_available")),
        "full_walk_forward_backtest_available": bool(full_walk_forward_backtest_available),
        "calibration_available": bool(calibration_available),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    safe_result = json_safe(result)
    output_path.write_text(json.dumps(safe_result, indent=2))
    return safe_result


def load_best_prediction_source_validation(path: Path = BEST_PREDICTION_SOURCE_VALIDATION_PATH) -> dict:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
