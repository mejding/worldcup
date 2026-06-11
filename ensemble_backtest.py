from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from backtest_paths import ENSEMBLE_BACKTEST_SUMMARY_PATH, ENSEMBLE_COMPARISON_PATH, ENSEMBLE_REPORT_PATH
from ensemble import (
    DRAW_MODEL_PROB_COLUMNS,
    MARKET_PROB_COLUMNS,
    MODEL_PROB_COLUMNS,
    calculate_ensemble_probabilities,
    create_weight_grid,
)
from evaluation import calculate_prediction_metrics


def evaluate_probability_source(
    predictions_df: pd.DataFrame,
    source_name: str,
    home_col: str,
    draw_col: str,
    away_col: str,
    actual_col: str = "actual_result",
) -> dict:
    if predictions_df.empty or any(column not in predictions_df.columns for column in [home_col, draw_col, away_col, actual_col]):
        metrics = calculate_prediction_metrics([], [])
        return {"source_name": source_name, "match_count": 0, **metrics}
    proba = predictions_df[[home_col, draw_col, away_col]].apply(pd.to_numeric, errors="coerce").fillna(0)
    totals = proba.sum(axis=1).where(lambda value: value > 0, 1.0)
    proba = proba.div(totals, axis=0)
    metrics = calculate_prediction_metrics(predictions_df[actual_col].tolist(), proba.to_numpy())
    return {"source_name": source_name, "match_count": int(len(predictions_df)), **metrics}


def compare_market_model_ensemble(
    predictions_df: pd.DataFrame,
    model_variant: str = "historical_model",
    weight_grid: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows = []
    if predictions_df.empty:
        return pd.DataFrame()
    weight_grid = weight_grid if weight_grid is not None else create_weight_grid()
    has_market = all(column in predictions_df.columns for column in MARKET_PROB_COLUMNS.values())
    has_model = all(column in predictions_df.columns for column in MODEL_PROB_COLUMNS.values())
    has_draw_model = all(column in predictions_df.columns for column in DRAW_MODEL_PROB_COLUMNS.values())
    if has_market:
        row = evaluate_probability_source(predictions_df, "market", *MARKET_PROB_COLUMNS.values())
        row.update({"model_variant": "market", "w_market": 1.0, "w_model": 0.0})
        rows.append(row)
    if has_model:
        row = evaluate_probability_source(predictions_df, "historical_model", *MODEL_PROB_COLUMNS.values())
        row.update({"model_variant": "historical_model", "w_market": 0.0, "w_model": 1.0})
        rows.append(row)
    if has_draw_model:
        row = evaluate_probability_source(predictions_df, "draw_context_model", *DRAW_MODEL_PROB_COLUMNS.values())
        row.update({"model_variant": "draw_context_model", "w_market": 0.0, "w_model": 1.0})
        rows.append(row)

    selected_model = DRAW_MODEL_PROB_COLUMNS if model_variant == "draw_context_model" and has_draw_model else MODEL_PROB_COLUMNS
    if has_market and all(column in predictions_df.columns for column in selected_model.values()):
        for _, weight in weight_grid.iterrows():
            combined = calculate_ensemble_probabilities(predictions_df, weight["w_market"], selected_model, MARKET_PROB_COLUMNS)
            source_name = f"ensemble_{weight['w_market']:.1f}_{weight['w_model']:.1f}"
            row = evaluate_probability_source(combined, source_name, "ensemble_home_prob", "ensemble_draw_prob", "ensemble_away_prob")
            row.update({"model_variant": model_variant, "w_market": float(weight["w_market"]), "w_model": float(weight["w_model"])})
            rows.append(row)
    result = pd.DataFrame(rows)
    keep = [
        "source_name",
        "model_variant",
        "w_market",
        "w_model",
        "match_count",
        "accuracy",
        "log_loss",
        "brier_score",
        "ece",
        "draw_calibration_gap",
        "avg_pred_draw_prob",
        "actual_draw_rate",
    ]
    return result[[column for column in keep if column in result.columns]]


def select_best_probability_source(comparison_df: pd.DataFrame, min_match_count: int = 500) -> dict:
    if comparison_df.empty:
        return {"recommended_source": "market", "w_market": 1.0, "w_model": 0.0, "reason": "No comparison rows available. Falling back to market.", "caveats": ["Run ensemble comparison first."]}
    candidates = comparison_df.copy()
    sufficient = candidates[candidates["match_count"] >= min_match_count]
    caveats = []
    if sufficient.empty:
        sufficient = candidates
        caveats.append("All comparison rows are below the preferred minimum sample size.")
    sufficient = sufficient.assign(_abs_draw_gap=sufficient["draw_calibration_gap"].abs())
    best = sufficient.sort_values(["log_loss", "brier_score", "ece", "_abs_draw_gap"]).iloc[0]
    market = sufficient[sufficient["source_name"] == "market"].head(1)
    if not market.empty:
        market_row = market.iloc[0]
        brier_clear = best["brier_score"] < market_row["brier_score"] - 0.001
        ece_clear = best["ece"] < market_row["ece"] - 0.001
        draw_gap_clear = abs(best["draw_calibration_gap"]) < abs(market_row["draw_calibration_gap"]) - 0.005
        if best["source_name"] != "market" and best["log_loss"] >= market_row["log_loss"] - 0.001 and not (brier_clear or ece_clear or draw_gap_clear):
            best = market_row
            caveats.append("Model/ensemble did not clearly beat market-only on log loss.")
    source = "ensemble" if str(best["source_name"]).startswith("ensemble") else str(best["source_name"])
    return {
        "recommended_source": source,
        "w_market": float(best.get("w_market", 1.0 if source == "market" else 0.0)),
        "w_model": float(best.get("w_model", 0.0 if source == "market" else 1.0)),
        "reason": f"Selected by log loss, then Brier, ECE and draw calibration: {best['source_name']}.",
        "caveats": caveats,
    }


def _write_report(path: Path, comparison_df: pd.DataFrame, recommendation: dict, status: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Ensemble Report",
        "",
        f"Run timestamp: {datetime.now(timezone.utc).isoformat()}",
        f"Status: {status}",
        "",
        "Bookmaker market probabilities are very strong baselines. The ensemble should only be preferred over market-only if it improves probability metrics such as log loss, Brier score and calibration on a proper validation/backtest set.",
        "",
        f"Best source: {recommendation.get('recommended_source')}",
        f"Weights: market={recommendation.get('w_market')}, model={recommendation.get('w_model')}",
        f"Reason: {recommendation.get('reason')}",
        "",
        "## Limitations",
        "- Historical market probabilities may be unavailable.",
        "- Accuracy is not used as the primary selection metric.",
    ]
    if not comparison_df.empty:
        lines.extend(["", "## Overall comparison"])
        for _, row in comparison_df.iterrows():
            lines.append(f"- {row['source_name']}: n={int(row['match_count'])}, log_loss={row['log_loss']:.4f}, brier={row['brier_score']:.4f}")
    path.write_text("\n".join(lines))


def run_ensemble_backtest_from_saved_predictions(backtest_predictions_path: Path, output_dir: Path) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = Path(backtest_predictions_path)
    if not predictions_path.exists() or predictions_path.stat().st_size == 0:
        comparison = pd.DataFrame()
        recommendation = select_best_probability_source(comparison)
        _write_report(output_dir / ENSEMBLE_REPORT_PATH.name, comparison, recommendation, "missing_backtest_predictions")
        return {"status": "missing_backtest_predictions", "comparison": comparison, "recommendation": recommendation}
    predictions = pd.read_csv(predictions_path)
    if not all(column in predictions.columns for column in MARKET_PROB_COLUMNS.values()):
        comparison = pd.DataFrame()
        recommendation = select_best_probability_source(comparison)
        _write_report(output_dir / ENSEMBLE_REPORT_PATH.name, comparison, recommendation, "market_probabilities_missing")
        return {"status": "market_probabilities_missing", "comparison": comparison, "recommendation": recommendation}
    comparison = compare_market_model_ensemble(predictions)
    recommendation = select_best_probability_source(comparison)
    comparison.to_csv(output_dir / ENSEMBLE_COMPARISON_PATH.name, index=False)
    comparison.to_csv(output_dir / ENSEMBLE_BACKTEST_SUMMARY_PATH.name, index=False)
    _write_report(output_dir / ENSEMBLE_REPORT_PATH.name, comparison, recommendation, "ok")
    return {"status": "ok", "comparison": comparison, "recommendation": recommendation}
