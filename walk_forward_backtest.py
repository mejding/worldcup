from __future__ import annotations

from pathlib import Path

import pandas as pd

from backtest import run_walk_forward_backtest
from backtest_paths import (
    BEST_PREDICTION_SOURCE_VALIDATION_PATH,
    FULL_BACKTEST_BY_FOLD_PATH,
    FULL_BACKTEST_BY_SEGMENT_PATH,
    FULL_BACKTEST_CALIBRATION_PATH,
    FULL_BACKTEST_DRAW_CALIBRATION_PATH,
    FULL_BACKTEST_MARKET_COMPARISON_PATH,
    FULL_BACKTEST_PREDICTIONS_PATH,
    FULL_BACKTEST_REPORT_PATH,
    FULL_BACKTEST_SUMMARY_PATH,
    PROCESSED_DATA_DIR,
)
from backtest_reports import evaluate_by_segment
from calibration import create_confidence_calibration_bins, create_draw_calibration_table
from validation import (
    MODEL_COLUMNS,
    compare_model_market_and_ensembles,
    calculate_probability_metrics,
    save_best_prediction_source_validation,
    select_best_source_from_validation,
)


def _prepare_for_full_backtest(historical_matches_df: pd.DataFrame) -> pd.DataFrame:
    df = historical_matches_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    required = ["date", "home_team", "away_team", "home_score", "away_score", "result"]
    df = df.dropna(subset=required).sort_values("date").reset_index(drop=True)
    return df


def _auto_initial_train_end_date(historical: pd.DataFrame, min_train_matches: int):
    if len(historical) < min_train_matches:
        return None
    return historical.iloc[min_train_matches - 1]["date"] + pd.Timedelta(days=1)


def _format_date(value):
    if pd.isna(value):
        return None
    return pd.to_datetime(value, utc=True).date().isoformat()


def _build_full_predictions(predictions: pd.DataFrame, fold_summary: pd.DataFrame, model_variant: str, data_features_used: str) -> pd.DataFrame:
    if predictions.empty:
        columns = [
            "fold_id",
            "train_start_date",
            "train_end_date",
            "test_start_date",
            "test_end_date",
            "match_date",
            "home_team",
            "away_team",
            "actual_result",
            "pred_home_prob",
            "pred_draw_prob",
            "pred_away_prob",
            "predicted_result",
            "confidence",
            "is_correct",
            "model_variant",
            "data_features_used",
        ]
        return pd.DataFrame(columns=columns)
    fold_dates = fold_summary[
        ["fold_id", "train_start_date", "train_end_date", "test_start_date", "test_end_date"]
    ].copy()
    result = predictions.merge(fold_dates, on="fold_id", how="left")
    result = result.rename(columns={"date": "match_date"})
    result["model_variant"] = model_variant
    result["data_features_used"] = data_features_used
    for target, source in [
        ("model_home_prob", "pred_home_prob"),
        ("model_draw_prob", "pred_draw_prob"),
        ("model_away_prob", "pred_away_prob"),
    ]:
        if target not in result.columns and source in result.columns:
            result[target] = result[source]
    preferred = [
        "fold_id",
        "train_start_date",
        "train_end_date",
        "test_start_date",
        "test_end_date",
        "match_date",
        "home_team",
        "away_team",
        "actual_result",
        "pred_home_prob",
        "pred_draw_prob",
        "pred_away_prob",
        "predicted_result",
        "confidence",
        "is_correct",
        "model_variant",
        "data_features_used",
        "model_home_prob",
        "model_draw_prob",
        "model_away_prob",
        "market_home_prob",
        "market_draw_prob",
        "market_away_prob",
        "tournament",
        "tournament_category",
        "neutral",
        "fifa_ranking_gap",
        "elo_gap",
        "draw_context_label",
    ]
    return result[[column for column in preferred if column in result.columns]]


def _build_overall_summary(predictions: pd.DataFrame, fold_summary: pd.DataFrame, model_variant: str) -> pd.DataFrame:
    metrics = calculate_probability_metrics(
        predictions["actual_result"].tolist() if not predictions.empty else [],
        predictions[["pred_home_prob", "pred_draw_prob", "pred_away_prob"]].to_numpy() if not predictions.empty else [],
    )
    return pd.DataFrame(
        [
            {
                "model_variant": model_variant,
                "match_count": metrics["match_count"],
                "accuracy": metrics["accuracy"],
                "log_loss": metrics["log_loss"],
                "brier_score": metrics["brier_score"],
                "ece": metrics["ece"],
                "draw_calibration_gap": metrics["draw_calibration_gap"],
                "home_calibration_gap": metrics["home_calibration_gap"],
                "away_calibration_gap": metrics["away_calibration_gap"],
                "avg_confidence": metrics["avg_confidence"],
                "evaluated_from": _format_date(predictions["match_date"].min()) if not predictions.empty else None,
                "evaluated_to": _format_date(predictions["match_date"].max()) if not predictions.empty else None,
                "folds": int(fold_summary["fold_id"].nunique()) if not fold_summary.empty and "fold_id" in fold_summary else 0,
            }
        ]
    )


def _build_fold_summary(fold_summary: pd.DataFrame) -> pd.DataFrame:
    if fold_summary.empty:
        return pd.DataFrame(
            columns=[
                "fold_id",
                "train_end_date",
                "test_start_date",
                "test_end_date",
                "match_count",
                "accuracy",
                "log_loss",
                "brier_score",
                "ece",
                "draw_calibration_gap",
            ]
        )
    result = fold_summary.rename(columns={"test_rows": "match_count"}).copy()
    columns = [
        "fold_id",
        "train_end_date",
        "test_start_date",
        "test_end_date",
        "match_count",
        "accuracy",
        "log_loss",
        "brier_score",
        "ece",
        "draw_calibration_gap",
    ]
    return result[[column for column in columns if column in result.columns]]


def _write_report(path: Path, result: dict) -> None:
    summary = result.get("summary", pd.DataFrame())
    comparison = result.get("market_comparison", pd.DataFrame())
    selection = result.get("best_source_validation", {})
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Full Model Validation Report",
        "",
        f"Status: {result.get('status')}",
        "",
        "This walk-forward validation tests the model as if it only knew matches played before each test window.",
        "",
    ]
    if not summary.empty:
        row = summary.iloc[0]
        lines.extend(
            [
                "## Overall model metrics",
                f"- Matches: {int(row.get('match_count', 0))}",
                f"- Accuracy: {row.get('accuracy')}",
                f"- Log loss: {row.get('log_loss')}",
                f"- Brier score: {row.get('brier_score')}",
                f"- ECE: {row.get('ece')}",
                "",
            ]
        )
    if comparison.empty:
        lines.extend(
            [
                "## Market comparison",
                "Market comparison cannot be calculated because historical market odds are not available.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Best prediction source",
                f"- Selected: {selection.get('selected_label')}",
                f"- Reason: {selection.get('reason')}",
                "",
            ]
        )
    caveats = selection.get("caveats") or []
    if caveats:
        lines.append("## Caveats")
        lines.extend(f"- {item}" for item in caveats)
        lines.append("")
    path.write_text("\n".join(lines))


def run_full_walk_forward_backtest(
    historical_matches_df: pd.DataFrame,
    initial_train_end_date=None,
    test_window_months: int = 12,
    step_months: int = 12,
    min_train_matches: int = 1000,
    model_variant: str = "best_available",
    include_fifa_ranking_features: bool = True,
    include_elo_features: bool = True,
    include_form_features: bool = True,
    output_dir: Path = PROCESSED_DATA_DIR,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    historical = _prepare_for_full_backtest(historical_matches_df)
    if historical.empty:
        return {"status": "validation_error", "error": "No valid historical matches available."}
    cutoff = pd.to_datetime(initial_train_end_date, utc=True) if initial_train_end_date is not None else _auto_initial_train_end_date(historical, min_train_matches)
    if cutoff is None:
        return {
            "status": "validation_error",
            "error": f"At least {min_train_matches} historical matches are required before the first test period.",
        }

    include_draw_context = model_variant == "draw_context_model"
    data_features_used = ", ".join(
        [
            item
            for item, enabled in [
                ("FIFA ranking", include_fifa_ranking_features),
                ("Elo", include_elo_features),
                ("form", include_form_features),
                ("draw context", include_draw_context),
            ]
            if enabled
        ]
    )
    base_result = run_walk_forward_backtest(
        historical,
        initial_train_end_date=cutoff.date().isoformat(),
        test_window=f"{int(test_window_months) * 30}D",
        step_size=f"{int(step_months) * 30}D",
        min_train_matches=int(min_train_matches),
        output_dir=output_dir,
        include_draw_context_features=include_draw_context,
    )
    fold_source = base_result.get("summary", pd.DataFrame())
    predictions = _build_full_predictions(
        base_result.get("predictions", pd.DataFrame()),
        fold_source,
        model_variant=model_variant,
        data_features_used=data_features_used,
    )
    by_fold = _build_fold_summary(fold_source)
    summary = _build_overall_summary(predictions, by_fold, model_variant=model_variant)
    segments = evaluate_by_segment(predictions.rename(columns={"match_date": "date"})) if not predictions.empty else pd.DataFrame()
    calibration = (
        create_confidence_calibration_bins(predictions["actual_result"], predictions[["pred_home_prob", "pred_draw_prob", "pred_away_prob"]].to_numpy())
        if not predictions.empty
        else create_confidence_calibration_bins([], [])
    )
    draw_calibration = (
        create_draw_calibration_table(predictions["actual_result"], predictions["pred_draw_prob"])
        if not predictions.empty
        else create_draw_calibration_table([], [])
    )
    market_comparison = compare_model_market_and_ensembles(predictions)
    market_available = not market_comparison.empty and market_comparison["source"].eq("market").any()
    selection = select_best_source_from_validation(market_comparison, market_comparison_available=market_available)
    best_source_validation = save_best_prediction_source_validation(
        selection,
        output_path=output_dir / BEST_PREDICTION_SOURCE_VALIDATION_PATH.name,
        metrics_source="full_backtest",
        full_walk_forward_backtest_available=not predictions.empty,
        calibration_available=summary["ece"].notna().any(),
    )

    paths = {
        "predictions": output_dir / FULL_BACKTEST_PREDICTIONS_PATH.name,
        "summary": output_dir / FULL_BACKTEST_SUMMARY_PATH.name,
        "by_fold": output_dir / FULL_BACKTEST_BY_FOLD_PATH.name,
        "by_segment": output_dir / FULL_BACKTEST_BY_SEGMENT_PATH.name,
        "calibration": output_dir / FULL_BACKTEST_CALIBRATION_PATH.name,
        "draw_calibration": output_dir / FULL_BACKTEST_DRAW_CALIBRATION_PATH.name,
        "market_comparison": output_dir / FULL_BACKTEST_MARKET_COMPARISON_PATH.name,
        "best_source_validation": output_dir / BEST_PREDICTION_SOURCE_VALIDATION_PATH.name,
        "report": FULL_BACKTEST_REPORT_PATH if output_dir == PROCESSED_DATA_DIR else output_dir / FULL_BACKTEST_REPORT_PATH.name,
    }
    predictions.to_csv(paths["predictions"], index=False)
    summary.to_csv(paths["summary"], index=False)
    by_fold.to_csv(paths["by_fold"], index=False)
    segments.to_csv(paths["by_segment"], index=False)
    calibration.to_csv(paths["calibration"], index=False)
    draw_calibration.to_csv(paths["draw_calibration"], index=False)
    market_comparison.to_csv(paths["market_comparison"], index=False)
    result = {
        "status": "ok" if not predictions.empty else "no_predictions",
        "predictions": predictions,
        "summary": summary,
        "by_fold": by_fold,
        "segments": segments,
        "calibration": calibration,
        "draw_calibration": draw_calibration,
        "market_comparison": market_comparison,
        "best_source_validation": best_source_validation,
        "paths": paths,
    }
    _write_report(paths["report"], result)
    return result
