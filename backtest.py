from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from backtest_reports import evaluate_by_segment, create_backtest_report
from calibration import create_confidence_calibration_bins, create_draw_calibration_table
from config import (
    BACKTEST_BY_SEGMENT_PATH,
    BACKTEST_CALIBRATION_BINS_PATH,
    BACKTEST_DRAW_CALIBRATION_PATH,
    BACKTEST_PREDICTIONS_PATH,
    BACKTEST_REPORT_PATH,
    BACKTEST_SUMMARY_PATH,
    PROCESSED_DATA_DIR,
    WORLD_CUP_BACKTEST_PREDICTIONS_PATH,
    WORLD_CUP_BACKTEST_SUMMARY_PATH,
)
from evaluation import calculate_prediction_metrics
from features import _empty_stats, _feature_row, _update_elo, _update_stats, build_training_dataset, categorize_tournament
from train_model import predict_with_model, train_model_in_memory


PREDICTION_COLUMNS = [
    "fold_id",
    "date",
    "home_team",
    "away_team",
    "tournament",
    "tournament_category",
    "neutral",
    "actual_result",
    "pred_home_prob",
    "pred_draw_prob",
    "pred_away_prob",
    "predicted_result",
    "confidence",
    "is_correct",
]


def _prepare_historical(df: pd.DataFrame) -> pd.DataFrame:
    historical = df.copy()
    historical["date"] = pd.to_datetime(historical["date"], errors="coerce", utc=True)
    historical = historical.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score", "result"])
    historical["tournament"] = historical["tournament"].fillna("Unknown") if "tournament" in historical.columns else "Unknown"
    historical["neutral"] = historical["neutral"].fillna(False).astype(bool) if "neutral" in historical.columns else False
    return historical.sort_values("date").reset_index(drop=True)


def _build_frozen_test_features(train_raw: pd.DataFrame, test_raw: pd.DataFrame) -> pd.DataFrame:
    team_stats = defaultdict(_empty_stats)
    elos = defaultdict(lambda: 1500.0)
    for _, match in train_raw.sort_values("date").iterrows():
        _update_stats(
            team_stats,
            match["home_team"],
            match["away_team"],
            match["home_score"],
            match["away_score"],
            match["result"],
        )
        _update_elo(elos, match["home_team"], match["away_team"], match["result"])
    rows = []
    for _, match in test_raw.sort_values("date").iterrows():
        rows.append(
            {
                "date": match["date"],
                "home_team": match["home_team"],
                "away_team": match["away_team"],
                "result": match["result"],
                "tournament": match.get("tournament", "Unknown"),
                **_feature_row(match, team_stats, elos),
            }
        )
    return pd.DataFrame(rows)


def _run_single_fold(fold_id: int, train_raw: pd.DataFrame, test_raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    with TemporaryDirectory() as tmpdir:
        training_df = build_training_dataset(train_raw, output_path=Path(tmpdir) / "training_dataset.csv")
    model, _ = train_model_in_memory(training_df)
    test_features = _build_frozen_test_features(train_raw, test_raw)
    predictions = predict_with_model(model, test_features)
    output = pd.concat(
        [
            test_features[["date", "home_team", "away_team", "tournament", "tournament_category", "neutral", "result"]].reset_index(drop=True),
            predictions.reset_index(drop=True),
        ],
        axis=1,
    )
    output = output.rename(columns={"result": "actual_result"})
    output.insert(0, "fold_id", fold_id)
    output["is_correct"] = output["predicted_result"] == output["actual_result"]
    metrics = calculate_prediction_metrics(output["actual_result"].tolist(), output[["pred_home_prob", "pred_draw_prob", "pred_away_prob"]].to_numpy())
    return output[PREDICTION_COLUMNS], metrics


def _summary_row(fold_id: int, train_raw: pd.DataFrame, test_raw: pd.DataFrame, metrics: dict) -> dict:
    row = {
        "fold_id": fold_id,
        "train_start_date": train_raw["date"].min().isoformat(),
        "train_end_date": train_raw["date"].max().isoformat(),
        "test_start_date": test_raw["date"].min().isoformat(),
        "test_end_date": test_raw["date"].max().isoformat(),
        "train_rows": int(len(train_raw)),
        "test_rows": int(len(test_raw)),
    }
    row.update(metrics)
    return row


def _save_backtest_outputs(predictions_df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: Path, setup: dict) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / BACKTEST_PREDICTIONS_PATH.name
    summary_path = output_dir / BACKTEST_SUMMARY_PATH.name
    segment_path = output_dir / BACKTEST_BY_SEGMENT_PATH.name
    draw_path = output_dir / BACKTEST_DRAW_CALIBRATION_PATH.name
    calibration_path = output_dir / BACKTEST_CALIBRATION_BINS_PATH.name
    report_path = BACKTEST_REPORT_PATH if output_dir == PROCESSED_DATA_DIR else output_dir / BACKTEST_REPORT_PATH.name

    segment_df = evaluate_by_segment(predictions_df)
    draw_df = create_draw_calibration_table(predictions_df["actual_result"], predictions_df["pred_draw_prob"]) if not predictions_df.empty else create_draw_calibration_table([], [])
    calibration_df = (
        create_confidence_calibration_bins(predictions_df["actual_result"], predictions_df[["pred_home_prob", "pred_draw_prob", "pred_away_prob"]].to_numpy())
        if not predictions_df.empty
        else create_confidence_calibration_bins([], [])
    )
    predictions_df.to_csv(predictions_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    segment_df.to_csv(segment_path, index=False)
    draw_df.to_csv(draw_path, index=False)
    calibration_df.to_csv(calibration_path, index=False)
    create_backtest_report(predictions_df, summary_df, segment_df, draw_df, calibration_df, report_path, setup=setup)
    return {
        "predictions": predictions_df,
        "summary": summary_df,
        "segments": segment_df,
        "draw_calibration": draw_df,
        "calibration_bins": calibration_df,
        "paths": {
            "predictions": predictions_path,
            "summary": summary_path,
            "segments": segment_path,
            "draw_calibration": draw_path,
            "calibration_bins": calibration_path,
            "report": report_path,
        },
    }


def run_walk_forward_backtest(
    historical_df: pd.DataFrame,
    initial_train_end_date,
    test_window="365D",
    step_size="365D",
    min_train_matches: int = 1000,
    output_dir: Path | None = None,
) -> dict:
    historical = _prepare_historical(historical_df)
    if historical.empty:
        return _save_backtest_outputs(
            pd.DataFrame(columns=PREDICTION_COLUMNS),
            pd.DataFrame(),
            Path(output_dir or PROCESSED_DATA_DIR),
            {"warning": "No valid historical matches"},
        )
    cutoff = pd.to_datetime(initial_train_end_date, utc=True)
    max_date = historical["date"].max()
    fold_rows = []
    prediction_frames = []
    fold_id = 1
    while cutoff <= max_date:
        test_end = cutoff + pd.Timedelta(test_window)
        train_raw = historical[historical["date"] < cutoff].copy()
        test_raw = historical[(historical["date"] >= cutoff) & (historical["date"] < test_end)].copy()
        if len(train_raw) >= min_train_matches and not test_raw.empty:
            try:
                predictions_df, metrics = _run_single_fold(fold_id, train_raw, test_raw)
                prediction_frames.append(predictions_df)
                fold_rows.append(_summary_row(fold_id, train_raw, test_raw, metrics))
                fold_id += 1
            except Exception as exc:
                fold_rows.append(
                    {
                        "fold_id": fold_id,
                        "train_start_date": train_raw["date"].min().isoformat() if not train_raw.empty else None,
                        "train_end_date": train_raw["date"].max().isoformat() if not train_raw.empty else None,
                        "test_start_date": test_raw["date"].min().isoformat() if not test_raw.empty else None,
                        "test_end_date": test_raw["date"].max().isoformat() if not test_raw.empty else None,
                        "train_rows": int(len(train_raw)),
                        "test_rows": int(len(test_raw)),
                        "error": str(exc),
                    }
                )
                fold_id += 1
        cutoff += pd.Timedelta(step_size)
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame(columns=PREDICTION_COLUMNS)
    summary = pd.DataFrame(fold_rows)
    setup = {
        "initial_train_end_date": str(initial_train_end_date),
        "test_window": str(test_window),
        "step_size": str(step_size),
        "min_train_matches": min_train_matches,
    }
    return _save_backtest_outputs(predictions, summary, Path(output_dir or PROCESSED_DATA_DIR), setup)


def run_world_cup_backtest(historical_df: pd.DataFrame, world_cup_years=None, output_dir: Path | None = None) -> dict:
    world_cup_years = world_cup_years or [2014, 2018, 2022]
    historical = _prepare_historical(historical_df)
    predictions = []
    summaries = []
    for fold_id, year in enumerate(world_cup_years, start=1):
        cutoff = pd.Timestamp(f"{year}-01-01", tz="UTC")
        train_raw = historical[historical["date"] < cutoff].copy()
        test_raw = historical[
            (historical["date"].dt.year == year)
            & (historical["tournament"].map(categorize_tournament) == "world_cup")
        ].copy()
        if len(train_raw) < 30 or test_raw.empty:
            summaries.append(
                {
                    "fold_id": fold_id,
                    "world_cup_year": year,
                    "train_rows": int(len(train_raw)),
                    "test_rows": int(len(test_raw)),
                    "warning": "Insufficient training data or no World Cup rows",
                }
            )
            continue
        try:
            fold_predictions, metrics = _run_single_fold(fold_id, train_raw, test_raw)
            predictions.append(fold_predictions)
            row = _summary_row(fold_id, train_raw, test_raw, metrics)
            row["world_cup_year"] = year
            summaries.append(row)
        except Exception as exc:
            summaries.append({"fold_id": fold_id, "world_cup_year": year, "train_rows": int(len(train_raw)), "test_rows": int(len(test_raw)), "error": str(exc)})
    output_dir = Path(output_dir or PROCESSED_DATA_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_df = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(columns=PREDICTION_COLUMNS)
    summary_df = pd.DataFrame(summaries)
    predictions_path = output_dir / WORLD_CUP_BACKTEST_PREDICTIONS_PATH.name
    summary_path = output_dir / WORLD_CUP_BACKTEST_SUMMARY_PATH.name
    predictions_df.to_csv(predictions_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    return {"predictions": predictions_df, "summary": summary_df, "paths": {"predictions": predictions_path, "summary": summary_path}}
