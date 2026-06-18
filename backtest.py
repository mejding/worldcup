from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from backtest_reports import evaluate_by_segment, create_backtest_report
from backtest_paths import (
    BACKTEST_BY_SEGMENT_PATH,
    BACKTEST_CALIBRATION_BINS_PATH,
    BACKTEST_DRAW_CALIBRATION_PATH,
    BACKTEST_PREDICTIONS_PATH,
    BACKTEST_PREDICTIONS_WITH_DRAW_FEATURES_PATH,
    BACKTEST_REPORT_PATH,
    BACKTEST_SUMMARY_PATH,
    BACKTEST_SUMMARY_WITH_DRAW_FEATURES_PATH,
    DRAW_FEATURE_COMPARISON_PATH,
    PROCESSED_DATA_DIR,
    WORLD_CUP_BACKTEST_PREDICTIONS_PATH,
    WORLD_CUP_BACKTEST_SUMMARY_PATH,
)
from calibration import create_confidence_calibration_bins, create_draw_calibration_table
from draw_hypothesis import recommend_draw_context_usage
from evaluation import calculate_prediction_metrics
from features import _empty_stats, _feature_row, _update_elo, _update_stats, build_training_dataset, categorize_tournament
from group_state import add_group_state_features
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


def _build_frozen_test_features(train_raw: pd.DataFrame, test_raw: pd.DataFrame, include_draw_context_features: bool = False) -> pd.DataFrame:
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
    test_source = test_raw.sort_values("date")
    if include_draw_context_features:
        combined = pd.concat([train_raw.assign(_is_test=False), test_raw.assign(_is_test=True)], ignore_index=True)
        enriched = add_group_state_features(combined)
        test_source = enriched[enriched["_is_test"]].sort_values("date")
    rows = []
    for _, match in test_source.iterrows():
        rows.append(
            {
                "date": match["date"],
                "home_team": match["home_team"],
                "away_team": match["away_team"],
                "result": match["result"],
                "tournament": match.get("tournament", "Unknown"),
                "stage": match.get("stage", pd.NA),
                "group": match.get("group", pd.NA),
                "matchday": match.get("matchday", match.get("group_matchday", 0)),
                "group_matchday": match.get("group_matchday", 0),
                "group_state_available": match.get("group_state_available", False),
                "home_must_win": match.get("home_must_win", False),
                "away_must_win": match.get("away_must_win", False),
                "one_team_must_win": match.get("one_team_must_win", False),
                "both_teams_need_win": match.get("both_teams_need_win", False),
                "home_draw_sufficient": match.get("home_draw_sufficient", False),
                "away_draw_sufficient": match.get("away_draw_sufficient", False),
                "both_teams_draw_satisfied": match.get("both_teams_draw_satisfied", False),
                **_feature_row(match, team_stats, elos),
            }
        )
    result = pd.DataFrame(rows)
    if include_draw_context_features:
        from draw_features import add_draw_context_features

        result = add_draw_context_features(result)
    return result


def _run_single_fold(fold_id: int, train_raw: pd.DataFrame, test_raw: pd.DataFrame, include_draw_context_features: bool = False) -> tuple[pd.DataFrame, dict]:
    with TemporaryDirectory() as tmpdir:
        training_df = build_training_dataset(
            train_raw,
            output_path=Path(tmpdir) / "training_dataset.csv",
            include_draw_context_features=include_draw_context_features,
        )
    model, _ = train_model_in_memory(training_df, include_draw_context_features=include_draw_context_features)
    test_features = _build_frozen_test_features(train_raw, test_raw, include_draw_context_features=include_draw_context_features)
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
    output["model_home_prob"] = output["pred_home_prob"]
    output["model_draw_prob"] = output["pred_draw_prob"]
    output["model_away_prob"] = output["pred_away_prob"]
    optional_columns = [
        "market_home_prob",
        "market_draw_prob",
        "market_away_prob",
        "fifa_ranking_gap",
        "elo_gap",
        "draw_context_label",
    ]
    for column in optional_columns:
        if column in test_raw.columns:
            output[column] = test_raw.sort_values("date")[column].reset_index(drop=True)
    metrics = calculate_prediction_metrics(output["actual_result"].tolist(), output[["pred_home_prob", "pred_draw_prob", "pred_away_prob"]].to_numpy())
    return output[PREDICTION_COLUMNS + [column for column in output.columns if column not in PREDICTION_COLUMNS]], metrics


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


def _save_backtest_outputs(
    predictions_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: Path,
    setup: dict,
    include_draw_context_features: bool = False,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_name = BACKTEST_PREDICTIONS_WITH_DRAW_FEATURES_PATH.name if include_draw_context_features else BACKTEST_PREDICTIONS_PATH.name
    summary_name = BACKTEST_SUMMARY_WITH_DRAW_FEATURES_PATH.name if include_draw_context_features else BACKTEST_SUMMARY_PATH.name
    predictions_path = output_dir / predictions_name
    summary_path = output_dir / summary_name
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
    include_draw_context_features: bool = False,
) -> dict:
    historical = _prepare_historical(historical_df)
    if historical.empty:
        return _save_backtest_outputs(
            pd.DataFrame(columns=PREDICTION_COLUMNS),
            pd.DataFrame(),
            Path(output_dir or PROCESSED_DATA_DIR),
            {"warning": "No valid historical matches"},
            include_draw_context_features=include_draw_context_features,
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
                predictions_df, metrics = _run_single_fold(fold_id, train_raw, test_raw, include_draw_context_features=include_draw_context_features)
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
        "include_draw_context_features": include_draw_context_features,
    }
    return _save_backtest_outputs(predictions, summary, Path(output_dir or PROCESSED_DATA_DIR), setup, include_draw_context_features=include_draw_context_features)


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


def _comparison_rows(result: dict, variant: str) -> list[dict]:
    rows = []
    segments = result.get("segments", pd.DataFrame())
    wanted = [
        ("overall", segments[(segments["segment_name"] == "Overall") & (segments["segment_value"] == "All")] if not segments.empty else pd.DataFrame()),
        ("major_tournament", segments[(segments["segment_name"] == "major_tournament") & (segments["segment_value"] == "major")] if not segments.empty else pd.DataFrame()),
        ("world_cup", segments[(segments["segment_name"] == "tournament_category") & (segments["segment_value"] == "world_cup")] if not segments.empty else pd.DataFrame()),
    ]
    for segment_name, segment_df in wanted:
        if segment_df.empty:
            continue
        row = segment_df.iloc[0]
        rows.append(
            {
                "model_variant": variant,
                "segment": segment_name,
                "match_count": int(row.get("match_count", 0)),
                "accuracy": row.get("accuracy", 0.0),
                "log_loss": row.get("log_loss", 0.0),
                "brier_score": row.get("brier_score", 0.0),
                "ece": row.get("ece", 0.0),
                "draw_calibration_gap": row.get("draw_calibration_gap", 0.0),
                "avg_pred_draw_prob": row.get("avg_pred_draw_prob", 0.0),
                "actual_draw_rate": row.get("actual_draw_rate", 0.0),
            }
        )
    return rows


def compare_baseline_vs_draw_context_model(historical_df: pd.DataFrame, backtest_config: dict) -> dict:
    from tempfile import TemporaryDirectory

    config = {
        "initial_train_end_date": backtest_config.get("initial_train_end_date", "2014-01-01"),
        "test_window": backtest_config.get("test_window", "365D"),
        "step_size": backtest_config.get("step_size", "365D"),
        "min_train_matches": int(backtest_config.get("min_train_matches", 1000)),
    }
    with TemporaryDirectory() as baseline_tmp:
        baseline = run_walk_forward_backtest(historical_df, output_dir=Path(baseline_tmp), include_draw_context_features=False, **config)
    draw_context = run_walk_forward_backtest(historical_df, output_dir=PROCESSED_DATA_DIR, include_draw_context_features=True, **config)
    comparison_df = pd.DataFrame(_comparison_rows(baseline, "baseline") + _comparison_rows(draw_context, "draw_context"))
    comparison_path = DRAW_FEATURE_COMPARISON_PATH
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(comparison_path, index=False)
    recommendation = recommend_draw_context_usage(comparison_df)
    return {
        "baseline": baseline,
        "draw_context": draw_context,
        "comparison": comparison_df,
        "recommendation": recommendation,
        "paths": {"comparison": comparison_path},
    }
