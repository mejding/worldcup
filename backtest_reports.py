from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from backtest_paths import BACKTEST_REPORT_PATH
from evaluation import calculate_prediction_metrics


PROBA_COLUMNS = ["pred_home_prob", "pred_draw_prob", "pred_away_prob"]
LABELS = ["H", "D", "A"]


def _metrics_for_predictions(df: pd.DataFrame) -> dict:
    if df.empty:
        return calculate_prediction_metrics([], [], LABELS)
    return calculate_prediction_metrics(df["actual_result"].tolist(), df[PROBA_COLUMNS].to_numpy(), LABELS)


def _segment_row(segment_name: str, segment_value, df: pd.DataFrame) -> dict:
    metrics = _metrics_for_predictions(df)
    return {
        "segment_name": segment_name,
        "segment_value": str(segment_value),
        "match_count": int(len(df)),
        **metrics,
    }


def evaluate_by_segment(predictions_df: pd.DataFrame) -> pd.DataFrame:
    rows = [_segment_row("Overall", "All", predictions_df)]
    if predictions_df.empty:
        return pd.DataFrame(rows)

    if "tournament_category" in predictions_df.columns:
        for value, segment_df in predictions_df.groupby(predictions_df["tournament_category"].fillna("unknown")):
            rows.append(_segment_row("tournament_category", value, segment_df))

    if "neutral" in predictions_df.columns:
        for value, segment_df in predictions_df.groupby(predictions_df["neutral"].fillna(False).astype(bool)):
            rows.append(_segment_row("neutral", value, segment_df))

    if "tournament_category" in predictions_df.columns:
        major_categories = {"world_cup", "euro", "copa_america", "afcon", "asian_cup", "gold_cup"}
        major_mask = predictions_df["tournament_category"].isin(major_categories)
        rows.append(_segment_row("major_tournament", "major", predictions_df[major_mask]))
        rows.append(_segment_row("major_tournament", "non_major", predictions_df[~major_mask]))

    dates = pd.to_datetime(predictions_df["date"], errors="coerce", utc=True)
    if dates.notna().any():
        year_df = predictions_df.assign(year=dates.dt.year)
        for value, segment_df in year_df.groupby("year"):
            rows.append(_segment_row("year", int(value), segment_df))

    draw_bins = pd.cut(
        predictions_df["pred_draw_prob"],
        bins=[0.0, 0.20, 0.25, 0.30, 0.35, 0.40, 1.0],
        include_lowest=True,
        right=False,
    )
    for value, segment_df in predictions_df.groupby(draw_bins, observed=False):
        if pd.isna(value):
            continue
        rows.append(_segment_row("draw_probability_bucket", value, segment_df))

    confidence_bins = pd.cut(
        predictions_df["confidence"],
        bins=[0.0, 0.40, 0.50, 0.60, 0.70, 0.80, 1.0],
        include_lowest=True,
        right=False,
    )
    for value, segment_df in predictions_df.groupby(confidence_bins, observed=False):
        if pd.isna(value):
            continue
        rows.append(_segment_row("confidence_bucket", value, segment_df))

    return pd.DataFrame(rows)


def create_backtest_report(
    predictions_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    segment_df: pd.DataFrame,
    draw_calibration_df: pd.DataFrame,
    calibration_bins_df: pd.DataFrame,
    output_path: Path = BACKTEST_REPORT_PATH,
    setup: dict | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    setup = setup or {}
    overall = segment_df[
        (segment_df["segment_name"] == "Overall") & (segment_df["segment_value"] == "All")
    ].head(1)
    overall_row = overall.iloc[0].to_dict() if not overall.empty else {}
    date_values = pd.to_datetime(predictions_df["date"], errors="coerce", utc=True) if not predictions_df.empty else pd.Series(dtype="datetime64[ns, UTC]")
    lines = [
        "# Backtest Report",
        "",
        f"Run timestamp: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Dataset",
        "",
        f"- Backtest predictions: {len(predictions_df)}",
        f"- Date range: {date_values.min() if not date_values.empty else '-'} to {date_values.max() if not date_values.empty else '-'}",
        f"- Folds: {len(summary_df)}",
        "",
        "## Backtest setup",
        "",
    ]
    for key, value in setup.items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Overall metrics",
            "",
            f"- Accuracy: {overall_row.get('accuracy', 0):.4f}",
            f"- Log loss: {overall_row.get('log_loss', 0):.4f}",
            f"- Brier score: {overall_row.get('brier_score', 0):.4f}",
            f"- ECE: {overall_row.get('ece', 0):.4f}",
            f"- Draw calibration gap: {overall_row.get('draw_calibration_gap', 0):.4f}",
            "",
            "## Segment highlights",
            "",
        ]
    )
    highlight = segment_df.sort_values(["match_count", "accuracy"], ascending=[False, False]).head(10)
    for _, row in highlight.iterrows():
        lines.append(
            f"- {row['segment_name']}={row['segment_value']}: n={int(row['match_count'])}, "
            f"accuracy={row['accuracy']:.4f}, log_loss={row['log_loss']:.4f}"
        )
    lines.extend(["", "## Draw calibration summary", ""])
    for _, row in draw_calibration_df.iterrows():
        lines.append(
            f"- {row['bin_label']}: n={int(row['count'])}, predicted={row['avg_predicted_draw_probability']:.4f}, "
            f"actual={row['actual_draw_rate']:.4f}, gap={row['calibration_gap']:.4f}"
        )
    lines.extend(
        [
            "",
            "## Confidence calibration",
            "",
            f"- Confidence bins: {len(calibration_bins_df)}",
            "",
            "## Limitations",
            "",
            "- This backtest currently evaluates the historical model only. Market-aware ensemble will be added in a later sprint.",
            "- Time-based walk-forward backtesting is used to avoid random-split leakage.",
            "- World Cup-only samples are small and should be treated as sanity checks, not definitive model evidence.",
            "- Historical odds are not included in this sprint, so staking profitability is not evaluated here.",
        ]
    )
    output_path.write_text("\n".join(lines))
    return output_path
