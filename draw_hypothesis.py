from __future__ import annotations

from math import sqrt
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from backtest_paths import (
    DRAW_FEATURE_COMPARISON_PATH,
    DRAW_HYPOTHESIS_BY_SEGMENT_PATH,
    DRAW_HYPOTHESIS_REPORT_PATH,
    DRAW_HYPOTHESIS_SUMMARY_PATH,
    GROUP_STATE_FEATURES_PATH,
    PROCESSED_DATA_DIR,
)
from draw_features import add_draw_context_features
from features import categorize_tournament
from group_state import add_group_state_features


def _result_column(df: pd.DataFrame) -> pd.Series:
    if "result" in df.columns:
        return df["result"]
    scores = df[["home_score", "away_score"]].apply(pd.to_numeric, errors="coerce")
    return scores.apply(lambda row: "H" if row["home_score"] > row["away_score"] else "A" if row["home_score"] < row["away_score"] else "D", axis=1)


def _ci(draw_count: int, match_count: int) -> tuple[float, float]:
    if match_count == 0:
        return 0.0, 0.0
    p = draw_count / match_count
    margin = 1.96 * sqrt(p * (1 - p) / match_count)
    return max(0.0, p - margin), min(1.0, p + margin)


def _segment_row(segment_name: str, segment_value, df: pd.DataFrame, baseline: float) -> dict:
    count = int(len(df))
    draws = int((df["result"] == "D").sum()) if count else 0
    rate = draws / count if count else 0.0
    low, high = _ci(draws, count)
    return {
        "segment_name": segment_name,
        "segment_value": str(segment_value),
        "match_count": count,
        "draw_count": draws,
        "draw_rate": rate,
        "baseline_draw_rate": baseline,
        "draw_rate_difference": rate - baseline,
        "confidence_interval_low": low,
        "confidence_interval_high": high,
    }


def _strength_bucket(value: float) -> str:
    value = abs(float(value or 0))
    if value <= 75:
        return "very_even"
    if value <= 150:
        return "slight_favorite"
    if value <= 250:
        return "moderate_favorite"
    return "heavy_favorite"


def prepare_draw_hypothesis_dataset(historical_df: pd.DataFrame) -> pd.DataFrame:
    df = historical_df.copy()
    df["date"] = pd.to_datetime(df.get("date"), errors="coerce", utc=True)
    df = df.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"])
    df["result"] = _result_column(df)
    df["tournament"] = df["tournament"].fillna("Unknown") if "tournament" in df.columns else "Unknown"
    df["neutral"] = df["neutral"].fillna(False).astype(bool) if "neutral" in df.columns else False
    df["tournament_category"] = df["tournament"].map(categorize_tournament)
    df["is_major_tournament"] = df["tournament_category"].isin({"world_cup", "euro", "copa_america", "afcon", "asian_cup", "gold_cup"})
    df = add_group_state_features(df)
    df = add_draw_context_features(df)
    df["strength_bucket"] = df.get("strength_gap_abs", pd.Series([0] * len(df))).map(_strength_bucket)
    return df


def run_draw_hypothesis_analysis(historical_df: pd.DataFrame, output_dir: Path | None = None) -> dict:
    output_dir = Path(output_dir or PROCESSED_DATA_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = prepare_draw_hypothesis_dataset(historical_df)
    baseline = float((df["result"] == "D").mean()) if not df.empty else 0.0
    rows = [_segment_row("overall", "all_matches", df, baseline)]

    segment_specs = [
        ("tournament_category", "tournament_category"),
        ("major_tournament_flag", "is_major_tournament"),
        ("world_cup_flag", "is_world_cup"),
        ("neutral_venue", "neutral"),
        ("group_stage_flag", "is_group_stage"),
        ("group_matchday", "group_matchday"),
        ("strength_difference_bucket", "strength_bucket"),
        ("both_teams_draw_satisfied", "both_teams_draw_satisfied"),
        ("one_team_must_win", "one_team_must_win"),
        ("both_teams_need_win", "both_teams_need_win"),
    ]
    df["is_world_cup"] = df["tournament_category"] == "world_cup"
    for segment_name, column in segment_specs:
        if column not in df.columns:
            continue
        series = df[column].fillna("unknown")
        for value, segment_df in df.groupby(series):
            value = "unknown" if value in {0, "0"} and column == "group_matchday" else value
            rows.append(_segment_row(segment_name, value, segment_df, baseline))

    segment_df = pd.DataFrame(rows)
    summary_df = pd.DataFrame(
        [
            {
                "metric": "overall_draw_rate",
                "value": baseline,
                "match_count": int(len(df)),
                "draw_count": int((df["result"] == "D").sum()) if not df.empty else 0,
            },
            {
                "metric": "group_metadata_available_rate",
                "value": float(df["is_group_stage"].mean()) if not df.empty else 0.0,
                "match_count": int(len(df)),
                "draw_count": 0,
            },
        ]
    )
    logistic = run_draw_logistic_regression_analysis(df)

    summary_path = output_dir / DRAW_HYPOTHESIS_SUMMARY_PATH.name
    segment_path = output_dir / DRAW_HYPOTHESIS_BY_SEGMENT_PATH.name
    group_state_path = output_dir / GROUP_STATE_FEATURES_PATH.name
    report_path = DRAW_HYPOTHESIS_REPORT_PATH if output_dir == PROCESSED_DATA_DIR else output_dir / DRAW_HYPOTHESIS_REPORT_PATH.name
    summary_df.to_csv(summary_path, index=False)
    segment_df.to_csv(segment_path, index=False)
    df.to_csv(group_state_path, index=False)
    _write_report(report_path, summary_df, segment_df, logistic)
    return {"summary": summary_df, "segments": segment_df, "logistic": logistic, "dataset": df, "paths": {"summary": summary_path, "segments": segment_path, "group_state_features": group_state_path, "report": report_path}}


def run_draw_logistic_regression_analysis(training_df: pd.DataFrame) -> dict:
    predictors = [
        "is_world_cup",
        "is_major_tournament",
        "is_group_stage",
        "group_matchday",
        "neutral",
        "strength_gap_abs",
        "both_teams_draw_satisfied",
        "one_team_must_win",
        "both_teams_need_win",
        "mutual_draw_acceptance",
    ]
    available = [column for column in predictors if column in training_df.columns]
    if len(training_df) < 30 or len(available) < 2 or training_df["result"].nunique() < 2:
        return {"available": False, "coefficients": {}, "odds_ratios": {}, "p_values": {}, "model_notes": "Not enough data for logistic coefficient analysis."}
    x = training_df[available].copy()
    for column in x.columns:
        if x[column].dtype == bool:
            x[column] = x[column].astype(int)
    y = (training_df["result"] == "D").astype(int)
    if y.nunique() < 2:
        return {"available": False, "coefficients": {}, "odds_ratios": {}, "p_values": {}, "model_notes": "Target has only one class."}
    model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("classifier", LogisticRegression(max_iter=1000))])
    model.fit(x, y)
    coefficients = dict(zip(available, model.named_steps["classifier"].coef_[0]))
    odds_ratios = {key: float(pd.Series([value]).map(lambda v: 2.718281828 ** v).iloc[0]) for key, value in coefficients.items()}
    return {"available": True, "coefficients": {k: float(v) for k, v in coefficients.items()}, "odds_ratios": odds_ratios, "p_values": {}, "model_notes": "Sklearn logistic regression coefficients; p-values not calculated."}


def recommend_draw_context_usage(comparison_df: pd.DataFrame) -> dict:
    if comparison_df.empty or "model_variant" not in comparison_df.columns:
        return {"recommended": False, "reason": "No comparison results available.", "caveats": ["Run the model comparison first."]}
    overall = comparison_df[comparison_df["segment"] == "overall"]
    if overall["model_variant"].nunique() < 2:
        return {"recommended": False, "reason": "Comparison is missing one model variant.", "caveats": ["Need both baseline and draw-context rows."]}
    baseline = overall[overall["model_variant"] == "baseline"].iloc[0]
    draw = overall[overall["model_variant"] == "draw_context"].iloc[0]
    caveats = []
    if int(draw.get("match_count", 0)) < 100:
        caveats.append("Sample size is below 100 matches.")
    log_loss_ok = draw["log_loss"] <= baseline["log_loss"] + 0.001
    brier_ok = draw["brier_score"] <= baseline["brier_score"] + 0.001
    draw_gap_ok = abs(draw["draw_calibration_gap"]) <= abs(baseline["draw_calibration_gap"]) + 0.01
    recommended = bool(log_loss_ok and brier_ok and draw_gap_ok and int(draw.get("match_count", 0)) >= 100)
    reason = "Draw-context model is not worse overall and draw calibration is acceptable." if recommended else "Draw-context model is not currently strong enough to recommend automatically."
    return {"recommended": recommended, "reason": reason, "caveats": caveats}


def _write_report(path: Path, summary_df: pd.DataFrame, segment_df: pd.DataFrame, logistic: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Draw Hypothesis Report",
        "",
        "Draw-context features are tested empirically. No manual draw bonus is applied.",
        "",
        "## Summary",
    ]
    for _, row in summary_df.iterrows():
        lines.append(f"- {row['metric']}: {row['value']:.4f}")
    lines.extend(["", "## Largest draw-rate differences"])
    highlights = segment_df.sort_values("draw_rate_difference", ascending=False).head(10)
    for _, row in highlights.iterrows():
        lines.append(f"- {row['segment_name']}={row['segment_value']}: n={int(row['match_count'])}, draw_rate={row['draw_rate']:.4f}, diff={row['draw_rate_difference']:.4f}")
    lines.extend(["", "## Logistic coefficient analysis", logistic.get("model_notes", "Unavailable."), "", "## Limitations", "- Group-state metadata may be incomplete.", "- Must-win and draw-sufficient flags are conservative approximations.", "- World Cup-only samples are small."])
    path.write_text("\n".join(lines))
