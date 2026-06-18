from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

import config as app_config
from evaluation import calculate_prediction_metrics
from features import build_training_dataset
from fifa_rankings import create_fifa_feature_coverage, load_fifa_rankings
from train_model import _split_train_test, train_model_in_memory


FIFA_FEATURE_COVERAGE_PATH = getattr(
    app_config,
    "FIFA_FEATURE_COVERAGE_PATH",
    app_config.PROCESSED_DATA_DIR / "fifa_feature_coverage.csv",
)
FIFA_RANKING_FEATURE_REPORT_PATH = getattr(
    app_config,
    "FIFA_RANKING_FEATURE_REPORT_PATH",
    app_config.REPORTS_DIR / "fifa_ranking_feature_report.md",
)
MODEL_VARIANT_COMPARISON_PATH = app_config.MODEL_VARIANT_COMPARISON_PATH


MODEL_VARIANTS = {
    "baseline_no_strength": {"includes_elo": False, "includes_fifa_ranking": False},
    "elo_only": {"includes_elo": True, "includes_fifa_ranking": False},
    "fifa_only": {"includes_elo": False, "includes_fifa_ranking": True},
    "elo_plus_fifa": {"includes_elo": True, "includes_fifa_ranking": True},
}


def _empty_comparison() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "model_variant",
            "includes_elo",
            "includes_fifa_ranking",
            "match_count",
            "accuracy",
            "log_loss",
            "brier_score",
            "ece",
            "draw_calibration_gap",
            "selected",
            "selection_reason",
        ]
    )


def select_model_variant(comparison_df: pd.DataFrame) -> dict:
    if comparison_df.empty:
        return {"model_variant": "elo_only", "selected_reason": "No variant comparison was available.", "includes_elo": True, "includes_fifa_ranking": False}
    candidates = comparison_df.copy()
    candidates = candidates[candidates["match_count"].fillna(0).astype(int) > 0]
    if candidates.empty:
        return {"model_variant": "elo_only", "selected_reason": "No variant had test matches.", "includes_elo": True, "includes_fifa_ranking": False}

    elo = candidates[candidates["model_variant"] == "elo_only"]
    plus = candidates[candidates["model_variant"] == "elo_plus_fifa"]
    fifa = candidates[candidates["model_variant"] == "fifa_only"]
    baseline = candidates[candidates["model_variant"] == "baseline_no_strength"]
    selected = elo.iloc[0] if not elo.empty else candidates.sort_values(["log_loss", "brier_score", "ece"]).iloc[0]
    reason = "Selected elo_only as the baseline team-strength model."

    if not plus.empty and not elo.empty:
        plus_row = plus.iloc[0]
        elo_row = elo.iloc[0]
        if plus_row["log_loss"] < elo_row["log_loss"] and plus_row["brier_score"] < elo_row["brier_score"]:
            selected = plus_row
            reason = "Selected elo_plus_fifa because it improved both log loss and Brier score versus elo_only."
        else:
            reason = "Kept elo_only because FIFA ranking did not improve both log loss and Brier score versus Elo."
    if not fifa.empty:
        fifa_row = fifa.iloc[0]
        if fifa_row["log_loss"] < selected["log_loss"] and fifa_row["brier_score"] <= selected["brier_score"]:
            selected = fifa_row
            reason = "Selected fifa_only because it beat the current selected model on log loss without worsening Brier score."
    if not baseline.empty:
        base_row = baseline.iloc[0]
        if base_row["log_loss"] < selected["log_loss"] and base_row["brier_score"] <= selected["brier_score"]:
            selected = base_row
            reason = "Selected baseline_no_strength because extra strength signals did not improve probability metrics."

    return {
        "model_variant": selected["model_variant"],
        "selected_reason": reason,
        "includes_elo": bool(selected["includes_elo"]),
        "includes_fifa_ranking": bool(selected["includes_fifa_ranking"]),
    }


def compare_model_variants(
    historical_df: pd.DataFrame,
    fifa_rankings_df: pd.DataFrame,
    test_start_date: str | None = None,
    output_path: Path = MODEL_VARIANT_COMPARISON_PATH,
    coverage_path: Path = FIFA_FEATURE_COVERAGE_PATH,
    report_path: Path = FIFA_RANKING_FEATURE_REPORT_PATH,
    allow_small_dataset: bool = False,
) -> dict:
    if len(historical_df) < 100 and not allow_small_dataset:
        comparison = _empty_comparison()
        recommendation = select_model_variant(comparison)
        _write_report(report_path, comparison, pd.DataFrame(), recommendation, ["Dataset is too small for production-ready ablation."])
        return {"comparison": comparison, "coverage": pd.DataFrame(), "recommendation": recommendation, "warnings": ["Dataset is too small for production-ready ablation."]}

    warnings = []
    rows = []
    coverage = pd.DataFrame()
    for variant, settings in MODEL_VARIANTS.items():
        with TemporaryDirectory() as tmpdir:
            training_df = build_training_dataset(
                historical_df,
                output_path=Path(tmpdir) / f"{variant}.csv",
                include_fifa_ranking_features=settings["includes_fifa_ranking"],
                fifa_rankings_df=fifa_rankings_df,
            )
        if settings["includes_fifa_ranking"] and coverage.empty:
            coverage = create_fifa_feature_coverage(training_df)
        train_df, test_df = _split_train_test(training_df, test_start_date)
        if train_df.empty or test_df.empty or train_df["result"].nunique() < 3:
            rows.append(_variant_row(variant, settings, 0, {}))
            continue
        feature_train = train_df.copy()
        feature_test = test_df.copy()
        model, _ = train_model_in_memory(
            feature_train,
            include_fifa_ranking_features=settings["includes_fifa_ranking"],
            include_elo_features=settings["includes_elo"],
        )
        labels = list(model.named_steps["classifier"].classes_)
        probabilities = model.predict_proba(feature_test[model.feature_columns_])
        metrics = calculate_prediction_metrics(
            feature_test["result"].tolist(),
            probabilities,
            labels,
        )
        rows.append(_variant_row(variant, settings, len(feature_test), metrics))

    comparison = pd.DataFrame(rows)
    recommendation = select_model_variant(comparison)
    comparison["selected"] = comparison["model_variant"].eq(recommendation["model_variant"])
    comparison["selection_reason"] = comparison["model_variant"].map(
        lambda value: recommendation["selected_reason"] if value == recommendation["model_variant"] else ""
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_path, index=False)
    coverage_path.parent.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(coverage_path, index=False)
    _write_report(report_path, comparison, coverage, recommendation, warnings)
    return {"comparison": comparison, "coverage": coverage, "recommendation": recommendation, "warnings": warnings}


def _variant_row(variant: str, settings: dict, match_count: int, metrics: dict) -> dict:
    return {
        "model_variant": variant,
        "includes_elo": bool(settings["includes_elo"]),
        "includes_fifa_ranking": bool(settings["includes_fifa_ranking"]),
        "match_count": int(match_count),
        "accuracy": float(metrics.get("accuracy", 0.0)),
        "log_loss": float(metrics.get("log_loss", 0.0)),
        "brier_score": float(metrics.get("brier_score", 0.0)),
        "ece": float(metrics.get("ece", 0.0)),
        "draw_calibration_gap": float(metrics.get("draw_calibration_gap", 0.0)),
        "selected": False,
        "selection_reason": "",
    }


def _write_report(path: Path, comparison: pd.DataFrame, coverage: pd.DataFrame, recommendation: dict, warnings: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ranking_available = not coverage.empty and int(coverage.get("matches_with_ranking", pd.Series(dtype=int)).sum()) > 0
    lines = [
        "# FIFA Ranking Feature Report",
        "",
        f"- FIFA ranking data available: {'yes' if ranking_available else 'no'}",
        f"- Selected variant: {recommendation.get('model_variant')}",
        f"- Recommendation: {recommendation.get('selected_reason')}",
        "- Warning: FIFA ranking and Elo are correlated team-strength signals. FIFA ranking is only used if backtesting shows it adds predictive value.",
    ]
    if warnings:
        lines.extend(["", "## Warnings", *[f"- {warning}" for warning in warnings]])
    if not coverage.empty:
        total_missing = coverage["matches_missing_ranking"].sum()
        total_matches = coverage["matches_with_ranking"].sum() + total_missing
        missing_rate = total_missing / total_matches if total_matches else 0.0
        lines.extend(["", "## Coverage", f"- Missing rate: {missing_rate:.2%}", f"- Teams covered: {int((coverage['matches_with_ranking'] > 0).sum())}"])
    if not comparison.empty:
        lines.extend(["", "## Variants Tested"])
        for _, row in comparison.iterrows():
            lines.append(
                f"- {row['model_variant']}: n={int(row['match_count'])}, log_loss={row['log_loss']:.4f}, "
                f"brier={row['brier_score']:.4f}, ece={row['ece']:.4f}"
            )
    path.write_text("\n".join(lines) + "\n")


def run_fifa_ablation_from_files(historical_df: pd.DataFrame, rankings_path: Path, **kwargs) -> dict:
    rankings_df, warnings = load_fifa_rankings(rankings_path)
    result = compare_model_variants(historical_df, rankings_df, **kwargs)
    result["warnings"] = warnings + result.get("warnings", [])
    return result
