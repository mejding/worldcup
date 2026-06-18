from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config as app_config
from evaluation import (
    calculate_accuracy,
    calculate_log_loss,
    calculate_multiclass_brier_score,
    calculate_prediction_metrics,
)
from features import FEATURE_COLUMNS, build_training_dataset, get_feature_columns
from fifa_rankings import load_fifa_rankings
from historical_data import load_historical_results, standardize_historical_results, validate_historical_results
from model_readiness import PRODUCTION_MIN_TEST_ROWS, PRODUCTION_MIN_TRAINING_ROWS


FEATURE_COLUMNS_PATH = app_config.FEATURE_COLUMNS_PATH
FIFA_RANKINGS_PATH = getattr(
    app_config,
    "FIFA_RANKINGS_PATH",
    app_config.REFERENCE_DATA_DIR / "fifa_rankings.csv",
)
MODEL_METADATA_PATH = app_config.MODEL_METADATA_PATH
MODEL_PATH = app_config.MODEL_PATH


def _split_train_test(training_df: pd.DataFrame, test_start_date: str = None):
    df = training_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df = df.sort_values("date").dropna(subset=["date"])
    if test_start_date:
        cutoff = pd.to_datetime(test_start_date, utc=True)
        train_df = df[df["date"] < cutoff]
        test_df = df[df["date"] >= cutoff]
    else:
        split_index = int(len(df) * 0.8)
        train_df = df.iloc[:split_index]
        test_df = df.iloc[split_index:]
    return train_df, test_df


def _build_model_pipeline(feature_columns=None) -> Pipeline:
    feature_columns = feature_columns or FEATURE_COLUMNS
    categorical_features = ["tournament_category"]
    numeric_features = [column for column in feature_columns if column not in categorical_features]
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]),
                categorical_features,
            ),
        ]
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )


def train_model_in_memory(
    training_df: pd.DataFrame,
    include_draw_context_features: bool = False,
    include_fifa_ranking_features: bool = False,
    include_elo_features: bool = True,
) -> tuple[Pipeline, dict]:
    if len(training_df) < 30:
        raise ValueError("Too little historical data to train a model. Add at least 30 matches.")
    if training_df["result"].nunique() < 3:
        raise ValueError("Training data must contain home wins, draws and away wins.")
    feature_columns = get_feature_columns(
        include_draw_context_features,
        include_fifa_ranking_features=include_fifa_ranking_features,
        include_elo_features=include_elo_features,
    )
    model = _build_model_pipeline(feature_columns)
    model.fit(training_df[feature_columns], training_df["result"])
    model.feature_columns_ = feature_columns
    model.include_draw_context_features_ = include_draw_context_features
    model.include_fifa_ranking_features_ = include_fifa_ranking_features
    model.include_elo_features_ = include_elo_features
    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "number_of_training_rows": int(len(training_df)),
        "date_min": str(training_df["date"].min()) if "date" in training_df.columns else None,
        "date_max": str(training_df["date"].max()) if "date" in training_df.columns else None,
        "feature_columns": feature_columns,
        "include_draw_context_features": include_draw_context_features,
        "include_fifa_ranking_features": include_fifa_ranking_features,
        "include_elo_features": include_elo_features,
        "target_classes": list(model.named_steps["classifier"].classes_),
    }
    return model, metadata


def predict_with_model(model, feature_df: pd.DataFrame) -> pd.DataFrame:
    if feature_df.empty:
        return pd.DataFrame(columns=["pred_home_prob", "pred_draw_prob", "pred_away_prob", "predicted_result", "confidence"])
    feature_columns = getattr(model, "feature_columns_", FEATURE_COLUMNS)
    probabilities = model.predict_proba(feature_df[feature_columns])
    labels = list(model.named_steps["classifier"].classes_)
    result = pd.DataFrame(0.0, index=feature_df.index, columns=["pred_home_prob", "pred_draw_prob", "pred_away_prob"])
    for label, column in [("H", "pred_home_prob"), ("D", "pred_draw_prob"), ("A", "pred_away_prob")]:
        if label in labels:
            result[column] = probabilities[:, labels.index(label)]
    label_columns = {"H": "pred_home_prob", "D": "pred_draw_prob", "A": "pred_away_prob"}
    result["predicted_result"] = result[list(label_columns.values())].idxmax(axis=1).map(
        {value: key for key, value in label_columns.items()}
    )
    result["confidence"] = result[list(label_columns.values())].max(axis=1)
    return result


def train_historical_model(
    training_df: pd.DataFrame,
    test_start_date: str = None,
    model_output_path: Union[str, Path] = MODEL_PATH,
    include_draw_context_features: bool = False,
    include_fifa_ranking_features: bool = False,
    include_elo_features: bool = True,
    fifa_ranking_metadata: dict | None = None,
    model_variant: str = "elo_only",
    selected_reason: str = "Selected by configured training command.",
    allow_demo_model: bool = True,
    training_data_source: str = "historical_international_results",
) -> dict:
    if len(training_df) < 30:
        raise ValueError("Too little historical data to train a model. Add at least 30 matches.")

    train_df, test_df = _split_train_test(training_df, test_start_date)
    if train_df.empty or test_df.empty:
        raise ValueError("Train/test split produced an empty train or test set.")
    if train_df["result"].nunique() < 3:
        raise ValueError("Training data must contain home wins, draws and away wins.")

    feature_columns = get_feature_columns(
        include_draw_context_features,
        include_fifa_ranking_features=include_fifa_ranking_features,
        include_elo_features=include_elo_features,
    )
    model = _build_model_pipeline(feature_columns)

    x_train = train_df[feature_columns]
    y_train = train_df["result"]
    x_test = test_df[feature_columns]
    y_test = test_df["result"]
    model.fit(x_train, y_train)
    model.feature_columns_ = feature_columns
    model.include_draw_context_features_ = include_draw_context_features
    model.include_fifa_ranking_features_ = include_fifa_ranking_features
    model.include_elo_features_ = include_elo_features
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)
    labels = list(model.named_steps["classifier"].classes_)
    draw_index = labels.index("D") if "D" in labels else 0
    metrics = calculate_prediction_metrics(y_test, y_proba, labels)
    metrics["draw_rate_actual"] = float((y_test == "D").mean())
    metrics["draw_rate_predicted"] = float(y_proba[:, draw_index].mean())
    is_demo_model = len(train_df) < PRODUCTION_MIN_TRAINING_ROWS or len(test_df) < PRODUCTION_MIN_TEST_ROWS
    if is_demo_model and not allow_demo_model:
        raise ValueError(
            "Training data volume is too small for a production model. "
            f"Need at least {PRODUCTION_MIN_TRAINING_ROWS} training rows and {PRODUCTION_MIN_TEST_ROWS} test rows."
        )

    model_output_path = Path(model_output_path)
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_output_path)
    FEATURE_COLUMNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEATURE_COLUMNS_PATH.write_text(json.dumps(feature_columns, indent=2))
    trained_at = datetime.now(timezone.utc).isoformat()
    training_start = training_df["date"].min()
    training_end = training_df["date"].max()
    training_year_span = (training_end - training_start).days / 365.25 if pd.notna(training_start) and pd.notna(training_end) else 0.0
    metadata = {
        "model_version": trained_at,
        "trained_at_utc": trained_at,
        "training_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "feature_count": len(feature_columns),
        "training_data_source": training_data_source,
        "training_data_start_date": str(training_start),
        "training_data_end_date": str(training_end),
        "training_year_span": float(training_year_span),
        "is_demo_model": is_demo_model,
        "model_variant": model_variant,
        "selected_reason": selected_reason,
        "includes_elo_features": "elo_diff" in feature_columns,
        "includes_fifa_ranking_features": include_fifa_ranking_features,
        "includes_form_features": "home_points_per_match_last5" in feature_columns,
        "includes_tournament_features": "is_world_cup" in feature_columns,
        "includes_neutral_venue": "neutral" in feature_columns,
        "includes_qualifiers": (
            "is_qualifier" in feature_columns
            and "is_qualifier" in training_df.columns
            and bool(training_df["is_qualifier"].fillna(False).any())
        ),
        "includes_world_cup_or_major_tournaments": (
            (
                "is_world_cup" in feature_columns
                and "is_world_cup" in training_df.columns
                and bool(training_df["is_world_cup"].fillna(False).any())
            )
            or (
                "is_major_tournament" in feature_columns
                and "is_major_tournament" in training_df.columns
                and bool(training_df["is_major_tournament"].fillna(False).any())
            )
        ),
        "includes_schedule_features": include_draw_context_features,
        "performance_accuracy": metrics.get("accuracy"),
        "performance_log_loss": metrics.get("log_loss"),
        "performance_brier_score": metrics.get("brier_score"),
        "performance_ece": metrics.get("ece"),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "number_of_training_rows": int(len(train_df)),
        "number_of_test_rows": int(len(test_df)),
        "date_min": str(training_df["date"].min()),
        "date_max": str(training_df["date"].max()),
        "feature_columns": feature_columns,
        "include_draw_context_features": include_draw_context_features,
        "include_fifa_ranking_features": include_fifa_ranking_features,
        "include_elo_features": include_elo_features,
        "target_classes": labels,
        "metrics": metrics,
    }
    if fifa_ranking_metadata:
        metadata.update(fifa_ranking_metadata)
    MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    return metadata


def train_from_historical_csv(
    input_path: Union[str, Path],
    test_start_date: str = None,
    include_draw_context_features: bool = False,
    include_fifa_ranking_features: bool = False,
    include_elo_features: bool = True,
    fifa_rankings_path: Union[str, Path] = FIFA_RANKINGS_PATH,
    model_variant: str = "elo_only",
    selected_reason: str = "Selected by configured training command.",
    allow_demo_model: bool = True,
) -> dict:
    raw = load_historical_results(input_path)
    warnings, errors = validate_historical_results(raw)
    if errors:
        raise ValueError("; ".join(errors))
    standardized = standardize_historical_results(raw)
    rankings_df, ranking_warnings = load_fifa_rankings(fifa_rankings_path)
    ranking_metadata = {
        "fifa_ranking_rows": int(len(rankings_df)),
        "fifa_ranking_start_date": None if rankings_df.empty else str(rankings_df["ranking_date"].min()),
        "fifa_ranking_end_date": None if rankings_df.empty else str(rankings_df["ranking_date"].max()),
        "fifa_ranking_source": None if rankings_df.empty or "source" not in rankings_df.columns else "; ".join(sorted(set(rankings_df["source"].dropna().astype(str))))[:200],
        "fifa_rank_missing_rate": None,
    }
    training_df = build_training_dataset(
        standardized,
        include_draw_context_features=include_draw_context_features,
        include_fifa_ranking_features=include_fifa_ranking_features,
        fifa_rankings_df=rankings_df,
    )
    if include_fifa_ranking_features and "home_fifa_rank_missing" in training_df.columns:
        ranking_metadata["fifa_rank_missing_rate"] = float(
            pd.concat([training_df["home_fifa_rank_missing"], training_df["away_fifa_rank_missing"]])
            .fillna(True)
            .astype(bool)
            .mean()
        )
    metadata = train_historical_model(
        training_df,
        test_start_date=test_start_date,
        include_draw_context_features=include_draw_context_features,
        include_fifa_ranking_features=include_fifa_ranking_features,
        include_elo_features=include_elo_features,
        fifa_ranking_metadata=ranking_metadata,
        model_variant=model_variant,
        selected_reason=selected_reason,
        allow_demo_model=allow_demo_model,
        training_data_source="historical_international_results",
    )
    metadata["warnings"] = warnings + ranking_warnings
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/historical/international_results.csv")
    parser.add_argument("--test-start-date", default=None)
    parser.add_argument("--include-draw-context-features", action="store_true")
    parser.add_argument("--include-fifa-ranking", action="store_true")
    parser.add_argument("--no-elo", action="store_true")
    parser.add_argument("--fifa-rankings-path", default=str(FIFA_RANKINGS_PATH))
    args = parser.parse_args()
    metadata = train_from_historical_csv(
        args.input,
        args.test_start_date,
        include_draw_context_features=args.include_draw_context_features,
        include_fifa_ranking_features=args.include_fifa_ranking,
        include_elo_features=not args.no_elo,
        fifa_rankings_path=args.fifa_rankings_path,
        model_variant="elo_plus_fifa" if args.include_fifa_ranking and not args.no_elo else "fifa_only" if args.include_fifa_ranking else "baseline_no_strength" if args.no_elo else "elo_only",
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
