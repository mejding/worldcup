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

from config import FEATURE_COLUMNS_PATH, MODEL_METADATA_PATH, MODEL_PATH
from evaluation import (
    calculate_accuracy,
    calculate_log_loss,
    calculate_multiclass_brier_score,
    calculate_prediction_metrics,
)
from features import FEATURE_COLUMNS, build_training_dataset
from historical_data import load_historical_results, standardize_historical_results, validate_historical_results


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


def _build_model_pipeline() -> Pipeline:
    categorical_features = ["tournament_category"]
    numeric_features = [column for column in FEATURE_COLUMNS if column not in categorical_features]
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


def train_model_in_memory(training_df: pd.DataFrame) -> tuple[Pipeline, dict]:
    if len(training_df) < 30:
        raise ValueError("Too little historical data to train a model. Add at least 30 matches.")
    if training_df["result"].nunique() < 3:
        raise ValueError("Training data must contain home wins, draws and away wins.")
    model = _build_model_pipeline()
    model.fit(training_df[FEATURE_COLUMNS], training_df["result"])
    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "number_of_training_rows": int(len(training_df)),
        "date_min": str(training_df["date"].min()) if "date" in training_df.columns else None,
        "date_max": str(training_df["date"].max()) if "date" in training_df.columns else None,
        "feature_columns": FEATURE_COLUMNS,
        "target_classes": list(model.named_steps["classifier"].classes_),
    }
    return model, metadata


def predict_with_model(model, feature_df: pd.DataFrame) -> pd.DataFrame:
    if feature_df.empty:
        return pd.DataFrame(columns=["pred_home_prob", "pred_draw_prob", "pred_away_prob", "predicted_result", "confidence"])
    probabilities = model.predict_proba(feature_df[FEATURE_COLUMNS])
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
) -> dict:
    if len(training_df) < 30:
        raise ValueError("Too little historical data to train a model. Add at least 30 matches.")

    train_df, test_df = _split_train_test(training_df, test_start_date)
    if train_df.empty or test_df.empty:
        raise ValueError("Train/test split produced an empty train or test set.")
    if train_df["result"].nunique() < 3:
        raise ValueError("Training data must contain home wins, draws and away wins.")

    model = _build_model_pipeline()

    x_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["result"]
    x_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["result"]
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)
    labels = list(model.named_steps["classifier"].classes_)
    draw_index = labels.index("D") if "D" in labels else 0
    metrics = calculate_prediction_metrics(y_test, y_proba, labels)
    metrics["draw_rate_actual"] = float((y_test == "D").mean())
    metrics["draw_rate_predicted"] = float(y_proba[:, draw_index].mean())

    model_output_path = Path(model_output_path)
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_output_path)
    FEATURE_COLUMNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEATURE_COLUMNS_PATH.write_text(json.dumps(FEATURE_COLUMNS, indent=2))
    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "number_of_training_rows": int(len(train_df)),
        "number_of_test_rows": int(len(test_df)),
        "date_min": str(training_df["date"].min()),
        "date_max": str(training_df["date"].max()),
        "feature_columns": FEATURE_COLUMNS,
        "target_classes": labels,
        "metrics": metrics,
    }
    MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    return metadata


def train_from_historical_csv(input_path: Union[str, Path], test_start_date: str = None) -> dict:
    raw = load_historical_results(input_path)
    warnings, errors = validate_historical_results(raw)
    if errors:
        raise ValueError("; ".join(errors))
    standardized = standardize_historical_results(raw)
    training_df = build_training_dataset(standardized)
    metadata = train_historical_model(training_df, test_start_date=test_start_date)
    metadata["warnings"] = warnings
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/historical/international_results.csv")
    parser.add_argument("--test-start-date", default=None)
    args = parser.parse_args()
    metadata = train_from_historical_csv(args.input, args.test_start_date)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
