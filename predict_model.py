from pathlib import Path
from typing import Union

import joblib
import pandas as pd

from config import LIVE_PREDICTIONS_WITH_MODEL_PATH, MODEL_PATH
from features import FEATURE_COLUMNS, build_upcoming_feature_dataset


def load_trained_model(model_path: Union[str, Path] = MODEL_PATH):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError("No trained model found. Using market probabilities as model probabilities.")
    return joblib.load(model_path)


def prediction_file_uses_market_as_model(path: Union[str, Path]) -> bool:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        df = pd.read_csv(path)
    except Exception:
        return False
    model_columns = ["model_home_prob", "model_draw_prob", "model_away_prob"]
    market_columns = ["market_home_prob", "market_draw_prob", "market_away_prob"]
    if df.empty or not set(model_columns + market_columns).issubset(df.columns):
        return False
    model = df[model_columns].apply(pd.to_numeric, errors="coerce")
    market = df[market_columns].apply(pd.to_numeric, errors="coerce")
    comparable = model.notna().all(axis=1) & market.notna().all(axis=1)
    if not comparable.any():
        return False
    equal_rows = (model[comparable].round(8).to_numpy() == market[comparable].round(8).to_numpy()).all(axis=1)
    return bool(equal_rows.mean() >= 0.8)


def apply_stored_model_predictions(
    upcoming_df: pd.DataFrame,
    stored_model_path: Union[str, Path],
    output_path: Union[str, Path],
) -> tuple[pd.DataFrame, list[str]]:
    stored_model_path = Path(stored_model_path)
    if not stored_model_path.exists() or stored_model_path.stat().st_size == 0:
        raise FileNotFoundError("Stored model predictions are missing.")
    if prediction_file_uses_market_as_model(stored_model_path):
        raise ValueError("Stored model predictions use market fallback.")

    stored = pd.read_csv(stored_model_path)
    merge_columns = [
        "match_id",
        "model_home_prob",
        "model_draw_prob",
        "model_away_prob",
        "model_probability_source",
        "draw_context_score",
        "draw_context_label",
        "home_draw_utility",
        "away_draw_utility",
        "mutual_draw_acceptance",
        "one_team_must_win",
        "both_teams_draw_satisfied",
    ]
    available_columns = [column for column in merge_columns if column in stored.columns]
    result = upcoming_df.drop(columns=[column for column in available_columns if column != "match_id" and column in upcoming_df.columns], errors="ignore")
    result = result.merge(stored[available_columns], on="match_id", how="left")
    for column in ["model_home_prob", "model_draw_prob", "model_away_prob"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    missing_model = result[["model_home_prob", "model_draw_prob", "model_away_prob"]].isna().any(axis=1)
    if missing_model.any():
        raise ValueError("Stored model predictions do not cover all upcoming matches.")
    if "model_probability_source" not in result.columns:
        result["model_probability_source"] = "historical_model"
    result["model_probability_source"] = result["model_probability_source"].fillna("historical_model")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result, []


def predict_upcoming_matches(
    upcoming_df: pd.DataFrame,
    historical_df: pd.DataFrame,
    model_path: Union[str, Path] = MODEL_PATH,
    output_path: Union[str, Path] = LIVE_PREDICTIONS_WITH_MODEL_PATH,
    include_draw_context_features: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    warnings = []
    try:
        model = load_trained_model(model_path)
    except FileNotFoundError as exc:
        result = upcoming_df.copy()
        result["model_home_prob"] = result["market_home_prob"]
        result["model_draw_prob"] = result["market_draw_prob"]
        result["model_away_prob"] = result["market_away_prob"]
        return result, [str(exc)]

    model_uses_draw_context = bool(getattr(model, "include_draw_context_features_", False))
    use_draw_context = include_draw_context_features or model_uses_draw_context
    features = build_upcoming_feature_dataset(upcoming_df, historical_df, include_draw_context_features=use_draw_context)
    feature_columns = getattr(model, "feature_columns_", FEATURE_COLUMNS)
    missing_columns = [column for column in feature_columns if column not in features.columns]
    if missing_columns:
        for column in missing_columns:
            features[column] = 0
    probabilities = model.predict_proba(features[feature_columns])
    classes = list(model.named_steps["classifier"].classes_)
    class_index = {label: index for index, label in enumerate(classes)}
    result = upcoming_df.copy()
    result["model_home_prob"] = probabilities[:, class_index["H"]]
    result["model_draw_prob"] = probabilities[:, class_index["D"]]
    result["model_away_prob"] = probabilities[:, class_index["A"]]
    result["model_probability_source"] = "historical_model"
    totals = result[["model_home_prob", "model_draw_prob", "model_away_prob"]].sum(axis=1)
    result["model_home_prob"] = result["model_home_prob"] / totals
    result["model_draw_prob"] = result["model_draw_prob"] / totals
    result["model_away_prob"] = result["model_away_prob"] / totals
    for column in ["draw_context_score", "draw_context_label", "mutual_draw_acceptance", "both_teams_draw_satisfied", "one_team_must_win"]:
        if column in features.columns:
            result[column] = features[column].values
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result, warnings
