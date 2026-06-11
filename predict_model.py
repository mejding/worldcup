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
