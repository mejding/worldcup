from __future__ import annotations

from pathlib import Path

import pandas as pd

from backtest_paths import ENSEMBLE_PREDICTIONS_PATH
from config import DEFAULT_ENSEMBLE_W_MARKET


MARKET_PROB_COLUMNS = {"home": "market_home_prob", "draw": "market_draw_prob", "away": "market_away_prob"}
MODEL_PROB_COLUMNS = {"home": "model_home_prob", "draw": "model_draw_prob", "away": "model_away_prob"}
DRAW_MODEL_PROB_COLUMNS = {"home": "draw_model_home_prob", "draw": "draw_model_draw_prob", "away": "draw_model_away_prob"}


def validate_probability_triplet(home, draw, away) -> bool:
    values = pd.Series([home, draw, away], dtype="float64")
    if values.isna().any() or (values < 0).any():
        return False
    return abs(float(values.sum()) - 1.0) <= 0.001


def normalize_probability_triplet(home, draw, away) -> tuple[float, float, float]:
    values = pd.Series([home, draw, away], dtype="float64").fillna(0).clip(lower=0)
    total = float(values.sum())
    if total <= 0:
        return 1 / 3, 1 / 3, 1 / 3
    values = values / total
    return float(values.iloc[0]), float(values.iloc[1]), float(values.iloc[2])


def create_weight_grid(market_weights: list[float] | None = None) -> pd.DataFrame:
    market_weights = market_weights or [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    return pd.DataFrame([{"w_market": float(weight), "w_model": float(1 - weight)} for weight in market_weights])


def _has_columns(df: pd.DataFrame, columns: dict) -> bool:
    return all(column in df.columns for column in columns.values())


def calculate_ensemble_probabilities(
    df: pd.DataFrame,
    w_market: float,
    model_prob_columns: dict,
    market_prob_columns: dict,
) -> pd.DataFrame:
    result = df.copy()
    w_market = max(0.0, min(float(w_market), 1.0))
    w_model = 1.0 - w_market
    missing = [column for column in list(model_prob_columns.values()) + list(market_prob_columns.values()) if column not in result.columns]
    if missing:
        raise ValueError("Missing probability columns: " + ", ".join(missing))
    for outcome in ["home", "draw", "away"]:
        result[f"ensemble_{outcome}_prob"] = (
            w_market * pd.to_numeric(result[market_prob_columns[outcome]], errors="coerce")
            + w_model * pd.to_numeric(result[model_prob_columns[outcome]], errors="coerce")
        )
    normalized = result[["ensemble_home_prob", "ensemble_draw_prob", "ensemble_away_prob"]].apply(
        lambda row: normalize_probability_triplet(row["ensemble_home_prob"], row["ensemble_draw_prob"], row["ensemble_away_prob"]),
        axis=1,
        result_type="expand",
    )
    result[["ensemble_home_prob", "ensemble_draw_prob", "ensemble_away_prob"]] = normalized
    result["ensemble_w_market"] = w_market
    result["ensemble_w_model"] = w_model
    return result


def apply_ensemble_to_upcoming_matches(
    upcoming_df: pd.DataFrame,
    w_market: float = DEFAULT_ENSEMBLE_W_MARKET,
    model_variant: str = "historical_model",
    output_path: Path = ENSEMBLE_PREDICTIONS_PATH,
) -> pd.DataFrame:
    warnings = []
    result = upcoming_df.copy()
    model_columns = DRAW_MODEL_PROB_COLUMNS if model_variant == "draw_context_model" else MODEL_PROB_COLUMNS
    has_market = _has_columns(result, MARKET_PROB_COLUMNS)
    has_model = _has_columns(result, model_columns)
    if not has_market and not has_model:
        raise ValueError("No market or model probabilities available for ensemble.")
    if not has_market:
        warnings.append("Market probabilities unavailable. Ensemble fell back to model probabilities.")
        for outcome, column in model_columns.items():
            result[f"ensemble_{outcome}_prob"] = result[column]
        result["ensemble_w_market"] = 0.0
        result["ensemble_w_model"] = 1.0
    elif not has_model:
        warnings.append("Model probabilities unavailable. Ensemble fell back to market probabilities.")
        for outcome, column in MARKET_PROB_COLUMNS.items():
            result[f"ensemble_{outcome}_prob"] = result[column]
        result["ensemble_w_market"] = 1.0
        result["ensemble_w_model"] = 0.0
    else:
        result = calculate_ensemble_probabilities(result, w_market, model_columns, MARKET_PROB_COLUMNS)
    result.attrs["warnings"] = warnings
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result
