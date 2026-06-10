from pathlib import Path
import shutil
from typing import Union

import pandas as pd

from config import REQUIRED_PREDICTION_COLUMNS

DATA_DIR = Path(__file__).resolve().parent / "data"
SAMPLE_PREDICTIONS_PATH = DATA_DIR / "sample_predictions.csv"
RUNTIME_FILE_PAIRS = {
    DATA_DIR / "bankroll_state.json": DATA_DIR / "bankroll_state.example.json",
    DATA_DIR / "bankroll_history.csv": DATA_DIR / "bankroll_history.example.csv",
    DATA_DIR / "bet_log.csv": DATA_DIR / "bet_log.example.csv",
}


def load_sample_predictions(path: Union[str, Path] = SAMPLE_PREDICTIONS_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def load_predictions(path: Union[str, Path] = SAMPLE_PREDICTIONS_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def validate_predictions(df: pd.DataFrame) -> list[str]:
    warnings = []
    missing_columns = [column for column in REQUIRED_PREDICTION_COLUMNS if column not in df.columns]
    if missing_columns:
        warnings.append(f"CRITICAL: Missing required columns: {', '.join(missing_columns)}")
        return warnings

    model_sums = df[["model_home_prob", "model_draw_prob", "model_away_prob"]].sum(axis=1)
    market_sums = df[["market_home_prob", "market_draw_prob", "market_away_prob"]].sum(axis=1)
    if not model_sums.sub(1.0).abs().le(0.001).all():
        warnings.append("Model probabilities do not sum to 1.0 for every match.")
    if not market_sums.sub(1.0).abs().le(0.001).all():
        warnings.append("Market probabilities do not sum to 1.0 for every match.")

    odds_columns = [
        "ds_home_odds",
        "ds_draw_odds",
        "ds_away_odds",
        "best_home_odds",
        "best_draw_odds",
        "best_away_odds",
    ]
    if not (df[odds_columns] > 0).all().all():
        warnings.append("One or more odds values are not positive.")

    if not df["draw_context_score"].between(0, 100).all():
        warnings.append("Draw-context score must be between 0 and 100.")

    return warnings


def ensure_runtime_data_files() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for runtime_path, example_path in RUNTIME_FILE_PAIRS.items():
        if runtime_path.exists():
            continue
        if example_path.exists():
            shutil.copyfile(example_path, runtime_path)
        else:
            runtime_path.touch()
