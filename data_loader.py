import shutil
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from pandas.errors import EmptyDataError

from config import (
    BANKROLL_HISTORY_EXAMPLE_PATH,
    BANKROLL_HISTORY_PATH,
    BANKROLL_STATE_EXAMPLE_PATH,
    BANKROLL_STATE_PATH,
    BET_LOG_EXAMPLE_PATH,
    BET_LOG_PATH,
    DATA_DIR,
    REQUIRED_PREDICTION_COLUMNS,
    SAMPLE_PREDICTIONS_PATH,
)
from bankroll import BANKROLL_HISTORY_COLUMNS, DEFAULT_BANKROLL_STATE, save_bankroll_state
from bet_log import BET_LOG_COLUMNS

RUNTIME_FILE_PAIRS = {
    BANKROLL_STATE_PATH: BANKROLL_STATE_EXAMPLE_PATH,
    BANKROLL_HISTORY_PATH: BANKROLL_HISTORY_EXAMPLE_PATH,
    BET_LOG_PATH: BET_LOG_EXAMPLE_PATH,
}


def load_sample_predictions(path: Union[str, Path] = SAMPLE_PREDICTIONS_PATH) -> pd.DataFrame:
    return load_predictions(path)


def load_predictions(path: Union[str, Path] = SAMPLE_PREDICTIONS_PATH) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"Sample predictions file not found: {path}")
    return pd.read_csv(path)


def validate_predictions(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    warnings = []
    errors = []
    if df.empty:
        return warnings, ["Predictions file is empty."]

    missing_columns = [column for column in REQUIRED_PREDICTION_COLUMNS if column not in df.columns]
    if missing_columns:
        return warnings, [f"Missing required columns: {', '.join(missing_columns)}"]

    probability_columns = [
        "model_home_prob",
        "model_draw_prob",
        "model_away_prob",
        "market_home_prob",
        "market_draw_prob",
        "market_away_prob",
    ]
    odds_columns = [
        "ds_home_odds",
        "ds_draw_odds",
        "ds_away_odds",
        "best_home_odds",
        "best_draw_odds",
        "best_away_odds",
    ]

    numeric_df = df.copy()
    for column in probability_columns + odds_columns + ["draw_context_score"]:
        numeric_df[column] = pd.to_numeric(numeric_df[column], errors="coerce")

    invalid_probability_columns = [
        column for column in probability_columns if numeric_df[column].isna().any()
    ]
    if invalid_probability_columns:
        errors.append(
            "Invalid probability columns: " + ", ".join(invalid_probability_columns)
        )
        return warnings, errors

    invalid_odds_columns = [column for column in odds_columns if numeric_df[column].isna().any()]
    if invalid_odds_columns:
        errors.append("Invalid odds columns: " + ", ".join(invalid_odds_columns))
        return warnings, errors

    model_sums = numeric_df[["model_home_prob", "model_draw_prob", "model_away_prob"]].sum(axis=1)
    market_sums = numeric_df[["market_home_prob", "market_draw_prob", "market_away_prob"]].sum(axis=1)
    if not model_sums.sub(1.0).abs().le(0.001).all():
        errors.append("Model probabilities must sum to 1.0 for every match.")
    if not market_sums.sub(1.0).abs().le(0.001).all():
        warnings.append("Market probabilities do not sum to 1.0 for every match.")

    if not (numeric_df[odds_columns] > 1.0).all().all():
        errors.append("All odds values must be numeric and greater than 1.0.")

    if pd.to_datetime(df["kickoff_time"], errors="coerce").isna().any():
        warnings.append("One or more kickoff_time values could not be parsed as datetime.")

    if numeric_df["draw_context_score"].isna().any() or not numeric_df["draw_context_score"].between(0, 100).all():
        warnings.append("Draw-context score must be between 0 and 100.")

    valid_draw_labels = {"Low", "Medium", "High"}
    invalid_draw_labels = set(df["draw_context_label"].dropna().unique()) - valid_draw_labels
    if invalid_draw_labels:
        warnings.append("Draw-context label should be one of Low, Medium, High.")

    return warnings, errors


def _write_csv_headers(path: Path, columns: list[str]) -> None:
    pd.DataFrame(columns=columns).to_csv(path, index=False)


def _create_default_runtime_file(runtime_path: Path) -> None:
    if runtime_path.name == BANKROLL_STATE_PATH.name:
        save_bankroll_state(DEFAULT_BANKROLL_STATE.copy(), runtime_path)
    elif runtime_path.name == BANKROLL_HISTORY_PATH.name:
        _write_csv_headers(runtime_path, BANKROLL_HISTORY_COLUMNS)
    elif runtime_path.name == BET_LOG_PATH.name:
        _write_csv_headers(runtime_path, BET_LOG_COLUMNS)
    else:
        runtime_path.touch()


def ensure_runtime_data_files(
    data_dir: Union[str, Path] = DATA_DIR,
    runtime_file_pairs: Optional[dict[Path, Path]] = None,
) -> list[str]:
    created = []
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    pairs = runtime_file_pairs or RUNTIME_FILE_PAIRS

    for runtime_path, example_path in pairs.items():
        runtime_path = Path(runtime_path)
        example_path = Path(example_path)
        if runtime_path.exists():
            continue
        runtime_path.parent.mkdir(parents=True, exist_ok=True)
        if example_path.exists():
            shutil.copyfile(example_path, runtime_path)
            created.append(str(runtime_path))
        else:
            _create_default_runtime_file(runtime_path)
            created.append(str(runtime_path))
    return created


def load_csv_with_columns(path: Union[str, Path], columns: list[str]) -> pd.DataFrame:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_csv_headers(path, columns)
        return pd.DataFrame(columns=columns)
    try:
        df = pd.read_csv(path)
    except EmptyDataError:
        _write_csv_headers(path, columns)
        return pd.DataFrame(columns=columns)
    missing_columns = [column for column in columns if column not in df.columns]
    if missing_columns:
        _write_csv_headers(path, columns)
        return pd.DataFrame(columns=columns)
    return df
