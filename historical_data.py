from pathlib import Path
from typing import Union

import pandas as pd

from config import HISTORICAL_RESULTS_PATH


STANDARD_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "neutral",
    "result",
]


def load_historical_results(path: Union[str, Path] = HISTORICAL_RESULTS_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            "No historical data file found. Add data/historical/international_results.csv to train the model."
        )
    return pd.read_csv(path)


def standardize_historical_results(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    lower_map = {column.lower().strip(): column for column in result.columns}
    rename_map = {}
    for expected in ["date", "home_team", "away_team", "home_score", "away_score", "tournament", "neutral"]:
        if expected in lower_map:
            rename_map[lower_map[expected]] = expected
    result = result.rename(columns=rename_map)

    if "tournament" not in result.columns:
        result["tournament"] = "Unknown"
    result["tournament"] = result["tournament"].fillna("Unknown")
    if "neutral" not in result.columns:
        result["neutral"] = False
    result["neutral"] = result["neutral"].fillna(False).astype(bool)

    if {"home_score", "away_score"}.issubset(result.columns):
        result["home_score"] = pd.to_numeric(result["home_score"], errors="coerce")
        result["away_score"] = pd.to_numeric(result["away_score"], errors="coerce")
        result["result"] = result.apply(_match_result, axis=1)

    for column in STANDARD_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA
    return result[STANDARD_COLUMNS]


def clean_historical_results_for_training(df: pd.DataFrame, as_of=None) -> pd.DataFrame:
    result = df.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce", utc=True)
    result["home_score"] = pd.to_numeric(result["home_score"], errors="coerce")
    result["away_score"] = pd.to_numeric(result["away_score"], errors="coerce")
    as_of_ts = pd.Timestamp(as_of, tz="UTC") if as_of is not None else pd.Timestamp.utcnow()
    result = result[
        result["date"].notna()
        & result["home_score"].notna()
        & result["away_score"].notna()
        & (result["date"] <= as_of_ts)
    ].copy()
    result["date"] = result["date"].dt.strftime("%Y-%m-%d")
    result["home_score"] = result["home_score"].astype(int)
    result["away_score"] = result["away_score"].astype(int)
    result = standardize_historical_results(result)
    return result.drop_duplicates().reset_index(drop=True)


def _match_result(row) -> str:
    if pd.isna(row["home_score"]) or pd.isna(row["away_score"]):
        return pd.NA
    if row["home_score"] > row["away_score"]:
        return "H"
    if row["home_score"] < row["away_score"]:
        return "A"
    return "D"


def validate_historical_results(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    warnings = []
    errors = []
    required = ["date", "home_team", "away_team", "home_score", "away_score"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        errors.append("Missing required historical columns: " + ", ".join(missing))
        return warnings, errors

    if "tournament" not in df.columns:
        warnings.append("Missing tournament column. It will be set to Unknown.")
    if "neutral" not in df.columns:
        warnings.append("Missing neutral column. It will be set to False.")

    dates = pd.to_datetime(df["date"], errors="coerce", utc=True)
    if dates.isna().any():
        warnings.append("One or more historical dates could not be parsed.")
    if (dates.dropna() > pd.Timestamp.utcnow()).any():
        warnings.append("Historical data contains future-dated matches.")
    if df.duplicated().any():
        warnings.append("Historical data contains duplicate rows.")

    scores = df[["home_score", "away_score"]].apply(pd.to_numeric, errors="coerce")
    if scores.isna().any().any():
        errors.append("Historical score columns must be numeric.")
    return warnings, errors
