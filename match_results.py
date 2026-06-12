from pathlib import Path
from typing import Optional, Union

import pandas as pd
from pandas.errors import EmptyDataError

from config import MATCH_RESULTS_PATH


MATCH_RESULTS_COLUMNS = [
    "match_id",
    "home_score",
    "away_score",
    "result_status",
    "result_favorite_outcome",
    "result_last_checked_utc",
    "result_source",
    "result_notes",
]


def load_match_results(path: Union[str, Path] = MATCH_RESULTS_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=MATCH_RESULTS_COLUMNS)
    try:
        df = pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame(columns=MATCH_RESULTS_COLUMNS)
    for column in MATCH_RESULTS_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA
    for column in ["match_id", "result_status", "result_favorite_outcome", "result_source", "result_notes"]:
        df[column] = df[column].astype("string").str.strip()
    return df[MATCH_RESULTS_COLUMNS]


def _actual_outcome(row) -> str:
    home_score = pd.to_numeric(row.get("home_score"), errors="coerce")
    away_score = pd.to_numeric(row.get("away_score"), errors="coerce")
    if pd.isna(home_score) or pd.isna(away_score):
        return pd.NA
    if home_score > away_score:
        return "home"
    if away_score > home_score:
        return "away"
    return "draw"


def _favorite_outcome(row) -> str:
    result_favorite = row.get("result_favorite_outcome")
    if not pd.isna(result_favorite) and str(result_favorite) in {"home", "away"}:
        return str(result_favorite)

    odds_pairs = [
        ("home", row.get("best_home_odds")),
        ("away", row.get("best_away_odds")),
        ("home", row.get("ds_home_odds")),
        ("away", row.get("ds_away_odds")),
    ]
    for home_key, away_key in [(odds_pairs[0], odds_pairs[1]), (odds_pairs[2], odds_pairs[3])]:
        home_odds = pd.to_numeric(home_key[1], errors="coerce")
        away_odds = pd.to_numeric(away_key[1], errors="coerce")
        if pd.notna(home_odds) and pd.notna(away_odds) and home_odds != away_odds:
            return "home" if home_odds < away_odds else "away"

    probability_groups = [
        ("market_home_prob", "market_away_prob"),
        ("active_home_prob", "active_away_prob"),
        ("model_home_prob", "model_away_prob"),
    ]
    for home_column, away_column in probability_groups:
        home_probability = pd.to_numeric(row.get(home_column), errors="coerce")
        away_probability = pd.to_numeric(row.get(away_column), errors="coerce")
        if pd.notna(home_probability) and pd.notna(away_probability) and home_probability != away_probability:
            return "home" if home_probability > away_probability else "away"

    return pd.NA


def outcome_label(outcome: str, home_team: str, away_team: str) -> str:
    if pd.isna(outcome):
        return "-"
    if outcome == "home":
        return str(home_team)
    if outcome == "away":
        return str(away_team)
    if outcome == "draw":
        return "Uafgjort"
    return "-"


def _favorite_result_status(row) -> str:
    if not bool(row.get("is_completed")):
        return "Upcoming"
    actual_outcome = row.get("actual_outcome")
    favorite_outcome = row.get("favorite_outcome")
    if pd.isna(actual_outcome):
        return "Resultat mangler"
    if pd.isna(favorite_outcome):
        return "Favorit ukendt"
    if actual_outcome == favorite_outcome:
        return "Favoritten gik hjem"
    return "Overraskelse"


def add_match_results(predictions: pd.DataFrame, results: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    result = predictions.copy()
    match_results = load_match_results() if results is None else results.copy()
    if match_results.empty:
        result["is_completed"] = False
        result["archive_status"] = "Upcoming"
        return result

    result = result.merge(match_results, on="match_id", how="left")
    result["is_completed"] = result["result_status"].fillna("").str.lower().eq("final")
    result["actual_outcome"] = result.apply(_actual_outcome, axis=1)
    result["favorite_outcome"] = result.apply(_favorite_outcome, axis=1)
    result["actual_outcome_label"] = result.apply(
        lambda row: outcome_label(row.get("actual_outcome"), row.get("home_team"), row.get("away_team")),
        axis=1,
    )
    result["favorite_outcome_label"] = result.apply(
        lambda row: outcome_label(row.get("favorite_outcome"), row.get("home_team"), row.get("away_team")),
        axis=1,
    )
    result["favorite_result_status"] = result.apply(_favorite_result_status, axis=1)
    result["full_time_score"] = result.apply(
        lambda row: f"{int(row['home_score'])}-{int(row['away_score'])}"
        if bool(row.get("is_completed")) and not pd.isna(row.get("home_score")) and not pd.isna(row.get("away_score"))
        else "",
        axis=1,
    )
    result["archive_status"] = result["favorite_result_status"]
    return result


def split_active_and_archived_matches(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "is_completed" not in df.columns:
        return df.copy(), pd.DataFrame(columns=df.columns)
    completed = df["is_completed"].fillna(False).astype(bool)
    return df.loc[~completed].copy(), df.loc[completed].copy()
