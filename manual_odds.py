from pathlib import Path
from typing import Union

import pandas as pd
from pandas.errors import EmptyDataError

from config import MANUAL_ODDS_PATH
from odds_normalizer import NORMALIZED_ODDS_COLUMNS, empty_normalized_odds


MANUAL_ODDS_REQUIRED_COLUMNS = [
    "match_id",
    "home_team",
    "away_team",
    "kickoff_utc",
    "bookmaker",
    "home_odds",
    "draw_odds",
    "away_odds",
    "odds_last_updated_utc",
]

MANUAL_ODDS_OPTIONAL_COLUMNS = [
    "bookmaker_key",
    "is_danske_spil",
    "source",
    "notes",
]

MANUAL_ODDS_COLUMNS = MANUAL_ODDS_REQUIRED_COLUMNS + MANUAL_ODDS_OPTIONAL_COLUMNS


def _empty_manual_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=MANUAL_ODDS_COLUMNS)


def _clean(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _is_truthy(value) -> bool:
    return _clean(value).lower() in {"1", "true", "yes", "y", "ja"}


def _bookmaker_key(value: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "_" for char in _clean(value))
    return "_".join(part for part in cleaned.split("_") if part)


def _validate_row(row) -> list[str]:
    warnings = []
    match_id = _clean(row.get("match_id"))
    for column in ["home_odds", "draw_odds", "away_odds"]:
        value = pd.to_numeric(row.get(column), errors="coerce")
        if pd.isna(value) or float(value) <= 1.0:
            warnings.append(f"{match_id or 'unknown match'}: {column} must be numeric and greater than 1.0.")
    for column in ["kickoff_utc", "odds_last_updated_utc"]:
        if pd.isna(pd.to_datetime(row.get(column), errors="coerce", utc=True)):
            warnings.append(f"{match_id or 'unknown match'}: {column} is not parseable as datetime.")
    return warnings


def load_manual_odds(path: Union[str, Path] = MANUAL_ODDS_PATH) -> tuple[pd.DataFrame, list[str]]:
    path = Path(path)
    if not path.exists():
        return _empty_manual_frame(), [f"Manual odds file missing: {path}"]
    if path.stat().st_size == 0:
        return _empty_manual_frame(), [f"Manual odds file is empty: {path}"]
    try:
        df = pd.read_csv(path)
    except EmptyDataError:
        return _empty_manual_frame(), [f"Manual odds file is empty: {path}"]
    except Exception as exc:
        return _empty_manual_frame(), [f"Manual odds file could not be read: {exc}"]

    if df.empty:
        return _empty_manual_frame(), [f"Manual odds file has no rows: {path}"]

    missing_columns = [column for column in MANUAL_ODDS_REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        return _empty_manual_frame(), ["Manual odds missing required columns: " + ", ".join(missing_columns)]

    for column in MANUAL_ODDS_OPTIONAL_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA

    warnings = []
    valid_mask = []
    for _, row in df.iterrows():
        row_warnings = _validate_row(row)
        warnings.extend(row_warnings)
        valid_mask.append(not row_warnings)

    valid = df.loc[valid_mask, MANUAL_ODDS_COLUMNS].copy()
    if valid.empty and not warnings:
        warnings.append("Manual odds file has no valid rows.")
    return valid, warnings


def normalize_manual_odds(manual_df: pd.DataFrame) -> pd.DataFrame:
    if manual_df.empty:
        return empty_normalized_odds()

    rows = []
    for _, row in manual_df.iterrows():
        is_danske_spil = _is_truthy(row.get("is_danske_spil"))
        bookmaker_title = _clean(row.get("bookmaker")) or ("Danske Spil" if is_danske_spil else "manual")
        bookmaker_key = _clean(row.get("bookmaker_key"))
        if not bookmaker_key:
            bookmaker_key = "danske_spil" if is_danske_spil else _bookmaker_key(bookmaker_title)
        if is_danske_spil and not _clean(row.get("bookmaker")):
            bookmaker_title = "Danske Spil"

        base = {
            "odds_source": "manual_csv",
            "provider": "manual",
            "fetched_at_utc": row["odds_last_updated_utc"],
            "event_id": row["match_id"],
            "commence_time_utc": row["kickoff_utc"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "bookmaker_key": bookmaker_key,
            "bookmaker_title": bookmaker_title,
            "bookmaker_last_update": row["odds_last_updated_utc"],
            "market_key": "h2h",
        }
        outcomes = [
            ("home", row["home_team"], row["home_odds"]),
            ("draw", "Draw", row["draw_odds"]),
            ("away", row["away_team"], row["away_odds"]),
        ]
        for outcome_type, outcome_name, price in outcomes:
            item = base.copy()
            item.update(
                {
                    "outcome_name": outcome_name,
                    "outcome_type": outcome_type,
                    "outcome_price": float(pd.to_numeric(price, errors="coerce")),
                }
            )
            rows.append(item)
    return pd.DataFrame(rows, columns=NORMALIZED_ODDS_COLUMNS)
