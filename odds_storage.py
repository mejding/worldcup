from pathlib import Path
from typing import Union

import pandas as pd
from pandas.errors import EmptyDataError

from config import PROCESSED_ODDS_PATH, RAW_ODDS_SNAPSHOT_PATH
from odds_normalizer import NORMALIZED_ODDS_COLUMNS, empty_normalized_odds


SNAPSHOT_DEDUPE_COLUMNS = [
    "fetched_at_utc",
    "event_id",
    "bookmaker_key",
    "market_key",
    "outcome_name",
]


def _read_odds_file(path: Union[str, Path]) -> pd.DataFrame:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return empty_normalized_odds()
    try:
        df = pd.read_csv(path)
    except EmptyDataError:
        return empty_normalized_odds()
    for column in NORMALIZED_ODDS_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA
    return df[NORMALIZED_ODDS_COLUMNS]


def save_odds_snapshot(df: pd.DataFrame, path: Union[str, Path] = RAW_ODDS_SNAPSHOT_PATH) -> None:
    if df.empty:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_odds_file(path)
    if existing.empty:
        combined = df[NORMALIZED_ODDS_COLUMNS].copy()
    else:
        combined = pd.concat([existing, df[NORMALIZED_ODDS_COLUMNS]], ignore_index=True)
    combined = combined.drop_duplicates(subset=SNAPSHOT_DEDUPE_COLUMNS, keep="last")
    combined.to_csv(path, index=False)


def load_latest_odds_snapshot(path: Union[str, Path] = RAW_ODDS_SNAPSHOT_PATH) -> pd.DataFrame:
    df = _read_odds_file(path)
    if df.empty or "fetched_at_utc" not in df.columns:
        return empty_normalized_odds()
    timestamps = pd.to_datetime(df["fetched_at_utc"], errors="coerce", utc=True)
    if timestamps.isna().all():
        return empty_normalized_odds()
    latest = timestamps.max()
    return df.loc[timestamps == latest, NORMALIZED_ODDS_COLUMNS].copy()


def save_latest_odds(df: pd.DataFrame, path: Union[str, Path] = PROCESSED_ODDS_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_latest_processed_odds(path: Union[str, Path] = PROCESSED_ODDS_PATH) -> pd.DataFrame:
    return _read_odds_file(path)
