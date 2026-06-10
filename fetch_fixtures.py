from datetime import datetime, timezone
from pathlib import Path
from typing import Union

import pandas as pd


FIXTURE_COLUMNS = [
    "match_id",
    "kickoff_time",
    "home_team",
    "away_team",
    "group",
    "matchday",
    "venue",
    "source",
    "fetched_at",
]


def _fetched_at() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_static_fixtures(path: Union[str, Path]) -> pd.DataFrame:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=FIXTURE_COLUMNS)
    df = pd.read_csv(path)
    for column in FIXTURE_COLUMNS:
        if column not in df.columns:
            df[column] = None
    return df[FIXTURE_COLUMNS]


def fixtures_from_odds_events(odds_df: pd.DataFrame) -> pd.DataFrame:
    if odds_df.empty:
        return pd.DataFrame(columns=FIXTURE_COLUMNS)
    rows = []
    for event_id, event_df in odds_df.groupby("event_id"):
        first = event_df.iloc[0]
        rows.append(
            {
                "match_id": event_id,
                "kickoff_time": first.get("commence_time"),
                "home_team": first.get("home_team"),
                "away_team": first.get("away_team"),
                "group": "TBD",
                "matchday": 0,
                "venue": "TBD",
                "source": "odds_api",
                "fetched_at": _fetched_at(),
            }
        )
    return pd.DataFrame(rows, columns=FIXTURE_COLUMNS)


def fetch_worldcup_fixtures(static_path: Union[str, Path, None] = None, odds_df: pd.DataFrame = None) -> pd.DataFrame:
    if odds_df is not None and not odds_df.empty:
        return fixtures_from_odds_events(odds_df)
    if static_path is not None:
        return load_static_fixtures(static_path)
    return pd.DataFrame(columns=FIXTURE_COLUMNS)

