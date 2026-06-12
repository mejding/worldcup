from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd
import requests

from config import MANUAL_ODDS_PATH, ODDS_API_BASE_URL, ODDS_SNAPSHOT_PATH


ODDS_COLUMNS = [
    "fetched_at",
    "event_id",
    "commence_time",
    "home_team",
    "away_team",
    "bookmaker_key",
    "bookmaker_title",
    "market_key",
    "outcome_name",
    "outcome_price",
]

MANUAL_ODDS_COLUMNS = [
    "match_id",
    "event_id",
    "commence_time",
    "home_team",
    "away_team",
    "bookmaker_key",
    "bookmaker_title",
    "market_key",
    "outcome_name",
    "outcome_price",
    "source",
    "source_last_checked",
]


def _fetched_at() -> str:
    return datetime.now(timezone.utc).isoformat()


def _events(raw_response: Union[dict, list]) -> list[dict]:
    if raw_response is None:
        return []
    if isinstance(raw_response, list):
        return raw_response
    if isinstance(raw_response, dict):
        if "data" in raw_response and isinstance(raw_response["data"], list):
            return raw_response["data"]
        if "events" in raw_response and isinstance(raw_response["events"], list):
            return raw_response["events"]
    return []


def is_draw_outcome(name: Any) -> bool:
    return str(name).strip().lower() in {"draw", "tie", "x", "uafgjort"}


def normalize_odds_response(raw_response: Union[dict, list]) -> pd.DataFrame:
    rows = []
    fetched_at = _fetched_at()
    for event in _events(raw_response):
        event_id = event.get("id") or event.get("event_id")
        commence_time = event.get("commence_time")
        home_team = event.get("home_team")
        away_team = event.get("away_team")
        bookmakers = event.get("bookmakers") or []
        if not bookmakers:
            continue
        for bookmaker in bookmakers:
            bookmaker_key = bookmaker.get("key")
            bookmaker_title = bookmaker.get("title") or bookmaker_key
            markets = bookmaker.get("markets") or []
            for market in markets:
                market_key = market.get("key")
                outcomes = market.get("outcomes") or []
                for outcome in outcomes:
                    price = outcome.get("price")
                    if price is None:
                        continue
                    rows.append(
                        {
                            "fetched_at": fetched_at,
                            "event_id": event_id,
                            "commence_time": commence_time,
                            "home_team": home_team,
                            "away_team": away_team,
                            "bookmaker_key": bookmaker_key,
                            "bookmaker_title": bookmaker_title,
                            "market_key": market_key,
                            "outcome_name": outcome.get("name"),
                            "outcome_price": float(price),
                        }
                    )
    return pd.DataFrame(rows, columns=ODDS_COLUMNS)


def _blank_odds_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=ODDS_COLUMNS)


def _clean_optional(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _bookmaker_key(value: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "_" for char in value.strip())
    return "_".join(part for part in cleaned.split("_") if part)


def _fixture_by_match_id(fixtures_df: Optional[pd.DataFrame]) -> dict:
    if fixtures_df is None or fixtures_df.empty or "match_id" not in fixtures_df.columns:
        return {}
    return {
        _clean_optional(row["match_id"]): row
        for _, row in fixtures_df.iterrows()
        if _clean_optional(row.get("match_id"))
    }


def load_manual_odds(
    path: Union[str, Path] = MANUAL_ODDS_PATH,
    fixtures_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return _blank_odds_frame()

    raw = pd.read_csv(path)
    if raw.empty:
        return _blank_odds_frame()

    for column in MANUAL_ODDS_COLUMNS:
        if column not in raw.columns:
            raw[column] = pd.NA

    fixture_lookup = _fixture_by_match_id(fixtures_df)
    rows = []
    fallback_fetched_at = _fetched_at()
    for _, row in raw.iterrows():
        outcome_name = _clean_optional(row.get("outcome_name"))
        outcome_price = pd.to_numeric(row.get("outcome_price"), errors="coerce")
        if not outcome_name or pd.isna(outcome_price):
            continue

        match_id = _clean_optional(row.get("match_id"))
        fixture = fixture_lookup.get(match_id)
        event_id = _clean_optional(row.get("event_id")) or match_id
        if not event_id:
            continue

        bookmaker_title = _clean_optional(row.get("bookmaker_title")) or _clean_optional(row.get("bookmaker_key"))
        bookmaker_key = _clean_optional(row.get("bookmaker_key")) or _bookmaker_key(bookmaker_title)
        commence_time = _clean_optional(row.get("commence_time"))
        home_team = _clean_optional(row.get("home_team"))
        away_team = _clean_optional(row.get("away_team"))

        if fixture is not None:
            commence_time = commence_time or _clean_optional(
                fixture.get("kickoff_utc", fixture.get("kickoff_time"))
            )
            home_team = home_team or _clean_optional(fixture.get("home_team"))
            away_team = away_team or _clean_optional(fixture.get("away_team"))

        if not all([commence_time, home_team, away_team, bookmaker_key, bookmaker_title]):
            continue

        fetched_at = (
            _clean_optional(row.get("source_last_checked"))
            or _clean_optional(row.get("fetched_at"))
            or fallback_fetched_at
        )
        rows.append(
            {
                "fetched_at": fetched_at,
                "event_id": event_id,
                "commence_time": commence_time,
                "home_team": home_team,
                "away_team": away_team,
                "bookmaker_key": bookmaker_key,
                "bookmaker_title": bookmaker_title,
                "market_key": _clean_optional(row.get("market_key")) or "h2h",
                "outcome_name": outcome_name,
                "outcome_price": float(outcome_price),
            }
        )

    return pd.DataFrame(rows, columns=ODDS_COLUMNS)


def fetch_odds_from_api(
    api_key: str,
    sport_key: str,
    region: str = "eu",
    market: str = "h2h",
    odds_format: str = "decimal",
) -> pd.DataFrame:
    if not api_key:
        raise ValueError("ODDS_API_KEY is not configured.")

    url = f"{ODDS_API_BASE_URL}/sports/{sport_key}/odds"
    response = requests.get(
        url,
        params={
            "apiKey": api_key,
            "regions": region,
            "markets": market,
            "oddsFormat": odds_format,
        },
        timeout=20,
    )
    if response.status_code in {401, 403}:
        raise ValueError("Odds API rejected the API key or access is not allowed.")
    if response.status_code == 429:
        raise ValueError("Odds API rate limit reached.")
    if response.status_code >= 400:
        raise ValueError(f"Odds API request failed with HTTP {response.status_code}.")
    return normalize_odds_response(response.json())


def append_odds_snapshot(df: pd.DataFrame, path: Union[str, Path] = ODDS_SNAPSHOT_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    df.to_csv(path, mode="a", header=write_header, index=False, columns=ODDS_COLUMNS)
