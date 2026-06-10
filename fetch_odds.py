from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Union

import pandas as pd
import requests

from config import ODDS_API_BASE_URL, ODDS_SNAPSHOT_PATH


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

