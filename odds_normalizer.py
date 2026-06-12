from typing import Any

import pandas as pd


NORMALIZED_ODDS_COLUMNS = [
    "odds_source",
    "provider",
    "fetched_at_utc",
    "event_id",
    "commence_time_utc",
    "home_team",
    "away_team",
    "bookmaker_key",
    "bookmaker_title",
    "bookmaker_last_update",
    "market_key",
    "outcome_name",
    "outcome_type",
    "outcome_price",
]


def empty_normalized_odds() -> pd.DataFrame:
    return pd.DataFrame(columns=NORMALIZED_ODDS_COLUMNS)


def _clean(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def detect_outcome_type(outcome_name: Any, home_team: Any, away_team: Any) -> str:
    outcome = _clean(outcome_name).lower()
    if outcome in {"draw", "tie"}:
        return "draw"
    if outcome == _clean(home_team).lower():
        return "home"
    if outcome == _clean(away_team).lower():
        return "away"
    return "unknown"


def normalize_the_odds_api_response(response_json: list[dict], fetched_at_utc: str) -> pd.DataFrame:
    if not response_json:
        return empty_normalized_odds()

    rows = []
    for event in response_json:
        event_id = event.get("id") or event.get("event_id")
        commence_time = event.get("commence_time")
        home_team = event.get("home_team")
        away_team = event.get("away_team")
        for bookmaker in event.get("bookmakers") or []:
            bookmaker_key = bookmaker.get("key")
            bookmaker_title = bookmaker.get("title") or bookmaker_key
            bookmaker_last_update = bookmaker.get("last_update")
            for market in bookmaker.get("markets") or []:
                market_key = market.get("key")
                for outcome in market.get("outcomes") or []:
                    price = pd.to_numeric(outcome.get("price"), errors="coerce")
                    if pd.isna(price):
                        continue
                    outcome_name = outcome.get("name")
                    rows.append(
                        {
                            "odds_source": "api",
                            "provider": "the_odds_api",
                            "fetched_at_utc": fetched_at_utc,
                            "event_id": event_id,
                            "commence_time_utc": commence_time,
                            "home_team": home_team,
                            "away_team": away_team,
                            "bookmaker_key": bookmaker_key,
                            "bookmaker_title": bookmaker_title,
                            "bookmaker_last_update": bookmaker_last_update,
                            "market_key": market_key,
                            "outcome_name": outcome_name,
                            "outcome_type": detect_outcome_type(outcome_name, home_team, away_team),
                            "outcome_price": float(price),
                        }
                    )
    return pd.DataFrame(rows, columns=NORMALIZED_ODDS_COLUMNS)


def normalized_to_legacy_odds(df: pd.DataFrame) -> pd.DataFrame:
    legacy_columns = [
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
    if df.empty:
        return pd.DataFrame(columns=legacy_columns)
    result = pd.DataFrame(
        {
            "fetched_at": df.get("fetched_at_utc"),
            "event_id": df.get("event_id"),
            "commence_time": df.get("commence_time_utc"),
            "home_team": df.get("home_team"),
            "away_team": df.get("away_team"),
            "bookmaker_key": df.get("bookmaker_key"),
            "bookmaker_title": df.get("bookmaker_title"),
            "market_key": df.get("market_key"),
            "outcome_name": df.get("outcome_name"),
            "outcome_price": df.get("outcome_price"),
        }
    )
    return result[legacy_columns]
