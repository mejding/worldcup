from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from config import (
    MANUAL_ODDS_PATH,
    ODDS_API_BASE_URL,
    ODDS_API_DATE_FORMAT,
    ODDS_API_MARKETS,
    ODDS_API_ODDS_FORMAT,
    ODDS_API_REGIONS,
    ODDS_API_SPORT_KEY,
    RAW_ODDS_SNAPSHOT_PATH,
)
from manual_odds import load_manual_odds
from odds_storage import load_latest_odds_snapshot


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_odds_api_key() -> Optional[str]:
    try:
        import streamlit as st

        if hasattr(st, "secrets") and "ODDS_API_KEY" in st.secrets:
            value = st.secrets["ODDS_API_KEY"]
            if value:
                return str(value)
    except Exception:
        pass

    import os

    value = os.environ.get("ODDS_API_KEY")
    return value if value else None


def _base_metadata(
    sport_key: str,
    regions: str,
    markets: str,
    provider: str = "the_odds_api",
) -> dict:
    return {
        "fetched_at_utc": _utc_now(),
        "provider": provider,
        "sport_key": sport_key,
        "regions": regions,
        "markets": markets,
        "status_code": None,
        "requests_remaining": None,
        "requests_used": None,
        "last_error": None,
        "source_status": "not_started",
    }


def fetch_odds_from_the_odds_api(
    sport_key: str = ODDS_API_SPORT_KEY,
    regions: str = ODDS_API_REGIONS,
    markets: str = ODDS_API_MARKETS,
    odds_format: str = ODDS_API_ODDS_FORMAT,
    date_format: str = ODDS_API_DATE_FORMAT,
    api_key: Optional[str] = None,
    commence_time_from: Optional[str] = None,
    commence_time_to: Optional[str] = None,
    bookmakers: Optional[str] = None,
    timeout: int = 20,
) -> tuple[list[dict], dict]:
    metadata = _base_metadata(sport_key, regions, markets)
    api_key = api_key or get_odds_api_key()
    if not api_key:
        metadata["source_status"] = "missing_api_key"
        metadata["last_error"] = "ODDS_API_KEY is not configured."
        return [], metadata

    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
        "dateFormat": date_format,
    }
    optional = {
        "commenceTimeFrom": commence_time_from,
        "commenceTimeTo": commence_time_to,
        "bookmakers": bookmakers,
    }
    params.update({key: value for key, value in optional.items() if value})

    try:
        response = requests.get(
            f"{ODDS_API_BASE_URL}/sports/{sport_key}/odds",
            params=params,
            timeout=timeout,
        )
        metadata["status_code"] = response.status_code
        metadata["requests_remaining"] = response.headers.get("x-requests-remaining")
        metadata["requests_used"] = response.headers.get("x-requests-used")
        if response.status_code >= 400:
            metadata["source_status"] = "http_error"
            metadata["last_error"] = f"Odds API request failed with HTTP {response.status_code}."
            return [], metadata
        try:
            payload = response.json()
        except ValueError as exc:
            metadata["source_status"] = "json_error"
            metadata["last_error"] = f"Could not parse Odds API JSON: {exc}"
            return [], metadata
        if not isinstance(payload, list):
            metadata["source_status"] = "json_error"
            metadata["last_error"] = "Odds API JSON response was not a list."
            return [], metadata
        metadata["source_status"] = "ok"
        return payload, metadata
    except requests.Timeout as exc:
        metadata["source_status"] = "timeout"
        metadata["last_error"] = str(exc)
        return [], metadata
    except Exception as exc:
        metadata["source_status"] = "exception"
        metadata["last_error"] = str(exc)
        return [], metadata


def _file_exists_with_rows(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def get_odds_source_status(
    manual_path: Path = MANUAL_ODDS_PATH,
    cache_path: Path = RAW_ODDS_SNAPSHOT_PATH,
) -> dict:
    has_api_key = bool(get_odds_api_key())
    manual_df, manual_warnings = load_manual_odds(manual_path)
    manual_exists = _file_exists_with_rows(Path(manual_path))
    manual_valid = not manual_df.empty
    cached_df = load_latest_odds_snapshot(cache_path)
    cached_exists = not cached_df.empty

    last_updated = None
    if manual_valid:
        last_updated = str(pd.to_datetime(manual_df["odds_last_updated_utc"], errors="coerce", utc=True).max())
    if cached_exists and not last_updated:
        last_updated = str(pd.to_datetime(cached_df["fetched_at_utc"], errors="coerce", utc=True).max())

    if has_api_key:
        active_source = "api"
        message = "Live odds source configured."
        warning = ""
    elif manual_valid:
        active_source = "manual"
        message = "Manual odds CSV loaded."
        warning = "Manual odds loaded. These are not live bookmaker odds."
    elif cached_exists:
        active_source = "cached"
        message = "Using cached odds snapshot."
        warning = "Cached odds are being used because latest live odds are unavailable."
    else:
        active_source = "missing"
        message = ""
        warning = "Odds data source missing. Add ODDS_API_KEY in Streamlit secrets or provide data/reference/manual_odds.csv."

    return {
        "has_api_key": has_api_key,
        "manual_odds_exists": manual_exists,
        "manual_odds_valid": manual_valid,
        "cached_odds_exists": cached_exists,
        "active_odds_source": active_source,
        "message": message,
        "warning": warning,
        "last_updated_utc": last_updated,
        "last_error": "; ".join(manual_warnings) if manual_warnings and manual_exists and not manual_valid else "",
    }
