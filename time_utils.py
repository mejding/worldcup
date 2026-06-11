from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd


DANISH_TIMEZONE = ZoneInfo("Europe/Copenhagen")
DANISH_MONTHS = {
    1: "januar",
    2: "februar",
    3: "marts",
    4: "april",
    5: "maj",
    6: "juni",
    7: "juli",
    8: "august",
    9: "september",
    10: "oktober",
    11: "november",
    12: "december",
}


def convert_to_danish_time(timestamp) -> datetime | None:
    try:
        parsed = pd.to_datetime(timestamp, errors="coerce", utc=True)
        if pd.isna(parsed):
            return None
        return parsed.to_pydatetime().astimezone(DANISH_TIMEZONE)
    except Exception:
        return None


def format_danish_kickoff(timestamp) -> str:
    converted = convert_to_danish_time(timestamp)
    if converted is None:
        return "" if pd.isna(timestamp) else str(timestamp)
    return f"{converted.day}. {DANISH_MONTHS[converted.month]} {converted.year}, {converted:%H:%M} dansk tid"


def add_danish_kickoff_column(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "kickoff_time" not in result.columns:
        result["kickoff_time_dk"] = ""
        return result
    result["kickoff_time_dk"] = result["kickoff_time"].map(format_danish_kickoff)
    return result
