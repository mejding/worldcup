#!/usr/bin/env python3
"""Import official FIFA men's ranking snapshots.

The Streamlit app does not call this script at runtime. It is a developer data
refresh step that writes a compact CSV used by the training pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import FIFA_RANKINGS_PATH


FIFA_RANKING_PAGE_URL = "https://inside.fifa.com/fifa-world-ranking/men"
FIFA_API_BASE_URL = "https://api.fifa.com/api/v3"
FIFA_RANKING_SCHEDULES_URL = f"{FIFA_API_BASE_URL}/rankingschedules/all?type=0&gender=1"
SOURCE_NAME = "FIFA official ranking API"
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; worldcup-prediction-app/1.0)",
    "Accept": "application/json,text/html;q=0.9,*/*;q=0.8",
}


class NextDataParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._inside_next_data = False
        self.data_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "script":
            return
        attrs_dict = {key: value for key, value in attrs}
        if attrs_dict.get("id") == "__NEXT_DATA__":
            self._inside_next_data = True

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "script" and self._inside_next_data:
            self._inside_next_data = False

    def handle_data(self, data: str) -> None:
        if self._inside_next_data:
            self.data_parts.append(data)


def _get_json(url: str, session: requests.Session, retries: int = 3, sleep_seconds: float = 0.3) -> Any:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, headers=REQUEST_HEADERS, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(sleep_seconds * attempt)
    raise RuntimeError(f"Could not fetch JSON from {url}: {last_error}") from last_error


def _get_text(url: str, session: requests.Session) -> str:
    response = session.get(url, headers=REQUEST_HEADERS, timeout=30)
    response.raise_for_status()
    return response.text


def _extract_next_data(html: str) -> dict:
    parser = NextDataParser()
    parser.feed(html)
    payload = "".join(parser.data_parts).strip()
    if not payload:
        raise RuntimeError("Could not find __NEXT_DATA__ on FIFA ranking page.")
    return json.loads(payload)


def _ranking_page_data(session: requests.Session) -> dict:
    html = _get_text(FIFA_RANKING_PAGE_URL, session)
    next_data = _extract_next_data(html)
    return next_data.get("props", {}).get("pageProps", {}).get("pageData", {}).get("ranking", {})


def _date_only(value: str | None) -> str | None:
    if not value:
        return None
    timestamp = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(timestamp):
        return None
    return timestamp.date().isoformat()


def _ranking_schedules(session: requests.Session) -> list[dict]:
    payload = _get_json(FIFA_RANKING_SCHEDULES_URL, session)
    schedules = []
    for item in payload.get("Results", []) or []:
        schedule_id = item.get("IdRankingSchedule")
        ranking_date = _date_only(item.get("OfficialDate") or item.get("VisibilityDate") or item.get("MatchWindowEndDate"))
        if schedule_id and ranking_date:
            schedules.append({"id": schedule_id, "ranking_date": ranking_date})
    return schedules


def _team_name(raw: dict) -> str:
    name = raw.get("TeamName")
    if isinstance(name, list) and name:
        for item in name:
            description = item.get("Description") if isinstance(item, dict) else None
            if description:
                return str(description)
    if isinstance(name, str):
        return name
    return str(raw.get("Name") or raw.get("CountryName") or raw.get("IdCountry") or "").strip()


def _first_number(raw: dict, *keys: str) -> float | None:
    for key in keys:
        value = raw.get(key)
        if value is None or value == "":
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _confederation(raw: dict) -> str:
    value = raw.get("Confederation") or raw.get("ConfederationCode") or raw.get("ConfederationName")
    if isinstance(value, dict):
        return str(value.get("Description") or value.get("Name") or value.get("Code") or "").strip()
    return str(value or "").strip()


def _ranking_rows(payload: dict, ranking_date: str, checked_at: str) -> list[dict]:
    results = payload.get("Results") or payload.get("results") or []
    rows = []
    for item in results:
        if not isinstance(item, dict):
            continue
        team = _team_name(item)
        rank = _first_number(item, "Rank", "rank")
        points = _first_number(item, "DecimalTotalPoints", "TotalPoints", "totalPoints", "Points")
        previous_rank = _first_number(item, "PrevRank", "PreviousRank", "previousRank")
        if not team or rank is None or points is None:
            continue
        rank_change = None if previous_rank is None else previous_rank - rank
        rows.append(
            {
                "ranking_date": ranking_date,
                "team": team,
                "fifa_rank": int(rank),
                "fifa_points": points,
                "confederation": _confederation(item),
                "previous_rank": "" if previous_rank is None else int(previous_rank),
                "rank_change": "" if rank_change is None else int(rank_change),
                "source": SOURCE_NAME,
                "source_last_checked": checked_at,
            }
        )
    return rows


def fetch_fifa_rankings(limit_schedules: int | None = None, pause_seconds: float = 0.05) -> pd.DataFrame:
    checked_at = datetime.now(timezone.utc).isoformat()
    session = requests.Session()
    schedules = _ranking_schedules(session)
    if not schedules:
        raise RuntimeError("FIFA ranking API did not expose any ranking schedules.")
    if limit_schedules:
        schedules = schedules[:limit_schedules]

    rows = []
    for index, schedule in enumerate(schedules, start=1):
        endpoint = f"{FIFA_API_BASE_URL}/rankingsbyschedule?rankingScheduleId={schedule['id']}"
        payload = _get_json(endpoint, session)
        schedule_rows = _ranking_rows(payload, schedule["ranking_date"], checked_at)
        rows.extend(schedule_rows)
        print(f"{index}/{len(schedules)} {schedule['ranking_date']} {schedule['id']}: {len(schedule_rows)} teams")
        if pause_seconds:
            time.sleep(pause_seconds)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("FIFA ranking import returned zero usable rows.")
    df["ranking_date"] = pd.to_datetime(df["ranking_date"], errors="coerce").dt.date.astype(str)
    df = df.dropna(subset=["ranking_date", "team", "fifa_rank", "fifa_points"])
    df = df.drop_duplicates(subset=["ranking_date", "team"], keep="first")
    return df.sort_values(["ranking_date", "fifa_rank", "team"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Import official FIFA men's ranking snapshots.")
    parser.add_argument("--output", default=str(FIFA_RANKINGS_PATH), help="Output CSV path.")
    parser.add_argument("--limit-schedules", type=int, default=None, help="Optional debug limit, latest schedules first.")
    parser.add_argument("--pause-seconds", type=float, default=0.05, help="Pause between FIFA API requests.")
    args = parser.parse_args()

    output_path = Path(args.output)
    df = fetch_fifa_rankings(limit_schedules=args.limit_schedules, pause_seconds=args.pause_seconds)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(
        f"Wrote {len(df)} ranking rows for {df['team'].nunique()} teams "
        f"from {df['ranking_date'].min()} to {df['ranking_date'].max()} -> {output_path}"
    )


if __name__ == "__main__":
    main()
