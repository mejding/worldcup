from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import requests
from pandas.errors import EmptyDataError

from config import MATCH_RESULTS_PATH, MATCH_RESULTS_UPDATES_PATH, REFERENCE_FIXTURES_PATH, get_secret_or_env
from fixture_data import load_fixture_dataset
from match_results import MATCH_RESULTS_COLUMNS, load_match_results


def _utc_now() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc))


def load_match_result_updates(path: Union[str, Path] = MATCH_RESULTS_UPDATES_PATH) -> pd.DataFrame:
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
    df = df[MATCH_RESULTS_COLUMNS].copy()
    df["match_id"] = df["match_id"].astype("string").str.strip()
    df["result_status"] = df["result_status"].astype("string").str.strip().str.lower()
    return df[df["match_id"].notna()]


def normalize_match_result_updates(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for column in MATCH_RESULTS_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA
    result = result[MATCH_RESULTS_COLUMNS].copy()
    result["match_id"] = result["match_id"].astype("string").str.strip()
    result["result_status"] = result["result_status"].astype("string").str.strip().str.lower()
    return result[result["match_id"].notna()]


def get_match_results_feed_url() -> Optional[str]:
    value = get_secret_or_env("MATCH_RESULTS_FEED_URL", None)
    return str(value).strip() if value else None


def fetch_remote_match_result_updates(
    url: Optional[str] = None,
    timeout: int = 15,
) -> tuple[pd.DataFrame, dict]:
    url = url or get_match_results_feed_url()
    metadata = {
        "source_status": "missing_url" if not url else "not_started",
        "url": url,
        "status_code": None,
        "last_error": None,
        "rows": 0,
    }
    if not url:
        return pd.DataFrame(columns=MATCH_RESULTS_COLUMNS), metadata

    try:
        response = requests.get(url, timeout=timeout)
        metadata["status_code"] = response.status_code
        if response.status_code >= 400:
            metadata["source_status"] = "http_error"
            metadata["last_error"] = f"Result feed request failed with HTTP {response.status_code}."
            return pd.DataFrame(columns=MATCH_RESULTS_COLUMNS), metadata
        remote = pd.read_csv(StringIO(response.text))
        result = normalize_match_result_updates(remote)
        metadata["source_status"] = "ok"
        metadata["rows"] = int(len(result))
        return result, metadata
    except EmptyDataError:
        metadata["source_status"] = "empty"
        metadata["last_error"] = "Result feed was empty."
        return pd.DataFrame(columns=MATCH_RESULTS_COLUMNS), metadata
    except requests.Timeout as exc:
        metadata["source_status"] = "timeout"
        metadata["last_error"] = str(exc)
        return pd.DataFrame(columns=MATCH_RESULTS_COLUMNS), metadata
    except Exception as exc:
        metadata["source_status"] = "exception"
        metadata["last_error"] = str(exc)
        return pd.DataFrame(columns=MATCH_RESULTS_COLUMNS), metadata


def load_combined_match_result_updates(
    local_path: Union[str, Path] = MATCH_RESULTS_UPDATES_PATH,
    remote_url: Optional[str] = None,
) -> tuple[pd.DataFrame, dict]:
    local = load_match_result_updates(local_path)
    remote, remote_metadata = fetch_remote_match_result_updates(remote_url)
    if local.empty and remote.empty:
        combined = pd.DataFrame(columns=MATCH_RESULTS_COLUMNS)
    elif local.empty:
        combined = remote.copy()
    elif remote.empty:
        combined = local.copy()
    else:
        combined = pd.concat([local, remote], ignore_index=True)
    if not combined.empty:
        combined = combined.drop_duplicates(subset=["match_id"], keep="last")
        combined = normalize_match_result_updates(combined)
    return combined, {
        "local_rows": int(len(local)),
        "remote_rows": int(len(remote)),
        "remote": remote_metadata,
    }


def save_match_results(df: pd.DataFrame, path: Union[str, Path] = MATCH_RESULTS_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result = df.copy()
    for column in MATCH_RESULTS_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA
    result[MATCH_RESULTS_COLUMNS].to_csv(path, index=False)


def refresh_match_results(
    fixtures_path: Union[str, Path] = REFERENCE_FIXTURES_PATH,
    updates_path: Union[str, Path] = MATCH_RESULTS_UPDATES_PATH,
    results_path: Union[str, Path] = MATCH_RESULTS_PATH,
    as_of: Optional[pd.Timestamp] = None,
    remote_updates_url: Optional[str] = None,
) -> dict:
    fixtures = load_fixture_dataset(fixtures_path)
    current_results = load_match_results(results_path)
    updates, update_metadata = load_combined_match_result_updates(updates_path, remote_updates_url)
    as_of = pd.to_datetime(as_of, utc=True) if as_of is not None else _utc_now()

    kickoff_column = "kickoff_time" if "kickoff_time" in fixtures.columns else "kickoff_utc"
    fixture_lookup = fixtures[["match_id", kickoff_column]].rename(columns={kickoff_column: "kickoff_time"}).copy()
    fixture_lookup["kickoff_time"] = pd.to_datetime(fixture_lookup["kickoff_time"], errors="coerce", utc=True)
    archived_ids = set(current_results["match_id"].dropna().astype(str)) if not current_results.empty else set()
    past_fixture_ids = set(
        fixture_lookup.loc[
            fixture_lookup["kickoff_time"].notna() & fixture_lookup["kickoff_time"].le(as_of),
            "match_id",
        ]
        .dropna()
        .astype(str)
    )
    missing_finished_results = past_fixture_ids - archived_ids

    if updates.empty:
        return {
            "status": "missing_updates",
            "checked": 0,
            "added": 0,
            "already_archived": int(len(current_results)),
            "missing_finished_results": int(len(missing_finished_results)),
            "source": update_metadata,
            "message": (
                "No match result updates found. "
                "Add data/reference/match_results_updates.csv or MATCH_RESULTS_FEED_URL. "
                f"{len(missing_finished_results)} finished fixtures still need a result source."
            ),
        }

    eligible = updates.merge(fixture_lookup, on="match_id", how="inner")
    eligible = eligible[
        eligible["result_status"].eq("final")
        & eligible["kickoff_time"].notna()
        & eligible["kickoff_time"].le(as_of)
    ].copy()

    existing_ids = set(current_results["match_id"].dropna().astype(str)) if not current_results.empty else set()
    new_results = eligible[~eligible["match_id"].astype(str).isin(existing_ids)][MATCH_RESULTS_COLUMNS].copy()

    if not new_results.empty:
        if current_results.empty:
            combined = new_results.copy()
        else:
            combined = pd.concat([current_results[MATCH_RESULTS_COLUMNS], new_results], ignore_index=True)
        combined = combined.drop_duplicates(subset=["match_id"], keep="last")
        save_match_results(combined, results_path)
    else:
        combined = current_results

    archived_ids = set(combined["match_id"].dropna().astype(str)) if not combined.empty else set()
    missing_finished_results = past_fixture_ids - archived_ids

    return {
        "status": "ok",
        "checked": int(len(eligible)),
        "added": int(len(new_results)),
        "already_archived": int(len(combined)),
        "missing_finished_results": int(len(missing_finished_results)),
        "source": update_metadata,
        "message": (
            f"Match refresh checked {len(eligible)} completed matches "
            f"({update_metadata['local_rows']} local, {update_metadata['remote_rows']} remote) "
            f"and added {len(new_results)} new results. "
            f"{len(missing_finished_results)} finished fixtures still need a result source."
        ),
    }
