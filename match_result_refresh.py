from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from pandas.errors import EmptyDataError

from config import MATCH_RESULTS_PATH, MATCH_RESULTS_UPDATES_PATH, REFERENCE_FIXTURES_PATH
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
) -> dict:
    fixtures = load_fixture_dataset(fixtures_path)
    current_results = load_match_results(results_path)
    updates = load_match_result_updates(updates_path)
    as_of = pd.to_datetime(as_of, utc=True) if as_of is not None else _utc_now()

    if updates.empty:
        return {
            "status": "missing_updates",
            "checked": 0,
            "added": 0,
            "already_archived": int(len(current_results)),
            "message": "No match result updates file found.",
        }

    kickoff_column = "kickoff_time" if "kickoff_time" in fixtures.columns else "kickoff_utc"
    fixture_lookup = fixtures[["match_id", kickoff_column]].rename(columns={kickoff_column: "kickoff_time"}).copy()
    fixture_lookup["kickoff_time"] = pd.to_datetime(fixture_lookup["kickoff_time"], errors="coerce", utc=True)
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

    return {
        "status": "ok",
        "checked": int(len(eligible)),
        "added": int(len(new_results)),
        "already_archived": int(len(combined)),
        "message": f"Match refresh checked {len(eligible)} completed matches and added {len(new_results)} new results.",
    }
