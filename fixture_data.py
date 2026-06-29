from pathlib import Path
from typing import Union

import pandas as pd

from config import REFERENCE_FIXTURES_PATH, REQUIRED_PREDICTION_COLUMNS
from time_utils import add_danish_kickoff_column


EXPECTED_WORLD_CUP_MATCH_COUNT = 104

REQUIRED_FIXTURE_COLUMNS = [
    "match_id",
    "match_number",
    "kickoff_utc",
    "kickoff_local",
    "kickoff_timezone",
    "home_team",
    "away_team",
    "group",
    "stage",
    "matchday",
    "city",
    "stadium",
    "source",
    "source_last_checked",
]


def load_fixture_dataset(path: Union[str, Path] = REFERENCE_FIXTURES_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=REQUIRED_FIXTURE_COLUMNS)
    return pd.read_csv(path)


def _is_complete_fixture_dataset(df: pd.DataFrame) -> bool:
    if "is_complete" not in df.columns or df.empty:
        return False
    return df["is_complete"].astype(str).str.lower().eq("true").all()


def validate_fixture_dataset(df: pd.DataFrame) -> tuple[bool, list[str]]:
    messages = []
    if df.empty:
        return False, ["Official fixture data is missing. Add or refresh World Cup 2026 fixtures before using live mode."]

    missing_columns = [column for column in REQUIRED_FIXTURE_COLUMNS if column not in df.columns]
    if missing_columns:
        return False, [f"Missing fixture columns: {', '.join(missing_columns)}"]

    valid = True
    if df["match_id"].duplicated().any():
        valid = False
        messages.append("Fixture dataset has duplicate match_id values.")

    if pd.to_datetime(df["kickoff_utc"], errors="coerce", utc=True).isna().any():
        valid = False
        messages.append("One or more kickoff_utc values are not parseable.")

    for column in ["home_team", "away_team"]:
        if df[column].isna().any() or df[column].astype(str).str.strip().eq("").any():
            valid = False
            messages.append(f"Fixture column {column} must not be empty.")

    group_stage = df["stage"].astype(str).str.lower().eq("group stage")
    if group_stage.any():
        if df.loc[group_stage, "group"].isna().any() or df.loc[group_stage, "group"].astype(str).str.strip().eq("").any():
            valid = False
            messages.append("Group-stage fixtures must include group.")
        if df.loc[group_stage, "stage"].isna().any():
            valid = False
            messages.append("Group-stage fixtures must include stage.")

    complete = _is_complete_fixture_dataset(df)
    expected_count = int(df.get("expected_match_count", pd.Series([EXPECTED_WORLD_CUP_MATCH_COUNT])).dropna().iloc[0])
    if complete and len(df) != expected_count:
        valid = False
        messages.append(f"Fixture dataset is marked complete but has {len(df)} of {expected_count} expected matches.")
    if not complete:
        valid = False
        messages.append(f"Fixture dataset is incomplete: {len(df)} of {expected_count} expected matches loaded.")

    parsed_dates = pd.to_datetime(df["kickoff_utc"], errors="coerce", utc=True).dt.date.astype(str)
    canada_bosnia = df[
        (df["home_team"].astype(str).str.casefold() == "canada")
        & (df["away_team"].astype(str).str.casefold() == "bosnia and herzegovina")
        & (parsed_dates == "2026-06-12")
    ]
    if canada_bosnia.empty:
        valid = False
        messages.append("Known fixture missing: Canada vs Bosnia and Herzegovina on 2026-06-12.")

    canada_switzerland_wrong_date = df[
        (df["home_team"].astype(str).str.casefold() == "canada")
        & (df["away_team"].astype(str).str.casefold() == "switzerland")
        & (parsed_dates == "2026-06-12")
    ]
    if not canada_switzerland_wrong_date.empty:
        valid = False
        messages.append("Invalid fixture: Canada vs Switzerland must not be on 2026-06-12.")

    group_b = df[df["group"].astype(str).str.upper() == "B"]
    if not group_b.empty:
        teams = set(group_b["home_team"].dropna()) | set(group_b["away_team"].dropna())
        expected_group_b = {"Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"}
        missing_group_b = sorted(expected_group_b - teams)
        if missing_group_b:
            valid = False
            messages.append("Group B fixture data is missing teams: " + ", ".join(missing_group_b))

    return valid, messages


def fixture_provenance(df: pd.DataFrame, mode: str) -> dict:
    if mode == "sample":
        return {
            "label": "Sample/demo data",
            "loaded": len(df),
            "expected": EXPECTED_WORLD_CUP_MATCH_COUNT,
            "last_checked": "not official",
            "is_complete": False,
        }
    if df.empty:
        return {
            "label": "Official reference missing",
            "loaded": 0,
            "expected": EXPECTED_WORLD_CUP_MATCH_COUNT,
            "last_checked": "-",
            "is_complete": False,
        }
    expected = int(df.get("expected_match_count", pd.Series([EXPECTED_WORLD_CUP_MATCH_COUNT])).dropna().iloc[0])
    complete = _is_complete_fixture_dataset(df)
    return {
        "label": "Official reference" if complete else "Incomplete official reference",
        "loaded": len(df),
        "expected": expected,
        "last_checked": str(df["source_last_checked"].dropna().max()) if "source_last_checked" in df.columns else "-",
        "is_complete": complete,
    }


def build_predictions_from_fixtures(fixtures_df: pd.DataFrame, fixture_source: str = "official_reference") -> pd.DataFrame:
    rows = []
    for _, fixture in fixtures_df.iterrows():
        rows.append(
            {
                "match_id": fixture["match_id"],
                "kickoff_time": fixture["kickoff_utc"],
                "kickoff_utc": fixture["kickoff_utc"],
                "fixture_source": fixture_source,
                "group": fixture["group"],
                "stage": fixture.get("stage", pd.NA),
                "matchday": fixture["matchday"],
                "home_team": fixture["home_team"],
                "away_team": fixture["away_team"],
                "model_home_prob": 1 / 3,
                "model_draw_prob": 1 / 3,
                "model_away_prob": 1 / 3,
                "market_home_prob": 1 / 3,
                "market_draw_prob": 1 / 3,
                "market_away_prob": 1 / 3,
                "ds_home_odds": pd.NA,
                "ds_draw_odds": pd.NA,
                "ds_away_odds": pd.NA,
                "best_home_odds": pd.NA,
                "best_home_bookmaker": pd.NA,
                "best_draw_odds": pd.NA,
                "best_draw_bookmaker": pd.NA,
                "best_away_odds": pd.NA,
                "best_away_bookmaker": pd.NA,
                "draw_context_score": 50,
                "draw_context_label": "Medium",
                "home_draw_utility": 0.0,
                "away_draw_utility": 0.0,
                "mutual_draw_acceptance": 0.0,
                "one_team_must_win": False,
                "both_teams_draw_satisfied": False,
            }
        )
    result = pd.DataFrame(rows)
    for column in REQUIRED_PREDICTION_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA
    return add_danish_kickoff_column(result)


def validate_prediction_fixture_consistency(predictions: pd.DataFrame, fixtures: pd.DataFrame) -> list[str]:
    warnings = []
    if fixtures.empty:
        return ["Missing fixture reference; predictions cannot be verified against official fixtures."]
    if predictions.empty:
        return warnings
    if "fixture_source" in predictions.columns and predictions["fixture_source"].astype(str).str.contains("sample", case=False, na=False).any():
        warnings.append("Predictions are based on sample fixtures and must not be used in live/official mode.")

    fixture_lookup = fixtures.set_index("match_id")
    for _, prediction in predictions.iterrows():
        match_id = prediction.get("match_id")
        if match_id not in fixture_lookup.index:
            warnings.append(f"Prediction match_id {match_id} does not exist in fixture source.")
            continue
        fixture = fixture_lookup.loc[match_id]
        if prediction.get("home_team") != fixture.get("home_team") or prediction.get("away_team") != fixture.get("away_team"):
            warnings.append(f"Prediction teams do not match fixture source for {match_id}.")
        prediction_kickoff = pd.to_datetime(prediction.get("kickoff_utc", prediction.get("kickoff_time")), errors="coerce", utc=True)
        fixture_kickoff = pd.to_datetime(fixture.get("kickoff_utc"), errors="coerce", utc=True)
        if pd.notna(prediction_kickoff) and pd.notna(fixture_kickoff):
            if abs((prediction_kickoff - fixture_kickoff).total_seconds()) > 60:
                warnings.append(f"Prediction kickoff is stale or mismatched for {match_id}.")
    return warnings
