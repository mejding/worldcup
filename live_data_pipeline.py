import re
import unicodedata
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from config import (
    LIVE_PREDICTIONS_PATH,
    ODDS_API_DATE_FORMAT,
    ODDS_API_MARKETS,
    ODDS_API_ODDS_FORMAT,
    ODDS_API_REGIONS,
    ODDS_API_SPORT_KEY,
    ODDS_REFRESH_MINUTES,
    PREFERRED_BOOKMAKER_NAMES,
    PROCESSED_ODDS_PATH,
    RAW_ODDS_SNAPSHOT_PATH,
    REQUIRED_PREDICTION_COLUMNS,
)
from fixture_data import load_fixture_dataset
from fixture_data import build_predictions_from_fixtures
from manual_odds import load_manual_odds as load_manual_odds_wide, normalize_manual_odds
from odds_normalizer import normalize_the_odds_api_response
from odds_mapping import (
    OUTCOME_ORDER,
    add_canonical_outcome,
    build_match_odds_table,
    calculate_market_fair_probabilities_from_best_or_consensus,
    identify_best_market_odds,
    identify_preferred_bookmaker_odds,
    map_odds_to_fixtures,
)
from odds_provider import fetch_odds_from_the_odds_api, get_odds_api_key, get_odds_source_status
from odds_storage import load_latest_odds_snapshot, save_latest_odds, save_odds_snapshot


TEAM_ALIASES = {
    "usa": "unitedstates",
    "us": "unitedstates",
    "unitedstatesofamerica": "unitedstates",
    "turkiye": "turkiye",
    "turkey": "turkiye",
    "iran": "iriran",
    "islamicrepublicofiran": "iriran",
    "drcongo": "congodr",
    "democraticrepublicofcongo": "congodr",
    "ivorycoast": "cotedivoire",
    "curacao": "curacao",
    "capeverde": "caboverde",
    "southkorea": "southkorea",
    "korearepublic": "southkorea",
    "republicofkorea": "southkorea",
}


def _normalize_team_name(value) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-z0-9]+", "", text.lower())
    return TEAM_ALIASES.get(text, text)


def _fixture_signature(home_team, away_team) -> tuple[str, str]:
    return (_normalize_team_name(home_team), _normalize_team_name(away_team))


def _fixture_date(value) -> str:
    timestamp = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(timestamp):
        return ""
    return timestamp.date().isoformat()


def _complete_event_ids(odds_df: pd.DataFrame) -> set:
    canonical = add_canonical_outcome(odds_df)
    complete = set()
    for event_id, event_df in canonical.groupby("event_id"):
        outcomes = set(event_df["canonical_outcome"].unique())
        if all(outcome in outcomes for outcome in OUTCOME_ORDER):
            complete.add(event_id)
    return complete


def _fixture_lookup(fixtures_df: Optional[pd.DataFrame]) -> dict:
    if fixtures_df is None or fixtures_df.empty:
        return {"by_id": {}, "by_signature_date": {}, "by_signature": {}}
    id_key = "match_id" if "match_id" in fixtures_df.columns else "event_id"
    by_id = {}
    by_signature_date = {}
    by_signature = {}
    for _, row in fixtures_df.iterrows():
        by_id[row[id_key]] = row
        signature = _fixture_signature(row.get("home_team"), row.get("away_team"))
        by_signature.setdefault(signature, row)
        by_signature_date[(signature, _fixture_date(row.get("kickoff_utc", row.get("kickoff_time"))))] = row
    return {
        "by_id": by_id,
        "by_signature_date": by_signature_date,
        "by_signature": by_signature,
    }


def _match_fixture(row, fixtures: dict):
    fixture = fixtures["by_id"].get(row["event_id"])
    if fixture is not None:
        return fixture
    signature = _fixture_signature(row.get("home_team"), row.get("away_team"))
    dated = fixtures["by_signature_date"].get((signature, _fixture_date(row.get("commence_time"))))
    if dated is not None:
        return dated
    return fixtures["by_signature"].get(signature)


def _empty_prediction_columns() -> list[str]:
    return REQUIRED_PREDICTION_COLUMNS + ["kickoff_utc", "fixture_source"]


def _write_live_predictions(df: pd.DataFrame, output_path: Union[str, Path]) -> pd.DataFrame:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def _snapshot_is_stale(path: Union[str, Path] = RAW_ODDS_SNAPSHOT_PATH, refresh_minutes: int = ODDS_REFRESH_MINUTES) -> bool:
    latest = load_latest_odds_snapshot(path)
    if latest.empty:
        return True
    timestamps = pd.to_datetime(latest["fetched_at_utc"], errors="coerce", utc=True)
    if timestamps.isna().all():
        return True
    age_minutes = (pd.Timestamp.utcnow() - timestamps.max()).total_seconds() / 60
    return age_minutes >= refresh_minutes


def _blank_prediction_row(fixture_row) -> dict:
    return {
        "match_id": fixture_row["match_id"],
        "kickoff_time": fixture_row.get("kickoff_utc"),
        "kickoff_utc": fixture_row.get("kickoff_utc"),
        "fixture_source": fixture_row.get("fixture_source", "official_reference"),
        "group": fixture_row.get("group", "TBD"),
        "matchday": fixture_row.get("matchday", 0),
        "home_team": fixture_row.get("home_team"),
        "away_team": fixture_row.get("away_team"),
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
        "odds_available": False,
        "odds_source": pd.NA,
        "odds_provider": pd.NA,
        "odds_last_updated_utc": pd.NA,
        "bookmaker_count": 0,
        "available_bookmakers": "",
    }


def _predictions_from_match_odds_table(match_odds_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, match in match_odds_df.iterrows():
        row = _blank_prediction_row(match)
        if bool(match.get("odds_available")) and not pd.isna(match.get("market_home_prob")):
            row.update(
                {
                    "model_home_prob": match["market_home_prob"],
                    "model_draw_prob": match["market_draw_prob"],
                    "model_away_prob": match["market_away_prob"],
                    "market_home_prob": match["market_home_prob"],
                    "market_draw_prob": match["market_draw_prob"],
                    "market_away_prob": match["market_away_prob"],
                    "ds_home_odds": match.get("ds_home_odds", pd.NA),
                    "ds_draw_odds": match.get("ds_draw_odds", pd.NA),
                    "ds_away_odds": match.get("ds_away_odds", pd.NA),
                    "best_home_odds": match.get("best_home_odds", pd.NA),
                    "best_home_bookmaker": match.get("best_home_bookmaker", pd.NA),
                    "best_draw_odds": match.get("best_draw_odds", pd.NA),
                    "best_draw_bookmaker": match.get("best_draw_bookmaker", pd.NA),
                    "best_away_odds": match.get("best_away_odds", pd.NA),
                    "best_away_bookmaker": match.get("best_away_bookmaker", pd.NA),
                    "odds_available": True,
                    "odds_source": match.get("odds_source", pd.NA),
                    "odds_provider": match.get("odds_provider", pd.NA),
                    "odds_last_updated_utc": match.get("odds_last_updated_utc", pd.NA),
                    "bookmaker_count": match.get("bookmaker_count", 0),
                    "available_bookmakers": match.get("available_bookmakers", ""),
                }
            )
        rows.append(row)

    result = pd.DataFrame(rows)
    for column in _empty_prediction_columns():
        if column not in result.columns:
            result[column] = pd.NA
    return result


def _select_odds_source(force_refresh: bool = False) -> tuple[pd.DataFrame, dict, list[str]]:
    warnings = []
    metadata = {}
    api_key = get_odds_api_key()
    if api_key and (force_refresh or _snapshot_is_stale()):
        response_json, metadata = fetch_odds_from_the_odds_api(
            sport_key=ODDS_API_SPORT_KEY,
            regions=ODDS_API_REGIONS,
            markets=ODDS_API_MARKETS,
            odds_format=ODDS_API_ODDS_FORMAT,
            date_format=ODDS_API_DATE_FORMAT,
            api_key=api_key,
        )
        normalized = normalize_the_odds_api_response(response_json, metadata.get("fetched_at_utc"))
        if not normalized.empty:
            save_odds_snapshot(normalized)
            normalized["odds_source"] = "api"
            return normalized, metadata | {"active_odds_source": "api"}, warnings
        if metadata.get("last_error"):
            warnings.append(metadata["last_error"])

    if api_key:
        cached = load_latest_odds_snapshot()
        if not cached.empty:
            cached = cached.copy()
            cached["odds_source"] = "cached_snapshot"
            return cached, metadata | {"active_odds_source": "cached"}, warnings

    manual_df, manual_warnings = load_manual_odds_wide()
    if not manual_df.empty:
        warnings.extend(manual_warnings)
        manual_normalized = normalize_manual_odds(manual_df)
        return manual_normalized, {"active_odds_source": "manual", "source_status": "ok"}, warnings
    warnings.extend(manual_warnings)

    cached = load_latest_odds_snapshot()
    if not cached.empty:
        cached = cached.copy()
        cached["odds_source"] = "cached_snapshot"
        return cached, {"active_odds_source": "cached", "source_status": "ok"}, warnings

    return pd.DataFrame(), {"active_odds_source": "missing", "source_status": "missing"}, warnings


def refresh_live_odds_and_predictions(force_refresh: bool = False) -> dict:
    fixtures_df = load_fixture_dataset()
    status = get_odds_source_status()
    odds_df, metadata, warnings = _select_odds_source(force_refresh=force_refresh)
    mapped_df, mapping_warnings = map_odds_to_fixtures(fixtures_df, odds_df)
    warnings.extend(mapping_warnings)
    match_odds_df = build_match_odds_table(mapped_df)
    save_latest_odds(match_odds_df, PROCESSED_ODDS_PATH)
    live_df = _predictions_from_match_odds_table(match_odds_df)
    _write_live_predictions(live_df, LIVE_PREDICTIONS_PATH)

    odds_available = int(match_odds_df["odds_available"].sum()) if "odds_available" in match_odds_df.columns else 0
    return {
        "status": "ok" if odds_available else "missing_odds",
        "active_odds_source": metadata.get("active_odds_source", status.get("active_odds_source", "missing")),
        "source_status": metadata.get("source_status"),
        "provider": metadata.get("provider"),
        "sport_key": metadata.get("sport_key", ODDS_API_SPORT_KEY),
        "regions": metadata.get("regions", ODDS_API_REGIONS),
        "markets": metadata.get("markets", ODDS_API_MARKETS),
        "requests_remaining": metadata.get("requests_remaining"),
        "requests_used": metadata.get("requests_used"),
        "last_error": metadata.get("last_error") or status.get("last_error"),
        "warnings": warnings,
        "matches_total": int(len(match_odds_df)),
        "matches_with_odds": odds_available,
        "matches_missing_odds": int(len(match_odds_df) - odds_available),
        "bookmaker_count": int(match_odds_df["bookmaker_count"].max()) if "bookmaker_count" in match_odds_df.columns and not match_odds_df.empty else 0,
        "latest_odds_path": str(PROCESSED_ODDS_PATH),
        "live_predictions_path": str(LIVE_PREDICTIONS_PATH),
    }


def build_live_predictions(
    odds_df: pd.DataFrame,
    fixtures_df: Optional[pd.DataFrame] = None,
    preferred_bookmaker_names: Optional[list[str]] = None,
    market_probability_method: str = "consensus",
    output_path: Union[str, Path] = LIVE_PREDICTIONS_PATH,
) -> pd.DataFrame:
    preferred_bookmaker_names = preferred_bookmaker_names or PREFERRED_BOOKMAKER_NAMES
    fixture_base = (
        build_predictions_from_fixtures(fixtures_df, fixture_source="official_reference")
        if fixtures_df is not None and not fixtures_df.empty
        else pd.DataFrame(columns=_empty_prediction_columns())
    )
    if odds_df.empty:
        return _write_live_predictions(fixture_base, output_path)

    complete_event_ids = _complete_event_ids(odds_df)
    if not complete_event_ids:
        return _write_live_predictions(fixture_base, output_path)

    complete_odds = odds_df[odds_df["event_id"].isin(complete_event_ids)].copy()
    preferred = identify_preferred_bookmaker_odds(complete_odds, preferred_bookmaker_names)
    best = identify_best_market_odds(complete_odds)
    probabilities = calculate_market_fair_probabilities_from_best_or_consensus(
        complete_odds,
        method=market_probability_method,
    )
    if probabilities.empty:
        return _write_live_predictions(fixture_base, output_path)

    merged = best.merge(preferred, on=["event_id", "commence_time", "home_team", "away_team"], how="left")
    merged = merged.merge(probabilities, on="event_id", how="inner")
    fixtures = _fixture_lookup(fixtures_df)

    rows_by_match_id = {}
    for _, row in merged.iterrows():
        fixture = _match_fixture(row, fixtures)
        match_id = fixture.get("match_id", row["event_id"]) if fixture is not None else row["event_id"]
        rows_by_match_id[match_id] = (
            {
                "match_id": match_id,
                "kickoff_time": fixture.get("kickoff_utc", row["commence_time"]) if fixture is not None else row["commence_time"],
                "kickoff_utc": fixture.get("kickoff_utc", row["commence_time"]) if fixture is not None else row["commence_time"],
                "fixture_source": "official_reference" if fixture is not None else "odds_api_event",
                "group": fixture.get("group", "TBD") if fixture is not None else "TBD",
                "matchday": fixture.get("matchday", 0) if fixture is not None else 0,
                "home_team": fixture.get("home_team", row["home_team"]) if fixture is not None else row["home_team"],
                "away_team": fixture.get("away_team", row["away_team"]) if fixture is not None else row["away_team"],
                "model_home_prob": row["market_home_prob"],
                "model_draw_prob": row["market_draw_prob"],
                "model_away_prob": row["market_away_prob"],
                "market_home_prob": row["market_home_prob"],
                "market_draw_prob": row["market_draw_prob"],
                "market_away_prob": row["market_away_prob"],
                "ds_home_odds": row.get("ds_home_odds", pd.NA),
                "ds_draw_odds": row.get("ds_draw_odds", pd.NA),
                "ds_away_odds": row.get("ds_away_odds", pd.NA),
                "best_home_odds": row["best_home_odds"],
                "best_home_bookmaker": row["best_home_bookmaker"],
                "best_draw_odds": row["best_draw_odds"],
                "best_draw_bookmaker": row["best_draw_bookmaker"],
                "best_away_odds": row["best_away_odds"],
                "best_away_bookmaker": row["best_away_bookmaker"],
                "draw_context_score": 50,
                "draw_context_label": "Medium",
                "home_draw_utility": 0.0,
                "away_draw_utility": 0.0,
                "mutual_draw_acceptance": 0.0,
                "one_team_must_win": False,
                "both_teams_draw_satisfied": False,
            }
        )

    if not fixture_base.empty:
        rows = []
        for _, fixture_row in fixture_base.iterrows():
            rows.append(rows_by_match_id.get(fixture_row["match_id"], fixture_row.to_dict()))
        orphan_rows = [
            row for match_id, row in rows_by_match_id.items()
            if match_id not in set(fixture_base["match_id"])
        ]
        rows.extend(orphan_rows)
        result = pd.DataFrame(rows)
    else:
        result = pd.DataFrame(rows_by_match_id.values())

    for column in _empty_prediction_columns():
        if column not in result.columns:
            result[column] = pd.NA
    result = result[_empty_prediction_columns()]
    return _write_live_predictions(result, output_path)
