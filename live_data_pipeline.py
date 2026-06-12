import re
import unicodedata
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from config import LIVE_PREDICTIONS_PATH, PREFERRED_BOOKMAKER_NAMES, REQUIRED_PREDICTION_COLUMNS
from fixture_data import build_predictions_from_fixtures
from odds_mapping import (
    OUTCOME_ORDER,
    add_canonical_outcome,
    calculate_market_fair_probabilities_from_best_or_consensus,
    identify_best_market_odds,
    identify_preferred_bookmaker_odds,
)


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
