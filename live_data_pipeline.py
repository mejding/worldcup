from pathlib import Path
from typing import Optional, Union

import pandas as pd

from config import LIVE_PREDICTIONS_PATH, PREFERRED_BOOKMAKER_NAMES, REQUIRED_PREDICTION_COLUMNS
from odds_mapping import (
    OUTCOME_ORDER,
    add_canonical_outcome,
    calculate_market_fair_probabilities_from_best_or_consensus,
    identify_best_market_odds,
    identify_preferred_bookmaker_odds,
)


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
        return {}
    key = "match_id" if "match_id" in fixtures_df.columns else "event_id"
    return {row[key]: row for _, row in fixtures_df.iterrows()}


def build_live_predictions(
    odds_df: pd.DataFrame,
    fixtures_df: Optional[pd.DataFrame] = None,
    preferred_bookmaker_names: Optional[list[str]] = None,
    market_probability_method: str = "consensus",
    output_path: Union[str, Path] = LIVE_PREDICTIONS_PATH,
) -> pd.DataFrame:
    preferred_bookmaker_names = preferred_bookmaker_names or PREFERRED_BOOKMAKER_NAMES
    if odds_df.empty:
        return pd.DataFrame(columns=REQUIRED_PREDICTION_COLUMNS)

    complete_event_ids = _complete_event_ids(odds_df)
    if not complete_event_ids:
        return pd.DataFrame(columns=REQUIRED_PREDICTION_COLUMNS)

    complete_odds = odds_df[odds_df["event_id"].isin(complete_event_ids)].copy()
    preferred = identify_preferred_bookmaker_odds(complete_odds, preferred_bookmaker_names)
    best = identify_best_market_odds(complete_odds)
    probabilities = calculate_market_fair_probabilities_from_best_or_consensus(
        complete_odds,
        method=market_probability_method,
    )
    if probabilities.empty:
        return pd.DataFrame(columns=REQUIRED_PREDICTION_COLUMNS)

    merged = best.merge(preferred, on=["event_id", "commence_time", "home_team", "away_team"], how="left")
    merged = merged.merge(probabilities, on="event_id", how="inner")
    fixtures = _fixture_lookup(fixtures_df)

    rows = []
    for _, row in merged.iterrows():
        fixture = fixtures.get(row["event_id"])
        rows.append(
            {
                "match_id": row["event_id"],
                "kickoff_time": row["commence_time"],
                "group": fixture.get("group", "TBD") if fixture is not None else "TBD",
                "matchday": fixture.get("matchday", 0) if fixture is not None else 0,
                "home_team": row["home_team"],
                "away_team": row["away_team"],
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

    result = pd.DataFrame(rows, columns=REQUIRED_PREDICTION_COLUMNS)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result

