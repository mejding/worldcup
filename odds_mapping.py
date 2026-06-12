import re
import unicodedata
from typing import Optional

import pandas as pd

from fetch_odds import is_draw_outcome
from odds_normalizer import NORMALIZED_ODDS_COLUMNS
from odds_utils import decimal_odds_to_implied_probability, remove_overround_proportional


OUTCOME_ORDER = ("home", "draw", "away")
DEFAULT_PREFERRED_BOOKMAKER_NAMES = ["Danske Spil", "DanskeSpil", "Danske Spil A/S", "danske_spil"]
BOOKMAKER_DS_ALIASES = {"danske spil", "danskespil", "danske_spil"}
TEAM_ALIASES = {
    "usa": "unitedstates",
    "us": "unitedstates",
    "usmnt": "unitedstates",
    "unitedstatesofamerica": "unitedstates",
    "bosniaherzegovina": "bosniaandherzegovina",
    "bosniaandherzegovina": "bosniaandherzegovina",
    "cotedivoire": "cotedivoire",
    "ivorycoast": "cotedivoire",
    "korearepublic": "southkorea",
    "southkorea": "southkorea",
    "republicofkorea": "southkorea",
    "iriran": "iran",
    "iran": "iran",
    "islamicrepublicofiran": "iran",
    "turkiye": "turkiye",
    "turkey": "turkiye",
}


MATCH_ODDS_COLUMNS = [
    "match_id",
    "kickoff_utc",
    "home_team",
    "away_team",
    "group",
    "stage",
    "matchday",
    "odds_last_updated_utc",
    "odds_provider",
    "odds_source",
    "odds_available",
    "bookmaker_count",
    "available_bookmakers",
    "ds_home_odds",
    "ds_draw_odds",
    "ds_away_odds",
    "ds_bookmaker_key",
    "ds_bookmaker_title",
    "best_home_odds",
    "best_home_bookmaker",
    "best_draw_odds",
    "best_draw_bookmaker",
    "best_away_odds",
    "best_away_bookmaker",
    "market_home_prob",
    "market_draw_prob",
    "market_away_prob",
]


def normalize_team_name(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-z0-9]+", "", text.lower())
    return TEAM_ALIASES.get(text, text)


def _fixture_kickoff(row):
    return row.get("kickoff_utc", row.get("kickoff_time"))


def _fixture_stage(row):
    return row.get("stage", "Group stage")


def _bookmaker_is_danske_spil(row) -> bool:
    title = str(row.get("bookmaker_title", "")).strip().lower()
    key = str(row.get("bookmaker_key", "")).strip().lower()
    compact_title = re.sub(r"[^a-z0-9]+", "", title)
    return title in BOOKMAKER_DS_ALIASES or key in BOOKMAKER_DS_ALIASES or compact_title == "danskespil"


def _canonical_outcome(row) -> Optional[str]:
    name = str(row["outcome_name"]).strip()
    if name == str(row["home_team"]).strip():
        return "home"
    if name == str(row["away_team"]).strip():
        return "away"
    if is_draw_outcome(name):
        return "draw"
    return None


def add_canonical_outcome(odds_df: pd.DataFrame) -> pd.DataFrame:
    if odds_df.empty:
        result = odds_df.copy()
        result["canonical_outcome"] = []
        return result
    result = odds_df.copy()
    result["canonical_outcome"] = result.apply(_canonical_outcome, axis=1)
    return result.dropna(subset=["canonical_outcome"])


def _event_base(event_id, event_df):
    first = event_df.iloc[0]
    return {
        "event_id": event_id,
        "commence_time": first["commence_time"],
        "home_team": first["home_team"],
        "away_team": first["away_team"],
    }


def identify_preferred_bookmaker_odds(
    odds_df: pd.DataFrame,
    preferred_bookmaker_names: list[str] = None,
) -> pd.DataFrame:
    preferred_bookmaker_names = preferred_bookmaker_names or DEFAULT_PREFERRED_BOOKMAKER_NAMES
    canonical = add_canonical_outcome(odds_df)
    rows = []
    for event_id, event_df in canonical.groupby("event_id"):
        row = _event_base(event_id, event_df)
        row["warning"] = ""
        preferred_mask = (
            event_df["bookmaker_title"].isin(preferred_bookmaker_names)
            | event_df["bookmaker_key"].isin(preferred_bookmaker_names)
        )
        preferred = event_df[preferred_mask]
        if preferred.empty:
            row.update({"ds_home_odds": pd.NA, "ds_draw_odds": pd.NA, "ds_away_odds": pd.NA})
            row["warning"] = "Danske Spil odds unavailable for this match."
        else:
            for outcome in OUTCOME_ORDER:
                prices = preferred[preferred["canonical_outcome"] == outcome]["outcome_price"]
                row[f"ds_{outcome}_odds"] = float(prices.iloc[0]) if not prices.empty else pd.NA
        rows.append(row)
    return pd.DataFrame(rows)


def identify_best_market_odds(odds_df: pd.DataFrame) -> pd.DataFrame:
    canonical = add_canonical_outcome(odds_df)
    rows = []
    for event_id, event_df in canonical.groupby("event_id"):
        row = _event_base(event_id, event_df)
        for outcome in OUTCOME_ORDER:
            outcome_df = event_df[event_df["canonical_outcome"] == outcome]
            if outcome_df.empty:
                row[f"best_{outcome}_odds"] = pd.NA
                row[f"best_{outcome}_bookmaker"] = pd.NA
            else:
                best = outcome_df.sort_values("outcome_price", ascending=False).iloc[0]
                row[f"best_{outcome}_odds"] = float(best["outcome_price"])
                row[f"best_{outcome}_bookmaker"] = best["bookmaker_title"]
        rows.append(row)
    return pd.DataFrame(rows)


def _complete_bookmaker_groups(canonical: pd.DataFrame):
    group_cols = ["event_id", "bookmaker_key", "bookmaker_title"]
    for group_key, group_df in canonical.groupby(group_cols):
        odds = {}
        for outcome in OUTCOME_ORDER:
            prices = group_df[group_df["canonical_outcome"] == outcome]["outcome_price"]
            if prices.empty:
                odds = {}
                break
            odds[outcome] = float(prices.iloc[0])
        if odds:
            yield group_key, group_df, odds


def calculate_market_fair_probabilities_from_best_or_consensus(
    odds_df: pd.DataFrame,
    method: str = "consensus",
) -> pd.DataFrame:
    canonical = add_canonical_outcome(odds_df)
    if canonical.empty:
        return pd.DataFrame(columns=["event_id", "market_home_prob", "market_draw_prob", "market_away_prob"])

    rows = []
    best_odds = identify_best_market_odds(canonical)
    preferred_odds = identify_preferred_bookmaker_odds(canonical)
    for event_id, event_df in canonical.groupby("event_id"):
        if method == "consensus":
            fair_probs = []
            for _, _, odds in _complete_bookmaker_groups(event_df):
                implied = [decimal_odds_to_implied_probability(odds[outcome]) for outcome in OUTCOME_ORDER]
                fair_probs.append(remove_overround_proportional(implied))
            if not fair_probs:
                continue
            avg_probs = pd.DataFrame(fair_probs, columns=OUTCOME_ORDER).mean().tolist()
            probs = remove_overround_proportional(avg_probs)
        elif method == "best":
            row = best_odds[best_odds["event_id"] == event_id]
            if row.empty or row[[f"best_{outcome}_odds" for outcome in OUTCOME_ORDER]].isna().any().any():
                continue
            odds = [float(row.iloc[0][f"best_{outcome}_odds"]) for outcome in OUTCOME_ORDER]
            probs = remove_overround_proportional([decimal_odds_to_implied_probability(odd) for odd in odds])
        elif method == "preferred":
            row = preferred_odds[preferred_odds["event_id"] == event_id]
            if row.empty or row[[f"ds_{outcome}_odds" for outcome in OUTCOME_ORDER]].isna().any().any():
                continue
            odds = [float(row.iloc[0][f"ds_{outcome}_odds"]) for outcome in OUTCOME_ORDER]
            probs = remove_overround_proportional([decimal_odds_to_implied_probability(odd) for odd in odds])
        else:
            raise ValueError("method must be consensus, best or preferred.")
        rows.append(
            {
                "event_id": event_id,
                "market_home_prob": probs[0],
                "market_draw_prob": probs[1],
                "market_away_prob": probs[2],
            }
        )
    return pd.DataFrame(rows)


def _empty_mapped_fixture_row(fixture) -> dict:
    return {
        "match_id": fixture["match_id"],
        "kickoff_utc": _fixture_kickoff(fixture),
        "home_team": fixture["home_team"],
        "away_team": fixture["away_team"],
        "group": fixture.get("group", "TBD"),
        "stage": _fixture_stage(fixture),
        "matchday": fixture.get("matchday", 0),
        "odds_available": False,
        **{column: pd.NA for column in NORMALIZED_ODDS_COLUMNS},
    }


def _fixture_signature(row) -> tuple[str, str]:
    return normalize_team_name(row.get("home_team")), normalize_team_name(row.get("away_team"))


def _match_odds_event_to_fixture(event_df: pd.DataFrame, fixtures_df: pd.DataFrame, tolerance_hours: int):
    event_id = str(event_df.iloc[0].get("event_id", ""))
    exact = fixtures_df[fixtures_df["match_id"].astype(str) == event_id]
    if not exact.empty:
        return exact.iloc[0], "match_id"

    first = event_df.iloc[0]
    odds_signature = _fixture_signature(first)
    odds_time = pd.to_datetime(first.get("commence_time_utc"), errors="coerce", utc=True)
    candidates = fixtures_df[
        fixtures_df.apply(lambda row: _fixture_signature(row) == odds_signature, axis=1)
    ].copy()
    if candidates.empty:
        return None, None
    if pd.isna(odds_time):
        return candidates.iloc[0], "teams"

    candidate_times = pd.to_datetime(
        candidates.apply(_fixture_kickoff, axis=1),
        errors="coerce",
        utc=True,
    )
    deltas = (candidate_times - odds_time).abs()
    within = candidates.loc[deltas <= pd.Timedelta(hours=tolerance_hours)]
    if within.empty:
        return None, None
    return within.iloc[0], "teams_kickoff"


def map_odds_to_fixtures(
    fixtures_df: pd.DataFrame,
    odds_df: pd.DataFrame,
    kickoff_tolerance_hours: int = 24,
) -> tuple[pd.DataFrame, list[str]]:
    warnings = []
    if fixtures_df.empty:
        return pd.DataFrame(), ["No fixtures available for odds mapping."]

    rows = []
    matched_match_ids = set()
    matched_event_ids = set()
    normalized_odds = odds_df.copy() if odds_df is not None else pd.DataFrame()
    if normalized_odds.empty:
        for _, fixture in fixtures_df.iterrows():
            rows.append(_empty_mapped_fixture_row(fixture))
        warnings.append("No odds rows available. Fixtures remain visible without odds.")
        return pd.DataFrame(rows), warnings

    for event_id, event_df in normalized_odds.groupby("event_id", dropna=False):
        fixture, match_method = _match_odds_event_to_fixture(event_df, fixtures_df, kickoff_tolerance_hours)
        if fixture is None:
            warnings.append(f"Odds event {event_id} could not be matched to an official fixture.")
            continue
        matched_match_ids.add(fixture["match_id"])
        matched_event_ids.add(event_id)
        for _, odds_row in event_df.iterrows():
            row = odds_row.to_dict()
            row.update(
                {
                    "match_id": fixture["match_id"],
                    "kickoff_utc": _fixture_kickoff(fixture),
                    "home_team": fixture["home_team"],
                    "away_team": fixture["away_team"],
                    "group": fixture.get("group", "TBD"),
                    "stage": _fixture_stage(fixture),
                    "matchday": fixture.get("matchday", 0),
                    "odds_available": True,
                    "odds_match_method": match_method,
                }
            )
            rows.append(row)

    for _, fixture in fixtures_df.iterrows():
        if fixture["match_id"] not in matched_match_ids:
            rows.append(_empty_mapped_fixture_row(fixture))
            warnings.append(f"No odds matched for fixture {fixture['match_id']}.")

    result = pd.DataFrame(rows)
    return result, warnings


def _complete_bookmaker_fair_probs(event_df: pd.DataFrame) -> list[list[float]]:
    fair_probs = []
    for _, bookmaker_df in event_df.groupby(["bookmaker_key", "bookmaker_title"], dropna=False):
        prices = {}
        for outcome in OUTCOME_ORDER:
            outcome_prices = pd.to_numeric(
                bookmaker_df[bookmaker_df["outcome_type"] == outcome]["outcome_price"],
                errors="coerce",
            ).dropna()
            if outcome_prices.empty:
                prices = {}
                break
            prices[outcome] = float(outcome_prices.iloc[0])
        if prices and all(value > 1.0 for value in prices.values()):
            implied = [decimal_odds_to_implied_probability(prices[outcome]) for outcome in OUTCOME_ORDER]
            fair_probs.append(remove_overround_proportional(implied))
    return fair_probs


def _market_probability_row(event_df: pd.DataFrame) -> dict:
    fair_probs = _complete_bookmaker_fair_probs(event_df)
    if not fair_probs:
        return {"market_home_prob": pd.NA, "market_draw_prob": pd.NA, "market_away_prob": pd.NA}
    avg_probs = pd.DataFrame(fair_probs, columns=OUTCOME_ORDER).mean().tolist()
    probs = remove_overround_proportional(avg_probs)
    return {
        "market_home_prob": probs[0],
        "market_draw_prob": probs[1],
        "market_away_prob": probs[2],
    }


def _best_outcome(event_df: pd.DataFrame, outcome: str) -> tuple[object, object]:
    outcome_df = event_df[event_df["outcome_type"] == outcome].copy()
    outcome_df["outcome_price"] = pd.to_numeric(outcome_df["outcome_price"], errors="coerce")
    outcome_df = outcome_df.dropna(subset=["outcome_price"])
    if outcome_df.empty:
        return pd.NA, pd.NA
    best = outcome_df.sort_values("outcome_price", ascending=False).iloc[0]
    return float(best["outcome_price"]), best["bookmaker_title"]


def _ds_outcome(event_df: pd.DataFrame, outcome: str) -> tuple[object, object, object]:
    ds_df = event_df[event_df.apply(_bookmaker_is_danske_spil, axis=1)]
    outcome_df = ds_df[ds_df["outcome_type"] == outcome].copy()
    outcome_df["outcome_price"] = pd.to_numeric(outcome_df["outcome_price"], errors="coerce")
    outcome_df = outcome_df.dropna(subset=["outcome_price"])
    if outcome_df.empty:
        return pd.NA, pd.NA, pd.NA
    selected = outcome_df.iloc[0]
    return float(selected["outcome_price"]), selected["bookmaker_key"], selected["bookmaker_title"]


def build_match_odds_table(mapped_df: pd.DataFrame) -> pd.DataFrame:
    if mapped_df.empty:
        return pd.DataFrame(columns=MATCH_ODDS_COLUMNS)

    rows = []
    for match_id, match_df in mapped_df.groupby("match_id", dropna=False):
        first = match_df.iloc[0]
        odds_rows = match_df[match_df.get("odds_available", False) == True].copy()
        row = {
            "match_id": match_id,
            "kickoff_utc": first.get("kickoff_utc"),
            "home_team": first.get("home_team"),
            "away_team": first.get("away_team"),
            "group": first.get("group", "TBD"),
            "stage": first.get("stage", "Group stage"),
            "matchday": first.get("matchday", 0),
            "odds_last_updated_utc": pd.NA,
            "odds_provider": pd.NA,
            "odds_source": pd.NA,
            "odds_available": False,
            "bookmaker_count": 0,
            "available_bookmakers": "",
            "ds_home_odds": pd.NA,
            "ds_draw_odds": pd.NA,
            "ds_away_odds": pd.NA,
            "ds_bookmaker_key": pd.NA,
            "ds_bookmaker_title": pd.NA,
        }
        if not odds_rows.empty:
            row["odds_available"] = True
            row["bookmaker_count"] = int(odds_rows["bookmaker_key"].nunique())
            row["available_bookmakers"] = ", ".join(sorted(odds_rows["bookmaker_title"].dropna().astype(str).unique()))
            row["odds_provider"] = ", ".join(sorted(odds_rows["provider"].dropna().astype(str).unique()))
            row["odds_source"] = ", ".join(sorted(odds_rows["odds_source"].dropna().astype(str).unique()))
            row["odds_last_updated_utc"] = pd.to_datetime(
                odds_rows["fetched_at_utc"],
                errors="coerce",
                utc=True,
            ).max()
            for outcome in OUTCOME_ORDER:
                best_odds, best_bookmaker = _best_outcome(odds_rows, outcome)
                row[f"best_{outcome}_odds"] = best_odds
                row[f"best_{outcome}_bookmaker"] = best_bookmaker
                ds_odds, ds_key, ds_title = _ds_outcome(odds_rows, outcome)
                row[f"ds_{outcome}_odds"] = ds_odds
                if not pd.isna(ds_key):
                    row["ds_bookmaker_key"] = ds_key
                if not pd.isna(ds_title):
                    row["ds_bookmaker_title"] = ds_title
            row.update(_market_probability_row(odds_rows))
        else:
            for outcome in OUTCOME_ORDER:
                row[f"best_{outcome}_odds"] = pd.NA
                row[f"best_{outcome}_bookmaker"] = pd.NA
            row.update({"market_home_prob": pd.NA, "market_draw_prob": pd.NA, "market_away_prob": pd.NA})
        rows.append(row)

    result = pd.DataFrame(rows)
    for column in MATCH_ODDS_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA
    return result[MATCH_ODDS_COLUMNS]
