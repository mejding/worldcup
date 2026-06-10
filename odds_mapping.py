from typing import Optional

import pandas as pd

from fetch_odds import is_draw_outcome
from odds_utils import decimal_odds_to_implied_probability, remove_overround_proportional


OUTCOME_ORDER = ("home", "draw", "away")
DEFAULT_PREFERRED_BOOKMAKER_NAMES = ["Danske Spil", "DanskeSpil", "Danske Spil A/S", "danske_spil"]


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
