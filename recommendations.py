from config import DEFAULT_STAKING_PROFILE
from kelly import calculate_final_stake_fraction, calculate_suggested_stake
from odds_utils import calculate_edge


OUTCOMES = ("home", "draw", "away")


def _outcome_label(outcome: str) -> str:
    return {
        "home": "Home",
        "draw": "Draw",
        "away": "Away",
    }[outcome]


def _probability_for_outcome(row, outcome: str) -> float:
    active_column = f"active_{outcome}_prob"
    if active_column in row.index:
        return float(row[active_column])
    return float(row[f"model_{outcome}_prob"])


def _build_candidate(row, outcome: str, odds_prefix: str, current_bankroll: float, profile: dict):
    probability = _probability_for_outcome(row, outcome)
    odds = float(row[f"{odds_prefix}_{outcome}_odds"])
    if odds != odds or odds <= 1:
        candidate = {
            "outcome": _outcome_label(outcome),
            "odds": None,
            "edge": 0,
            "full_kelly": 0,
            "fractional_kelly": 0,
            "final_stake_fraction": 0,
            "stake": 0,
            "is_valid": False,
        }
        if odds_prefix == "best":
            candidate["bookmaker"] = row[f"best_{outcome}_bookmaker"]
        return candidate
    edge = calculate_edge(probability, odds)
    kelly_values = calculate_final_stake_fraction(
        probability,
        odds,
        fractional_kelly_multiplier=profile["fractional_kelly_multiplier"],
        max_stake_pct_of_bankroll=profile["max_stake_pct_of_bankroll"],
    )
    stake = calculate_suggested_stake(current_bankroll, kelly_values["final_stake_fraction"])

    candidate = {
        "outcome": _outcome_label(outcome),
        "odds": odds,
        "edge": edge,
        "full_kelly": kelly_values["full_kelly"],
        "fractional_kelly": kelly_values["fractional_kelly"],
        "final_stake_fraction": kelly_values["final_stake_fraction"],
        "stake": stake,
        "is_valid": edge >= profile["min_edge_threshold"]
        and kelly_values["final_stake_fraction"] >= profile["min_stake_pct_threshold"],
    }

    if odds_prefix == "best":
        candidate["bookmaker"] = row[f"best_{outcome}_bookmaker"]

    return candidate


def _select_recommendation(candidates: list[dict]):
    valid_candidates = [candidate for candidate in candidates if candidate["is_valid"]]
    if not valid_candidates:
        return None
    return sorted(
        valid_candidates,
        key=lambda candidate: (candidate["final_stake_fraction"], candidate["edge"]),
        reverse=True,
    )[0]


def _empty_market_recommendation(prefix: str) -> dict:
    empty = {
        f"recommended_outcome_{prefix}": "No bet",
        f"recommended_odds_{prefix}": None,
        f"recommended_edge_{prefix}": 0,
        f"recommended_full_kelly_{prefix}": 0,
        f"recommended_fractional_kelly_{prefix}": 0,
        f"recommended_stake_{prefix}": 0,
    }
    if prefix == "best":
        empty["recommended_bookmaker_best"] = None
    return empty


def _format_market_recommendation(prefix: str, recommendation) -> dict:
    if recommendation is None:
        return _empty_market_recommendation(prefix)

    formatted = {
        f"recommended_outcome_{prefix}": recommendation["outcome"],
        f"recommended_odds_{prefix}": recommendation["odds"],
        f"recommended_edge_{prefix}": recommendation["edge"],
        f"recommended_full_kelly_{prefix}": recommendation["full_kelly"],
        f"recommended_fractional_kelly_{prefix}": recommendation["fractional_kelly"],
        f"recommended_stake_{prefix}": recommendation["stake"],
    }
    if prefix == "best":
        formatted["recommended_bookmaker_best"] = recommendation["bookmaker"]
    return formatted


def recommend_for_match(
    row,
    current_bankroll: float,
    staking_profile: dict = None,
    probability_source: str = "active",
) -> dict:
    profile = DEFAULT_STAKING_PROFILE | (staking_profile or {})

    ds_candidates = [
        _build_candidate(row, outcome, "ds", current_bankroll, profile) for outcome in OUTCOMES
    ]
    best_candidates = [
        _build_candidate(row, outcome, "best", current_bankroll, profile) for outcome in OUTCOMES
    ]

    ds_recommendation = _select_recommendation(ds_candidates)
    best_recommendation = _select_recommendation(best_candidates)

    result = {
        **_format_market_recommendation("ds", ds_recommendation),
        **_format_market_recommendation("best", best_recommendation),
    }

    if ds_recommendation is not None:
        result["status"] = "Playable at Danske Spil"
    elif best_recommendation is not None:
        result["status"] = "Better elsewhere"
    else:
        result["status"] = "No bet"

    return result


def add_recommendations(predictions, current_bankroll: float, staking_profile: dict = None):
    recommendations = predictions.apply(
        lambda row: recommend_for_match(row, current_bankroll, staking_profile),
        axis=1,
        result_type="expand",
    )
    return predictions.join(recommendations)
