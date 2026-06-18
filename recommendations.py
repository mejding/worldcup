from __future__ import annotations

import pandas as pd

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


def calculate_fair_odds(probability) -> float | None:
    probability = pd.to_numeric(probability, errors="coerce")
    if pd.isna(probability) or probability <= 0:
        return None
    return float(1 / probability)


def calculate_expected_value(probability, odds) -> float | None:
    probability = pd.to_numeric(probability, errors="coerce")
    odds = pd.to_numeric(odds, errors="coerce")
    if pd.isna(probability) or pd.isna(odds) or odds <= 1:
        return None
    return float(probability * odds - 1)


def calculate_probability_edge(model_probability, market_probability) -> float | None:
    model_probability = pd.to_numeric(model_probability, errors="coerce")
    market_probability = pd.to_numeric(market_probability, errors="coerce")
    if pd.isna(model_probability) or pd.isna(market_probability):
        return None
    return float(model_probability - market_probability)


def is_playable_at_bookmaker(
    probability,
    odds,
    min_edge_threshold,
    min_stake_threshold,
    bankroll,
    kelly_fraction,
    max_stake_pct,
) -> dict:
    probability = pd.to_numeric(probability, errors="coerce")
    odds = pd.to_numeric(odds, errors="coerce")
    fair_odds = calculate_fair_odds(probability)
    if pd.isna(probability) or pd.isna(odds) or odds <= 1:
        return {
            "is_playable": False,
            "reason": "Odds missing at selected bookmaker.",
            "fair_odds": fair_odds,
            "edge_pct": None,
            "kelly_pct": 0,
            "stake_dkk": 0,
        }
    edge_pct = calculate_expected_value(probability, odds)
    kelly_values = calculate_final_stake_fraction(
        float(probability),
        float(odds),
        fractional_kelly_multiplier=kelly_fraction,
        max_stake_pct_of_bankroll=max_stake_pct,
    )
    stake = calculate_suggested_stake(float(bankroll), kelly_values["final_stake_fraction"])
    if edge_pct is None or edge_pct <= 0:
        reason = "Bookmaker odds are below the model fair odds."
    elif edge_pct < min_edge_threshold:
        reason = "Positive value is below the minimum edge threshold."
    elif kelly_values["final_stake_fraction"] < min_stake_threshold:
        reason = "Kelly stake is below the minimum stake threshold."
    else:
        reason = "Model probability is higher than the probability implied by bookmaker odds."
    return {
        "is_playable": bool(edge_pct is not None and edge_pct >= min_edge_threshold and kelly_values["final_stake_fraction"] >= min_stake_threshold),
        "reason": reason,
        "fair_odds": fair_odds,
        "edge_pct": edge_pct,
        "kelly_pct": kelly_values["fractional_kelly"],
        "stake_dkk": stake,
    }


def _probability_for_outcome(row, outcome: str) -> float:
    active_column = f"active_{outcome}_prob"
    if active_column in row.index:
        return float(row[active_column])
    return float(row[f"model_{outcome}_prob"])


def _market_probability_for_outcome(row, outcome: str):
    value = row.get(f"market_{outcome}_prob")
    value = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(value) else float(value)


def _build_candidate(row, outcome: str, odds_prefix: str, current_bankroll: float, profile: dict):
    probability = _probability_for_outcome(row, outcome)
    odds = pd.to_numeric(row.get(f"{odds_prefix}_{outcome}_odds"), errors="coerce")
    if pd.isna(odds) or odds <= 1:
        candidate = {
            "outcome": _outcome_label(outcome),
            "outcome_key": outcome,
            "probability": probability,
            "odds": None,
            "fair_odds": calculate_fair_odds(probability),
            "implied_probability": None,
            "probability_edge": calculate_probability_edge(probability, _market_probability_for_outcome(row, outcome)),
            "edge": 0,
            "full_kelly": 0,
            "fractional_kelly": 0,
            "final_stake_fraction": 0,
            "stake": 0,
            "is_valid": False,
        }
        if odds_prefix == "best":
            candidate["bookmaker"] = row.get(f"best_{outcome}_bookmaker")
        return candidate
    odds = float(odds)
    edge = calculate_edge(probability, odds)
    market_probability = _market_probability_for_outcome(row, outcome)
    kelly_values = calculate_final_stake_fraction(
        probability,
        odds,
        fractional_kelly_multiplier=profile["fractional_kelly_multiplier"],
        max_stake_pct_of_bankroll=profile["max_stake_pct_of_bankroll"],
    )
    stake = calculate_suggested_stake(current_bankroll, kelly_values["final_stake_fraction"])

    candidate = {
        "outcome": _outcome_label(outcome),
        "outcome_key": outcome,
        "probability": probability,
        "odds": odds,
        "fair_odds": calculate_fair_odds(probability),
        "implied_probability": 1 / odds,
        "probability_edge": calculate_probability_edge(probability, market_probability),
        "edge": edge,
        "full_kelly": kelly_values["full_kelly"],
        "fractional_kelly": kelly_values["fractional_kelly"],
        "final_stake_fraction": kelly_values["final_stake_fraction"],
        "stake": stake,
        "is_valid": edge >= profile["min_edge_threshold"]
        and kelly_values["final_stake_fraction"] >= profile["min_stake_pct_threshold"],
    }

    if odds_prefix == "best":
        candidate["bookmaker"] = row.get(f"best_{outcome}_bookmaker")

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
        f"recommended_fair_odds_{prefix}": None,
        f"recommended_probability_{prefix}": None,
        f"recommended_implied_probability_{prefix}": None,
        f"recommended_probability_edge_{prefix}": None,
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
        f"recommended_fair_odds_{prefix}": recommendation["fair_odds"],
        f"recommended_probability_{prefix}": recommendation["probability"],
        f"recommended_implied_probability_{prefix}": recommendation["implied_probability"],
        f"recommended_probability_edge_{prefix}": recommendation["probability_edge"],
        f"recommended_edge_{prefix}": recommendation["edge"],
        f"recommended_full_kelly_{prefix}": recommendation["full_kelly"],
        f"recommended_fractional_kelly_{prefix}": recommendation["fractional_kelly"],
        f"recommended_stake_{prefix}": recommendation["stake"],
    }
    if prefix == "best":
        formatted["recommended_bookmaker_best"] = recommendation["bookmaker"]
    return formatted


def _favorite(row) -> dict:
    probabilities = {
        "home": _probability_for_outcome(row, "home"),
        "draw": _probability_for_outcome(row, "draw"),
        "away": _probability_for_outcome(row, "away"),
    }
    outcome = max(probabilities, key=probabilities.get)
    label = {"home": row.get("home_team", "Home"), "draw": "Draw", "away": row.get("away_team", "Away")}[outcome]
    return {"outcome": outcome, "label": label, "probability": probabilities[outcome]}


def _reason_for_market(row, market: str, recommendation, candidates: list[dict]) -> str:
    bookmaker_label = "Danske Spil" if market == "ds" else "Best market"
    if recommendation is not None:
        return f"Model fair odds are {recommendation['fair_odds']:.2f} and {bookmaker_label} offers {recommendation['odds']:.2f}."
    if not candidates or all(candidate["odds"] is None for candidate in candidates):
        return f"{bookmaker_label} odds are missing for this match."
    best_edge = max(candidates, key=lambda candidate: candidate["edge"])
    if best_edge["edge"] <= 0:
        return f"{bookmaker_label} odds are too low compared with model fair odds."
    return f"{bookmaker_label} value is below the current edge or stake threshold."


def _primary_payload(row, market: str, recommendation, candidates, status: str, bookmaker_mode: str) -> dict:
    bookmaker = "Danske Spil" if market == "ds" else (recommendation.get("bookmaker") if recommendation else "Best market")
    payload = {
        "bookmaker": bookmaker,
        "status": status,
        "outcome": None,
        "label": "No bet",
        "odds": None,
        "fair_odds": None,
        "edge_pct": None,
        "kelly_pct": 0,
        "stake_dkk": 0,
        "reason": _reason_for_market(row, market, recommendation, candidates),
    }
    if recommendation is not None:
        payload.update(
            {
                "outcome": recommendation["outcome_key"],
                "label": recommendation["outcome"],
                "odds": recommendation["odds"],
                "fair_odds": recommendation["fair_odds"],
                "edge_pct": recommendation["edge"],
                "kelly_pct": recommendation["fractional_kelly"],
                "stake_dkk": recommendation["stake"],
                "reason": _reason_for_market(row, market, recommendation, candidates),
            }
        )
    if status == "odds_missing":
        payload["reason"] = "The app does not currently have odds for your selected bookmaker."
    return payload


def recommend_for_match(
    row,
    current_bankroll: float,
    staking_profile: dict = None,
    probability_source: str = "active",
    preferred_bookmaker_mode: str = "danske_spil",
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

    ds_odds_missing = all(candidate["odds"] is None for candidate in ds_candidates)
    best_odds_missing = all(candidate["odds"] is None for candidate in best_candidates)
    favorite = _favorite(row)
    selected_market = "best" if preferred_bookmaker_mode == "best_market" else "ds"
    comparison_market = "ds" if selected_market == "best" else "best"
    selected_recommendation = best_recommendation if selected_market == "best" else ds_recommendation
    selected_candidates = best_candidates if selected_market == "best" else ds_candidates
    selected_missing = best_odds_missing if selected_market == "best" else ds_odds_missing
    comparison_recommendation = ds_recommendation if comparison_market == "ds" else best_recommendation
    comparison_candidates = ds_candidates if comparison_market == "ds" else best_candidates

    primary_status = "play" if selected_recommendation is not None else ("odds_missing" if selected_missing else "no_bet")
    if selected_market == "ds":
        if best_recommendation is not None and ds_recommendation is None:
            comparison_status = "better_elsewhere"
        elif best_odds_missing:
            comparison_status = "comparison_missing"
        elif best_recommendation is not None and ds_recommendation is not None:
            comparison_status = "better_elsewhere" if best_recommendation["odds"] > ds_recommendation["odds"] + 0.01 else "same_or_similar"
        else:
            comparison_status = "no_value_elsewhere"
    else:
        if ds_recommendation is not None and best_recommendation is None:
            comparison_status = "same_or_similar"
        elif ds_odds_missing:
            comparison_status = "comparison_missing"
        elif ds_recommendation is None:
            comparison_status = "no_value_elsewhere"
        else:
            comparison_status = "same_or_similar"

    result.update(
        {
            "model_favorite_outcome": favorite["outcome"],
            "model_favorite_label": favorite["label"],
            "model_favorite_probability": favorite["probability"],
            "selected_bookmaker": preferred_bookmaker_mode,
            "primary_status": primary_status,
            "comparison_status": comparison_status,
            "primary_bookmaker": "Best market" if selected_market == "best" else "Danske Spil",
            "primary_outcome": selected_recommendation["outcome"] if selected_recommendation else "No bet",
            "primary_odds": selected_recommendation["odds"] if selected_recommendation else None,
            "primary_fair_odds": selected_recommendation["fair_odds"] if selected_recommendation else None,
            "primary_edge": selected_recommendation["edge"] if selected_recommendation else 0,
            "primary_kelly": selected_recommendation["fractional_kelly"] if selected_recommendation else 0,
            "primary_stake": selected_recommendation["stake"] if selected_recommendation else 0,
            "primary_reason": _reason_for_market(row, selected_market, selected_recommendation, selected_candidates),
            "comparison_bookmaker": "Danske Spil" if comparison_market == "ds" else (best_recommendation.get("bookmaker") if best_recommendation else "Best market"),
            "comparison_outcome": comparison_recommendation["outcome"] if comparison_recommendation else "No bet",
            "comparison_odds": comparison_recommendation["odds"] if comparison_recommendation else None,
            "comparison_fair_odds": comparison_recommendation["fair_odds"] if comparison_recommendation else None,
            "comparison_edge": comparison_recommendation["edge"] if comparison_recommendation else 0,
            "comparison_kelly": comparison_recommendation["fractional_kelly"] if comparison_recommendation else 0,
            "comparison_stake": comparison_recommendation["stake"] if comparison_recommendation else 0,
            "comparison_reason": _reason_for_market(row, comparison_market, comparison_recommendation, comparison_candidates),
            "primary_recommendation": _primary_payload(row, selected_market, selected_recommendation, selected_candidates, primary_status, preferred_bookmaker_mode),
            "comparison_recommendation": _primary_payload(row, comparison_market, comparison_recommendation, comparison_candidates, comparison_status, preferred_bookmaker_mode),
        }
    )

    return result


def add_recommendations(predictions, current_bankroll: float, staking_profile: dict = None, preferred_bookmaker_mode: str = "danske_spil"):
    recommendations = predictions.apply(
        lambda row: recommend_for_match(row, current_bankroll, staking_profile, preferred_bookmaker_mode=preferred_bookmaker_mode),
        axis=1,
        result_type="expand",
    )
    return predictions.join(recommendations)
