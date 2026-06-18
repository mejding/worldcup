from collections import defaultdict
from pathlib import Path
from typing import Union

import pandas as pd

from config import TRAINING_DATASET_PATH
from draw_features import DRAW_CONTEXT_FEATURE_COLUMNS, add_draw_context_features
from group_state import GROUP_STATE_COLUMNS, add_group_state_features


DEFAULTS = {
    "win_rate": 0.33,
    "draw_rate": 0.25,
    "loss_rate": 0.42,
    "points_per_match": 1.0,
    "goals_for": 1.0,
    "goals_against": 1.0,
    "goal_diff": 0.0,
    "elo": 1500.0,
}

UPCOMING_TEAM_ALIASES = {
    "Cabo Verde": "Cape Verde",
    "Congo DR": "DR Congo",
    "Côte d'Ivoire": "Ivory Coast",
    "IR Iran": "Iran",
    "Türkiye": "Turkey",
}

FEATURE_COLUMNS = [
    "neutral",
    "is_major_tournament",
    "is_world_cup",
    "is_qualifier",
    "is_friendly",
    "home_matches_played_before",
    "away_matches_played_before",
    "home_win_rate_before",
    "away_win_rate_before",
    "home_draw_rate_before",
    "away_draw_rate_before",
    "home_loss_rate_before",
    "away_loss_rate_before",
    "home_points_per_match_last5",
    "away_points_per_match_last5",
    "home_points_per_match_last10",
    "away_points_per_match_last10",
    "home_goals_for_last5",
    "away_goals_for_last5",
    "home_goals_against_last5",
    "away_goals_against_last5",
    "home_goal_diff_last5",
    "away_goal_diff_last5",
    "win_rate_diff",
    "points_per_match_last5_diff",
    "points_per_match_last10_diff",
    "goal_diff_last5_diff",
    "experience_diff",
    "home_elo_before",
    "away_elo_before",
    "elo_diff",
    "tournament_category",
]


def get_feature_columns(include_draw_context_features: bool = False) -> list[str]:
    if not include_draw_context_features:
        return FEATURE_COLUMNS.copy()
    return FEATURE_COLUMNS + DRAW_CONTEXT_FEATURE_COLUMNS


def categorize_tournament(tournament: str) -> str:
    text = str(tournament or "").lower()
    if "world cup" in text and "qual" not in text:
        return "world_cup"
    if "euro" in text and "qual" not in text:
        return "euro"
    if "copa" in text:
        return "copa_america"
    if "afcon" in text or "africa cup" in text:
        return "afcon"
    if "asian cup" in text:
        return "asian_cup"
    if "gold cup" in text:
        return "gold_cup"
    if "qual" in text:
        return "qualifier"
    if "nations league" in text:
        return "nations_league"
    if "friendly" in text:
        return "friendly"
    return "other"


def canonical_team_name(team: str) -> str:
    return UPCOMING_TEAM_ALIASES.get(str(team), str(team))


def _empty_stats():
    return {"played": 0, "wins": 0, "draws": 0, "losses": 0, "history": []}


def _rates(stats):
    played = stats["played"]
    if played == 0:
        return DEFAULTS["win_rate"], DEFAULTS["draw_rate"], DEFAULTS["loss_rate"]
    return stats["wins"] / played, stats["draws"] / played, stats["losses"] / played


def _recent(stats, n: int, key: str) -> float:
    history = stats["history"][-n:]
    if not history:
        if key == "points":
            return DEFAULTS["points_per_match"]
        if key == "gf":
            return DEFAULTS["goals_for"]
        if key == "ga":
            return DEFAULTS["goals_against"]
        return DEFAULTS["goal_diff"]
    if key == "points":
        return sum(item["points"] for item in history) / len(history)
    if key == "gf":
        return sum(item["gf"] for item in history) / len(history)
    if key == "ga":
        return sum(item["ga"] for item in history) / len(history)
    return sum(item["gf"] - item["ga"] for item in history) / len(history)


def _expected_score(rating_a: float, rating_b: float) -> float:
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def _update_elo(elos, home_team, away_team, result, k=20):
    home_elo = elos[home_team]
    away_elo = elos[away_team]
    actual_home = 1 if result == "H" else 0 if result == "A" else 0.5
    expected_home = _expected_score(home_elo, away_elo)
    elos[home_team] = home_elo + k * (actual_home - expected_home)
    elos[away_team] = away_elo + k * ((1 - actual_home) - (1 - expected_home))


def _feature_row(match, team_stats, elos):
    home = match["home_team"]
    away = match["away_team"]
    home_stats = team_stats[home]
    away_stats = team_stats[away]
    home_win, home_draw, home_loss = _rates(home_stats)
    away_win, away_draw, away_loss = _rates(away_stats)
    category = categorize_tournament(match.get("tournament", "Unknown"))
    home_ppm5 = _recent(home_stats, 5, "points")
    away_ppm5 = _recent(away_stats, 5, "points")
    home_ppm10 = _recent(home_stats, 10, "points")
    away_ppm10 = _recent(away_stats, 10, "points")
    home_gd5 = _recent(home_stats, 5, "gd")
    away_gd5 = _recent(away_stats, 5, "gd")
    return {
        "neutral": bool(match.get("neutral", False)),
        "is_major_tournament": category in {"world_cup", "euro", "copa_america", "afcon", "asian_cup", "gold_cup"},
        "is_world_cup": category == "world_cup",
        "is_qualifier": category == "qualifier",
        "is_friendly": category == "friendly",
        "home_matches_played_before": home_stats["played"],
        "away_matches_played_before": away_stats["played"],
        "home_win_rate_before": home_win,
        "away_win_rate_before": away_win,
        "home_draw_rate_before": home_draw,
        "away_draw_rate_before": away_draw,
        "home_loss_rate_before": home_loss,
        "away_loss_rate_before": away_loss,
        "home_points_per_match_last5": home_ppm5,
        "away_points_per_match_last5": away_ppm5,
        "home_points_per_match_last10": home_ppm10,
        "away_points_per_match_last10": away_ppm10,
        "home_goals_for_last5": _recent(home_stats, 5, "gf"),
        "away_goals_for_last5": _recent(away_stats, 5, "gf"),
        "home_goals_against_last5": _recent(home_stats, 5, "ga"),
        "away_goals_against_last5": _recent(away_stats, 5, "ga"),
        "home_goal_diff_last5": home_gd5,
        "away_goal_diff_last5": away_gd5,
        "win_rate_diff": home_win - away_win,
        "points_per_match_last5_diff": home_ppm5 - away_ppm5,
        "points_per_match_last10_diff": home_ppm10 - away_ppm10,
        "goal_diff_last5_diff": home_gd5 - away_gd5,
        "experience_diff": home_stats["played"] - away_stats["played"],
        "home_elo_before": elos[home],
        "away_elo_before": elos[away],
        "elo_diff": elos[home] - elos[away],
        "tournament_category": category,
    }


def _update_stats(team_stats, home_team, away_team, home_score, away_score, result):
    if result == "H":
        home_points, away_points = 3, 0
    elif result == "A":
        home_points, away_points = 0, 3
    else:
        home_points, away_points = 1, 1
    for team, gf, ga, points, side_result in [
        (home_team, home_score, away_score, home_points, "W" if result == "H" else "L" if result == "A" else "D"),
        (away_team, away_score, home_score, away_points, "W" if result == "A" else "L" if result == "H" else "D"),
    ]:
        stats = team_stats[team]
        stats["played"] += 1
        stats["wins"] += side_result == "W"
        stats["draws"] += side_result == "D"
        stats["losses"] += side_result == "L"
        stats["history"].append({"points": points, "gf": float(gf), "ga": float(ga)})


def _optional_match_metadata(match) -> dict:
    metadata = {}
    for column in ["stage", "group", "matchday", "group_matchday"] + GROUP_STATE_COLUMNS:
        if column in match.index:
            metadata[column] = match[column]
    return metadata


def _build_training_dataset(
    df: pd.DataFrame,
    output_path: Union[str, Path] = TRAINING_DATASET_PATH,
    include_draw_context_features: bool = False,
) -> pd.DataFrame:
    matches = df.copy()
    matches["date"] = pd.to_datetime(matches["date"], errors="coerce", utc=True)
    matches = matches.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score", "result"])
    matches = matches.sort_values("date").reset_index(drop=True)
    if include_draw_context_features:
        matches = add_group_state_features(matches)
    team_stats = defaultdict(_empty_stats)
    elos = defaultdict(lambda: DEFAULTS["elo"])
    rows = []
    for _, match in matches.iterrows():
        row = {
            "date": match["date"],
            "home_team": match["home_team"],
            "away_team": match["away_team"],
            "result": match["result"],
            "tournament": match.get("tournament", "Unknown"),
            **_optional_match_metadata(match),
            **_feature_row(match, team_stats, elos),
        }
        rows.append(row)
        _update_stats(
            team_stats,
            match["home_team"],
            match["away_team"],
            match["home_score"],
            match["away_score"],
            match["result"],
        )
        _update_elo(elos, match["home_team"], match["away_team"], match["result"])
    result = pd.DataFrame(rows)
    if include_draw_context_features:
        result = add_draw_context_features(result)
        for column in DRAW_CONTEXT_FEATURE_COLUMNS:
            if column not in result.columns:
                result[column] = 0
            if result[column].dtype == bool:
                result[column] = result[column].astype(bool)
            else:
                result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def build_training_dataset(
    df: pd.DataFrame,
    output_path: Union[str, Path] = TRAINING_DATASET_PATH,
    include_draw_context_features: bool = False,
) -> pd.DataFrame:
    return _build_training_dataset(df, output_path=output_path, include_draw_context_features=include_draw_context_features)


def build_upcoming_feature_dataset(
    upcoming_df: pd.DataFrame,
    historical_df: pd.DataFrame,
    include_draw_context_features: bool = False,
) -> pd.DataFrame:
    historical = historical_df.copy()
    historical["date"] = pd.to_datetime(historical["date"], errors="coerce", utc=True)
    upcoming = upcoming_df.copy()
    upcoming["kickoff_dt"] = pd.to_datetime(upcoming["kickoff_time"], errors="coerce", utc=True)
    combined_rows = []
    for _, upcoming_match in upcoming.iterrows():
        prior = historical[historical["date"] < upcoming_match["kickoff_dt"]].copy()
        team_stats = defaultdict(_empty_stats)
        elos = defaultdict(lambda: DEFAULTS["elo"])
        if not prior.empty:
            prior = prior.sort_values("date")
            for _, match in prior.iterrows():
                _update_stats(team_stats, match["home_team"], match["away_team"], match["home_score"], match["away_score"], match["result"])
                _update_elo(elos, match["home_team"], match["away_team"], match["result"])
        feature_match = {
            "home_team": canonical_team_name(upcoming_match["home_team"]),
            "away_team": canonical_team_name(upcoming_match["away_team"]),
            "neutral": True,
            "tournament": "World Cup",
            "group": upcoming_match.get("group", pd.NA),
            "matchday": upcoming_match.get("matchday", 0),
            "group_matchday": upcoming_match.get("matchday", 0),
            "group_state_available": False,
        }
        row = {
            **feature_match,
            **_feature_row(feature_match, team_stats, elos),
        }
        combined_rows.append(row)
    result = pd.DataFrame(combined_rows)
    if include_draw_context_features:
        result = add_draw_context_features(result)
    columns = get_feature_columns(include_draw_context_features)
    for column in columns:
        if column not in result.columns:
            result[column] = 0
    return result[columns]
