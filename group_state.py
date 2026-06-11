import pandas as pd


GROUP_STATE_COLUMNS = [
    "home_points_before",
    "away_points_before",
    "home_goal_difference_before",
    "away_goal_difference_before",
    "home_goals_for_before",
    "away_goals_for_before",
    "home_group_position_before",
    "away_group_position_before",
    "group_matchday",
    "home_matches_played_in_group_before",
    "away_matches_played_in_group_before",
    "group_state_available",
    "home_must_win",
    "away_must_win",
    "one_team_must_win",
    "both_teams_need_win",
    "home_draw_sufficient",
    "away_draw_sufficient",
    "both_teams_draw_satisfied",
]


def _neutral_defaults(matchday=0) -> dict:
    return {
        "home_points_before": 0,
        "away_points_before": 0,
        "home_goal_difference_before": 0,
        "away_goal_difference_before": 0,
        "home_goals_for_before": 0,
        "away_goals_for_before": 0,
        "home_group_position_before": 0,
        "away_group_position_before": 0,
        "group_matchday": int(matchday or 0),
        "home_matches_played_in_group_before": 0,
        "away_matches_played_in_group_before": 0,
        "group_state_available": False,
        "home_must_win": False,
        "away_must_win": False,
        "one_team_must_win": False,
        "both_teams_need_win": False,
        "home_draw_sufficient": False,
        "away_draw_sufficient": False,
        "both_teams_draw_satisfied": False,
    }


def _coerce_matchday(df: pd.DataFrame) -> pd.Series:
    if "group_matchday" in df.columns:
        source = df["group_matchday"]
    elif "matchday" in df.columns:
        source = df["matchday"]
    else:
        return pd.Series([0] * len(df), index=df.index)
    return pd.to_numeric(source, errors="coerce").fillna(0).astype(int)


def _is_group_stage(row) -> bool:
    if pd.notna(row.get("group")) and str(row.get("group")).strip():
        return True
    stage = str(row.get("stage", "")).lower()
    return "group" in stage


def _new_team_state() -> dict:
    return {"points": 0, "gf": 0, "ga": 0, "played": 0}


def _position(team: str, table: dict) -> int:
    standings = sorted(
        table.items(),
        key=lambda item: (-item[1]["points"], -(item[1]["gf"] - item[1]["ga"]), -item[1]["gf"], item[0]),
    )
    for index, (candidate, _) in enumerate(standings, start=1):
        if candidate == team:
            return index
    return 0


def _strategic_flags(home_points: int, away_points: int, matchday: int) -> dict:
    home_must_win = matchday == 3 and home_points <= 1
    away_must_win = matchday == 3 and away_points <= 1
    home_draw_sufficient = matchday == 3 and home_points >= 4
    away_draw_sufficient = matchday == 3 and away_points >= 4
    return {
        "home_must_win": home_must_win,
        "away_must_win": away_must_win,
        "one_team_must_win": home_must_win ^ away_must_win,
        "both_teams_need_win": home_must_win and away_must_win,
        "home_draw_sufficient": home_draw_sufficient,
        "away_draw_sufficient": away_draw_sufficient,
        "both_teams_draw_satisfied": home_draw_sufficient and away_draw_sufficient,
    }


def _update_table(table: dict, home_team: str, away_team: str, home_score, away_score) -> None:
    if pd.isna(home_score) or pd.isna(away_score):
        return
    home_score = int(home_score)
    away_score = int(away_score)
    home = table.setdefault(home_team, _new_team_state())
    away = table.setdefault(away_team, _new_team_state())
    if home_score > away_score:
        home_points, away_points = 3, 0
    elif home_score < away_score:
        home_points, away_points = 0, 3
    else:
        home_points, away_points = 1, 1
    home["points"] += home_points
    away["points"] += away_points
    home["gf"] += home_score
    home["ga"] += away_score
    away["gf"] += away_score
    away["ga"] += home_score
    home["played"] += 1
    away["played"] += 1


def add_group_state_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if result.empty:
        for column in GROUP_STATE_COLUMNS:
            result[column] = pd.Series(dtype="object")
        return result

    result["date"] = pd.to_datetime(result["date"], errors="coerce", utc=True) if "date" in result.columns else pd.NaT
    result["year"] = result["year"] if "year" in result.columns else result["date"].dt.year
    result["group_matchday"] = _coerce_matchday(result)
    if "group" not in result.columns:
        result["group"] = pd.NA
    if "stage" not in result.columns:
        result["stage"] = pd.NA
    result["_group_stage_candidate"] = result.apply(_is_group_stage, axis=1)

    feature_rows = {}
    groupable = result[result["_group_stage_candidate"] & result["group"].notna() & (result["group_matchday"] > 0)].copy()
    for _, tournament_df in groupable.sort_values(["date", "group_matchday"]).groupby(
        [groupable["tournament"].fillna("Unknown"), groupable["year"].fillna(0), groupable["group"].astype(str)],
        dropna=False,
    ):
        table = {}
        for index, row in tournament_df.sort_values(["group_matchday", "date"]).iterrows():
            home = row["home_team"]
            away = row["away_team"]
            home_state = table.setdefault(home, _new_team_state())
            away_state = table.setdefault(away, _new_team_state())
            matchday = int(row["group_matchday"])
            features = {
                "home_points_before": int(home_state["points"]),
                "away_points_before": int(away_state["points"]),
                "home_goal_difference_before": int(home_state["gf"] - home_state["ga"]),
                "away_goal_difference_before": int(away_state["gf"] - away_state["ga"]),
                "home_goals_for_before": int(home_state["gf"]),
                "away_goals_for_before": int(away_state["gf"]),
                "home_group_position_before": _position(home, table),
                "away_group_position_before": _position(away, table),
                "group_matchday": matchday,
                "home_matches_played_in_group_before": int(home_state["played"]),
                "away_matches_played_in_group_before": int(away_state["played"]),
                "group_state_available": True,
            }
            features.update(_strategic_flags(home_state["points"], away_state["points"], matchday))
            feature_rows[index] = features
            _update_table(table, home, away, row.get("home_score"), row.get("away_score"))

    defaults = []
    for index, row in result.iterrows():
        defaults.append(feature_rows.get(index, _neutral_defaults(row.get("group_matchday", 0))))
    features_df = pd.DataFrame(defaults, index=result.index)
    for column in GROUP_STATE_COLUMNS:
        result[column] = features_df[column]
    return result.drop(columns=["_group_stage_candidate"])
