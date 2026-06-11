import pandas as pd


DRAW_CONTEXT_FEATURE_COLUMNS = [
    "is_major_tournament_group",
    "is_world_cup_group",
    "is_group_matchday_1",
    "is_group_matchday_2",
    "is_group_matchday_3",
    "teams_evenly_matched",
    "heavy_favorite_match",
    "home_must_win",
    "away_must_win",
    "one_team_must_win",
    "both_teams_need_win",
    "home_draw_sufficient",
    "away_draw_sufficient",
    "both_teams_draw_satisfied",
    "mutual_draw_acceptance",
    "draw_context_score",
]


MAJOR_CATEGORIES = {"world_cup", "euro", "copa_america", "afcon", "asian_cup", "gold_cup"}


def _categorize_series(df: pd.DataFrame) -> pd.Series:
    if "tournament_category" in df.columns:
        return df["tournament_category"].fillna("other")
    from features import categorize_tournament

    return df.get("tournament", pd.Series(["Unknown"] * len(df), index=df.index)).map(categorize_tournament)


def _group_matchday(df: pd.DataFrame) -> pd.Series:
    if "group_matchday" in df.columns:
        return pd.to_numeric(df["group_matchday"], errors="coerce").fillna(0).astype(int)
    if "matchday" in df.columns:
        return pd.to_numeric(df["matchday"], errors="coerce").fillna(0).astype(int)
    return pd.Series([0] * len(df), index=df.index)


def _group_stage_flag(df: pd.DataFrame) -> pd.Series:
    if "is_group_stage" in df.columns:
        return df["is_group_stage"].fillna(False).astype(bool)
    group_available = df["group"].notna() if "group" in df.columns else pd.Series([False] * len(df), index=df.index)
    stage_group = df["stage"].astype(str).str.lower().str.contains("group", na=False) if "stage" in df.columns else pd.Series([False] * len(df), index=df.index)
    state_available = df["group_state_available"].fillna(False).astype(bool) if "group_state_available" in df.columns else pd.Series([False] * len(df), index=df.index)
    return group_available | stage_group | state_available


def _strength_gap(df: pd.DataFrame) -> pd.Series:
    if "elo_diff" in df.columns:
        return pd.to_numeric(df["elo_diff"], errors="coerce").abs().fillna(0)
    if "win_rate_diff" in df.columns:
        return pd.to_numeric(df["win_rate_diff"], errors="coerce").abs().fillna(0) * 400
    return pd.Series([0.0] * len(df), index=df.index)


def _label(score: float) -> str:
    if score <= 33:
        return "Low"
    if score <= 66:
        return "Medium"
    return "High"


def add_draw_context_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if result.empty:
        for column in DRAW_CONTEXT_FEATURE_COLUMNS + ["draw_context_label", "strength_gap_abs"]:
            result[column] = pd.Series(dtype="float64")
        return result

    category = _categorize_series(result)
    matchday = _group_matchday(result)
    is_group_stage = _group_stage_flag(result)
    strength_gap = _strength_gap(result)

    result["tournament_category"] = category
    result["is_group_stage"] = is_group_stage
    result["group_matchday"] = matchday
    result["strength_gap_abs"] = strength_gap
    result["is_major_tournament_group"] = category.isin(MAJOR_CATEGORIES) & is_group_stage
    result["is_world_cup_group"] = (category == "world_cup") & is_group_stage
    result["is_group_matchday_1"] = is_group_stage & (matchday == 1)
    result["is_group_matchday_2"] = is_group_stage & (matchday == 2)
    result["is_group_matchday_3"] = is_group_stage & (matchday == 3)
    result["teams_evenly_matched"] = strength_gap <= 100
    result["heavy_favorite_match"] = strength_gap >= 250

    bool_defaults = [
        "home_must_win",
        "away_must_win",
        "one_team_must_win",
        "both_teams_need_win",
        "home_draw_sufficient",
        "away_draw_sufficient",
        "both_teams_draw_satisfied",
    ]
    for column in bool_defaults:
        if column not in result.columns:
            result[column] = False
        result[column] = result[column].fillna(False).astype(bool)

    mutual = 0.5
    mutual += result["teams_evenly_matched"].astype(float) * 0.2
    mutual += result["both_teams_draw_satisfied"].astype(float) * 0.3
    mutual += result["is_group_matchday_1"].astype(float) * 0.1
    mutual -= result["one_team_must_win"].astype(float) * 0.25
    mutual -= result["both_teams_need_win"].astype(float) * 0.35
    mutual -= result["heavy_favorite_match"].astype(float) * 0.15
    result["mutual_draw_acceptance"] = mutual.clip(0, 1)

    score = pd.Series([50.0] * len(result), index=result.index)
    score += result["teams_evenly_matched"].astype(float) * 15
    score += result["is_major_tournament_group"].astype(float) * 10
    score += result["is_group_matchday_1"].astype(float) * 10
    score += result["both_teams_draw_satisfied"].astype(float) * 15
    score += (result["mutual_draw_acceptance"] >= 0.75).astype(float) * 10
    score -= result["one_team_must_win"].astype(float) * 15
    score -= result["both_teams_need_win"].astype(float) * 25
    score -= result["heavy_favorite_match"].astype(float) * 10
    result["draw_context_score"] = score.clip(0, 100)
    result["draw_context_label"] = result["draw_context_score"].map(_label)

    for column in DRAW_CONTEXT_FEATURE_COLUMNS:
        if column not in result.columns:
            result[column] = 0
        if result[column].dtype == bool:
            result[column] = result[column].astype(bool)
        else:
            result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0)
    return result
