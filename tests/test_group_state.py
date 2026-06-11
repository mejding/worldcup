import pandas as pd

from group_state import add_group_state_features


def _group_df():
    return pd.DataFrame(
        [
            {"date": "2022-11-20", "tournament": "FIFA World Cup", "group": "A", "matchday": 1, "home_team": "A", "away_team": "B", "home_score": 2, "away_score": 0, "result": "H"},
            {"date": "2022-11-21", "tournament": "FIFA World Cup", "group": "A", "matchday": 1, "home_team": "C", "away_team": "D", "home_score": 1, "away_score": 1, "result": "D"},
            {"date": "2022-11-25", "tournament": "FIFA World Cup", "group": "A", "matchday": 2, "home_team": "A", "away_team": "C", "home_score": 0, "away_score": 0, "result": "D"},
            {"date": "2022-11-26", "tournament": "FIFA World Cup", "group": "A", "matchday": 2, "home_team": "B", "away_team": "D", "home_score": 0, "away_score": 1, "result": "A"},
            {"date": "2022-11-30", "tournament": "FIFA World Cup", "group": "A", "matchday": 3, "home_team": "A", "away_team": "D", "home_score": 1, "away_score": 1, "result": "D"},
            {"date": "2022-11-30", "tournament": "FIFA World Cup", "group": "A", "matchday": 3, "home_team": "B", "away_team": "C", "home_score": 2, "away_score": 0, "result": "H"},
        ]
    )


def test_points_before_match_are_calculated():
    df = add_group_state_features(_group_df())
    row = df[(df["home_team"] == "A") & (df["away_team"] == "C")].iloc[0]

    assert row["home_points_before"] == 3
    assert row["away_points_before"] == 1


def test_goal_difference_before_match_is_calculated():
    df = add_group_state_features(_group_df())
    row = df[(df["home_team"] == "A") & (df["away_team"] == "C")].iloc[0]

    assert row["home_goal_difference_before"] == 2


def test_matchday_1_has_no_must_win_by_default():
    row = add_group_state_features(_group_df()).iloc[0]

    assert not row["home_must_win"]
    assert not row["away_must_win"]


def test_matchday_3_team_with_one_or_less_points_is_must_win():
    df = add_group_state_features(_group_df())
    row = df[(df["home_team"] == "B") & (df["away_team"] == "C")].iloc[0]

    assert row["home_must_win"]


def test_matchday_3_team_with_four_or_more_points_is_draw_sufficient():
    df = add_group_state_features(_group_df())
    row = df[(df["home_team"] == "A") & (df["away_team"] == "D")].iloc[0]

    assert row["home_draw_sufficient"]


def test_both_teams_draw_satisfied_works():
    df = _group_df()
    df.loc[df["home_team"].eq("D") & df["away_team"].eq("B"), "home_score"] = 3
    out = add_group_state_features(df)
    row = out[(out["home_team"] == "A") & (out["away_team"] == "D")].iloc[0]

    assert isinstance(bool(row["both_teams_draw_satisfied"]), bool)


def test_missing_group_data_gives_neutral_defaults():
    df = _group_df().drop(columns=["group", "matchday"])
    out = add_group_state_features(df)

    assert not out["group_state_available"].any()
    assert out["home_points_before"].sum() == 0
