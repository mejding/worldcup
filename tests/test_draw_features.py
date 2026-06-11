import pandas as pd

from draw_features import DRAW_CONTEXT_FEATURE_COLUMNS, add_draw_context_features


def test_draw_context_score_between_zero_and_100():
    df = add_draw_context_features(pd.DataFrame({"tournament_category": ["world_cup"], "group": ["A"], "group_matchday": [1], "elo_diff": [0]}))

    assert df["draw_context_score"].between(0, 100).all()


def test_labels_are_low_medium_high():
    df = add_draw_context_features(pd.DataFrame({"tournament_category": ["friendly"], "elo_diff": [500]}))

    assert set(df["draw_context_label"]).issubset({"Low", "Medium", "High"})


def test_evenly_matched_teams_increase_score():
    base = pd.DataFrame({"tournament_category": ["friendly", "friendly"], "elo_diff": [0, 400]})
    df = add_draw_context_features(base)

    assert df.iloc[0]["draw_context_score"] > df.iloc[1]["draw_context_score"]


def test_one_team_must_win_lowers_score():
    df = add_draw_context_features(pd.DataFrame({"tournament_category": ["world_cup", "world_cup"], "group": ["A", "A"], "group_matchday": [3, 3], "elo_diff": [0, 0], "one_team_must_win": [False, True]}))

    assert df.iloc[1]["draw_context_score"] < df.iloc[0]["draw_context_score"]


def test_both_teams_draw_satisfied_increases_score():
    df = add_draw_context_features(pd.DataFrame({"tournament_category": ["world_cup", "world_cup"], "group": ["A", "A"], "group_matchday": [3, 3], "elo_diff": [0, 0], "both_teams_draw_satisfied": [False, True]}))

    assert df.iloc[1]["draw_context_score"] > df.iloc[0]["draw_context_score"]


def test_no_nans_in_generated_draw_features():
    df = add_draw_context_features(pd.DataFrame({"tournament": ["Unknown"], "elo_diff": [None]}))

    assert not df[DRAW_CONTEXT_FEATURE_COLUMNS].isna().any().any()
