import pandas as pd

from match_results import add_match_results, split_active_and_archived_matches


def _predictions():
    return pd.DataFrame(
        [
            {
                "match_id": "m1",
                "home_team": "Mexico",
                "away_team": "South Africa",
                "active_home_prob": 0.6,
                "active_draw_prob": 0.25,
                "active_away_prob": 0.15,
            },
            {
                "match_id": "m2",
                "home_team": "South Korea",
                "away_team": "Czechia",
                "active_home_prob": 0.2,
                "active_draw_prob": 0.3,
                "active_away_prob": 0.5,
            },
            {
                "match_id": "m3",
                "home_team": "Canada",
                "away_team": "Bosnia and Herzegovina",
                "active_home_prob": 0.4,
                "active_draw_prob": 0.3,
                "active_away_prob": 0.3,
            },
        ]
    )


def test_add_match_results_marks_completed_and_favorite_won():
    results = pd.DataFrame(
        [
            {
                "match_id": "m1",
                "home_score": 2,
                "away_score": 0,
                "result_status": "final",
                "result_last_checked_utc": "2026-06-12T08:30:00Z",
                "result_source": "test",
                "result_notes": "",
            }
        ]
    )

    df = add_match_results(_predictions(), results)
    row = df[df["match_id"] == "m1"].iloc[0]

    assert row["is_completed"] == True
    assert row["actual_outcome"] == "home"
    assert row["favorite_outcome"] == "home"
    assert row["favorite_result_status"] == "Favoritten gik hjem"
    assert row["full_time_score"] == "2-0"


def test_add_match_results_marks_surprise():
    results = pd.DataFrame(
        [
            {
                "match_id": "m2",
                "home_score": 2,
                "away_score": 1,
                "result_status": "final",
                "result_last_checked_utc": "2026-06-12T08:30:00Z",
                "result_source": "test",
                "result_notes": "",
            }
        ]
    )

    df = add_match_results(_predictions(), results)
    row = df[df["match_id"] == "m2"].iloc[0]

    assert row["actual_outcome"] == "home"
    assert row["favorite_outcome"] == "away"
    assert row["favorite_result_status"] == "Overraskelse"


def test_favorite_outcome_ignores_draw_probability():
    predictions = pd.DataFrame(
        [
            {
                "match_id": "m1",
                "home_team": "Mexico",
                "away_team": "South Africa",
                "active_home_prob": 0.4,
                "active_draw_prob": 0.5,
                "active_away_prob": 0.1,
            }
        ]
    )
    results = pd.DataFrame(
        [
            {
                "match_id": "m1",
                "home_score": 2,
                "away_score": 0,
                "result_status": "final",
                "result_last_checked_utc": "2026-06-12T08:30:00Z",
                "result_source": "test",
                "result_notes": "",
            }
        ]
    )

    df = add_match_results(predictions, results)
    row = df.iloc[0]

    assert row["favorite_outcome"] == "home"
    assert row["favorite_outcome_label"] == "Mexico"
    assert row["favorite_result_status"] == "Favoritten gik hjem"


def test_completed_match_without_clear_favorite_does_not_crash():
    predictions = pd.DataFrame(
        [
            {
                "match_id": "m1",
                "home_team": "Mexico",
                "away_team": "South Africa",
                "active_home_prob": 1 / 3,
                "active_draw_prob": 1 / 3,
                "active_away_prob": 1 / 3,
                "market_home_prob": 1 / 3,
                "market_draw_prob": 1 / 3,
                "market_away_prob": 1 / 3,
                "model_home_prob": 1 / 3,
                "model_draw_prob": 1 / 3,
                "model_away_prob": 1 / 3,
            }
        ]
    )
    results = pd.DataFrame(
        [
            {
                "match_id": "m1",
                "home_score": 2,
                "away_score": 0,
                "result_status": "final",
                "result_last_checked_utc": "2026-06-12T08:30:00Z",
                "result_source": "test",
                "result_notes": "",
            }
        ]
    )

    df = add_match_results(predictions, results)
    row = df.iloc[0]

    assert pd.isna(row["favorite_outcome"])
    assert row["actual_outcome"] == "home"
    assert row["favorite_result_status"] == "Favorit ukendt"


def test_result_favorite_override_is_used():
    predictions = pd.DataFrame(
        [
            {
                "match_id": "m1",
                "home_team": "Mexico",
                "away_team": "South Africa",
                "active_home_prob": 1 / 3,
                "active_draw_prob": 1 / 3,
                "active_away_prob": 1 / 3,
            }
        ]
    )
    results = pd.DataFrame(
        [
            {
                "match_id": "m1",
                "home_score": 2,
                "away_score": 0,
                "result_status": "final",
                "result_favorite_outcome": "home",
                "result_last_checked_utc": "2026-06-12T08:30:00Z",
                "result_source": "test",
                "result_notes": "",
            }
        ]
    )

    df = add_match_results(predictions, results)
    row = df.iloc[0]

    assert row["favorite_outcome"] == "home"
    assert row["favorite_result_status"] == "Favoritten gik hjem"


def test_split_active_and_archived_matches():
    results = pd.DataFrame(
        [
            {
                "match_id": "m1",
                "home_score": 2,
                "away_score": 0,
                "result_status": "final",
                "result_last_checked_utc": "2026-06-12T08:30:00Z",
                "result_source": "test",
                "result_notes": "",
            }
        ]
    )

    df = add_match_results(_predictions(), results)
    active, archived = split_active_and_archived_matches(df)

    assert set(active["match_id"]) == {"m2", "m3"}
    assert set(archived["match_id"]) == {"m1"}


def test_split_archives_past_kickoff_without_result():
    predictions = pd.DataFrame(
        [
            {
                "match_id": "past",
                "home_team": "Netherlands",
                "away_team": "Sweden",
                "kickoff_time": "2026-06-20T17:00:00Z",
                "active_home_prob": 0.55,
                "active_draw_prob": 0.25,
                "active_away_prob": 0.20,
            },
            {
                "match_id": "future",
                "home_team": "Japan",
                "away_team": "Sweden",
                "kickoff_time": "2099-06-25T23:00:00Z",
                "active_home_prob": 0.40,
                "active_draw_prob": 0.30,
                "active_away_prob": 0.30,
            },
        ]
    )

    df = add_match_results(predictions, pd.DataFrame())
    active, archived = split_active_and_archived_matches(df)

    assert set(active["match_id"]) == {"future"}
    assert set(archived["match_id"]) == {"past"}
    assert archived.iloc[0]["favorite_result_status"] == "Resultat mangler"
