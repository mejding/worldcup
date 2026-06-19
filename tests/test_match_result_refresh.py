import pandas as pd

from match_result_refresh import refresh_match_results
from match_results import load_match_results


def _fixtures(path):
    pd.DataFrame(
        [
            {
                "match_id": "m1",
                "match_number": 1,
                "kickoff_utc": "2026-06-18T20:00:00Z",
                "kickoff_local": "2026-06-18 22:00",
                "kickoff_timezone": "Europe/Copenhagen",
                "home_team": "Mexico",
                "away_team": "South Korea",
                "group": "A",
                "stage": "Group stage",
                "matchday": 2,
                "city": "Guadalajara",
                "stadium": "Estadio Guadalajara",
                "source": "test",
                "source_last_checked": "2026-06-19",
            }
        ]
    ).to_csv(path, index=False)


def _updates(path):
    pd.DataFrame(
        [
            {
                "match_id": "m1",
                "home_score": 1,
                "away_score": 0,
                "result_status": "final",
                "result_favorite_outcome": "home",
                "result_last_checked_utc": "2026-06-19T03:00:00Z",
                "result_source": "test",
                "result_notes": "Mexico 1-0 South Korea",
            }
        ]
    ).to_csv(path, index=False)


def test_refresh_match_results_adds_completed_match(tmp_path):
    fixtures_path = tmp_path / "fixtures.csv"
    updates_path = tmp_path / "updates.csv"
    results_path = tmp_path / "results.csv"
    _fixtures(fixtures_path)
    _updates(updates_path)

    result = refresh_match_results(
        fixtures_path=fixtures_path,
        updates_path=updates_path,
        results_path=results_path,
        as_of=pd.Timestamp("2026-06-19T04:00:00Z"),
    )

    saved = load_match_results(results_path)
    assert result["added"] == 1
    assert saved.iloc[0]["match_id"] == "m1"
    assert int(saved.iloc[0]["home_score"]) == 1


def test_refresh_match_results_does_not_duplicate_existing_match(tmp_path):
    fixtures_path = tmp_path / "fixtures.csv"
    updates_path = tmp_path / "updates.csv"
    results_path = tmp_path / "results.csv"
    _fixtures(fixtures_path)
    _updates(updates_path)
    _updates(results_path)

    result = refresh_match_results(
        fixtures_path=fixtures_path,
        updates_path=updates_path,
        results_path=results_path,
        as_of=pd.Timestamp("2026-06-19T04:00:00Z"),
    )

    saved = load_match_results(results_path)
    assert result["added"] == 0
    assert len(saved) == 1
