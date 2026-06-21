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


def test_refresh_match_results_adds_remote_feed_match(tmp_path, monkeypatch):
    fixtures_path = tmp_path / "fixtures.csv"
    updates_path = tmp_path / "updates.csv"
    results_path = tmp_path / "results.csv"
    _fixtures(fixtures_path)
    pd.DataFrame(columns=[
        "match_id",
        "home_score",
        "away_score",
        "result_status",
        "result_favorite_outcome",
        "result_last_checked_utc",
        "result_source",
        "result_notes",
    ]).to_csv(updates_path, index=False)

    class Response:
        status_code = 200
        text = (
            "match_id,home_score,away_score,result_status,result_favorite_outcome,"
            "result_last_checked_utc,result_source,result_notes\n"
            "m1,3,2,final,home,2026-06-19T23:00:00Z,remote feed,Mexico 3-2 South Korea\n"
        )

    monkeypatch.setattr("match_result_refresh.requests.get", lambda *args, **kwargs: Response())

    result = refresh_match_results(
        fixtures_path=fixtures_path,
        updates_path=updates_path,
        results_path=results_path,
        remote_updates_url="https://example.com/results.csv",
        as_of=pd.Timestamp("2026-06-20T04:00:00Z"),
    )

    saved = load_match_results(results_path)
    assert result["added"] == 1
    assert result["source"]["remote_rows"] == 1
    assert saved.iloc[0]["match_id"] == "m1"
    assert int(saved.iloc[0]["home_score"]) == 3
    assert saved.iloc[0]["result_source"] == "remote feed"


def test_refresh_match_results_reports_missing_finished_results(tmp_path):
    fixtures_path = tmp_path / "fixtures.csv"
    updates_path = tmp_path / "updates.csv"
    results_path = tmp_path / "results.csv"
    _fixtures(fixtures_path)
    pd.DataFrame(columns=[
        "match_id",
        "home_score",
        "away_score",
        "result_status",
        "result_favorite_outcome",
        "result_last_checked_utc",
        "result_source",
        "result_notes",
    ]).to_csv(updates_path, index=False)

    result = refresh_match_results(
        fixtures_path=fixtures_path,
        updates_path=updates_path,
        results_path=results_path,
        as_of=pd.Timestamp("2026-06-20T04:00:00Z"),
    )

    assert result["status"] == "missing_updates"
    assert result["added"] == 0
    assert result["missing_finished_results"] == 1
    assert "1 finished fixtures still need a result source" in result["message"]
