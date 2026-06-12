import pandas as pd

from odds_provider import get_odds_source_status
from odds_storage import save_odds_snapshot


def _manual_row():
    return pd.DataFrame(
        [
            {
                "match_id": "WC2026-GRA-001",
                "home_team": "Mexico",
                "away_team": "South Africa",
                "kickoff_utc": "2026-06-11T19:00:00Z",
                "bookmaker": "Book",
                "home_odds": 1.5,
                "draw_odds": 4.0,
                "away_odds": 7.0,
                "odds_last_updated_utc": "2026-06-10T12:00:00Z",
            }
        ]
    )


def _normalized_snapshot():
    return pd.DataFrame(
        [
            {
                "odds_source": "api",
                "provider": "the_odds_api",
                "fetched_at_utc": "2026-06-10T12:00:00Z",
                "event_id": "evt1",
                "commence_time_utc": "2026-06-11T19:00:00Z",
                "home_team": "Mexico",
                "away_team": "South Africa",
                "bookmaker_key": "book",
                "bookmaker_title": "Book",
                "bookmaker_last_update": "2026-06-10T12:00:00Z",
                "market_key": "h2h",
                "outcome_name": "Mexico",
                "outcome_type": "home",
                "outcome_price": 1.5,
            }
        ]
    )


def test_no_source_returns_missing_warning(tmp_path, monkeypatch):
    monkeypatch.delenv("ODDS_API_KEY", raising=False)

    status = get_odds_source_status(tmp_path / "manual.csv", tmp_path / "cache.csv")

    assert status["active_odds_source"] == "missing"
    assert status["warning"]


def test_api_key_present_sets_active_source_api(tmp_path, monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", "key")

    status = get_odds_source_status(tmp_path / "manual.csv", tmp_path / "cache.csv")

    assert status["active_odds_source"] == "api"
    assert not status["warning"]


def test_valid_manual_csv_sets_active_source_manual(tmp_path, monkeypatch):
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    manual_path = tmp_path / "manual.csv"
    _manual_row().to_csv(manual_path, index=False)

    status = get_odds_source_status(manual_path, tmp_path / "cache.csv")

    assert status["active_odds_source"] == "manual"
    assert status["manual_odds_valid"]
    assert "missing" not in status["warning"].lower()


def test_cached_odds_sets_active_source_cached(tmp_path, monkeypatch):
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    cache_path = tmp_path / "cache.csv"
    save_odds_snapshot(_normalized_snapshot(), cache_path)

    status = get_odds_source_status(tmp_path / "manual.csv", cache_path)

    assert status["active_odds_source"] == "cached"
    assert status["cached_odds_exists"]
    assert "missing" not in status["warning"].lower()
