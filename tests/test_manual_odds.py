import pandas as pd

from manual_odds import load_manual_odds, normalize_manual_odds


def _valid_manual_df():
    return pd.DataFrame(
        [
            {
                "match_id": "WC2026-GRA-001",
                "home_team": "Mexico",
                "away_team": "South Africa",
                "kickoff_utc": "2026-06-11T19:00:00Z",
                "bookmaker": "Danske Spil",
                "home_odds": 1.44,
                "draw_odds": 4.0,
                "away_odds": 7.5,
                "odds_last_updated_utc": "2026-06-10T12:00:00Z",
                "is_danske_spil": True,
            }
        ]
    )


def test_missing_manual_odds_file_handled_gracefully(tmp_path):
    df, warnings = load_manual_odds(tmp_path / "manual_odds.csv")

    assert df.empty
    assert warnings


def test_valid_manual_odds_loads(tmp_path):
    path = tmp_path / "manual_odds.csv"
    _valid_manual_df().to_csv(path, index=False)

    df, warnings = load_manual_odds(path)

    assert len(df) == 1
    assert warnings == []


def test_invalid_manual_odds_returns_warnings(tmp_path):
    path = tmp_path / "manual_odds.csv"
    df = _valid_manual_df()
    df.loc[0, "home_odds"] = 1.0
    df.to_csv(path, index=False)

    loaded, warnings = load_manual_odds(path)

    assert loaded.empty
    assert warnings


def test_manual_odds_convert_to_normalized_long_format():
    normalized = normalize_manual_odds(_valid_manual_df())

    assert len(normalized) == 3
    assert set(normalized["outcome_type"]) == {"home", "draw", "away"}
    assert set(normalized["bookmaker_key"]) == {"danske_spil"}
    assert set(normalized["bookmaker_title"]) == {"Danske Spil"}
    assert set(normalized["odds_source"]) == {"manual_csv"}


def test_example_file_is_not_loaded_automatically(tmp_path):
    example_path = tmp_path / "manual_odds.example.csv"
    _valid_manual_df().to_csv(example_path, index=False)

    df, warnings = load_manual_odds(tmp_path / "manual_odds.csv")

    assert df.empty
    assert warnings
