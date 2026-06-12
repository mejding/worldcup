import pandas as pd

from fixture_data import (
    REQUIRED_FIXTURE_COLUMNS,
    load_fixture_dataset,
    validate_fixture_dataset,
)


def test_reference_fixture_file_has_required_columns():
    fixtures = load_fixture_dataset()

    for column in REQUIRED_FIXTURE_COLUMNS:
        assert column in fixtures.columns


def test_reference_fixture_file_contains_known_group_b_correction():
    fixtures = load_fixture_dataset()
    valid, messages = validate_fixture_dataset(fixtures)

    assert valid is False
    assert any("incomplete" in message.lower() for message in messages)
    assert not any("Known fixture missing" in message for message in messages)
    assert not any("Canada vs Switzerland must not be on 2026-06-12" in message for message in messages)


def test_canada_switzerland_on_june_12_is_rejected():
    fixtures = load_fixture_dataset()
    broken = fixtures.copy()
    broken.loc[broken.index[0], "home_team"] = "Canada"
    broken.loc[broken.index[0], "away_team"] = "Switzerland"
    broken.loc[broken.index[0], "kickoff_utc"] = "2026-06-12T19:00:00Z"

    _, messages = validate_fixture_dataset(broken)

    assert "Invalid fixture: Canada vs Switzerland must not be on 2026-06-12." in messages


def test_group_b_expected_teams_are_present_when_group_data_exists():
    fixtures = load_fixture_dataset()
    group_b = fixtures[fixtures["group"] == "B"]
    teams = set(group_b["home_team"]) | set(group_b["away_team"])

    assert {"Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"} <= teams


def test_empty_fixture_dataset_is_invalid():
    valid, messages = validate_fixture_dataset(pd.DataFrame(columns=REQUIRED_FIXTURE_COLUMNS))

    assert valid is False
    assert any("missing" in message.lower() for message in messages)
