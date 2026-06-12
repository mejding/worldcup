from fixture_data import (
    build_predictions_from_fixtures,
    load_fixture_dataset,
    validate_prediction_fixture_consistency,
)


def test_predictions_match_reference_fixtures_without_consistency_warnings():
    fixtures = load_fixture_dataset()
    predictions = build_predictions_from_fixtures(fixtures)

    warnings = validate_prediction_fixture_consistency(predictions, fixtures)

    assert warnings == []


def test_sample_predictions_are_flagged_as_not_live_or_official():
    fixtures = load_fixture_dataset()
    predictions = build_predictions_from_fixtures(fixtures, fixture_source="sample_demo")

    warnings = validate_prediction_fixture_consistency(predictions, fixtures)

    assert any("sample fixtures" in warning for warning in warnings)


def test_missing_fixture_reference_is_flagged():
    fixtures = load_fixture_dataset()
    predictions = build_predictions_from_fixtures(fixtures)

    warnings = validate_prediction_fixture_consistency(predictions, fixtures.iloc[0:0])

    assert warnings == ["Missing fixture reference; predictions cannot be verified against official fixtures."]


def test_prediction_match_id_must_exist_in_fixture_source():
    fixtures = load_fixture_dataset()
    predictions = build_predictions_from_fixtures(fixtures)
    predictions.loc[predictions.index[0], "match_id"] = "STALE-SAMPLE-ID"

    warnings = validate_prediction_fixture_consistency(predictions, fixtures)

    assert any("STALE-SAMPLE-ID does not exist" in warning for warning in warnings)


def test_prediction_teams_must_match_fixture_source():
    fixtures = load_fixture_dataset()
    predictions = build_predictions_from_fixtures(fixtures)
    predictions.loc[predictions.index[0], "away_team"] = "Switzerland"

    warnings = validate_prediction_fixture_consistency(predictions, fixtures)

    assert any("teams do not match" in warning for warning in warnings)
