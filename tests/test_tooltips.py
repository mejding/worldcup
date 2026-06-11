from tooltip_definitions import REQUIRED_TOOLTIP_KEYS, TOOLTIPS


def test_required_tooltips_exist_and_have_text():
    for key in REQUIRED_TOOLTIP_KEYS:
        assert key in TOOLTIPS
        assert isinstance(TOOLTIPS[key], str)
        assert TOOLTIPS[key].strip()


def test_recommendation_tooltips_cover_statuses():
    assert "minimum edge" in TOOLTIPS["no_bet"]
    assert "Danske Spil" in TOOLTIPS["better_elsewhere"]
    assert "Danske Spil" in TOOLTIPS["playable_ds"]
