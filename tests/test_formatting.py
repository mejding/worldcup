from components import format_edge, format_kelly, format_probability, outcome_label


def test_probability_formatting_is_compact_percentage():
    assert format_probability(0.456) == "45.6%"


def test_edge_formatting_includes_positive_sign():
    assert format_edge(0.042) == "+4.2%"


def test_kelly_formatting_uses_two_decimals():
    assert format_kelly(0.00348) == "0.35%"


def test_outcome_label_uses_team_names_for_bet_display():
    assert outcome_label("Away", "Denmark", "Japan") == "Japan win"
    assert outcome_label("Draw", "Denmark", "Japan", language="da") == "Uafgjort"
