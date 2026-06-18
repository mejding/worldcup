from model_performance_summary import display_metric_value, metric_interpretation


def test_no_metric_is_displayed_as_bare_dash():
    assert display_metric_value("ece", None) == "Not calculated yet"
    assert display_metric_value("log_loss", None) != "-"


def test_market_comparison_missing_message_is_user_friendly():
    message = "Market comparison cannot be calculated because historical market odds are not available."

    assert "historical market odds" in message
    assert "technical" not in message.lower()


def test_technical_warnings_are_not_main_headline():
    headline = "Model confidence: Medium"

    assert headline != "The model is usable as a baseline, but full validation is not complete."
    assert "Full walk-forward backtest has not been run yet" not in headline


def test_advanced_details_not_required_for_normal_summary():
    text = metric_interpretation("accuracy", 0.594)

    assert "1X2 football prediction" in text
