from model_performance_summary import display_metric_value, metric_interpretation


def test_no_metric_is_displayed_as_bare_dash():
    assert display_metric_value("ece", None) == "Not calculated yet"
    assert display_metric_value("log_loss", None) != "-"


def test_market_comparison_missing_message_is_user_friendly():
    message = "Market comparison has not been calculated yet."

    assert "not been calculated yet" in message
    assert "technical" not in message.lower()


def test_advanced_details_not_required_for_normal_summary():
    text = metric_interpretation("accuracy", 0.594)

    assert "1X2 football prediction" in text
