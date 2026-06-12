from best_prediction_source import select_best_prediction_source


PRODUCTION_MODEL = {"status": "production_ready", "is_usable_as_best_available": True}
DEMO_MODEL = {"status": "demo_model", "is_usable_as_best_available": False}
MISSING_MODEL = {"status": "missing", "is_usable_as_best_available": False}
PRODUCTION_ENSEMBLE = {"status": "production_ready", "is_usable_as_best_available": True}


def test_production_ensemble_is_selected_before_model():
    result = select_best_prediction_source(PRODUCTION_MODEL, market_available=True, ensemble_readiness=PRODUCTION_ENSEMBLE)

    assert result["resolved_source"] == "ensemble"


def test_production_model_is_selected_before_market():
    result = select_best_prediction_source(PRODUCTION_MODEL, market_available=True)

    assert result["resolved_source"] == "historical_model"


def test_demo_model_falls_back_to_market():
    result = select_best_prediction_source(DEMO_MODEL, market_available=True)

    assert result["resolved_source"] == "market"
    assert result["status"] == "fallback_market"


def test_missing_model_falls_back_to_market():
    result = select_best_prediction_source(MISSING_MODEL, market_available=True)

    assert result["resolved_source"] == "market"


def test_no_market_and_no_model_gives_unavailable_status():
    result = select_best_prediction_source(MISSING_MODEL, market_available=False)

    assert result["resolved_source"] == "unavailable"
    assert result["status"] == "unavailable"
