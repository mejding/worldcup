def select_best_prediction_source(
    model_readiness: dict,
    market_available: bool,
    ensemble_readiness: dict = None,
) -> dict:
    ensemble_readiness = ensemble_readiness or {}
    if ensemble_readiness.get("status") == "production_ready" and ensemble_readiness.get("is_usable_as_best_available", True):
        return {
            "resolved_source": "ensemble",
            "status": "ready",
            "reason": "Production-ready ensemble selected before model.",
        }
    if model_readiness.get("status") == "production_ready" and model_readiness.get("is_usable_as_best_available"):
        return {
            "resolved_source": "historical_model",
            "status": "ready",
            "reason": "Production-ready pre-trained model selected.",
        }
    if market_available:
        reason = "Market probabilities selected."
        if model_readiness.get("status") == "demo_model":
            reason = "Demo model is not production-ready. Falling back to market probabilities."
        elif model_readiness.get("status") in {"missing", "invalid"}:
            reason = "Production model unavailable. Falling back to market probabilities."
        return {
            "resolved_source": "market",
            "status": "fallback_market",
            "reason": reason,
        }
    return {
        "resolved_source": "unavailable",
        "status": "unavailable",
        "reason": "No production-ready model and no market probabilities available.",
    }
