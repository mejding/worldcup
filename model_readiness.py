from datetime import datetime


PRODUCTION_MIN_TRAINING_ROWS = 1000
PRODUCTION_MIN_TEST_ROWS = 200
PRODUCTION_MIN_FEATURE_COUNT = 10
PRODUCTION_MIN_YEAR_SPAN = 8
PRODUCTION_TRAINING_DATA_SOURCE = "historical_international_results"

REQUIRED_COVERAGE_FLAGS = [
    "includes_elo_features",
    "includes_form_features",
    "includes_tournament_features",
    "includes_neutral_venue",
    "includes_qualifiers",
    "includes_world_cup_or_major_tournaments",
]


REQUIRED_METADATA_FIELDS = [
    "model_version",
    "trained_at_utc",
    "training_rows",
    "test_rows",
    "feature_count",
    "training_data_source",
    "training_data_start_date",
    "training_data_end_date",
    "is_demo_model",
    "includes_elo_features",
    "includes_form_features",
    "includes_tournament_features",
    "includes_neutral_venue",
    "includes_qualifiers",
    "includes_world_cup_or_major_tournaments",
    "includes_schedule_features",
    "performance_accuracy",
    "performance_log_loss",
    "performance_brier_score",
    "performance_ece",
]


def _as_int(value, default=0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _metadata_value(metadata: dict, primary: str, legacy: str = None, default=None):
    if primary in metadata:
        return metadata.get(primary)
    if legacy and legacy in metadata:
        return metadata.get(legacy)
    return default


def _metric(metadata: dict, primary: str, legacy: str):
    if primary in metadata:
        return metadata.get(primary)
    return metadata.get("metrics", {}).get(legacy)


def _feature_count(metadata: dict) -> int:
    if "feature_count" in metadata:
        return _as_int(metadata.get("feature_count"))
    features = metadata.get("feature_columns", [])
    return len(features) if isinstance(features, list) else 0


def _looks_like_demo_source(value) -> bool:
    text = str(value or "").lower()
    return any(marker in text for marker in ["sample", "demo", "synthetic", "generated", "fallback", "bundled_baseline"])


def _parse_date(value):
    if not value:
        return None
    text = str(value).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        try:
            return datetime.fromisoformat(text.split(" ")[0])
        except ValueError:
            return None


def _training_year_span(start_date, end_date) -> float:
    start = _parse_date(start_date)
    end = _parse_date(end_date)
    if not start or not end or end < start:
        return 0.0
    return (end - start).days / 365.25


def normalize_model_metadata(metadata: dict) -> dict:
    metadata = metadata or {}
    feature_columns = metadata.get("feature_columns", [])
    if not isinstance(feature_columns, list):
        feature_columns = []
    start_date = _metadata_value(metadata, "training_data_start_date", "date_min")
    end_date = _metadata_value(metadata, "training_data_end_date", "date_max")
    return {
        "model_version": _metadata_value(metadata, "model_version", "trained_at"),
        "trained_at": _metadata_value(metadata, "trained_at_utc", "trained_at"),
        "training_rows": _as_int(_metadata_value(metadata, "training_rows", "number_of_training_rows")),
        "test_rows": _as_int(_metadata_value(metadata, "test_rows", "number_of_test_rows")),
        "feature_count": _feature_count(metadata),
        "training_data_source": metadata.get("training_data_source", metadata.get("model_artifact_type")),
        "training_data_start_date": start_date,
        "training_data_end_date": end_date,
        "training_year_span": float(metadata.get("training_year_span") or _training_year_span(start_date, end_date)),
        "is_demo_model": bool(metadata.get("is_demo_model", False)),
        "model_variant": metadata.get("model_variant", "unknown"),
        "selected_reason": metadata.get("selected_reason", ""),
        "includes_elo_features": bool(metadata.get("includes_elo_features", "elo_diff" in feature_columns)),
        "includes_fifa_ranking_features": bool(metadata.get("includes_fifa_ranking_features", "fifa_points_diff" in feature_columns)),
        "fifa_ranking_rows": _as_int(metadata.get("fifa_ranking_rows")),
        "fifa_rank_missing_rate": metadata.get("fifa_rank_missing_rate"),
        "includes_form_features": bool(metadata.get("includes_form_features", "home_points_per_match_last5" in feature_columns)),
        "includes_tournament_features": bool(metadata.get("includes_tournament_features", "is_world_cup" in feature_columns or "is_major_tournament" in feature_columns)),
        "includes_neutral_venue": bool(metadata.get("includes_neutral_venue", "neutral" in feature_columns)),
        "includes_qualifiers": bool(metadata.get("includes_qualifiers", "is_qualifier" in feature_columns)),
        "includes_world_cup_or_major_tournaments": bool(
            metadata.get(
                "includes_world_cup_or_major_tournaments",
                "is_world_cup" in feature_columns or "is_major_tournament" in feature_columns,
            )
        ),
        "includes_schedule_features": bool(metadata.get("includes_schedule_features", False)),
        "performance_accuracy": _metric(metadata, "performance_accuracy", "accuracy"),
        "performance_log_loss": _metric(metadata, "performance_log_loss", "log_loss"),
        "performance_brier_score": _metric(metadata, "performance_brier_score", "brier_score"),
        "performance_ece": _metric(metadata, "performance_ece", "ece"),
    }


def validate_model_artifact(metadata: dict) -> dict:
    metadata = metadata or {}
    normalized = normalize_model_metadata(metadata)
    warnings = []
    missing_fields = [field for field in REQUIRED_METADATA_FIELDS if field not in metadata]

    model_file_exists = bool(metadata.get("_model_file_exists", True))
    metadata_exists = bool(metadata.get("_metadata_exists", bool(metadata)))
    feature_columns_exists = bool(metadata.get("_feature_columns_exists", True))

    if not model_file_exists:
        warnings.append("Model artifact is missing.")
    if not metadata_exists:
        warnings.append("Model metadata is missing.")
    if not feature_columns_exists:
        warnings.append("Feature columns file is missing.")
    if missing_fields:
        warnings.append("Model metadata is incomplete: " + ", ".join(missing_fields))
    if normalized["training_rows"] < PRODUCTION_MIN_TRAINING_ROWS:
        warnings.append("This model appears to be trained on sample/demo data and is not production-ready.")
    if normalized["test_rows"] < PRODUCTION_MIN_TEST_ROWS:
        warnings.append("Model test set is too small for reliable production metrics.")
    if normalized["feature_count"] < PRODUCTION_MIN_FEATURE_COUNT:
        warnings.append("Model feature set is too small for production readiness.")
    if normalized["training_data_source"] != PRODUCTION_TRAINING_DATA_SOURCE:
        warnings.append("Production model must be trained from historical international results.")
    if normalized["training_year_span"] < PRODUCTION_MIN_YEAR_SPAN:
        warnings.append("Historical training period is too short for production readiness.")
    missing_coverage = [flag for flag in REQUIRED_COVERAGE_FLAGS if not normalized[flag]]
    if missing_coverage:
        warnings.append(
            "Production model requires historical coverage for qualifiers, major tournaments, Elo/form and match context."
        )
    if normalized["is_demo_model"] or _looks_like_demo_source(normalized["training_data_source"]):
        warnings.append("Model metadata marks this as a demo/sample artifact.")

    missing_or_invalid_artifacts = not model_file_exists or not metadata_exists or not feature_columns_exists
    incomplete_metadata_but_demo_sized = (
        missing_fields
        and (
            normalized["training_rows"] < PRODUCTION_MIN_TRAINING_ROWS
            or normalized["test_rows"] < PRODUCTION_MIN_TEST_ROWS
            or normalized["is_demo_model"]
            or _looks_like_demo_source(normalized["training_data_source"])
        )
    )
    if missing_or_invalid_artifacts:
        status = "missing" if not model_file_exists or not metadata_exists else "invalid"
    elif missing_fields and not incomplete_metadata_but_demo_sized:
        status = "invalid"
    elif (
        normalized["training_rows"] < PRODUCTION_MIN_TRAINING_ROWS
        or normalized["test_rows"] < PRODUCTION_MIN_TEST_ROWS
        or normalized["feature_count"] < PRODUCTION_MIN_FEATURE_COUNT
        or normalized["training_data_source"] != PRODUCTION_TRAINING_DATA_SOURCE
        or normalized["training_year_span"] < PRODUCTION_MIN_YEAR_SPAN
        or missing_coverage
        or normalized["is_demo_model"]
        or _looks_like_demo_source(normalized["training_data_source"])
    ):
        status = "demo_model"
    else:
        status = "production_ready"

    return {
        "status": status,
        "is_usable_for_predictions": status in {"production_ready", "demo_model"},
        "is_usable_as_best_available": status == "production_ready",
        "warnings": warnings,
        **normalized,
    }


def is_production_performance_available(readiness: dict, backtest_status: dict) -> bool:
    return (
        readiness.get("status") == "production_ready"
        and bool(backtest_status.get("backtest_exists"))
        and int(backtest_status.get("prediction_count") or 0) > 0
    )
