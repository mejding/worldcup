import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from backtest_paths import ACTIVE_PROBABILITY_SOURCE_PATH
from best_prediction_source import select_best_prediction_source
from model_registry import get_model_readiness


PROBABILITY_SOURCE_LABELS = {
    "market": "Market only",
    "historical_model": "Historical model",
    "draw_context_model": "Draw-context model",
    "ensemble": "Market-aware ensemble",
    "best_validated": "Best validated source",
}

DEFAULT_ACTIVE_SOURCE = {
    "source": "best_validated",
    "resolved_source": "market",
    "w_market": 1.0,
    "w_model": 0.0,
    "reason": "No validated ensemble available yet. Falling back to market probabilities.",
}


def get_probability_columns(source: str) -> dict:
    if source == "best_validated":
        source = load_active_probability_source().get("resolved_source", "market")
    mapping = {
        "market": {"home": "market_home_prob", "draw": "market_draw_prob", "away": "market_away_prob"},
        "historical_model": {"home": "model_home_prob", "draw": "model_draw_prob", "away": "model_away_prob"},
        "draw_context_model": {"home": "draw_model_home_prob", "draw": "draw_model_draw_prob", "away": "draw_model_away_prob"},
        "ensemble": {"home": "ensemble_home_prob", "draw": "ensemble_draw_prob", "away": "ensemble_away_prob"},
    }
    if source not in mapping:
        raise ValueError(f"Unknown probability source: {source}")
    return mapping[source]


def save_active_probability_source(config: dict, path: Path = ACTIVE_PROBABILITY_SOURCE_PATH) -> None:
    payload = DEFAULT_ACTIVE_SOURCE | config
    payload["last_updated_at"] = datetime.now(timezone.utc).isoformat()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def load_active_probability_source(path: Path = ACTIVE_PROBABILITY_SOURCE_PATH) -> dict:
    path = Path(path)
    if not path.exists():
        return DEFAULT_ACTIVE_SOURCE.copy()
    try:
        return DEFAULT_ACTIVE_SOURCE | json.loads(path.read_text())
    except Exception:
        return DEFAULT_ACTIVE_SOURCE.copy()


def _columns_available(df: pd.DataFrame, columns: dict) -> bool:
    return all(column in df.columns and not pd.to_numeric(df[column], errors="coerce").isna().all() for column in columns.values())


def _normalize_active(result: pd.DataFrame) -> pd.DataFrame:
    totals = result[["active_home_prob", "active_draw_prob", "active_away_prob"]].sum(axis=1)
    totals = totals.where(totals > 0, 1.0)
    for column in ["active_home_prob", "active_draw_prob", "active_away_prob"]:
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0) / totals
    return result


def _has_priced_odds(df: pd.DataFrame) -> pd.Series:
    best_columns = ["best_home_odds", "best_draw_odds", "best_away_odds"]
    ds_columns = ["ds_home_odds", "ds_draw_odds", "ds_away_odds"]
    result = pd.Series(False, index=df.index)
    for columns in [best_columns, ds_columns]:
        if all(column in df.columns for column in columns):
            odds = df[columns].apply(pd.to_numeric, errors="coerce")
            result = result | odds.gt(1.0).all(axis=1)
    return result


def _triplet_is_uniform(df: pd.DataFrame, prefix: str) -> pd.Series:
    columns = [f"{prefix}_home_prob", f"{prefix}_draw_prob", f"{prefix}_away_prob"]
    if not all(column in df.columns for column in columns):
        return pd.Series(False, index=df.index)
    values = df[columns].apply(pd.to_numeric, errors="coerce")
    return values.sub(1 / 3).abs().lt(0.0001).all(axis=1)


def _replace_unpriced_market_placeholders(result: pd.DataFrame, source: str, requested: str) -> pd.DataFrame:
    if source != "best_validated" or requested != "market":
        return result
    readiness = get_model_readiness()
    if not readiness["is_usable_as_best_available"]:
        result.attrs.setdefault("warnings", []).append(readiness["normal_user_message"])
        return result
    model_columns = get_probability_columns("historical_model")
    if not _columns_available(result, model_columns):
        return result

    use_model = (
        ~_has_priced_odds(result)
        & _triplet_is_uniform(result, "market")
        & _triplet_is_uniform(result, "active")
        & ~_triplet_is_uniform(result, "model")
    )
    if not use_model.any():
        return result

    result.loc[use_model, "active_home_prob"] = pd.to_numeric(result.loc[use_model, model_columns["home"]], errors="coerce")
    result.loc[use_model, "active_draw_prob"] = pd.to_numeric(result.loc[use_model, model_columns["draw"]], errors="coerce")
    result.loc[use_model, "active_away_prob"] = pd.to_numeric(result.loc[use_model, model_columns["away"]], errors="coerce")
    result.loc[use_model, "active_probability_source"] = "historical_model"
    result.loc[use_model, "probability_source_warning"] = "Market odds unavailable. Using historical model for this match."
    result.attrs.setdefault("warnings", []).append(
        "Market odds are unavailable for some fixtures. Best validated source uses the historical model for those matches."
    )
    return result


def resolve_probability_source(source: str) -> dict:
    if source == "best_validated":
        return load_active_probability_source()
    return {"source": source, "resolved_source": source, "reason": f"Manual source selected: {PROBABILITY_SOURCE_LABELS.get(source, source)}"}


def apply_probability_source(df: pd.DataFrame, source: str) -> pd.DataFrame:
    result = df.copy()
    warnings = []
    resolved = resolve_probability_source(source)
    requested = resolved.get("resolved_source", "market")
    if source == "best_validated":
        readiness = get_model_readiness()
        market_columns = get_probability_columns("market")
        market_available = _columns_available(result, market_columns)
        best_source = select_best_prediction_source(readiness, market_available)
        requested = best_source["resolved_source"]
        if requested == "unavailable":
            raise ValueError("No usable probability source columns available.")
        if best_source["status"] == "fallback_market":
            warnings.append(best_source["reason"])
    try:
        columns = get_probability_columns(requested)
    except ValueError:
        columns = get_probability_columns("market")
        requested = "market"
        warnings.append("Unknown probability source. Falling back to market probabilities.")

    if not _columns_available(result, columns):
        for fallback in ["historical_model", "market"]:
            fallback_columns = get_probability_columns(fallback)
            if _columns_available(result, fallback_columns):
                columns = fallback_columns
                requested = fallback
                warnings.append(f"Selected probability source unavailable. Falling back to {PROBABILITY_SOURCE_LABELS[fallback]}.")
                break
        else:
            raise ValueError("No usable probability source columns available.")

    result["active_home_prob"] = pd.to_numeric(result[columns["home"]], errors="coerce")
    result["active_draw_prob"] = pd.to_numeric(result[columns["draw"]], errors="coerce")
    result["active_away_prob"] = pd.to_numeric(result[columns["away"]], errors="coerce")
    result = _normalize_active(result)
    result["active_probability_source"] = requested
    result = _replace_unpriced_market_placeholders(result, source, requested)
    result = _normalize_active(result)
    warnings.extend(warning for warning in result.attrs.get("warnings", []) if warning not in warnings)
    if warnings:
        if "probability_source_warning" not in result.columns:
            result["probability_source_warning"] = ""
        empty_warning = result["probability_source_warning"].fillna("").eq("")
        result.loc[empty_warning, "probability_source_warning"] = "; ".join(warnings)
    result.attrs["warnings"] = warnings
    return result
