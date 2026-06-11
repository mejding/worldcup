import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from backtest_paths import ACTIVE_PROBABILITY_SOURCE_PATH


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


def resolve_probability_source(source: str) -> dict:
    if source == "best_validated":
        return load_active_probability_source()
    return {"source": source, "resolved_source": source, "reason": f"Manual source selected: {PROBABILITY_SOURCE_LABELS.get(source, source)}"}


def apply_probability_source(df: pd.DataFrame, source: str) -> pd.DataFrame:
    result = df.copy()
    warnings = []
    resolved = resolve_probability_source(source)
    requested = resolved.get("resolved_source", "market")
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
    if warnings:
        result["probability_source_warning"] = "; ".join(warnings)
    result.attrs["warnings"] = warnings
    return result
