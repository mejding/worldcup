import pandas as pd
import pytest

import probability_sources
from probability_sources import (
    apply_probability_source,
    get_probability_columns,
    load_active_probability_source,
    save_active_probability_source,
)
from recommendations import recommend_for_match


def _df():
    return pd.DataFrame(
        {
            "market_home_prob": [0.4],
            "market_draw_prob": [0.3],
            "market_away_prob": [0.3],
            "model_home_prob": [0.7],
            "model_draw_prob": [0.2],
            "model_away_prob": [0.1],
            "ensemble_home_prob": [0.5],
            "ensemble_draw_prob": [0.3],
            "ensemble_away_prob": [0.2],
        }
    )


def test_probability_source_column_mapping_works():
    assert get_probability_columns("market")["home"] == "market_home_prob"
    assert get_probability_columns("ensemble")["draw"] == "ensemble_draw_prob"


def test_active_probabilities_are_created():
    result = apply_probability_source(_df(), "historical_model")

    assert result.iloc[0]["active_home_prob"] == pytest.approx(0.7)
    assert result.iloc[0]["active_probability_source"] == "historical_model"


def test_unavailable_source_falls_back_to_market():
    result = apply_probability_source(_df(), "draw_context_model")

    assert result.iloc[0]["active_probability_source"] == "historical_model"
    assert result.attrs["warnings"]


def test_best_validated_uses_model_when_market_is_unpriced_placeholder(monkeypatch):
    monkeypatch.setattr(
        probability_sources,
        "load_active_probability_source",
        lambda: {"source": "best_validated", "resolved_source": "market"},
    )
    df = pd.DataFrame(
        {
            "market_home_prob": [1 / 3],
            "market_draw_prob": [1 / 3],
            "market_away_prob": [1 / 3],
            "model_home_prob": [0.45],
            "model_draw_prob": [0.30],
            "model_away_prob": [0.25],
            "best_home_odds": [pd.NA],
            "best_draw_odds": [pd.NA],
            "best_away_odds": [pd.NA],
            "ds_home_odds": [pd.NA],
            "ds_draw_odds": [pd.NA],
            "ds_away_odds": [pd.NA],
        }
    )

    result = apply_probability_source(df, "best_validated")

    assert result.iloc[0]["active_probability_source"] == "historical_model"
    assert result.iloc[0]["active_home_prob"] == pytest.approx(0.45)
    assert result.iloc[0]["active_draw_prob"] == pytest.approx(0.30)
    assert result.attrs["warnings"]


def test_active_probability_source_state_can_be_saved_and_loaded(tmp_path):
    path = tmp_path / "active.json"
    save_active_probability_source({"source": "ensemble", "resolved_source": "ensemble", "w_market": 0.8, "w_model": 0.2}, path=path)

    assert load_active_probability_source(path)["resolved_source"] == "ensemble"


def test_recommendations_can_use_active_probabilities():
    row = pd.Series(
        {
            "model_home_prob": 0.1,
            "model_draw_prob": 0.1,
            "model_away_prob": 0.8,
            "active_home_prob": 0.6,
            "active_draw_prob": 0.2,
            "active_away_prob": 0.2,
            "ds_home_odds": 2.0,
            "ds_draw_odds": 3.0,
            "ds_away_odds": 2.0,
            "best_home_odds": 2.1,
            "best_home_bookmaker": "A",
            "best_draw_odds": 3.0,
            "best_draw_bookmaker": "B",
            "best_away_odds": 2.0,
            "best_away_bookmaker": "C",
        }
    )

    recommendation = recommend_for_match(row, 10000)

    assert recommendation["recommended_outcome_ds"] == "Home"
