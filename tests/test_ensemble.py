import pandas as pd
import pytest

from ensemble import (
    MODEL_PROB_COLUMNS,
    MARKET_PROB_COLUMNS,
    apply_ensemble_to_upcoming_matches,
    calculate_ensemble_probabilities,
    create_weight_grid,
    normalize_probability_triplet,
    validate_probability_triplet,
)


def _df():
    return pd.DataFrame(
        {
            "market_home_prob": [0.5],
            "market_draw_prob": [0.3],
            "market_away_prob": [0.2],
            "model_home_prob": [0.4],
            "model_draw_prob": [0.4],
            "model_away_prob": [0.2],
        }
    )


def test_weight_grid_sums_to_one():
    grid = create_weight_grid()

    assert ((grid["w_market"] + grid["w_model"]).round(10) == 1).all()


def test_ensemble_probabilities_are_calculated_correctly():
    result = calculate_ensemble_probabilities(_df(), 0.8, MODEL_PROB_COLUMNS, MARKET_PROB_COLUMNS)

    assert result.iloc[0]["ensemble_home_prob"] == pytest.approx(0.48)
    assert result.iloc[0]["ensemble_draw_prob"] == pytest.approx(0.32)


def test_ensemble_probabilities_sum_to_one():
    result = calculate_ensemble_probabilities(_df(), 0.8, MODEL_PROB_COLUMNS, MARKET_PROB_COLUMNS)

    assert result[["ensemble_home_prob", "ensemble_draw_prob", "ensemble_away_prob"]].sum(axis=1).iloc[0] == pytest.approx(1)


def test_normalization_works():
    assert sum(normalize_probability_triplet(2, 1, 1)) == pytest.approx(1)


def test_invalid_probabilities_handled():
    assert not validate_probability_triplet(0.5, -0.1, 0.6)


def test_missing_model_probabilities_fall_back_to_market(tmp_path):
    result = apply_ensemble_to_upcoming_matches(_df().drop(columns=["model_home_prob"]), output_path=tmp_path / "ensemble.csv")

    assert result.iloc[0]["ensemble_w_market"] == 1.0
    assert result.attrs["warnings"]
