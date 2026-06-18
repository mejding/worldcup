import pandas as pd

from validation import select_best_source_from_validation


def _comparison(**overrides):
    data = {
        "source": ["market", "model", "ensemble_0.8_0.2"],
        "match_count": [1000, 1000, 1000],
        "accuracy": [0.55, 0.56, 0.57],
        "log_loss": [0.90, 0.88, 0.87],
        "brier_score": [0.54, 0.53, 0.52],
        "ece": [0.04, 0.03, 0.02],
        "draw_calibration_gap": [0.04, 0.03, 0.01],
        "w_market": [1.0, 0.0, 0.8],
        "w_model": [0.0, 1.0, 0.2],
    }
    data.update(overrides)
    return pd.DataFrame(data)


def test_lowest_log_loss_wins():
    result = select_best_source_from_validation(_comparison())

    assert result["selected_source"] == "ensemble_0.8_0.2"
    assert result["selected_label"] == "Ensemble"


def test_brier_score_used_as_tie_breaker():
    comparison = _comparison(log_loss=[0.90, 0.88, 0.88], brier_score=[0.54, 0.51, 0.52])

    result = select_best_source_from_validation(comparison)

    assert result["selected_source"] == "model"


def test_market_comparison_missing_prevents_beats_market_claim():
    result = select_best_source_from_validation(pd.DataFrame(), market_comparison_available=False)

    assert result["market_comparison_available"] is False
    assert any("historical market odds" in caveat for caveat in result["caveats"])


def test_ensemble_selected_only_when_improvement_exists():
    comparison = _comparison(log_loss=[0.8000, 0.8100, 0.7995], brier_score=[0.30, 0.31, 0.30], ece=[0.03, 0.04, 0.03])

    result = select_best_source_from_validation(comparison)

    assert result["selected_source"] == "market"
