import pandas as pd

from fifa_feature_ablation import MODEL_VARIANTS, compare_model_variants, select_model_variant


def test_model_variants_are_defined():
    assert {"baseline_no_strength", "elo_only", "fifa_only", "elo_plus_fifa"}.issubset(MODEL_VARIANTS)


def test_elo_plus_fifa_not_selected_if_log_loss_worsens():
    comparison = pd.DataFrame(
        [
            {"model_variant": "elo_only", "includes_elo": True, "includes_fifa_ranking": False, "match_count": 100, "log_loss": 0.9, "brier_score": 0.55, "ece": 0.1, "draw_calibration_gap": 0.0},
            {"model_variant": "elo_plus_fifa", "includes_elo": True, "includes_fifa_ranking": True, "match_count": 100, "log_loss": 0.91, "brier_score": 0.54, "ece": 0.1, "draw_calibration_gap": 0.0},
        ]
    )

    selected = select_model_variant(comparison)

    assert selected["model_variant"] == "elo_only"


def test_elo_plus_fifa_can_be_selected_if_it_improves_log_loss_and_brier():
    comparison = pd.DataFrame(
        [
            {"model_variant": "elo_only", "includes_elo": True, "includes_fifa_ranking": False, "match_count": 100, "log_loss": 0.9, "brier_score": 0.55, "ece": 0.1, "draw_calibration_gap": 0.0},
            {"model_variant": "elo_plus_fifa", "includes_elo": True, "includes_fifa_ranking": True, "match_count": 100, "log_loss": 0.88, "brier_score": 0.53, "ece": 0.1, "draw_calibration_gap": 0.0},
        ]
    )

    selected = select_model_variant(comparison)

    assert selected["model_variant"] == "elo_plus_fifa"


def test_small_dataset_does_not_produce_production_ready_ablation(tmp_path):
    historical = pd.DataFrame(
        [
            {"date": "2020-01-01", "home_team": "A", "away_team": "B", "home_score": 1, "away_score": 0, "result": "H", "tournament": "Friendly", "neutral": True},
            {"date": "2020-01-02", "home_team": "B", "away_team": "A", "home_score": 1, "away_score": 1, "result": "D", "tournament": "Friendly", "neutral": True},
            {"date": "2020-01-03", "home_team": "C", "away_team": "A", "home_score": 2, "away_score": 3, "result": "A", "tournament": "Friendly", "neutral": True},
        ]
    )

    result = compare_model_variants(
        historical,
        pd.DataFrame(),
        output_path=tmp_path / "comparison.csv",
        coverage_path=tmp_path / "coverage.csv",
        report_path=tmp_path / "report.md",
    )

    assert result["comparison"].empty
    assert result["warnings"]
