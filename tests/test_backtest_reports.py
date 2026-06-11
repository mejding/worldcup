import pandas as pd

from backtest_reports import create_backtest_report, evaluate_by_segment
from calibration import create_confidence_calibration_bins, create_draw_calibration_table


def _predictions():
    return pd.DataFrame(
        {
            "fold_id": [1, 1, 2],
            "date": ["2020-01-01", "2020-02-01", "2021-01-01"],
            "home_team": ["A", "B", "C"],
            "away_team": ["D", "E", "F"],
            "tournament": ["Friendly", "World Cup", "Friendly"],
            "tournament_category": ["friendly", "world_cup", "friendly"],
            "neutral": [False, True, False],
            "actual_result": ["H", "D", "A"],
            "pred_home_prob": [0.7, 0.2, 0.2],
            "pred_draw_prob": [0.2, 0.5, 0.2],
            "pred_away_prob": [0.1, 0.3, 0.6],
            "predicted_result": ["H", "D", "A"],
            "confidence": [0.7, 0.5, 0.6],
            "is_correct": [True, True, True],
        }
    )


def test_evaluate_by_segment_returns_expected_segments():
    result = evaluate_by_segment(_predictions())

    assert "Overall" in result["segment_name"].tolist()
    assert "tournament_category" in result["segment_name"].tolist()
    assert "major_tournament" in result["segment_name"].tolist()


def test_small_sample_segments_included_with_match_count():
    result = evaluate_by_segment(_predictions())
    wc = result[(result["segment_name"] == "tournament_category") & (result["segment_value"] == "world_cup")]

    assert wc.iloc[0]["match_count"] == 1


def test_report_markdown_file_is_created(tmp_path):
    predictions = _predictions()
    segments = evaluate_by_segment(predictions)
    draw = create_draw_calibration_table(predictions["actual_result"], predictions["pred_draw_prob"])
    bins = create_confidence_calibration_bins(predictions["actual_result"], predictions[["pred_home_prob", "pred_draw_prob", "pred_away_prob"]].to_numpy())
    path = create_backtest_report(predictions, pd.DataFrame({"fold_id": [1]}), segments, draw, bins, tmp_path / "report.md")

    assert path.exists()


def test_report_includes_limitations_text(tmp_path):
    predictions = _predictions()
    segments = evaluate_by_segment(predictions)
    draw = create_draw_calibration_table(predictions["actual_result"], predictions["pred_draw_prob"])
    bins = create_confidence_calibration_bins(predictions["actual_result"], predictions[["pred_home_prob", "pred_draw_prob", "pred_away_prob"]].to_numpy())
    path = create_backtest_report(predictions, pd.DataFrame({"fold_id": [1]}), segments, draw, bins, tmp_path / "report.md")

    assert "Limitations" in path.read_text()
