import pytest

from evaluation import (
    calculate_accuracy,
    calculate_draw_calibration,
    calculate_log_loss,
    calculate_multiclass_brier_score,
)


def test_accuracy_calculation():
    assert calculate_accuracy(["H", "D", "A"], ["H", "A", "A"]) == pytest.approx(2 / 3)


def test_log_loss_calculation():
    value = calculate_log_loss(["H", "D"], [[0.8, 0.1, 0.1], [0.2, 0.6, 0.2]], ["H", "D", "A"])
    assert value > 0


def test_multiclass_brier_score():
    value = calculate_multiclass_brier_score(["H"], [[0.8, 0.1, 0.1]], ["H", "D", "A"])
    assert value == pytest.approx(0.06)


def test_draw_calibration_buckets():
    df = calculate_draw_calibration(["D", "H", "D"], [0.22, 0.28, 0.45])

    assert "0.20-0.25" in df["bucket"].tolist()
    assert df["count"].sum() == 3

