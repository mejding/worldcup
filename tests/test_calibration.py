import pytest

from calibration import create_confidence_calibration_bins, create_draw_calibration_table


def test_confidence_calibration_bins_output_columns():
    df = create_confidence_calibration_bins(
        ["H", "D"],
        [[0.8, 0.1, 0.1], [0.2, 0.6, 0.2]],
        n_bins=5,
    )

    assert set(["bin_lower", "bin_upper", "count", "avg_confidence", "accuracy", "calibration_gap"]).issubset(df.columns)
    assert len(df) == 5


def test_draw_calibration_table_output_columns():
    df = create_draw_calibration_table(["D", "H"], [0.22, 0.28])

    assert set(["bin_label", "count", "avg_predicted_draw_probability", "actual_draw_rate", "calibration_gap"]).issubset(df.columns)


def test_buckets_handle_empty_bins():
    df = create_draw_calibration_table(["H"], [0.10])

    assert df["count"].sum() == 1
    assert (df["count"] == 0).any()


def test_calibration_gap_calculation():
    df = create_draw_calibration_table(["D", "H"], [0.22, 0.23])
    row = df[df["bin_label"] == "0.20-0.25"].iloc[0]

    assert row["calibration_gap"] == pytest.approx(0.225 - 0.5)
