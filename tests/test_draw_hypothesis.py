import pandas as pd

from draw_hypothesis import run_draw_hypothesis_analysis


def _historical():
    rows = []
    scores = [(1, 1), (2, 0), (0, 1), (0, 0)]
    for i in range(40):
        home_score, away_score = scores[i % len(scores)]
        rows.append(
            {
                "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=i),
                "tournament": "FIFA World Cup" if i < 12 else "Friendly",
                "group": "A" if i < 12 else pd.NA,
                "matchday": (i % 3) + 1 if i < 12 else pd.NA,
                "home_team": f"H{i % 6}",
                "away_team": f"A{i % 6}",
                "home_score": home_score,
                "away_score": away_score,
                "neutral": i % 2 == 0,
            }
        )
    return pd.DataFrame(rows)


def test_draw_rates_are_calculated_correctly(tmp_path):
    result = run_draw_hypothesis_analysis(_historical(), output_dir=tmp_path)
    overall = result["segments"][result["segments"]["segment_name"] == "overall"].iloc[0]

    assert overall["draw_count"] == 20
    assert overall["draw_rate"] == 0.5


def test_baseline_draw_rate_is_calculated(tmp_path):
    result = run_draw_hypothesis_analysis(_historical(), output_dir=tmp_path)

    assert result["segments"]["baseline_draw_rate"].iloc[0] == 0.5


def test_confidence_intervals_are_present(tmp_path):
    result = run_draw_hypothesis_analysis(_historical(), output_dir=tmp_path)

    assert "confidence_interval_low" in result["segments"].columns
    assert "confidence_interval_high" in result["segments"].columns


def test_small_sample_segments_handled(tmp_path):
    result = run_draw_hypothesis_analysis(_historical().head(3), output_dir=tmp_path)

    assert not result["segments"].empty


def test_missing_group_metadata_does_not_crash(tmp_path):
    result = run_draw_hypothesis_analysis(_historical().drop(columns=["group", "matchday"]), output_dir=tmp_path)

    assert not result["segments"].empty
