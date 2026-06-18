import pandas as pd

from fifa_rankings import (
    add_fifa_ranking_features,
    get_latest_fifa_ranking_before_date,
    load_fifa_rankings,
    normalize_team_name,
)


def test_load_valid_fifa_rankings(tmp_path):
    path = tmp_path / "rankings.csv"
    path.write_text(
        "ranking_date,team,fifa_rank,fifa_points\n"
        "2026-04-01,Belgium,8,1740.5\n"
    )

    df, warnings = load_fifa_rankings(path)

    assert warnings == []
    assert len(df) == 1
    assert df.iloc[0]["fifa_rank"] == 8


def test_missing_csv_handled_gracefully(tmp_path):
    df, warnings = load_fifa_rankings(tmp_path / "missing.csv")

    assert df.empty
    assert warnings


def test_required_columns_validated(tmp_path):
    path = tmp_path / "rankings.csv"
    path.write_text("ranking_date,team,fifa_rank\n2026-04-01,A,1\n")

    df, warnings = load_fifa_rankings(path)

    assert df.empty
    assert "fifa_points" in warnings[0]


def test_latest_ranking_before_match_date_selected(tmp_path):
    path = tmp_path / "rankings.csv"
    path.write_text(
        "ranking_date,team,fifa_rank,fifa_points\n"
        "2026-01-01,Iran,22,1560\n"
        "2026-07-01,Iran,5,1800\n"
    )
    df, _ = load_fifa_rankings(path)

    ranking = get_latest_fifa_ranking_before_date(df, "IR Iran", "2026-06-01")

    assert ranking["fifa_rank"] == 22


def test_team_name_normalization_works():
    assert normalize_team_name("USA") == normalize_team_name("United States")
    assert normalize_team_name("Türkiye") == normalize_team_name("Turkey")
    assert normalize_team_name("Côte d'Ivoire") == normalize_team_name("Ivory Coast")
    assert normalize_team_name("Korea Republic") == normalize_team_name("South Korea")


def test_ranking_features_added_correctly(tmp_path):
    path = tmp_path / "rankings.csv"
    path.write_text(
        "ranking_date,team,fifa_rank,fifa_points\n"
        "2026-04-01,Belgium,8,1740\n"
        "2026-04-01,Iran,20,1580\n"
    )
    rankings, _ = load_fifa_rankings(path)
    matches = pd.DataFrame(
        [
            {
                "date": "2026-06-01",
                "home_team": "Belgium",
                "away_team": "IR Iran",
            }
        ]
    )

    enriched, warnings = add_fifa_ranking_features(matches, rankings)

    assert warnings == []
    assert enriched.iloc[0]["fifa_rank_diff"] == 12
    assert enriched.iloc[0]["fifa_points_diff"] == 160
    assert enriched.iloc[0]["home_fifa_rank_missing"] == False


def test_missing_ranking_flags_created():
    matches = pd.DataFrame([{"date": "2026-06-01", "home_team": "A", "away_team": "B"}])

    enriched, warnings = add_fifa_ranking_features(matches, pd.DataFrame())

    assert warnings
    assert enriched.iloc[0]["home_fifa_rank_missing"] == True
    assert enriched.iloc[0]["away_fifa_rank_missing"] == True
