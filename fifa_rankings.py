from __future__ import annotations

import math
import unicodedata
from pathlib import Path
from typing import Union

import pandas as pd
from pandas.errors import EmptyDataError

import config as app_config


FIFA_RANKINGS_PATH = getattr(
    app_config,
    "FIFA_RANKINGS_PATH",
    app_config.REFERENCE_DATA_DIR / "fifa_rankings.csv",
)


REQUIRED_COLUMNS = ["ranking_date", "team", "fifa_rank", "fifa_points"]
OPTIONAL_COLUMNS = ["confederation", "previous_rank", "rank_change", "source", "source_last_checked"]
FIFA_RANKING_FEATURE_COLUMNS = [
    "home_fifa_rank",
    "away_fifa_rank",
    "fifa_rank_diff",
    "fifa_rank_gap_abs",
    "home_fifa_points",
    "away_fifa_points",
    "fifa_points_diff",
    "fifa_points_gap_abs",
    "home_fifa_rank_missing",
    "away_fifa_rank_missing",
]

TEAM_ALIASES = {
    "bosniaandherzegovina": "bosniaandherzegovina",
    "bosniaherzegovina": "bosniaandherzegovina",
    "bosnia": "bosniaandherzegovina",
    "caboverde": "capeverde",
    "capeverde": "capeverde",
    "congodr": "drcongo",
    "democraticrepublicofcongo": "drcongo",
    "drcongo": "drcongo",
    "cotedivoire": "ivorycoast",
    "ivorycoast": "ivorycoast",
    "holland": "netherlands",
    "netherlands": "netherlands",
    "iran": "iran",
    "iriran": "iran",
    "korearepublic": "southkorea",
    "republicofkorea": "southkorea",
    "southkorea": "southkorea",
    "turkey": "turkey",
    "turkiye": "turkey",
    "unitedstates": "unitedstates",
    "unitedstatesofamerica": "unitedstates",
    "usa": "unitedstates",
    "us": "unitedstates",
}


def normalize_team_name(value) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = "".join(char.lower() if char.isalnum() else "" for char in text)
    return TEAM_ALIASES.get(text, text)


def _empty_rankings() -> pd.DataFrame:
    return pd.DataFrame(columns=REQUIRED_COLUMNS + OPTIONAL_COLUMNS + ["team_normalized"])


def load_fifa_rankings(path: Union[str, Path] = FIFA_RANKINGS_PATH) -> tuple[pd.DataFrame, list[str]]:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return _empty_rankings(), [f"FIFA rankings file missing: {path}"]
    try:
        df = pd.read_csv(path)
    except EmptyDataError:
        return _empty_rankings(), [f"FIFA rankings file is empty: {path}"]
    except Exception as exc:
        return _empty_rankings(), [f"FIFA rankings file could not be read: {exc}"]

    if df.empty:
        return _empty_rankings(), [f"FIFA rankings file has no rows: {path}"]
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        return _empty_rankings(), ["FIFA rankings missing required columns: " + ", ".join(missing)]

    warnings = []
    result = df.copy()
    for column in OPTIONAL_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA
    result["ranking_date"] = pd.to_datetime(result["ranking_date"], errors="coerce", utc=True)
    result["fifa_rank"] = pd.to_numeric(result["fifa_rank"], errors="coerce")
    result["fifa_points"] = pd.to_numeric(result["fifa_points"], errors="coerce")
    result["team"] = result["team"].astype("string").str.strip()
    result["team_normalized"] = result["team"].map(normalize_team_name)

    valid = (
        result["ranking_date"].notna()
        & result["team"].notna()
        & result["team_normalized"].astype(str).str.len().gt(0)
        & result["fifa_rank"].notna()
        & result["fifa_points"].notna()
        & result["fifa_rank"].gt(0)
    )
    invalid_count = int((~valid).sum())
    if invalid_count:
        warnings.append(f"Ignored {invalid_count} invalid FIFA ranking rows.")
    clean = result.loc[valid, REQUIRED_COLUMNS + OPTIONAL_COLUMNS + ["team_normalized"]].copy()
    clean["fifa_rank"] = clean["fifa_rank"].astype(float)
    clean["fifa_points"] = clean["fifa_points"].astype(float)
    return clean.sort_values(["team_normalized", "ranking_date"]).reset_index(drop=True), warnings


def get_latest_fifa_ranking_before_date(rankings_df: pd.DataFrame, team: str, match_date) -> dict | None:
    if rankings_df is None or rankings_df.empty:
        return None
    timestamp = pd.to_datetime(match_date, errors="coerce", utc=True)
    if pd.isna(timestamp):
        return None
    normalized = normalize_team_name(team)
    team_rows = rankings_df[
        (rankings_df["team_normalized"] == normalized)
        & (pd.to_datetime(rankings_df["ranking_date"], errors="coerce", utc=True) <= timestamp)
    ].copy()
    if team_rows.empty:
        return None
    latest = team_rows.sort_values("ranking_date").iloc[-1]
    return {
        "fifa_rank": float(latest["fifa_rank"]),
        "fifa_points": float(latest["fifa_points"]),
        "ranking_date": latest["ranking_date"],
    }


def _neutral_rankings(rankings_df: pd.DataFrame) -> tuple[float, float]:
    if rankings_df is None or rankings_df.empty:
        return 100.0, 1300.0
    return float(rankings_df["fifa_rank"].median()), float(rankings_df["fifa_points"].median())


def add_fifa_ranking_features(matches_df: pd.DataFrame, rankings_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    result = matches_df.copy()
    warnings = []
    neutral_rank, neutral_points = _neutral_rankings(rankings_df)
    home_values = []
    away_values = []
    rankings_by_team = {}
    if rankings_df is not None and not rankings_df.empty:
        ranking_lookup = rankings_df.copy()
        ranking_lookup["ranking_date"] = pd.to_datetime(ranking_lookup["ranking_date"], errors="coerce", utc=True)
        ranking_lookup = ranking_lookup.dropna(subset=["ranking_date", "team_normalized", "fifa_rank", "fifa_points"])
        for team_key, team_df in ranking_lookup.sort_values("ranking_date").groupby("team_normalized", sort=False):
            rankings_by_team[team_key] = team_df.reset_index(drop=True)

    def lookup(team, match_date):
        timestamp = pd.to_datetime(match_date, errors="coerce", utc=True)
        if pd.isna(timestamp):
            return None
        team_rows = rankings_by_team.get(normalize_team_name(team))
        if team_rows is None or team_rows.empty:
            return None
        position = team_rows["ranking_date"].searchsorted(timestamp, side="right") - 1
        if position < 0:
            return None
        latest = team_rows.iloc[int(position)]
        return {
            "fifa_rank": float(latest["fifa_rank"]),
            "fifa_points": float(latest["fifa_points"]),
            "ranking_date": latest["ranking_date"],
        }

    date_column = "date" if "date" in result.columns else "kickoff_time"
    for _, row in result.iterrows():
        match_date = row.get(date_column)
        home = lookup(row.get("home_team"), match_date)
        away = lookup(row.get("away_team"), match_date)
        if home is None:
            warnings.append(f"Missing FIFA ranking for {row.get('home_team')} before {match_date}.")
            home = {"fifa_rank": neutral_rank, "fifa_points": neutral_points, "ranking_date": pd.NaT, "missing": True}
        else:
            home["missing"] = False
        if away is None:
            warnings.append(f"Missing FIFA ranking for {row.get('away_team')} before {match_date}.")
            away = {"fifa_rank": neutral_rank, "fifa_points": neutral_points, "ranking_date": pd.NaT, "missing": True}
        else:
            away["missing"] = False
        home_values.append(home)
        away_values.append(away)

    result["home_fifa_rank"] = [item["fifa_rank"] for item in home_values]
    result["home_fifa_points"] = [item["fifa_points"] for item in home_values]
    result["home_fifa_ranking_date"] = [item["ranking_date"] for item in home_values]
    result["home_fifa_rank_missing"] = [bool(item["missing"]) for item in home_values]
    result["away_fifa_rank"] = [item["fifa_rank"] for item in away_values]
    result["away_fifa_points"] = [item["fifa_points"] for item in away_values]
    result["away_fifa_ranking_date"] = [item["ranking_date"] for item in away_values]
    result["away_fifa_rank_missing"] = [bool(item["missing"]) for item in away_values]
    result["fifa_rank_diff"] = result["away_fifa_rank"] - result["home_fifa_rank"]
    result["fifa_points_diff"] = result["home_fifa_points"] - result["away_fifa_points"]
    result["fifa_rank_gap_abs"] = result["fifa_rank_diff"].abs()
    result["fifa_points_gap_abs"] = result["fifa_points_diff"].abs()
    result["fifa_rank_log_gap"] = result["fifa_rank_gap_abs"].map(lambda value: math.log1p(float(value)))
    result["fifa_points_diff_scaled"] = result["fifa_points_diff"] / 100.0
    return result, warnings


def create_fifa_feature_coverage(matches_df: pd.DataFrame) -> pd.DataFrame:
    if matches_df.empty or "home_fifa_rank_missing" not in matches_df.columns:
        return pd.DataFrame(columns=["team", "matches_with_ranking", "matches_missing_ranking", "missing_rate", "earliest_ranking_date", "latest_ranking_date"])
    rows = []
    for side in ["home", "away"]:
        subset = matches_df[
            [
                f"{side}_team",
                f"{side}_fifa_rank_missing",
                f"{side}_fifa_ranking_date",
            ]
        ].rename(
            columns={
                f"{side}_team": "team",
                f"{side}_fifa_rank_missing": "missing",
                f"{side}_fifa_ranking_date": "ranking_date",
            }
        )
        rows.append(subset)
    combined = pd.concat(rows, ignore_index=True)
    coverage_rows = []
    for team, team_df in combined.groupby("team"):
        missing = team_df["missing"].fillna(True).astype(bool)
        dates = pd.to_datetime(team_df.loc[~missing, "ranking_date"], errors="coerce", utc=True)
        total = len(team_df)
        missing_count = int(missing.sum())
        coverage_rows.append(
            {
                "team": team,
                "matches_with_ranking": int(total - missing_count),
                "matches_missing_ranking": missing_count,
                "missing_rate": float(missing_count / total) if total else 0.0,
                "earliest_ranking_date": "" if dates.dropna().empty else dates.min().date().isoformat(),
                "latest_ranking_date": "" if dates.dropna().empty else dates.max().date().isoformat(),
            }
        )
    return pd.DataFrame(coverage_rows).sort_values("team").reset_index(drop=True)
