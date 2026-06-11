from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
HISTORICAL_DATA_DIR = DATA_DIR / "historical"
MODELS_DIR = DATA_DIR / "models"

SAMPLE_PREDICTIONS_PATH = DATA_DIR / "sample_predictions.csv"
LIVE_PREDICTIONS_PATH = PROCESSED_DATA_DIR / "live_predictions.csv"
MODEL_PREDICTIONS_PATH = PROCESSED_DATA_DIR / "model_predictions.csv"
LIVE_PREDICTIONS_WITH_MODEL_PATH = PROCESSED_DATA_DIR / "live_predictions_with_model.csv"
TRAINING_DATASET_PATH = PROCESSED_DATA_DIR / "training_dataset.csv"
MODEL_EVALUATION_PATH = PROCESSED_DATA_DIR / "model_evaluation.csv"
HISTORICAL_RESULTS_PATH = HISTORICAL_DATA_DIR / "international_results.csv"
MODEL_PATH = MODELS_DIR / "model.pkl"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.json"
BANKROLL_STATE_PATH = DATA_DIR / "bankroll_state.json"
BANKROLL_HISTORY_PATH = DATA_DIR / "bankroll_history.csv"
BET_LOG_PATH = DATA_DIR / "bet_log.csv"
FIXTURES_SNAPSHOT_PATH = RAW_DATA_DIR / "fixtures_snapshots.csv"
ODDS_SNAPSHOT_PATH = RAW_DATA_DIR / "odds_snapshots.csv"

BANKROLL_STATE_EXAMPLE_PATH = DATA_DIR / "bankroll_state.example.json"
BANKROLL_HISTORY_EXAMPLE_PATH = DATA_DIR / "bankroll_history.example.csv"
BET_LOG_EXAMPLE_PATH = DATA_DIR / "bet_log.example.csv"

DATA_MODE_OPTIONS = ("sample", "live")
DATA_MODE = "sample"
MODEL_SOURCE_OPTIONS = ("market_only", "historical_model", "historical_model_if_available")
MODEL_SOURCE = "historical_model_if_available"

ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"
ODDS_API_SPORT_KEY = "soccer_fifa_world_cup"
ODDS_API_REGION = "eu"
ODDS_API_MARKET = "h2h"
ODDS_API_ODDS_FORMAT = "decimal"
PREFERRED_BOOKMAKER_NAMES = ["Danske Spil", "DanskeSpil", "Danske Spil A/S", "danske_spil"]

STAKING_PROFILES = {
    "Conservative": {
        "fractional_kelly_multiplier": 0.25,
        "max_stake_pct_of_bankroll": 0.015,
        "min_edge_threshold": 0.03,
        "min_stake_pct_threshold": 0.0025,
    },
    "Standard": {
        "fractional_kelly_multiplier": 0.25,
        "max_stake_pct_of_bankroll": 0.025,
        "min_edge_threshold": 0.025,
        "min_stake_pct_threshold": 0.0025,
    },
    "Offensive": {
        "fractional_kelly_multiplier": 0.33,
        "max_stake_pct_of_bankroll": 0.03,
        "min_edge_threshold": 0.02,
        "min_stake_pct_threshold": 0.0025,
    },
    "Aggressive": {
        "fractional_kelly_multiplier": 0.50,
        "max_stake_pct_of_bankroll": 0.04,
        "min_edge_threshold": 0.02,
        "min_stake_pct_threshold": 0.0025,
    },
}

DEFAULT_PROFILE_NAME = "Standard"
DEFAULT_STAKING_PROFILE = STAKING_PROFILES[DEFAULT_PROFILE_NAME].copy()

PREFERRED_BOOKMAKER = "Danske Spil"

REQUIRED_PREDICTION_COLUMNS = [
    "match_id",
    "kickoff_time",
    "group",
    "matchday",
    "home_team",
    "away_team",
    "model_home_prob",
    "model_draw_prob",
    "model_away_prob",
    "market_home_prob",
    "market_draw_prob",
    "market_away_prob",
    "ds_home_odds",
    "ds_draw_odds",
    "ds_away_odds",
    "best_home_odds",
    "best_home_bookmaker",
    "best_draw_odds",
    "best_draw_bookmaker",
    "best_away_odds",
    "best_away_bookmaker",
    "draw_context_score",
    "draw_context_label",
    "home_draw_utility",
    "away_draw_utility",
    "mutual_draw_acceptance",
    "one_team_must_win",
    "both_teams_draw_satisfied",
]


def get_staking_profile(profile_name: str = DEFAULT_PROFILE_NAME) -> dict:
    return STAKING_PROFILES.get(profile_name, STAKING_PROFILES[DEFAULT_PROFILE_NAME]).copy()


def validate_staking_profile(profile: dict) -> list[str]:
    errors = []
    fractional_kelly = float(profile.get("fractional_kelly_multiplier", -1))
    max_stake = float(profile.get("max_stake_pct_of_bankroll", -1))
    min_edge = float(profile.get("min_edge_threshold", -1))
    min_stake = float(profile.get("min_stake_pct_threshold", -1))

    if not 0 <= fractional_kelly <= 1:
        errors.append("Fractional Kelly must be between 0 and 1.")
    if not 0 < max_stake <= 0.20:
        errors.append("Max stake must be greater than 0 and no more than 20%.")
    if not 0 <= min_edge <= 0.50:
        errors.append("Minimum edge must be between 0 and 50%.")
    if not 0 <= min_stake <= max_stake:
        errors.append("Minimum stake must be between 0 and max stake.")
    return errors


def get_secret_or_env(name: str, default=None):
    try:
        import streamlit as st

        if hasattr(st, "secrets") and name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name, default)
