from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
HISTORICAL_DATA_DIR = DATA_DIR / "historical"
MODELS_DIR = DATA_DIR / "models"
REPORTS_DIR = DATA_DIR / "reports"
REFERENCE_DATA_DIR = DATA_DIR / "reference"

SAMPLE_PREDICTIONS_PATH = DATA_DIR / "sample_predictions.csv"
REFERENCE_FIXTURES_PATH = REFERENCE_DATA_DIR / "worldcup_2026_fixtures.csv"
MATCH_RESULTS_PATH = REFERENCE_DATA_DIR / "match_results.csv"
FIFA_RANKINGS_PATH = REFERENCE_DATA_DIR / "fifa_rankings.csv"
MANUAL_ODDS_PATH = REFERENCE_DATA_DIR / "manual_odds.csv"
MANUAL_ODDS_EXAMPLE_PATH = REFERENCE_DATA_DIR / "manual_odds.example.csv"
LIVE_PREDICTIONS_PATH = PROCESSED_DATA_DIR / "live_predictions.csv"
PROCESSED_ODDS_PATH = PROCESSED_DATA_DIR / "latest_odds.csv"
MODEL_PREDICTIONS_PATH = PROCESSED_DATA_DIR / "model_predictions.csv"
LIVE_PREDICTIONS_WITH_MODEL_PATH = PROCESSED_DATA_DIR / "live_predictions_with_model.csv"
TRAINING_DATASET_PATH = PROCESSED_DATA_DIR / "training_dataset.csv"
MODEL_EVALUATION_PATH = PROCESSED_DATA_DIR / "model_evaluation.csv"
BACKTEST_PREDICTIONS_PATH = PROCESSED_DATA_DIR / "backtest_predictions.csv"
BACKTEST_SUMMARY_PATH = PROCESSED_DATA_DIR / "backtest_summary.csv"
BACKTEST_BY_SEGMENT_PATH = PROCESSED_DATA_DIR / "backtest_by_segment.csv"
BACKTEST_DRAW_CALIBRATION_PATH = PROCESSED_DATA_DIR / "backtest_draw_calibration.csv"
BACKTEST_CALIBRATION_BINS_PATH = PROCESSED_DATA_DIR / "backtest_calibration_bins.csv"
MODEL_VARIANT_COMPARISON_PATH = PROCESSED_DATA_DIR / "model_variant_comparison.csv"
FIFA_FEATURE_COVERAGE_PATH = PROCESSED_DATA_DIR / "fifa_feature_coverage.csv"
WORLD_CUP_BACKTEST_PREDICTIONS_PATH = PROCESSED_DATA_DIR / "world_cup_backtest_predictions.csv"
WORLD_CUP_BACKTEST_SUMMARY_PATH = PROCESSED_DATA_DIR / "world_cup_backtest_summary.csv"
BACKTEST_REPORT_PATH = REPORTS_DIR / "backtest_report.md"
FIFA_RANKING_FEATURE_REPORT_PATH = REPORTS_DIR / "fifa_ranking_feature_report.md"
DRAW_HYPOTHESIS_SUMMARY_PATH = PROCESSED_DATA_DIR / "draw_hypothesis_summary.csv"
DRAW_HYPOTHESIS_BY_SEGMENT_PATH = PROCESSED_DATA_DIR / "draw_hypothesis_by_segment.csv"
DRAW_FEATURE_COMPARISON_PATH = PROCESSED_DATA_DIR / "draw_feature_comparison.csv"
BACKTEST_PREDICTIONS_WITH_DRAW_FEATURES_PATH = PROCESSED_DATA_DIR / "backtest_predictions_with_draw_features.csv"
BACKTEST_SUMMARY_WITH_DRAW_FEATURES_PATH = PROCESSED_DATA_DIR / "backtest_summary_with_draw_features.csv"
GROUP_STATE_FEATURES_PATH = PROCESSED_DATA_DIR / "group_state_features.csv"
DRAW_HYPOTHESIS_REPORT_PATH = REPORTS_DIR / "draw_hypothesis_report.md"
ENSEMBLE_COMPARISON_PATH = PROCESSED_DATA_DIR / "ensemble_comparison.csv"
ENSEMBLE_PREDICTIONS_PATH = PROCESSED_DATA_DIR / "ensemble_predictions.csv"
ENSEMBLE_BACKTEST_SUMMARY_PATH = PROCESSED_DATA_DIR / "ensemble_backtest_summary.csv"
ENSEMBLE_BACKTEST_BY_SEGMENT_PATH = PROCESSED_DATA_DIR / "ensemble_backtest_by_segment.csv"
ACTIVE_PROBABILITY_SOURCE_PATH = PROCESSED_DATA_DIR / "active_probability_source.json"
ENSEMBLE_REPORT_PATH = REPORTS_DIR / "ensemble_report.md"
HISTORICAL_RESULTS_PATH = HISTORICAL_DATA_DIR / "international_results.csv"
MODEL_PATH = MODELS_DIR / "model.pkl"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.json"
BANKROLL_STATE_PATH = DATA_DIR / "bankroll_state.json"
BANKROLL_HISTORY_PATH = DATA_DIR / "bankroll_history.csv"
BET_LOG_PATH = DATA_DIR / "bet_log.csv"
FIXTURES_SNAPSHOT_PATH = RAW_DATA_DIR / "fixtures_snapshots.csv"
ODDS_SNAPSHOT_PATH = RAW_DATA_DIR / "odds_snapshots.csv"
RAW_ODDS_SNAPSHOT_PATH = ODDS_SNAPSHOT_PATH

BANKROLL_STATE_EXAMPLE_PATH = DATA_DIR / "bankroll_state.example.json"
BANKROLL_HISTORY_EXAMPLE_PATH = DATA_DIR / "bankroll_history.example.csv"
BET_LOG_EXAMPLE_PATH = DATA_DIR / "bet_log.example.csv"

DATA_MODE_OPTIONS = ("official", "sample", "live")
DATA_MODE = "official"
MODEL_SOURCE_OPTIONS = ("market_only", "historical_model", "historical_model_if_available")
MODEL_SOURCE = "historical_model_if_available"
PROBABILITY_SOURCE_OPTIONS = ("best_validated", "market", "historical_model", "draw_context_model", "ensemble")
DEFAULT_PROBABILITY_SOURCE = "best_validated"
DEFAULT_ENSEMBLE_W_MARKET = 0.8
DEFAULT_ENSEMBLE_W_MODEL = 0.2

ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"
ODDS_API_SPORT_KEY = "soccer_fifa_world_cup"
ODDS_API_REGIONS = "eu"
ODDS_API_MARKETS = "h2h"
ODDS_API_ODDS_FORMAT = "decimal"
ODDS_API_DATE_FORMAT = "iso"
ODDS_REFRESH_MINUTES = 30
ODDS_API_REGION = ODDS_API_REGIONS
ODDS_API_MARKET = ODDS_API_MARKETS
PREFERRED_BOOKMAKER_NAMES = ["Danske Spil", "DanskeSpil", "Danske Spil A/S", "danske_spil"]

STAKING_PROFILES = {
    "Conservative": {
        "fractional_kelly_multiplier": 0.20,
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
    "Aggressive": {
        "fractional_kelly_multiplier": 0.33,
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
