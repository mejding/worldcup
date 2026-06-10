from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

SAMPLE_PREDICTIONS_PATH = DATA_DIR / "sample_predictions.csv"
BANKROLL_STATE_PATH = DATA_DIR / "bankroll_state.json"
BANKROLL_HISTORY_PATH = DATA_DIR / "bankroll_history.csv"
BET_LOG_PATH = DATA_DIR / "bet_log.csv"

BANKROLL_STATE_EXAMPLE_PATH = DATA_DIR / "bankroll_state.example.json"
BANKROLL_HISTORY_EXAMPLE_PATH = DATA_DIR / "bankroll_history.example.csv"
BET_LOG_EXAMPLE_PATH = DATA_DIR / "bet_log.example.csv"

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
