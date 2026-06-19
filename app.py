import html
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

import backtest_paths as backtest_path_config
import config as app_config
from backtest import compare_baseline_vs_draw_context_model, run_walk_forward_backtest, run_world_cup_backtest
from bankroll import load_bankroll_history, load_bankroll_state, reset_bankroll, update_bankroll
from bet_log import add_bet, calculate_bet_summary, load_bet_log, reset_bet_settlement, settle_bet
from charts import (
    active_vs_market_model_chart,
    backtest_metric_by_fold_chart,
    bankroll_history_chart,
    confidence_calibration_chart,
    draw_calibration_chart,
    draw_context_score_distribution_chart,
    draw_feature_comparison_chart,
    draw_rate_by_segment_chart,
    ensemble_weight_metric_chart,
    probability_comparison_chart,
    probability_source_comparison_chart,
    profit_loss_by_bookmaker_chart,
    profit_loss_by_outcome_chart,
    render_chart,
    segment_metric_chart,
)
from components import (
    best_source_card,
    calibration_gap_badge,
    draw_context_badge,
    draw_context_card_v2,
    draw_context_decision_card,
    draw_context_score_badge,
    draw_hypothesis_summary_card,
    empty_state,
    ensemble_weight_badge,
    format_dkk,
    format_odds,
    format_percentage,
    format_probability,
    metric_card,
    metric_improvement_badge,
    metric_row,
    model_metric_explanation,
    odds_comparison_table,
    outcome_label,
    recommendation_card_v2,
    probability_source_badge,
    small_sample_caveat,
    small_sample_warning,
    status_badge,
)
from backtest_paths import (
    ACTIVE_PROBABILITY_SOURCE_PATH,
    BACKTEST_BY_SEGMENT_PATH,
    BACKTEST_CALIBRATION_BINS_PATH,
    BACKTEST_DRAW_CALIBRATION_PATH,
    BACKTEST_PREDICTIONS_PATH,
    BACKTEST_PREDICTIONS_WITH_DRAW_FEATURES_PATH,
    BACKTEST_REPORT_PATH,
    BACKTEST_SUMMARY_PATH,
    BACKTEST_SUMMARY_WITH_DRAW_FEATURES_PATH,
    DRAW_FEATURE_COMPARISON_PATH,
    DRAW_HYPOTHESIS_BY_SEGMENT_PATH,
    DRAW_HYPOTHESIS_REPORT_PATH,
    DRAW_HYPOTHESIS_SUMMARY_PATH,
    ENSEMBLE_COMPARISON_PATH,
    ENSEMBLE_PREDICTIONS_PATH,
    ENSEMBLE_REPORT_PATH,
    PROCESSED_DATA_DIR,
    WORLD_CUP_BACKTEST_PREDICTIONS_PATH,
    WORLD_CUP_BACKTEST_SUMMARY_PATH,
)

PROJECT_ROOT = getattr(app_config, "PROJECT_ROOT", Path(__file__).resolve().parent)
DATA_DIR = getattr(app_config, "DATA_DIR", PROJECT_ROOT / "data")
RAW_DATA_DIR = getattr(app_config, "RAW_DATA_DIR", DATA_DIR / "raw")
REFERENCE_DATA_DIR = getattr(app_config, "REFERENCE_DATA_DIR", DATA_DIR / "reference")
HISTORICAL_DATA_DIR = getattr(app_config, "HISTORICAL_DATA_DIR", DATA_DIR / "historical")
MODELS_DIR = getattr(app_config, "MODELS_DIR", DATA_DIR / "models")
REPORTS_DIR = getattr(app_config, "REPORTS_DIR", DATA_DIR / "reports")
FULL_BACKTEST_BY_FOLD_PATH = getattr(
    backtest_path_config,
    "FULL_BACKTEST_BY_FOLD_PATH",
    PROCESSED_DATA_DIR / "full_backtest_by_fold.csv",
)
FULL_BACKTEST_MARKET_COMPARISON_PATH = getattr(
    backtest_path_config,
    "FULL_BACKTEST_MARKET_COMPARISON_PATH",
    PROCESSED_DATA_DIR / "full_backtest_market_comparison.csv",
)
FULL_BACKTEST_SUMMARY_PATH = getattr(
    backtest_path_config,
    "FULL_BACKTEST_SUMMARY_PATH",
    PROCESSED_DATA_DIR / "full_backtest_summary.csv",
)

BANKROLL_STATE_PATH = getattr(app_config, "BANKROLL_STATE_PATH", DATA_DIR / "bankroll_state.json")
DATA_MODE = getattr(app_config, "DATA_MODE", "official")
DEFAULT_ENSEMBLE_W_MARKET = getattr(app_config, "DEFAULT_ENSEMBLE_W_MARKET", 0.8)
DEFAULT_PROFILE_NAME = getattr(app_config, "DEFAULT_PROFILE_NAME", "Standard")
DEFAULT_PROBABILITY_SOURCE = getattr(app_config, "DEFAULT_PROBABILITY_SOURCE", "best_validated")
FIFA_RANKINGS_PATH = getattr(
    app_config,
    "FIFA_RANKINGS_PATH",
    REFERENCE_DATA_DIR / "fifa_rankings.csv",
)
FIFA_FEATURE_COVERAGE_PATH = getattr(
    app_config,
    "FIFA_FEATURE_COVERAGE_PATH",
    PROCESSED_DATA_DIR / "fifa_feature_coverage.csv",
)
FIFA_RANKING_FEATURE_REPORT_PATH = getattr(
    app_config,
    "FIFA_RANKING_FEATURE_REPORT_PATH",
    REPORTS_DIR / "fifa_ranking_feature_report.md",
)
HISTORICAL_RESULTS_PATH = getattr(app_config, "HISTORICAL_RESULTS_PATH", HISTORICAL_DATA_DIR / "international_results.csv")
LIVE_PREDICTIONS_PATH = getattr(app_config, "LIVE_PREDICTIONS_PATH", PROCESSED_DATA_DIR / "live_predictions.csv")
LIVE_PREDICTIONS_WITH_MODEL_PATH = getattr(
    app_config,
    "LIVE_PREDICTIONS_WITH_MODEL_PATH",
    PROCESSED_DATA_DIR / "live_predictions_with_model.csv",
)
MANUAL_ODDS_PATH = getattr(app_config, "MANUAL_ODDS_PATH", REFERENCE_DATA_DIR / "manual_odds.csv")
MATCH_RESULTS_PATH = getattr(app_config, "MATCH_RESULTS_PATH", REFERENCE_DATA_DIR / "match_results.csv")
MATCH_RESULTS_UPDATES_PATH = getattr(app_config, "MATCH_RESULTS_UPDATES_PATH", REFERENCE_DATA_DIR / "match_results_updates.csv")
MODEL_PATH = getattr(app_config, "MODEL_PATH", MODELS_DIR / "model.pkl")
MODEL_METADATA_PATH = getattr(app_config, "MODEL_METADATA_PATH", MODELS_DIR / "model_metadata.json")
MODEL_VARIANT_COMPARISON_PATH = getattr(
    app_config,
    "MODEL_VARIANT_COMPARISON_PATH",
    PROCESSED_DATA_DIR / "model_variant_comparison.csv",
)
MODEL_PREDICTIONS_PATH = getattr(app_config, "MODEL_PREDICTIONS_PATH", PROCESSED_DATA_DIR / "model_predictions.csv")
MODEL_SOURCE = getattr(app_config, "MODEL_SOURCE", "historical_model_if_available")
ODDS_API_MARKETS = getattr(app_config, "ODDS_API_MARKETS", "h2h")
ODDS_API_REGIONS = getattr(app_config, "ODDS_API_REGIONS", "eu")
ODDS_API_SPORT_KEY = getattr(app_config, "ODDS_API_SPORT_KEY", "soccer_fifa_world_cup")
ODDS_SNAPSHOT_PATH = getattr(app_config, "ODDS_SNAPSHOT_PATH", RAW_DATA_DIR / "odds_snapshots.csv")
PREFERRED_BOOKMAKER = getattr(app_config, "PREFERRED_BOOKMAKER", "Danske Spil")
PREFERRED_BOOKMAKER_NAMES = getattr(
    app_config,
    "PREFERRED_BOOKMAKER_NAMES",
    ["Danske Spil", "DanskeSpil", "Danske Spil A/S", "danske_spil"],
)
PROCESSED_ODDS_PATH = getattr(app_config, "PROCESSED_ODDS_PATH", PROCESSED_DATA_DIR / "latest_odds.csv")
REFERENCE_FIXTURES_PATH = getattr(app_config, "REFERENCE_FIXTURES_PATH", REFERENCE_DATA_DIR / "worldcup_2026_fixtures.csv")
SAMPLE_PREDICTIONS_PATH = getattr(app_config, "SAMPLE_PREDICTIONS_PATH", DATA_DIR / "sample_predictions.csv")
TRAINING_DATASET_PATH = getattr(app_config, "TRAINING_DATASET_PATH", PROCESSED_DATA_DIR / "training_dataset.csv")
STAKING_PROFILES = getattr(
    app_config,
    "STAKING_PROFILES",
    {
        "Standard": {
            "fractional_kelly_multiplier": 0.25,
            "max_stake_pct_of_bankroll": 0.025,
            "min_edge_threshold": 0.025,
            "min_stake_pct_threshold": 0.0025,
        }
    },
)
get_staking_profile = getattr(app_config, "get_staking_profile", lambda profile_name=DEFAULT_PROFILE_NAME: STAKING_PROFILES.get(profile_name, STAKING_PROFILES["Standard"]).copy())
get_secret_or_env = getattr(app_config, "get_secret_or_env", lambda name, default=None: os.environ.get(name, default))
validate_staking_profile = getattr(app_config, "validate_staking_profile", lambda profile: [])
from data_loader import (
    ensure_runtime_data_files,
    get_data_freshness,
    load_odds_snapshot,
    load_predictions,
    load_predictions_by_mode,
    validate_predictions,
)
from draw_features import add_draw_context_features
from draw_hypothesis import run_draw_hypothesis_analysis
from ensemble import apply_ensemble_to_upcoming_matches
from ensemble_backtest import run_ensemble_backtest_from_saved_predictions, select_best_probability_source
from fifa_rankings import load_fifa_rankings
from fixture_data import fixture_provenance, load_fixture_dataset, validate_fixture_dataset
from features import build_training_dataset
from historical_data import load_historical_results, standardize_historical_results, validate_historical_results
from kelly import calculate_final_stake_fraction, calculate_suggested_stake
from live_data_pipeline import live_odds_refresh_needed, refresh_live_odds_and_predictions
from match_results import add_match_results, split_active_and_archived_matches
from match_result_refresh import refresh_match_results
from model_readiness import is_production_performance_available
from model_registry import get_active_model_status, get_latest_backtest_status, get_latest_draw_context_status, get_model_readiness
from model_performance_summary import (
    build_model_quality_summary,
    display_metric_value,
    load_model_performance_summary,
    metric_interpretation,
    metric_status,
    metric_tooltip,
    validation_checklist,
)
from odds_provider import get_odds_source_status
from odds_utils import calculate_edge, decimal_odds_to_implied_probability, remove_overround_proportional
from probability_sources import (
    PROBABILITY_SOURCE_LABELS,
    apply_probability_source,
    load_active_probability_source,
    save_active_probability_source,
)
from predict_model import apply_stored_model_predictions, predict_upcoming_matches, prediction_file_uses_market_as_model
from recommendations import add_recommendations
from time_utils import add_danish_kickoff_column, format_danish_kickoff
from walk_forward_backtest import run_full_walk_forward_backtest
from tooltip_definitions import TOOLTIPS
from train_model import train_historical_model
from ui_navigation import PAGES, apply_navigation_css, render_sidebar_navigation, render_sidebar_status_card
from ui_theme import apply_custom_theme


st.set_page_config(
    page_title="VM 2026 Prediction & Kelly",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_custom_theme()
apply_navigation_css()
ensure_runtime_data_files()


def init_session_state() -> None:
    if "page" not in st.session_state:
        st.session_state.page = "Match Overview"
    if "current_page" not in st.session_state:
        st.session_state.current_page = st.session_state.page
    if "kelly_profile_name" not in st.session_state:
        st.session_state.kelly_profile_name = DEFAULT_PROFILE_NAME
    if "staking_profile" not in st.session_state:
        st.session_state.staking_profile = get_staking_profile(DEFAULT_PROFILE_NAME)
    if st.session_state.kelly_profile_name not in STAKING_PROFILES:
        st.session_state.kelly_profile_name = DEFAULT_PROFILE_NAME
        st.session_state.staking_profile = get_staking_profile(DEFAULT_PROFILE_NAME)
    if "preferred_bookmaker" not in st.session_state:
        st.session_state.preferred_bookmaker = PREFERRED_BOOKMAKER
    if "preferred_bookmaker_mode" not in st.session_state:
        st.session_state.preferred_bookmaker_mode = "danske_spil"
    if "selected_match_id" not in st.session_state:
        st.session_state.selected_match_id = None
    if "data_mode" not in st.session_state:
        st.session_state.data_mode = DATA_MODE
    if "active_data_mode" not in st.session_state:
        st.session_state.active_data_mode = DATA_MODE
    if "model_source" not in st.session_state:
        st.session_state.model_source = MODEL_SOURCE
    if "active_model_source" not in st.session_state:
        st.session_state.active_model_source = "market_only"
    if "use_draw_context_features" not in st.session_state:
        st.session_state.use_draw_context_features = False
    if "probability_source" not in st.session_state:
        st.session_state.probability_source = DEFAULT_PROBABILITY_SOURCE
    if "ensemble_w_market" not in st.session_state:
        st.session_state.ensemble_w_market = DEFAULT_ENSEMBLE_W_MARKET
    if "ensemble_model_variant" not in st.session_state:
        st.session_state.ensemble_model_variant = "historical_model"
    if "bet_slip" not in st.session_state:
        st.session_state.bet_slip = []


def current_profile() -> dict:
    return st.session_state.staking_profile.copy()


def _empty_historical_results() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "result",
            "tournament",
            "neutral",
        ]
    )


def rebuild_model_from_historical_data(include_draw_context_features: bool = False) -> str:
    if not HISTORICAL_RESULTS_PATH.exists():
        raise FileNotFoundError("Historical training data is missing.")
    raw = load_historical_results(HISTORICAL_RESULTS_PATH)
    hist_warnings, hist_errors = validate_historical_results(raw)
    if hist_errors:
        raise ValueError("; ".join(hist_errors))
    standardized = standardize_historical_results(raw)
    training_df = build_training_dataset(
        standardized,
        include_draw_context_features=include_draw_context_features,
    )
    metadata = train_historical_model(
        training_df,
        include_draw_context_features=include_draw_context_features,
        allow_demo_model=False,
        training_data_source="historical_international_results",
    )
    warning_text = f" Warnings: {'; '.join(hist_warnings)}" if hist_warnings else ""
    return (
        "Runtime model rebuilt from historical data. "
        f"Training rows: {metadata['training_rows']}; test rows: {metadata['test_rows']}."
        f"{warning_text}"
    )


def _path_signature(path: Path) -> tuple[str, bool, float, int]:
    path = Path(path)
    if not path.exists():
        return (str(path), False, 0.0, 0)
    stat = path.stat()
    return (str(path), True, stat.st_mtime, stat.st_size)


def _prediction_prepare_signature() -> tuple:
    return (
        st.session_state.data_mode,
        st.session_state.model_source,
        _path_signature(MODEL_PATH),
        _path_signature(MODEL_METADATA_PATH),
        _path_signature(MODEL_PREDICTIONS_PATH),
        _path_signature(LIVE_PREDICTIONS_PATH),
        _path_signature(LIVE_PREDICTIONS_WITH_MODEL_PATH),
        _path_signature(REFERENCE_FIXTURES_PATH),
        _path_signature(SAMPLE_PREDICTIONS_PATH),
    )


def _prediction_data_signature() -> tuple:
    return (
        _path_signature(MODEL_PATH),
        _path_signature(MODEL_METADATA_PATH),
        _path_signature(MODEL_PREDICTIONS_PATH),
        _path_signature(LIVE_PREDICTIONS_PATH),
        _path_signature(LIVE_PREDICTIONS_WITH_MODEL_PATH),
        _path_signature(ENSEMBLE_PREDICTIONS_PATH),
        _path_signature(REFERENCE_FIXTURES_PATH),
        _path_signature(SAMPLE_PREDICTIONS_PATH),
        _path_signature(MATCH_RESULTS_PATH),
        _path_signature(MATCH_RESULTS_UPDATES_PATH),
        _path_signature(BANKROLL_STATE_PATH),
    )


def prepare_best_available_predictions() -> list[str]:
    signature = _prediction_prepare_signature()
    if st.session_state.get("_prediction_prepare_signature") == signature:
        return []
    messages = []
    model_status = get_active_model_status()
    readiness = get_model_readiness(predictions_exist=MODEL_PREDICTIONS_PATH.exists() or LIVE_PREDICTIONS_WITH_MODEL_PATH.exists())
    if not readiness["is_usable_as_best_available"] and st.session_state.data_mode != "sample":
        st.session_state.model_source = "market_only"
        st.session_state["_prediction_prepare_signature"] = signature
        messages.append(readiness["normal_user_message"])
        return messages
    if not model_status["artifacts_ready"]:
        try:
            messages.append(
                rebuild_model_from_historical_data(
                    include_draw_context_features=bool(model_status.get("include_draw_context_features", False))
                )
            )
            model_status = get_active_model_status()
        except Exception as exc:
            st.session_state.model_source = "market_only"
            st.session_state["_prediction_prepare_signature"] = signature
            messages.append(
                "Predictions are currently based on market odds because the pre-trained model is unavailable "
                f"and could not be rebuilt from historical data. Details: {exc}"
            )
            return messages

    try:
        base_df, base_warnings, actual_mode = load_predictions_by_mode(
            st.session_state.data_mode,
            model_source="market_only",
        )
        messages.extend(base_warnings)
        output_path = LIVE_PREDICTIONS_WITH_MODEL_PATH if actual_mode == "live" else MODEL_PREDICTIONS_PATH
        should_generate = not output_path.exists() or output_path.stat().st_size == 0
        if not should_generate:
            if actual_mode == "live":
                source_path = LIVE_PREDICTIONS_PATH
            elif actual_mode == "sample":
                source_path = SAMPLE_PREDICTIONS_PATH
            else:
                source_path = REFERENCE_FIXTURES_PATH
            should_generate = output_path.stat().st_mtime < max(
                source_path.stat().st_mtime if source_path.exists() else 0,
                MODEL_PATH.stat().st_mtime,
            )
        if (
            not should_generate
            and readiness["is_usable_as_best_available"]
            and prediction_file_uses_market_as_model(output_path)
        ):
            should_generate = True
            messages.append("Existing model prediction file used market probabilities as fallback. Regenerating ML predictions.")
        if should_generate:
            if actual_mode == "live" and MODEL_PREDICTIONS_PATH.exists():
                try:
                    _, model_warnings = apply_stored_model_predictions(
                        base_df,
                        MODEL_PREDICTIONS_PATH,
                        output_path,
                    )
                    messages.extend(model_warnings)
                    messages.append("Model loaded. Stored ML predictions were matched to refreshed odds.")
                    st.session_state.model_source = "historical_model_if_available"
                    return messages
                except Exception as exc:
                    messages.append(f"Stored ML predictions could not be reused. Regenerating ML predictions. Details: {exc}")
            if HISTORICAL_RESULTS_PATH.exists():
                raw = load_historical_results(HISTORICAL_RESULTS_PATH)
                standardized = standardize_historical_results(raw)
            else:
                standardized = _empty_historical_results()
            try:
                _, model_warnings = predict_upcoming_matches(
                    base_df,
                    standardized,
                    output_path=output_path,
                    include_draw_context_features=bool(model_status.get("include_draw_context_features", False)),
                )
            except Exception:
                messages.append(
                    rebuild_model_from_historical_data(
                        include_draw_context_features=bool(model_status.get("include_draw_context_features", False))
                    )
                )
                _, model_warnings = predict_upcoming_matches(
                    base_df,
                    standardized,
                    output_path=output_path,
                    include_draw_context_features=bool(model_status.get("include_draw_context_features", False)),
                )
            messages.extend(model_warnings)
            messages.append("Model loaded. Predictions are being generated for upcoming matches.")
        else:
            messages.append("Pre-trained model loaded.")
        st.session_state.model_source = "historical_model_if_available"
    except Exception as exc:
        st.session_state.model_source = "market_only"
        messages.append(f"Pre-trained model could not be applied. Using market-implied probabilities as fallback. Details: {exc}")
    st.session_state["_prediction_prepare_signature"] = _prediction_prepare_signature()
    return messages


def auto_refresh_live_odds_on_reload() -> list[str]:
    if st.session_state.data_mode == "sample":
        return []
    status = get_odds_source_status()
    auto_api_refresh = str(get_secret_or_env("AUTO_REFRESH_API_ODDS_ON_RELOAD", "false")).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
    has_refresh_source = (
        status.get("has_api_key")
        or status.get("manual_odds_valid")
        or status.get("cached_odds_exists")
    )
    if not has_refresh_source:
        return []
    if not live_odds_refresh_needed(force_refresh=False):
        if st.session_state.data_mode == "official" and LIVE_PREDICTIONS_PATH.exists():
            st.session_state.data_mode = "live"
        return []
    if status.get("has_api_key") and not auto_api_refresh and not status.get("manual_odds_valid"):
        if st.session_state.data_mode == "official" and LIVE_PREDICTIONS_PATH.exists():
            st.session_state.data_mode = "live"
        return []

    try:
        result = refresh_live_odds_and_predictions(
            force_refresh=False,
            allow_api_fetch=auto_api_refresh,
        )
    except Exception as exc:
        return [f"Automatic odds refresh failed. Using last available data. Details: {exc}"]

    if result.get("matches_with_odds", 0) > 0:
        st.session_state.data_mode = "live"
        return [
            "Odds refreshed automatically. "
            f"{result['matches_with_odds']} / {result['matches_total']} matches have odds."
        ]
    if result.get("warnings"):
        return ["Automatic odds refresh ran, but no match odds were available."]
    return []


@st.cache_data(show_spinner=False, ttl=300)
def _load_enriched_predictions_cached(
    data_mode: str,
    model_source: str,
    probability_source: str,
    current_bankroll: float,
    staking_profile_items: tuple,
    preferred_bookmaker_mode: str,
    data_signature: tuple,
) -> tuple[pd.DataFrame, list[str], list[str], str, str]:
    try:
        predictions, mode_warnings, actual_mode = load_predictions_by_mode(
            data_mode,
            model_source=model_source,
        )
        predictions = add_danish_kickoff_column(predictions)
        active_model_source = (
            "market_only"
            if model_source == "market_only"
            or any("market probabilities" in warning.lower() for warning in mode_warnings)
            else "historical_model"
        )
    except FileNotFoundError as exc:
        return pd.DataFrame(), [], [str(exc)], data_mode, model_source
    except pd.errors.EmptyDataError:
        return pd.DataFrame(), [], ["Sample predictions file is empty or malformed."], data_mode, model_source

    warnings, errors = validate_predictions(predictions)
    warnings = mode_warnings + warnings
    if errors:
        return predictions, warnings, errors, actual_mode, active_model_source
    if ENSEMBLE_PREDICTIONS_PATH.exists() and ENSEMBLE_PREDICTIONS_PATH.stat().st_size > 0:
        try:
            ensemble_df = pd.read_csv(ENSEMBLE_PREDICTIONS_PATH)
            ensemble_columns = [
                "match_id",
                "ensemble_home_prob",
                "ensemble_draw_prob",
                "ensemble_away_prob",
                "ensemble_w_market",
                "ensemble_w_model",
            ]
            if set(ensemble_columns).issubset(ensemble_df.columns):
                predictions = predictions.drop(
                    columns=[column for column in ensemble_columns[1:] if column in predictions.columns],
                    errors="ignore",
                ).merge(ensemble_df[ensemble_columns], on="match_id", how="left")
        except Exception:
            warnings.append("Could not load saved ensemble predictions.")
    try:
        predictions = apply_probability_source(predictions, probability_source)
        warnings.extend(predictions.attrs.get("warnings", []))
    except ValueError as exc:
        return predictions, warnings, [str(exc)], actual_mode, active_model_source
    predictions = add_match_results(predictions)
    enriched = add_recommendations(
        predictions,
        current_bankroll,
        dict(staking_profile_items),
        preferred_bookmaker_mode=preferred_bookmaker_mode,
    )
    return enriched.rename(columns={"status": "recommendation_status"}), warnings, errors, actual_mode, active_model_source


def load_enriched_predictions() -> tuple[pd.DataFrame, list[str], list[str]]:
    startup_messages = auto_refresh_live_odds_on_reload()
    startup_messages.extend(prepare_best_available_predictions())
    bankroll = load_bankroll_state()["current_bankroll"]
    df, warnings, errors, active_data_mode, active_model_source = _load_enriched_predictions_cached(
        st.session_state.data_mode,
        st.session_state.model_source,
        st.session_state.probability_source,
        float(bankroll),
        tuple(sorted(current_profile().items())),
        st.session_state.preferred_bookmaker_mode,
        _prediction_data_signature(),
    )
    st.session_state.active_data_mode = active_data_mode
    st.session_state.active_model_source = active_model_source
    return df, startup_messages + warnings, errors


def probability_for_outcome(row, outcome: str) -> float:
    active_column = f"active_{outcome.lower()}_prob"
    if active_column in row.index:
        return float(row[active_column])
    return float(row[f"model_{outcome.lower()}_prob"])


def match_label(row) -> str:
    return f"{row['home_team']} vs {row['away_team']}"


def recommendation_outcome_label(row, market: str) -> str:
    outcome = row.get(f"recommended_outcome_{market}", "No bet")
    return outcome_label(outcome, row["home_team"], row["away_team"])


def recommendation_summary(row, market: str) -> str:
    outcome = row.get(f"recommended_outcome_{market}", "No bet")
    if outcome == "No bet" or pd.isna(outcome):
        return "No bet"
    bookmaker = row.get("recommended_bookmaker_best") if market == "best" else None
    bookmaker_text = f" · {bookmaker}" if bookmaker else ""
    return (
        f"{recommendation_outcome_label(row, market)} @ {format_odds(row[f'recommended_odds_{market}'])}"
        f"{bookmaker_text} · Stake {format_dkk(row[f'recommended_stake_{market}'])}"
    )


def preferred_bookmaker_label() -> str:
    return "Best market" if st.session_state.get("preferred_bookmaker_mode") == "best_market" else "Danske Spil"


def primary_decision_title() -> str:
    return "Best market decision" if st.session_state.get("preferred_bookmaker_mode") == "best_market" else "Danske Spil decision"


def primary_decision_text(row) -> str:
    mode = st.session_state.get("preferred_bookmaker_mode", "danske_spil")
    if mode == "best_market":
        if row.get("primary_status") == "play":
            return f"Play {recommendation_outcome_label(row, 'best')} @ {format_odds(row.get('recommended_odds_best'))}"
        if row.get("primary_status") == "odds_missing":
            return "Best market odds missing"
        return "No bet at best market"
    if row.get("primary_status") == "play":
        return f"Play {recommendation_outcome_label(row, 'ds')} @ {format_odds(row.get('recommended_odds_ds'))}"
    if row.get("primary_status") == "odds_missing":
        return "Odds missing at Danske Spil"
    return "No bet at Danske Spil"


def primary_stake_text(row) -> str:
    if row.get("primary_status") != "play":
        return ""
    return f"Stake {format_dkk(row.get('primary_stake', 0))}"


def best_market_note(row) -> str:
    if row.get("recommended_outcome_best") == "No bet":
        if all(pd.isna(row.get(f"best_{outcome}_odds")) or float(row.get(f"best_{outcome}_odds") or 0) <= 1 for outcome in ["home", "draw", "away"]):
            return "Best market odds missing"
        return "No better market value found"
    summary = recommendation_summary(row, "best")
    if st.session_state.get("preferred_bookmaker_mode", "danske_spil") == "danske_spil" and row.get("recommended_outcome_ds") == "No bet":
        return f"{summary}. Value exists elsewhere; not playable at your selected bookmaker."
    if row.get("recommended_outcome_ds") != "No bet" and row.get("recommended_odds_best") and row.get("recommended_odds_ds"):
        if float(row.get("recommended_odds_best")) > float(row.get("recommended_odds_ds")) + 0.01:
            return f"{summary}. Better odds available elsewhere."
        return f"{summary}. Same or similar odds."
    return summary


def no_bet_reason(row, market: str) -> str:
    profile = current_profile()
    prefix = "ds" if market == "ds" else "best"
    outcomes = ["home", "draw", "away"]
    odds_values = [row.get(f"{prefix}_{outcome}_odds") for outcome in outcomes]
    if all(pd.isna(odds) or float(odds) <= 1 for odds in odds_values):
        if market == "ds":
            return "Danske Spil odds missing."
        if market == "best":
            return "Best market odds missing."
        return "Odds missing at selected bookmaker."

    candidates = []
    for outcome in outcomes:
        odds = row.get(f"{prefix}_{outcome}_odds")
        if pd.isna(odds) or float(odds) <= 1:
            continue
        probability = probability_for_outcome(row, outcome)
        edge = calculate_edge(probability, float(odds))
        kelly_values = calculate_final_stake_fraction(
            probability,
            float(odds),
            profile["fractional_kelly_multiplier"],
            profile["max_stake_pct_of_bankroll"],
        )
        candidates.append(
            {
                "edge": edge,
                "final_stake_fraction": kelly_values["final_stake_fraction"],
            }
        )
    if not candidates:
        return "Probability source unavailable or odds missing."
    best_edge = max(candidate["edge"] for candidate in candidates)
    best_stake_fraction = max(candidate["final_stake_fraction"] for candidate in candidates)
    if best_edge < profile["min_edge_threshold"]:
        return (
            f"Edge below minimum threshold. Best edge is {format_percentage(best_edge)}; "
            f"minimum is {format_percentage(profile['min_edge_threshold'])}."
        )
    if best_stake_fraction < profile["min_stake_pct_threshold"]:
        return (
            f"Kelly stake below minimum. Best stake fraction is {format_percentage(best_stake_fraction)}; "
            f"minimum is {format_percentage(profile['min_stake_pct_threshold'])}."
        )
    return "No outcome passes both edge and Kelly thresholds."


def row_has_priced_odds(row) -> bool:
    odds_sets = [
        ["best_home_odds", "best_draw_odds", "best_away_odds"],
        ["ds_home_odds", "ds_draw_odds", "ds_away_odds"],
    ]
    for columns in odds_sets:
        values = [pd.to_numeric(row.get(column), errors="coerce") for column in columns]
        if all(not pd.isna(value) and float(value) > 1.0 for value in values):
            return True
    return False


def _probability_triplet_is_uniform(row, prefix: str) -> bool:
    values = [
        pd.to_numeric(row.get(f"{prefix}_home_prob"), errors="coerce"),
        pd.to_numeric(row.get(f"{prefix}_draw_prob"), errors="coerce"),
        pd.to_numeric(row.get(f"{prefix}_away_prob"), errors="coerce"),
    ]
    if any(pd.isna(value) for value in values):
        return False
    return all(abs(float(value) - (1 / 3)) < 0.0001 for value in values)


def row_has_displayable_probabilities(row) -> bool:
    if row_has_priced_odds(row):
        return True
    return not (
        _probability_triplet_is_uniform(row, "market")
        and _probability_triplet_is_uniform(row, "model")
        and _probability_triplet_is_uniform(row, "active")
    )


def probability_triplets_are_identical(row, prefixes: tuple[str, ...] = ("market", "model", "active")) -> bool:
    triplets = []
    for prefix in prefixes:
        values = [
            pd.to_numeric(row.get(f"{prefix}_home_prob"), errors="coerce"),
            pd.to_numeric(row.get(f"{prefix}_draw_prob"), errors="coerce"),
            pd.to_numeric(row.get(f"{prefix}_away_prob"), errors="coerce"),
        ]
        if any(pd.isna(value) for value in values):
            return False
        triplets.append([float(value) for value in values])
    first = triplets[0]
    return all(all(abs(a - b) < 0.0001 for a, b in zip(first, triplet)) for triplet in triplets[1:])


def format_probability_for_row(row, column: str) -> str:
    if not row_has_displayable_probabilities(row):
        return "-"
    return format_probability(row.get(column))


def row_has_distinct_ml_model(row) -> bool:
    source = str(row.get("model_probability_source", "") or "").lower()
    if source == "historical_model":
        return True
    if source == "market_fallback":
        return False
    model_values = [
        pd.to_numeric(row.get("model_home_prob"), errors="coerce"),
        pd.to_numeric(row.get("model_draw_prob"), errors="coerce"),
        pd.to_numeric(row.get("model_away_prob"), errors="coerce"),
    ]
    market_values = [
        pd.to_numeric(row.get("market_home_prob"), errors="coerce"),
        pd.to_numeric(row.get("market_draw_prob"), errors="coerce"),
        pd.to_numeric(row.get("market_away_prob"), errors="coerce"),
    ]
    if any(pd.isna(value) for value in model_values):
        return False
    if any(pd.isna(value) for value in market_values):
        return not _probability_triplet_is_uniform(row, "model")
    return any(abs(float(model) - float(market)) >= 0.0001 for model, market in zip(model_values, market_values))


def probability_source_label(row) -> str:
    if not row_has_displayable_probabilities(row):
        return "Afventer odds/model"
    return "Best available"


def technical_probability_source_label(row) -> str:
    source = row.get("active_probability_source", st.session_state.probability_source)
    label = PROBABILITY_SOURCE_LABELS.get(source, source)
    if source == "ensemble" and not pd.isna(row.get("ensemble_w_market")):
        return f"{label} {float(row.get('ensemble_w_market')):.0%}/{float(row.get('ensemble_w_model')):.0%}"
    return label


def match_prediction_summary(row) -> dict:
    if not row_has_displayable_probabilities(row):
        return {
            "favorite": "Afventer odds",
            "line": "Sandsynligheder mangler",
        }
    probabilities = {
        row["home_team"]: probability_for_outcome(row, "home"),
        "Draw": probability_for_outcome(row, "draw"),
        row["away_team"]: probability_for_outcome(row, "away"),
    }
    favorite = max(probabilities, key=probabilities.get)
    return {
        "favorite": favorite,
        "line": (
            f"{row['home_team']} {format_probability(probabilities[row['home_team']])} · "
            f"Draw {format_probability(probabilities['Draw'])} · "
            f"{row['away_team']} {format_probability(probabilities[row['away_team']])}"
        ),
    }


def probability_line_for_source(row, prefix: str) -> str:
    if prefix == "model" and not row_has_distinct_ml_model(row):
        return "Afventer ML model"
    if prefix == "market" and not row_has_priced_odds(row):
        return "Afventer odds"
    if prefix in {"market", "model"} and _probability_triplet_is_uniform(row, prefix) and not row_has_priced_odds(row):
        return "Afventer data"
    values = [
        pd.to_numeric(row.get(f"{prefix}_home_prob"), errors="coerce"),
        pd.to_numeric(row.get(f"{prefix}_draw_prob"), errors="coerce"),
        pd.to_numeric(row.get(f"{prefix}_away_prob"), errors="coerce"),
    ]
    if any(pd.isna(value) for value in values):
        return "Afventer data"
    return (
        f"{row['home_team']} {format_probability(values[0])} · "
        f"Draw {format_probability(values[1])} · "
        f"{row['away_team']} {format_probability(values[2])}"
    )


def danske_spil_probability_values(row) -> Optional[list[float]]:
    odds_values = [
        pd.to_numeric(row.get("ds_home_odds"), errors="coerce"),
        pd.to_numeric(row.get("ds_draw_odds"), errors="coerce"),
        pd.to_numeric(row.get("ds_away_odds"), errors="coerce"),
    ]
    if any(pd.isna(odds) or float(odds) <= 1.0 for odds in odds_values):
        return None
    implied = [decimal_odds_to_implied_probability(float(odds)) for odds in odds_values]
    return remove_overround_proportional(implied)


def danske_spil_probability_line(row) -> str:
    probabilities = danske_spil_probability_values(row)
    if probabilities is None:
        return "Afventer Danske Spil odds"
    return (
        f"{row['home_team']} {format_probability(probabilities[0])} · "
        f"Draw {format_probability(probabilities[1])} · "
        f"{row['away_team']} {format_probability(probabilities[2])}"
    )


def model_vs_danske_spil_difference_note(row, threshold: float = 0.05) -> str:
    ds_probabilities = danske_spil_probability_values(row)
    if ds_probabilities is None or not row_has_distinct_ml_model(row):
        return ""
    model_probabilities = [
        pd.to_numeric(row.get("model_home_prob"), errors="coerce"),
        pd.to_numeric(row.get("model_draw_prob"), errors="coerce"),
        pd.to_numeric(row.get("model_away_prob"), errors="coerce"),
    ]
    if any(pd.isna(probability) for probability in model_probabilities):
        return ""
    labels = [row["home_team"], "Draw", row["away_team"]]
    differences = [
        (label, float(model) - float(ds))
        for label, model, ds in zip(labels, model_probabilities, ds_probabilities)
    ]
    label, difference = max(differences, key=lambda item: abs(item[1]))
    if abs(difference) < threshold:
        return ""
    direction = "higher" if difference > 0 else "lower"
    return f"Differs from Danske Spil: {label} is {format_percentage(abs(difference))} {direction} in ML."


def recommended_bet_is_danske_spil_favorite(row) -> bool:
    if row.get("primary_status") != "play":
        return False
    odds_by_outcome = {
        "Home": pd.to_numeric(row.get("ds_home_odds"), errors="coerce"),
        "Draw": pd.to_numeric(row.get("ds_draw_odds"), errors="coerce"),
        "Away": pd.to_numeric(row.get("ds_away_odds"), errors="coerce"),
    }
    valid_odds = {
        outcome: float(odds)
        for outcome, odds in odds_by_outcome.items()
        if not pd.isna(odds) and float(odds) > 1.0
    }
    if len(valid_odds) != 3:
        return False
    favorite_outcome = min(valid_odds, key=valid_odds.get)
    recommended = row.get("recommended_outcome_ds")
    if st.session_state.get("preferred_bookmaker_mode") == "best_market":
        recommended = row.get("recommended_outcome_best")
    return recommended == favorite_outcome


def favorite_for_source(row, prefix: str) -> str:
    line = probability_line_for_source(row, prefix)
    if line.startswith("Afventer"):
        return line
    probabilities = {
        row["home_team"]: float(row.get(f"{prefix}_home_prob")),
        "Draw": float(row.get(f"{prefix}_draw_prob")),
        row["away_team"]: float(row.get(f"{prefix}_away_prob")),
    }
    return max(probabilities, key=probabilities.get)


def betting_decision_summary(row) -> str:
    if row["recommendation_status"] == "No bet":
        return "No bet"
    if row["recommended_outcome_ds"] != "No bet":
        return f"{recommendation_outcome_label(row, 'ds')} · {format_dkk(row['recommended_stake_ds'])}"
    return f"{recommendation_outcome_label(row, 'best')} · {format_dkk(row['recommended_stake_best'])}"


def format_bet_log_table(df: pd.DataFrame) -> pd.DataFrame:
    display_columns = [
        "timestamp",
        "match",
        "kickoff_time_dk",
        "bookmaker",
        "outcome",
        "odds",
        "edge",
        "fractional_kelly",
        "stake_dkk",
        "result",
        "profit_loss_dkk",
        "settled",
    ]
    result = df[[column for column in display_columns if column in df.columns]].copy()
    rename = {
        "timestamp": "Logged",
        "match": "Match",
        "kickoff_time_dk": "Kickoff DK",
        "bookmaker": "Bookmaker",
        "outcome": "Outcome",
        "odds": "Odds",
        "edge": "Edge",
        "fractional_kelly": "Kelly",
        "stake_dkk": "Stake",
        "result": "Result",
        "profit_loss_dkk": "P/L",
        "settled": "Settled",
    }
    result = result.rename(columns=rename)
    for column in ["Odds"]:
        if column in result.columns:
            result[column] = result[column].map(format_odds)
    for column in ["Edge", "Kelly"]:
        if column in result.columns:
            result[column] = result[column].map(format_percentage)
    for column in ["Stake", "P/L"]:
        if column in result.columns:
            result[column] = result[column].map(format_dkk)
    return result


def app_health_rows(df: pd.DataFrame) -> pd.DataFrame:
    model_status = get_active_model_status()
    backtest_status = get_latest_backtest_status()
    odds_freshness = get_data_freshness(ODDS_SNAPSHOT_PATH)
    live_freshness = get_data_freshness(LIVE_PREDICTIONS_PATH)
    bet_log = load_bet_log()
    bankroll_loaded = True
    try:
        load_bankroll_state()
    except Exception:
        bankroll_loaded = False
    checks = [
        ("Data mode", st.session_state.active_data_mode.title(), fixture_provenance_text(st.session_state.active_data_mode, df)),
        ("Matches loaded", str(len(df)), "Upcoming matches available in the current app mode."),
        ("Active probability source", PROBABILITY_SOURCE_LABELS.get(st.session_state.probability_source, st.session_state.probability_source), "Used for edge and Kelly."),
        ("Model readiness", model_status["readiness_status"], "Only production-ready models can become Best available."),
        ("Ensemble available", "Yes" if ENSEMBLE_PREDICTIONS_PATH.exists() else "No", "Run or apply ensemble from Ensemble page."),
        ("Bankroll loaded", "Yes" if bankroll_loaded else "No", "Runtime bankroll JSON is readable."),
        ("Bet log loaded", "Yes", f"{len(bet_log)} bets logged."),
        ("Live predictions", f"{live_freshness['row_count']} rows", live_freshness["last_modified"] or "No live predictions file."),
        ("Latest odds update", odds_freshness["last_modified"] or "No odds snapshots yet", "Fetch odds in Settings."),
        ("Latest backtest update", backtest_status["last_modified"] or "No backtest yet", "Run backtest from Backtest & Metrics."),
    ]
    return pd.DataFrame(checks, columns=["Check", "Status", "What to do"])


def add_recommended_bet(row, market: str) -> None:
    if market == "ds":
        outcome = row["recommended_outcome_ds"]
        bookmaker = "Danske Spil"
    else:
        outcome = row["recommended_outcome_best"]
        bookmaker = row["recommended_bookmaker_best"]

    if outcome == "No bet" or pd.isna(outcome):
        st.warning("Der er ingen gyldig anbefaling at tilføje.")
        return

    outcome_key = {"Home": "home", "Draw": "draw", "Away": "away"}[outcome]
    try:
        bet = add_bet(
            match_id=row["match_id"],
            match=match_label(row),
            kickoff_time_dk=row.get("kickoff_time_dk", ""),
            bookmaker=bookmaker,
            outcome=outcome,
            odds=row[f"recommended_odds_{market}"],
            model_probability=probability_for_outcome(row, outcome_key),
            edge=row[f"recommended_edge_{market}"],
            full_kelly=row[f"recommended_full_kelly_{market}"],
            fractional_kelly=row[f"recommended_fractional_kelly_{market}"],
            stake_dkk=row[f"recommended_stake_{market}"],
        )
        st.success(
            f"Bet tilføjet til Bet Log. Bankroll ændres først, når bettet afregnes. Bet ID: {bet['bet_id']}"
        )
    except ValueError as exc:
        st.error(str(exc))


def recommended_market(row) -> str:
    if row.get("recommended_outcome_ds") != "No bet":
        return "ds"
    if row.get("recommended_outcome_best") != "No bet":
        return "best"
    return "none"


def bet_payload_from_recommendation(row, market: str) -> dict:
    if market == "ds":
        outcome = row["recommended_outcome_ds"]
        bookmaker = "Danske Spil"
    else:
        outcome = row["recommended_outcome_best"]
        bookmaker = row["recommended_bookmaker_best"]
    outcome_key = {"Home": "home", "Draw": "draw", "Away": "away"}[outcome]
    return {
        "match_id": row["match_id"],
        "match": match_label(row),
        "kickoff_time_dk": row.get("kickoff_time_dk", ""),
        "bookmaker": bookmaker,
        "outcome": outcome,
        "odds": row[f"recommended_odds_{market}"],
        "model_probability": probability_for_outcome(row, outcome_key),
        "edge": row[f"recommended_edge_{market}"],
        "full_kelly": row[f"recommended_full_kelly_{market}"],
        "fractional_kelly": row[f"recommended_fractional_kelly_{market}"],
        "stake_dkk": row[f"recommended_stake_{market}"],
        "market": market,
        "status": row["recommendation_status"],
    }


def add_recommendation_to_bet_slip(row, market: str) -> None:
    outcome = row.get(f"recommended_outcome_{market}")
    if outcome == "No bet" or pd.isna(outcome):
        st.warning("Der er ingen gyldig anbefaling at tilføje.")
        return
    if st.session_state.get("preferred_bookmaker_mode", "danske_spil") == "danske_spil" and market == "best":
        st.warning("This bet is not available at Danske Spil. It is added as a best-market reference bet.")
    payload = bet_payload_from_recommendation(row, market)
    slip_key = f"{payload['match_id']}|{payload['market']}|{payload['outcome']}|{payload['bookmaker']}"
    existing_keys = {
        f"{item['match_id']}|{item['market']}|{item['outcome']}|{item['bookmaker']}"
        for item in st.session_state.bet_slip
    }
    if slip_key not in existing_keys:
        st.session_state.bet_slip.append(payload)
    st.success("Tilføjet til bet slip. Bankroll ændres ikke.")


def commit_bet_slip_to_log() -> None:
    added = 0
    for item in st.session_state.bet_slip:
        try:
            add_bet(
                match_id=item["match_id"],
                match=item["match"],
                kickoff_time_dk=item.get("kickoff_time_dk", ""),
                bookmaker=item["bookmaker"],
                outcome=item["outcome"],
                odds=item["odds"],
                model_probability=item["model_probability"],
                edge=item["edge"],
                full_kelly=item["full_kelly"],
                fractional_kelly=item["fractional_kelly"],
                stake_dkk=item["stake_dkk"],
            )
            added += 1
        except ValueError as exc:
            st.error(str(exc))
    if added:
        st.session_state.bet_slip = []
        st.success(f"{added} bet(s) lagt i Bet Log. Bankroll ændres først ved settlement.")


def format_bet_slip_table(items: list[dict]) -> pd.DataFrame:
    if not items:
        return pd.DataFrame()
    df = pd.DataFrame(items)
    display = df[["match", "kickoff_time_dk", "bookmaker", "outcome", "odds", "edge", "fractional_kelly", "stake_dkk", "status"]].copy()
    display = display.rename(
        columns={
            "match": "Match",
            "kickoff_time_dk": "Kickoff",
            "bookmaker": "Bookmaker",
            "outcome": "Outcome",
            "odds": "Odds",
            "edge": "Edge",
            "fractional_kelly": "Kelly",
            "stake_dkk": "Stake",
            "status": "Status",
        }
    )
    display["Odds"] = display["Odds"].map(format_odds)
    display["Edge"] = display["Edge"].map(format_percentage)
    display["Kelly"] = display["Kelly"].map(format_percentage)
    display["Stake"] = display["Stake"].map(format_dkk)
    return display


def outcome_kelly_table(row) -> pd.DataFrame:
    rows = []
    profile = current_profile()
    bankroll = load_bankroll_state()["current_bankroll"]
    for outcome_key, outcome_name in [("home", "Home"), ("draw", "Draw"), ("away", "Away")]:
        model_probability = float(row.get(f"active_{outcome_key}_prob", row[f"model_{outcome_key}_prob"]))
        ds_odds = pd.to_numeric(row.get(f"ds_{outcome_key}_odds"), errors="coerce")
        best_odds = pd.to_numeric(row.get(f"best_{outcome_key}_odds"), errors="coerce")
        ds_odds_for_calc = float(ds_odds) if pd.notna(ds_odds) and ds_odds > 1 else 1.0
        best_odds_for_calc = float(best_odds) if pd.notna(best_odds) and best_odds > 1 else 1.0
        ds_kelly = (
            calculate_final_stake_fraction(
                model_probability,
                ds_odds_for_calc,
                profile["fractional_kelly_multiplier"],
                profile["max_stake_pct_of_bankroll"],
            )
            if ds_odds_for_calc > 1
            else {"full_kelly": 0, "fractional_kelly": 0, "final_stake_fraction": 0}
        )
        best_kelly = (
            calculate_final_stake_fraction(
                model_probability,
                best_odds_for_calc,
                profile["fractional_kelly_multiplier"],
                profile["max_stake_pct_of_bankroll"],
            )
            if best_odds_for_calc > 1
            else {"full_kelly": 0, "fractional_kelly": 0, "final_stake_fraction": 0}
        )
        rows.append(
            {
                "Outcome": outcome_name,
                "Model probability": model_probability,
                "DS odds": ds_odds if pd.notna(ds_odds) else None,
                "DS edge": calculate_edge(model_probability, ds_odds_for_calc) if ds_odds_for_calc > 1 else 0,
                "DS full Kelly": ds_kelly["full_kelly"],
                "DS fractional Kelly": ds_kelly["fractional_kelly"],
                "DS suggested stake": calculate_suggested_stake(bankroll, ds_kelly["final_stake_fraction"]),
                "Best odds": best_odds if pd.notna(best_odds) else None,
                "Best bookmaker": row[f"best_{outcome_key}_bookmaker"],
                "Best edge": calculate_edge(model_probability, best_odds_for_calc) if best_odds_for_calc > 1 else 0,
                "Best full Kelly": best_kelly["full_kelly"],
                "Best fractional Kelly": best_kelly["fractional_kelly"],
                "Best suggested stake": calculate_suggested_stake(
                    bankroll, best_kelly["final_stake_fraction"]
                ),
            }
        )
    return pd.DataFrame(rows)


def style_edge_table(df: pd.DataFrame):
    edge_columns = ["DS edge", "Best edge"]

    def color_edge(value):
        numeric_value = pd.to_numeric(value, errors="coerce")
        color = "green" if not pd.isna(numeric_value) and float(numeric_value) > 0 else "red"
        return f"color: {color}"

    return df.style.format(
        {
            "Model probability": format_probability,
            "DS odds": format_odds,
            "DS edge": format_percentage,
            "DS full Kelly": format_percentage,
            "DS fractional Kelly": format_percentage,
            "DS suggested stake": format_dkk,
            "Best odds": format_odds,
            "Best edge": format_percentage,
            "Best full Kelly": format_percentage,
            "Best fractional Kelly": format_percentage,
            "Best suggested stake": format_dkk,
        }
    ).map(color_edge, subset=edge_columns)


def format_overview_table(df: pd.DataFrame) -> pd.DataFrame:
    display_df = df.copy()
    display_df["match"] = display_df["home_team"] + " vs " + display_df["away_team"]
    display_df["DS"] = display_df.apply(lambda row: recommendation_summary(row, "ds"), axis=1)
    display_df["Best market"] = display_df.apply(lambda row: recommendation_summary(row, "best"), axis=1)
    display_df = display_df.rename(
        columns={
            "kickoff_time_dk": "Kickoff DK",
            "model_home_prob": "Model H",
            "model_draw_prob": "Model U",
            "model_away_prob": "Model A",
            "market_home_prob": "Market H",
            "market_draw_prob": "Market U",
            "market_away_prob": "Market A",
            "active_home_prob": "Active H",
            "active_draw_prob": "Active U",
            "active_away_prob": "Active A",
            "ds_home_odds": "DS H",
            "ds_draw_odds": "DS U",
            "ds_away_odds": "DS A",
            "best_home_odds": "Best H",
            "best_draw_odds": "Best U",
            "best_away_odds": "Best A",
            "recommended_outcome_ds": "DS rec",
            "recommended_outcome_best": "Best rec",
            "recommended_stake_ds": "Stake DS",
            "recommended_stake_best": "Stake Best",
            "recommendation_status": "Status",
            "draw_context_label": "Draw context",
        }
    )
    probability_display_columns = {
        "Active H": "active_home_prob",
        "Active U": "active_draw_prob",
        "Active A": "active_away_prob",
        "Model H": "model_home_prob",
        "Model U": "model_draw_prob",
        "Model A": "model_away_prob",
        "Market H": "market_home_prob",
        "Market U": "market_draw_prob",
        "Market A": "market_away_prob",
    }
    for display_column, source_column in probability_display_columns.items():
        if display_column in display_df.columns:
            display_df[display_column] = df.apply(
                lambda row: format_probability_for_row(row, source_column),
                axis=1,
            )
    for column in ["DS H", "DS U", "DS A", "Best H", "Best U", "Best A"]:
        display_df[column] = display_df[column].map(format_odds)
    for column in ["Stake DS", "Stake Best"]:
        display_df[column] = display_df[column].map(format_dkk)
    columns = [
        "Kickoff DK",
        "group",
        "matchday",
        "match",
        "Active H",
        "Active U",
        "Active A",
        "Model H",
        "Model U",
        "Model A",
        "Market H",
        "Market U",
        "Market A",
        "DS",
        "Best market",
        "Status",
        "Draw context",
    ]
    return display_df[[column for column in columns if column in display_df.columns]]


def show_sidebar() -> None:
    state = load_bankroll_state()
    net = state["current_bankroll"] - state["starting_bankroll"]
    ret = net / state["starting_bankroll"] if state["starting_bankroll"] else 0
    page_aliases = {
        "Overview": "Match Overview",
        "Bet Log": "My Bets",
        "Bankroll": "My Bets",
        "Analytics": "My Bets",
        "About": "Model Performance",
        "Model & Data": "Advanced / Admin",
        "Admin / Settings": "Advanced / Admin",
        "Backtest & Metrics": "Model Performance",
        "Draw Hypothesis": "Advanced / Admin",
        "Ensemble": "Advanced / Admin",
    }
    page_keys = [page["key"] for page in PAGES]
    allowed_pages = set(page_keys) | {"Match Detail", "Match Archive", "Model & Data"}
    requested_page = page_aliases.get(st.session_state.get("page", "Match Overview"), st.session_state.get("page", "Match Overview"))
    active_page = page_aliases.get(st.session_state.get("current_page", requested_page), st.session_state.get("current_page", requested_page))
    if requested_page != active_page and requested_page in allowed_pages:
        active_page = requested_page
    if active_page not in allowed_pages:
        active_page = "Match Overview"
    st.session_state.current_page = active_page
    st.session_state.page = active_page

    st.sidebar.markdown(
        """
        <div class="wc-sidebar-brand">
          <div class="wc-sidebar-brand-main">VM 2026</div>
          <div class="wc-sidebar-brand-sub">Prediction & Kelly</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected_page = render_sidebar_navigation(PAGES, active_page)
    if selected_page != active_page:
        st.session_state.current_page = selected_page
        st.session_state.page = selected_page
        st.rerun()

    if st.session_state.active_data_mode == "live":
        mode_path = LIVE_PREDICTIONS_PATH
    elif st.session_state.active_data_mode == "sample":
        mode_path = SAMPLE_PREDICTIONS_PATH
    else:
        mode_path = REFERENCE_FIXTURES_PATH
    freshness = get_data_freshness(mode_path)
    odds_status = get_odds_source_status()
    odds_label = {
        "api": "The Odds API",
        "manual": "Manual CSV",
        "cached": "Cached snapshot",
        "missing": "Missing",
    }.get(odds_status["active_odds_source"], odds_status["active_odds_source"].title())
    if freshness["last_modified"] and odds_status["active_odds_source"] != "missing":
        odds_label = f"{odds_label} · {freshness['last_modified']}"

    st.sidebar.markdown('<div class="wc-sidebar-status-title">Status</div>', unsafe_allow_html=True)
    render_sidebar_status_card("Bankroll", f"{format_dkk(state['current_bankroll'])} · {format_dkk(net)} / {format_percentage(ret)}")
    render_sidebar_status_card("Prediction", "Best available")
    render_sidebar_status_card("Odds", odds_label)
    render_sidebar_status_card(
        "Data",
        f"Live data · {freshness['row_count']} rows",
    )
    if st.session_state.active_data_mode == "live":
        render_sidebar_status_card("API key", "Configured" if get_secret_or_env("ODDS_API_KEY") else "Missing")


def show_validation_messages(warnings: list[str], errors: list[str]) -> None:
    unique_warnings = list(dict.fromkeys(str(warning) for warning in warnings))
    normal_pages = {
        "Match Overview",
        "Match Archive",
        "Betting Center",
        "My Bets",
        "Match Detail",
        "Model Performance",
        "Settings",
    }
    active_page = st.session_state.get("current_page", st.session_state.get("page", "Match Overview"))
    if unique_warnings and active_page in normal_pages:
        user_facing = [
            warning for warning in unique_warnings
            if "model unavailable" in warning.lower()
            or "pre-trained model unavailable" in warning.lower()
            or "pre-trained model is unavailable" in warning.lower()
            or "market-implied probabilities" in warning.lower()
            or "market probabilities" in warning.lower()
        ]
        for warning in user_facing[:1]:
            if "demo model" in warning.lower():
                st.info("Predictions are based on market odds because the available model is only a demo model.")
            else:
                st.info("Predictions are currently based on market odds because the pre-trained model is unavailable.")
    else:
        for warning in unique_warnings:
            text = str(warning)
            if "fallback" in text.lower() or "market probabilities" in text.lower() or "market-implied" in text.lower():
                st.info(text)
            else:
                st.warning(text)
    for error in errors:
        st.error(error)
    if errors:
        st.stop()


def fixture_provenance_text(mode: str, df: Optional[pd.DataFrame] = None) -> str:
    source_df = load_fixture_dataset() if mode != "sample" else (df if df is not None else pd.DataFrame())
    provenance = fixture_provenance(source_df, mode)
    return (
        f"Fixture source: {provenance['label']} | "
        f"{provenance['loaded']} / {provenance['expected']} matches | "
        f"checked: {provenance['last_checked']}"
    )


def data_freshness_path_for_mode(mode: str):
    if mode == "live":
        return LIVE_PREDICTIONS_PATH
    if mode == "sample":
        return SAMPLE_PREDICTIONS_PATH
    return REFERENCE_FIXTURES_PATH


def odds_availability_message(df: pd.DataFrame) -> Optional[str]:
    odds_columns = ["best_home_odds", "best_draw_odds", "best_away_odds"]
    has_best_odds = (
        all(column in df.columns for column in odds_columns)
        and df[odds_columns].apply(pd.to_numeric, errors="coerce").notna().any().any()
    )
    if has_best_odds:
        return None
    status = get_odds_source_status()
    if status["active_odds_source"] == "missing":
        return status["warning"]
    if status["active_odds_source"] == "api":
        return "The Odds API er konfigureret, men der er ikke bygget odds til kampene endnu. Kør Refresh odds now under Advanced / Admin."
    if status["active_odds_source"] == "manual":
        return "Manual odds CSV er fundet. Kør Refresh odds now under Advanced / Admin for at bygge kampodds."
    if status["active_odds_source"] == "cached":
        return "Cached odds snapshot findes. Kør Refresh odds now under Advanced / Admin for at bruge seneste cache."
    live_freshness = get_data_freshness(LIVE_PREDICTIONS_PATH)
    if not live_freshness["file_exists"] or live_freshness["row_count"] == 0:
        return "Odds source er konfigureret, men live odds er ikke bygget endnu. Kør Refresh odds now under Advanced / Admin."
    return "Live predictions er indlæst, men ingen 1X2-odds matchede fixtures. Tjek odds-providerens sport key/regions og bookmaker coverage."


def refresh_odds_from_ui(button_label: str, key: str, use_container_width: bool = False) -> None:
    if not st.button(button_label, key=key, use_container_width=use_container_width):
        return
    try:
        with st.spinner("Refreshing odds..."):
            result = refresh_live_odds_and_predictions(force_refresh=True)
        for warning in result.get("warnings", []):
            if "Ignored" in str(warning) and "provider odds event" in str(warning):
                st.info(warning)
            else:
                st.warning(warning)
        if result.get("last_error"):
            st.warning(result["last_error"])
        if result.get("matches_with_odds", 0) > 0:
            st.session_state.data_mode = "live"
        st.session_state.pop("_prediction_prepare_signature", None)
        _load_enriched_predictions_cached.clear()
        model_refresh_messages = []
        if result.get("matches_with_odds", 0) > 0:
            try:
                model_refresh_messages = prepare_best_available_predictions()
            except Exception as exc:
                model_refresh_messages = [f"Model predictions could not be regenerated after odds refresh: {exc}"]
            st.session_state.pop("_prediction_prepare_signature", None)
            _load_enriched_predictions_cached.clear()
        source = result.get("active_odds_source", "missing")
        if source == "manual":
            source_label = "Danske Spil CSV"
        elif source == "api":
            source_label = "The Odds API"
        elif source == "cached":
            source_label = "cached odds snapshot"
        else:
            source_label = "missing odds source"
        if result.get("matches_with_odds", 0) > 0:
            refresh_message = (
                f"Odds opdateret via {source_label}. "
                f"{result['matches_with_odds']} / {result['matches_total']} kampe har odds."
            )
        else:
            refresh_message = (
                f"Ingen odds blev opdateret via {source_label}. "
                "Tjek ODDS_API_KEY eller data/reference/manual_odds.csv."
            )
        if model_refresh_messages:
            refresh_message = f"{refresh_message} {' '.join(model_refresh_messages)}"
        st.session_state["last_odds_refresh_message"] = refresh_message
        st.session_state["last_odds_refresh_result"] = result
        st.rerun()
    except Exception as exc:
        st.error(f"Could not refresh odds: {exc}")


def refresh_matches_from_ui(button_label: str, key: str, use_container_width: bool = False) -> None:
    if not st.button(button_label, key=key, use_container_width=use_container_width):
        return
    try:
        with st.spinner("Refreshing match results..."):
            result = refresh_match_results()
        st.session_state.pop("_prediction_prepare_signature", None)
        _load_enriched_predictions_cached.clear()
        st.session_state["last_match_refresh_message"] = result["message"]
        st.session_state["last_match_refresh_result"] = result
        st.rerun()
    except Exception as exc:
        st.error(f"Could not refresh matches: {exc}")


def odds_provenance_text(df: pd.DataFrame) -> str:
    if st.session_state.active_data_mode == "sample":
        return "Odds: Sample/demo odds - not real bookmaker odds."
    if "odds_available" in df.columns and pd.to_numeric(df["odds_available"], errors="coerce").fillna(False).astype(bool).any():
        priced = df[pd.to_numeric(df["odds_available"], errors="coerce").fillna(False).astype(bool)]
        source = str(priced["odds_source"].dropna().iloc[0]) if "odds_source" in priced.columns and not priced["odds_source"].dropna().empty else "unknown"
        provider = str(priced["odds_provider"].dropna().iloc[0]) if "odds_provider" in priced.columns and not priced["odds_provider"].dropna().empty else source
        bookmaker_count = int(pd.to_numeric(priced.get("bookmaker_count", pd.Series([0])), errors="coerce").max())
        updated = ""
        if "odds_last_updated_utc" in priced.columns and not priced["odds_last_updated_utc"].dropna().empty:
            updated = f" · updated {format_danish_kickoff(priced['odds_last_updated_utc'].dropna().iloc[0])}"
        label = {
            "api": "The Odds API",
            "manual_csv": "Manual CSV",
            "cached_snapshot": "Cached snapshot",
        }.get(source, provider)
        return f"Odds: {label} · {bookmaker_count} bookmakers{updated}"
    status = get_odds_source_status()
    if status["active_odds_source"] == "manual":
        return "Odds: Manual CSV configured · refresh required"
    if status["active_odds_source"] == "cached":
        return "Odds: Cached snapshot available · refresh required"
    if status["active_odds_source"] == "api":
        return "Odds: The Odds API configured · refresh required"
    return "Odds missing"


def compact_match_card(row) -> None:
    prediction = match_prediction_summary(row)
    model_line = probability_line_for_source(row, "model")
    ds_odds_line = danske_spil_probability_line(row)
    difference_note = model_vs_danske_spil_difference_note(row)
    difference_line = (
        f"<div class=\"wc-match-reason\"><b>Model/odds gap:</b> {html.escape(difference_note)}</div>"
        if difference_note
        else ""
    )
    status_label = {
        "play": "Play",
        "no_bet": "No bet",
        "odds_missing": "Odds missing",
    }.get(str(row.get("primary_status")), str(row["recommendation_status"]))
    status = html.escape(status_label)
    status_class = {
        "play": "wc-status-green",
        "no_bet": "wc-status-muted",
        "odds_missing": "wc-status-amber",
    }.get(str(row.get("primary_status")), "wc-status-muted")
    favorite_label = (
        " <span class=\"wc-mini-label\">Favorite bet</span>"
        if recommended_bet_is_danske_spil_favorite(row)
        else ""
    )
    stake = primary_stake_text(row)
    stake_line = f"<div class=\"wc-match-line\"><b>Suggested stake:</b> {html.escape(stake)}</div>" if stake else ""
    st.markdown(
        f"""
        <div class="wc-match-compact">
          <div class="wc-match-main">
            <div>
              <div class="wc-match-title">{html.escape(match_label(row))}</div>
              <div class="wc-match-meta">{html.escape(str(row.get('kickoff_time_dk', row['kickoff_time'])))} · Group {html.escape(str(row['group']))} · MD {html.escape(str(row['matchday']))}</div>
            </div>
            <div class="{status_class} wc-match-status">{status}{favorite_label}</div>
          </div>
          <div class="wc-match-line"><b>Favorite:</b> {html.escape(str(row.get('model_favorite_label', prediction['favorite'])))} · {html.escape(format_probability(row.get('model_favorite_probability', 0)))}</div>
          <div class="wc-match-line"><b>ML model:</b> {html.escape(model_line)}</div>
          <div class="wc-match-line"><b>Danske Spil odds:</b> {html.escape(ds_odds_line)}</div>
          {difference_line}
          <div class="wc-match-line"><b>{html.escape(primary_decision_title())}:</b> {html.escape(primary_decision_text(row))}</div>
          {stake_line}
          <div class="wc-match-reason"><b>Reason:</b> {html.escape(str(row.get('primary_reason') or no_bet_reason(row, 'ds')))}</div>
          <div class="wc-match-line"><b>Best market:</b> {html.escape(best_market_note(row))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_overview(df: pd.DataFrame) -> None:
    active_df, archived_df = split_active_and_archived_matches(df)
    st.markdown(
        """
        <div class="wc-hero-title">VM 2026 Prediction & Kelly</div>
        <div class="wc-hero-subtitle">Se favorit, sandsynligheder, betting decision og anbefalet stake for kommende VM-kampe.</div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"Live data · {len(active_df)} upcoming matches · Preferred bookmaker: {preferred_bookmaker_label()}")
    st.caption(odds_provenance_text(active_df))
    refresh_message = st.session_state.pop("last_odds_refresh_message", None)
    if refresh_message:
        st.success(refresh_message)
    match_refresh_message = st.session_state.pop("last_match_refresh_message", None)
    if match_refresh_message:
        st.success(match_refresh_message)
    archive_col, odds_refresh_col, matches_refresh_col = st.columns([1, 1, 1])
    with archive_col:
        st.caption(f"{len(archived_df)} afviklede kampe er flyttet til Match Archive.")
        if st.button("View archive", key="overview_view_archive", disabled=archived_df.empty, use_container_width=True):
            st.session_state.page = "Match Archive"
            st.session_state.current_page = "Match Archive"
            st.rerun()
    with odds_refresh_col:
        st.caption("Opdater odds og live predictions.")
        refresh_odds_from_ui("Refresh odds", key="overview_refresh_odds", use_container_width=True)
    with matches_refresh_col:
        st.caption("Opdater resultater og arkiv.")
        refresh_matches_from_ui("Refresh matches", key="overview_refresh_matches", use_container_width=True)
    if st.session_state.active_data_mode == "sample":
        st.warning("Sample/demo data is selected manually. These are not official World Cup fixtures.")
    if active_df.empty:
        empty_state(
            "No upcoming matches loaded. Completed matches are available in Match Archive."
        )
        return
    df = active_df
    counts = df["recommendation_status"].value_counts()
    kpi_cols = st.columns(6)
    with kpi_cols[0]:
        st.metric("Bankroll", format_dkk(load_bankroll_state()["current_bankroll"]), help=TOOLTIPS["stake"])
    with kpi_cols[1]:
        st.metric("Kampe", str(len(df)), st.session_state.active_data_mode.title())
    with kpi_cols[2]:
        st.metric("Playable DS", str(counts.get("Playable at Danske Spil", 0)), help=TOOLTIPS["playable_ds"])
    with kpi_cols[3]:
        st.metric("Better elsewhere", str(counts.get("Better elsewhere", 0)), help=TOOLTIPS["better_elsewhere"])
    with kpi_cols[4]:
        st.metric("No bet", str(counts.get("No bet", 0)), help=TOOLTIPS["no_bet"])
    with kpi_cols[5]:
        st.metric("High draw", str((df["draw_context_label"] == "High").sum()), help=TOOLTIPS["draw_context"])
    freshness = get_data_freshness(data_freshness_path_for_mode(st.session_state.active_data_mode))
    st.caption(f"Sidst opdateret: {freshness['last_modified'] or 'not available'} | Rækker: {len(df)}")
    odds_message = odds_availability_message(df)
    if odds_message:
        st.info(odds_message)
    if st.session_state.active_data_mode == "live":
        st.info(
            "Live mode uses fetched odds and the best available prediction source. If model predictions cannot be generated, market probabilities are used as fallback."
        )
    st.caption(f"Recommendations use {preferred_bookmaker_label()} first. Best market odds are shown as comparison.")

    with st.expander("Filtre", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        groups = c1.multiselect("Group", sorted(df["group"].unique()), default=sorted(df["group"].unique()))
        matchdays = c2.multiselect(
            "Matchday", sorted(df["matchday"].unique()), default=sorted(df["matchday"].unique())
        )
        recommended_only = c3.checkbox("Show recommended bets only")
        positive_edge_only = c4.checkbox("Show positive edge only")
        draw_value_only = c1.checkbox("Show draw-value opportunities only")
        high_draw_only = c2.checkbox("Show high draw-context only")
        playable_ds_only = c3.checkbox("Show bets playable at Danske Spil")
        better_elsewhere_only = c4.checkbox("Show bets better elsewhere")
        if st.button("Reset filters"):
            st.rerun()

    filtered = df[df["group"].isin(groups) & df["matchday"].isin(matchdays)].copy()
    if recommended_only:
        filtered = filtered[filtered["primary_status"] == "play"]
    if positive_edge_only:
        filtered = filtered[(filtered["recommended_edge_ds"] > 0) | (filtered["recommended_edge_best"] > 0)]
    if draw_value_only:
        filtered = filtered[
            (filtered["recommended_outcome_ds"] == "Draw") | (filtered["recommended_outcome_best"] == "Draw")
        ]
    if high_draw_only:
        filtered = filtered[filtered["draw_context_label"] == "High"]
    if playable_ds_only:
        filtered = filtered[filtered["recommended_outcome_ds"] != "No bet"]
    if better_elsewhere_only:
        filtered = filtered[filtered["recommendation_status"] == "Better elsewhere"]
    if "kickoff_time" in filtered.columns:
        filtered["_kickoff_sort"] = pd.to_datetime(filtered["kickoff_time"], errors="coerce", utc=True)
        filtered = filtered.sort_values(["_kickoff_sort", "group", "matchday", "match_id"], na_position="last")
        filtered = filtered.drop(columns=["_kickoff_sort"])

    for _, row in filtered.iterrows():
        compact_match_card(row)
        action_cols = st.columns([0.8, 1, 1, 6])
        if action_cols[0].button("Detail", key=f"select_{row['match_id']}"):
            st.session_state.selected_match_id = row["match_id"]
            st.session_state.page = "Match Detail"
            st.rerun()
        action_cols[1].button(
            "Add DS",
            key=f"add_ds_{row['match_id']}",
            disabled=row["recommended_outcome_ds"] == "No bet",
            on_click=add_recommendation_to_bet_slip,
            args=(row, "ds"),
        )
        action_cols[2].button(
            "Add best",
            key=f"add_best_{row['match_id']}",
            disabled=row["recommended_outcome_best"] == "No bet",
            on_click=add_recommendation_to_bet_slip,
            args=(row, "best"),
        )

    table_columns = [
        "kickoff_time_dk",
        "group",
        "matchday",
        "home_team",
        "away_team",
        "active_home_prob",
        "active_draw_prob",
        "active_away_prob",
        "model_home_prob",
        "model_draw_prob",
        "model_away_prob",
        "ds_home_odds",
        "ds_draw_odds",
        "ds_away_odds",
        "best_home_odds",
        "best_draw_odds",
        "best_away_odds",
        "recommended_outcome_ds",
        "recommended_odds_ds",
        "recommended_stake_ds",
        "recommended_outcome_best",
        "recommended_odds_best",
        "recommended_bookmaker_best",
        "recommended_stake_best",
        "recommendation_status",
        "draw_context_label",
    ]
    st.dataframe(format_overview_table(filtered[table_columns]), width="stretch", hide_index=True)


def render_value_bet_card(row, market: str, key_suffix: str) -> None:
    bookmaker = "Danske Spil" if market == "ds" else row.get("recommended_bookmaker_best")
    status = "Playable at Danske Spil" if market == "ds" else row["recommendation_status"]
    reason = (
        "Danske Spil odds are high enough to create positive value."
        if market == "ds"
        else "Value exists at best market odds, but not at Danske Spil."
    )
    with st.container(border=True):
        c1, c2 = st.columns([3, 1])
        c1.markdown(
            f"**{recommendation_outcome_label(row, market)} @ {format_odds(row[f'recommended_odds_{market}'])} · {bookmaker}**"
        )
        c1.caption(f"{match_label(row)} · {row.get('kickoff_time_dk', row['kickoff_time'])}")
        c1.caption(
            f"Edge {format_percentage(row[f'recommended_edge_{market}'])} · "
            f"Kelly {format_percentage(row[f'recommended_fractional_kelly_{market}'])} · "
            f"Stake {format_dkk(row[f'recommended_stake_{market}'])}"
        )
        c1.caption(f"Reason: {reason}")
        c2.markdown(status_badge(status), unsafe_allow_html=True)
        c2.button(
            "Add to bet slip",
            key=f"slip_{market}_{key_suffix}_{row['match_id']}",
            on_click=add_recommendation_to_bet_slip,
            args=(row, market),
        )
        if c2.button("View match", key=f"view_{market}_{key_suffix}_{row['match_id']}"):
            st.session_state.selected_match_id = row["match_id"]
            st.session_state.page = "Match Detail"
            st.rerun()


def render_ds_no_bet_row(row) -> None:
    note = "Better odds available elsewhere." if row["recommended_outcome_best"] != "No bet" else ""
    with st.container(border=True):
        c1, c2 = st.columns([3, 1])
        c1.markdown(f"**{match_label(row)}**")
        c1.caption(f"{row.get('kickoff_time_dk', row['kickoff_time'])} · Reason: {no_bet_reason(row, 'ds')}")
        if note:
            c1.caption(note)
        c2.markdown(status_badge("No bet"), unsafe_allow_html=True)


def page_betting_center(df: pd.DataFrame) -> None:
    df, archived = split_active_and_archived_matches(df)
    st.title("Betting Center")
    st.caption(f"Recommendations are based on {preferred_bookmaker_label()} first. Best market odds are kept as reference.")
    if not archived.empty:
        st.caption(f"{len(archived)} afviklede kampe er flyttet til Match Archive og er ikke spilbare her.")
    ds_tab, elsewhere_tab, best_tab, slip_tab = st.tabs(["Danske Spil", "Value elsewhere", "Best market", "Bet slip"])

    with ds_tab:
        st.subheader("Playable at Danske Spil")
        playable = df[df["recommended_outcome_ds"] != "No bet"].copy()
        if playable.empty:
            empty_state("No playable bets at Danske Spil with the current odds and risk settings.")
        else:
            playable = playable.sort_values(["recommended_edge_ds", "recommended_stake_ds", "kickoff_time"], ascending=[False, False, True])
            for _, row in playable.iterrows():
                render_value_bet_card(row, "ds", "ds")

    with elsewhere_tab:
        st.subheader("Value elsewhere")
        st.caption("These are value bets elsewhere, not at your selected bookmaker.")
        elsewhere = df[(df["recommended_outcome_ds"] == "No bet") & (df["recommended_outcome_best"] != "No bet")].copy()
        if elsewhere.empty:
            empty_state("No value bets elsewhere that are unavailable or unplayable at Danske Spil.")
        else:
            elsewhere = elsewhere.sort_values(["recommended_edge_best", "recommended_stake_best", "kickoff_time"], ascending=[False, False, True])
            for _, row in elsewhere.iterrows():
                render_value_bet_card(row, "best", "elsewhere")
                st.caption("Stake if betting elsewhere: " + format_dkk(row["recommended_stake_best"]))

    with best_tab:
        st.subheader("Best market")
        best = df[df["recommended_outcome_best"] != "No bet"].copy()
        if best.empty:
            empty_state("No best-market value bets found.")
        else:
            best = best.sort_values(["recommended_edge_best", "recommended_stake_best", "kickoff_time"], ascending=[False, False, True])
            for _, row in best.iterrows():
                render_value_bet_card(row, "best", "best")
                st.caption("Danske Spil: Playable" if row["recommended_outcome_ds"] != "No bet" else "Danske Spil: No bet")

    with slip_tab:
        st.subheader("Bet slip")
        if not st.session_state.bet_slip:
            empty_state("No bets in the slip yet. Add value bets from Match Overview or Betting Center.")
        else:
            total_stake = sum(float(item["stake_dkk"]) for item in st.session_state.bet_slip)
            bankroll = load_bankroll_state()["current_bankroll"]
            cols = st.columns(4)
            cols[0].metric("Bets", str(len(st.session_state.bet_slip)))
            cols[1].metric("Total stake", format_dkk(total_stake))
            cols[2].metric("Bankroll impact", format_percentage(total_stake / bankroll if bankroll else 0))
            cols[3].metric("Max exposure", format_dkk(total_stake))
            if bankroll and total_stake / bankroll > current_profile()["max_stake_pct_of_bankroll"] * max(1, len(st.session_state.bet_slip)):
                st.warning("Total exposure is high compared with the active stake settings.")
            st.dataframe(format_bet_slip_table(st.session_state.bet_slip), width="stretch", hide_index=True)
            c1, c2 = st.columns(2)
            if c1.button("Add selected bets to Bet Log"):
                commit_bet_slip_to_log()
                st.rerun()
            if c2.button("Clear bet slip"):
                st.session_state.bet_slip = []
                st.rerun()


def page_match_detail(df: pd.DataFrame) -> None:
    if df.empty:
        empty_state("No matches loaded yet.")
        return
    options = {f"{row.match_id} | {row.home_team} vs {row.away_team}": row.match_id for row in df.itertuples()}
    selected_label = st.selectbox(
        "Vælg kamp",
        list(options.keys()),
        index=list(options.values()).index(st.session_state.selected_match_id)
        if st.session_state.selected_match_id in options.values()
        else 0,
    )
    st.session_state.selected_match_id = options[selected_label]
    row = df[df["match_id"] == st.session_state.selected_match_id].iloc[0]

    st.title(match_label(row))
    st.caption(f"{row.get('kickoff_time_dk', row['kickoff_time'])} · Group {row['group']} · Matchday {row['matchday']}")
    prediction = match_prediction_summary(row)

    h1, h2, h3 = st.columns(3)
    with h1:
        metric_card("Favorite", f"{row.get('model_favorite_label', prediction['favorite'])} · {format_probability(row.get('model_favorite_probability', 0))}", "The most likely outcome according to the model.")
    with h2:
        metric_card("Prediction", prediction["line"])
    with h3:
        metric_card("Preferred bookmaker", preferred_bookmaker_label(), "Choose where you normally place bets. Recommendations use this bookmaker first.")
    st.caption("Prediction source: Best available")

    source_cols = st.columns(2)
    with source_cols[0]:
        metric_card(
            "ML model prediction",
            favorite_for_source(row, "model"),
            probability_line_for_source(row, "model"),
        )
    with source_cols[1]:
        metric_card(
            "Market-implied prediction",
            favorite_for_source(row, "market"),
            probability_line_for_source(row, "market"),
        )

    has_displayable_probabilities = row_has_displayable_probabilities(row)
    if has_displayable_probabilities:
        probability_columns = {
            "Outcome": ["Home", "Draw", "Away"],
            "Model": [row["model_home_prob"], row["model_draw_prob"], row["model_away_prob"]],
            "Active": [
                row.get("active_home_prob", row["model_home_prob"]),
                row.get("active_draw_prob", row["model_draw_prob"]),
                row.get("active_away_prob", row["model_away_prob"]),
            ],
        }
        if row_has_priced_odds(row):
            probability_columns["Market"] = [row["market_home_prob"], row["market_draw_prob"], row["market_away_prob"]]
        prob_df = pd.DataFrame(probability_columns)
        if "ensemble_home_prob" in row.index:
            prob_df["Ensemble"] = [row.get("ensemble_home_prob"), row.get("ensemble_draw_prob"), row.get("ensemble_away_prob")]
    else:
        prob_df = pd.DataFrame()

    st.subheader(primary_decision_title())
    c1, c2 = st.columns(2)
    with c1:
        recommendation_card_v2(
            "Danske Spil decision",
            "Play at Danske Spil" if row["recommended_outcome_ds"] != "No bet" else ("Odds missing at Danske Spil" if row.get("primary_status") == "odds_missing" else "No bet at Danske Spil"),
            recommendation_outcome_label(row, "ds"),
            row["recommended_odds_ds"],
            "Danske Spil",
            row["recommended_edge_ds"],
            row["recommended_fractional_kelly_ds"],
            row["recommended_stake_ds"],
            probability_source_label(row),
            reason=row.get("primary_reason") if st.session_state.get("preferred_bookmaker_mode", "danske_spil") == "danske_spil" else no_bet_reason(row, "ds"),
        )
        if row["recommended_outcome_ds"] == "No bet":
            st.caption(f"Why disabled: {no_bet_reason(row, 'ds')}")
        st.button(
            "Add Danske Spil to bet slip",
            disabled=row["recommended_outcome_ds"] == "No bet",
            on_click=add_recommendation_to_bet_slip,
            args=(row, "ds"),
        )
    with c2:
        recommendation_card_v2(
            "Best market comparison",
            "Better odds available elsewhere" if row.get("comparison_status") == "better_elsewhere" else ("Best market odds missing" if row.get("comparison_status") == "comparison_missing" else "No better market value found" if row["recommended_outcome_best"] == "No bet" else "Same or similar odds"),
            recommendation_outcome_label(row, "best"),
            row["recommended_odds_best"],
            row["recommended_bookmaker_best"],
            row["recommended_edge_best"],
            row["recommended_fractional_kelly_best"],
            row["recommended_stake_best"],
            probability_source_label(row),
            reason=best_market_note(row),
        )
        if row["recommended_outcome_best"] == "No bet":
            st.caption(f"Why disabled: {no_bet_reason(row, 'best')}")
        st.button(
            "Add Best Market to bet slip",
            disabled=row["recommended_outcome_best"] == "No bet",
            on_click=add_recommendation_to_bet_slip,
            args=(row, "best"),
        )

    draw_context_card_v2(
        row["draw_context_score"],
        row["draw_context_label"],
        row["mutual_draw_acceptance"],
        row["both_teams_draw_satisfied"],
        row["one_team_must_win"],
    )

    with st.expander("Advanced probability and value details"):
        value_rows = []
        for market, bookmaker in [("ds", "Danske Spil"), ("best", row.get("recommended_bookmaker_best") or "Best market")]:
            value_rows.append(
                {
                    "Bookmaker": bookmaker,
                    "Outcome": recommendation_outcome_label(row, market),
                    "Odds": format_odds(row.get(f"recommended_odds_{market}")),
                    "Model probability": format_probability(row.get(f"recommended_probability_{market}")),
                    "Market probability": format_probability(row.get(f"recommended_implied_probability_{market}")),
                    "Fair odds": format_odds(row.get(f"recommended_fair_odds_{market}")),
                    "Edge": format_percentage(row.get(f"recommended_edge_{market}", 0)),
                    "Kelly": format_percentage(row.get(f"recommended_fractional_kelly_{market}", 0)),
                    "Stake": format_dkk(row.get(f"recommended_stake_{market}", 0)),
                }
            )
        st.dataframe(pd.DataFrame(value_rows), width="stretch", hide_index=True)
        st.caption(f"Technical source: {technical_probability_source_label(row)}")
        if probability_triplets_are_identical(row):
            st.info("These values are identical because the app is currently using market probabilities as fallback.")
        if not has_displayable_probabilities:
            empty_state("Sandsynligheder afventer odds eller en modelkørsel. 33/33/33-placeholders vises ikke som rigtige prognoser.")
        else:
            c1, c2 = st.columns([1, 1.2])
            c1.dataframe(
                prob_df.style.format({column: "{:.1%}" for column in prob_df.columns if column != "Outcome"}),
                width="stretch",
                hide_index=True,
            )
            chart = active_vs_market_model_chart(row)
            if chart is None:
                c2.info("Ingen sandsynlighedsdata at vise endnu.")
            else:
                c2.plotly_chart(chart, width="stretch")

    with st.expander("Odds and Kelly details"):
        st.dataframe(odds_comparison_table(row), width="stretch", hide_index=True)
        st.dataframe(style_edge_table(outcome_kelly_table(row)), width="stretch", hide_index=True)


def format_archive_table(df: pd.DataFrame) -> pd.DataFrame:
    display = df.copy()
    display["match"] = display["home_team"].astype(str) + " vs " + display["away_team"].astype(str)
    columns = [
        "kickoff_time_dk",
        "group",
        "matchday",
        "match",
        "full_time_score",
        "favorite_outcome_label",
        "actual_outcome_label",
        "favorite_result_status",
        "result_source",
    ]
    rename = {
        "kickoff_time_dk": "Kickoff DK",
        "group": "Group",
        "matchday": "MD",
        "full_time_score": "Resultat",
        "favorite_outcome_label": "Favorit",
        "actual_outcome_label": "Udfald",
        "favorite_result_status": "Status",
        "result_source": "Kilde",
    }
    return display[[column for column in columns if column in display.columns]].rename(columns=rename)


def page_match_archive(df: pd.DataFrame) -> None:
    _, archived = split_active_and_archived_matches(df)
    st.title("Match Archive")
    st.caption("Afviklede kampe flyttes hertil, så Match Overview kun viser kommende kampe.")
    if archived.empty:
        empty_state("Ingen afviklede kampe i arkivet endnu.")
        return

    favorite_hits = int((archived["favorite_result_status"] == "Favoritten gik hjem").sum())
    surprises = int((archived["favorite_result_status"] == "Overraskelse").sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Afviklede kampe", str(len(archived)))
    c2.metric("Favoritten gik hjem", str(favorite_hits))
    c3.metric("Overraskelser", str(surprises))

    for _, row in archived.sort_values("kickoff_time").iterrows():
        status_class = "wc-status-green" if row["favorite_result_status"] == "Favoritten gik hjem" else "wc-status-amber"
        st.markdown(
            f"""
            <div class="wc-match-compact">
              <div class="wc-match-main">
                <div>
                  <div class="wc-match-title">{html.escape(match_label(row))}</div>
                  <div class="wc-match-meta">{html.escape(str(row.get('kickoff_time_dk', row['kickoff_time'])))} · Group {html.escape(str(row['group']))} · MD {html.escape(str(row['matchday']))}</div>
                </div>
                <div class="{status_class} wc-match-status">{html.escape(str(row['favorite_result_status']))}</div>
              </div>
              <div class="wc-match-line"><b>Resultat:</b> {html.escape(str(row['full_time_score']))} · <b>Favorit:</b> {html.escape(str(row['favorite_outcome_label']))} · <b>Udfald:</b> {html.escape(str(row['actual_outcome_label']))}</div>
              <div class="wc-match-reason">Kilde: {html.escape(str(row.get('result_source', '-')))} · checked: {html.escape(str(row.get('result_last_checked_utc', '-')))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.dataframe(format_archive_table(archived), width="stretch", hide_index=True)


def page_bankroll() -> None:
    state = load_bankroll_state()
    net = state["current_bankroll"] - state["starting_bankroll"]
    ret = net / state["starting_bankroll"] if state["starting_bankroll"] else 0
    st.title("Bankroll")
    cols = st.columns(4)
    with cols[0]:
        metric_card("Starting bankroll", format_dkk(state["starting_bankroll"]))
    with cols[1]:
        metric_card("Current bankroll", format_dkk(state["current_bankroll"]))
    with cols[2]:
        metric_card("Net profit/loss", format_dkk(net))
    with cols[3]:
        metric_card("Return", format_percentage(ret))

    history = load_bankroll_history()
    if history.empty:
        empty_state("No bankroll history yet. Deposits, withdrawals and bet settlements will appear here.")
    else:
        render_chart(bankroll_history_chart(history))
        st.dataframe(history, width="stretch", hide_index=True)

    with st.form("manual_bankroll_update"):
        st.subheader("Manual update")
        transaction_type = st.selectbox("Transaction type", ["deposit", "withdrawal", "manual correction"])
        amount = st.number_input("Amount", value=0.0, step=10.0)
        note = st.text_input("Note")
        submitted = st.form_submit_button("Update bankroll")
        if submitted:
            signed_amount = abs(amount) if transaction_type == "deposit" else -abs(amount)
            if transaction_type == "manual correction":
                signed_amount = amount
            try:
                update_bankroll(signed_amount, transaction_type, note=note)
                st.success("Bankroll opdateret.")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

    with st.expander("Reset bankroll"):
        st.warning("Resetting bankroll changes the starting and current bankroll. Existing bet log entries are not deleted.")
        new_start = st.number_input("New starting bankroll", min_value=0.0, value=1000.0, step=100.0)
        confirm = st.checkbox("I understand this resets starting and current bankroll")
        if st.button("Reset bankroll", disabled=not confirm):
            try:
                reset_bankroll(new_start)
                st.success("Bankroll nulstillet.")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))


def page_bet_log() -> None:
    st.title("Bet Log")
    st.caption(
        "Adding a bet does not change bankroll. Settlement updates bankroll exactly once: won adds profit, "
        "lost subtracts stake, and void leaves bankroll unchanged. Kickoff is shown in Danish time when available."
    )
    summary = calculate_bet_summary()
    cols = st.columns(6)
    with cols[0]:
        metric_card("Total bets", str(summary["total_bets"]))
    with cols[1]:
        metric_card("Pending", str(summary["pending_bets"]))
    with cols[2]:
        metric_card("Settled", str(summary["settled_bets"]))
    with cols[3]:
        metric_card("Total P/L", format_dkk(summary["total_profit_loss"]))
    with cols[4]:
        metric_card("ROI", format_percentage(summary["roi"]))
    with cols[5]:
        metric_card("Win rate", format_percentage(summary["win_rate"]))

    df = load_bet_log()
    pending = df[df["result"] == "pending"] if not df.empty else df
    settled = df[df["result"].isin(["won", "lost", "void"])] if not df.empty else df

    tab_all, tab_pending, tab_settled, tab_manual, tab_settlement = st.tabs(
        ["All bets", "Pending", "Settled", "Manual entry", "Settlement"]
    )

    with tab_all:
        if df.empty:
            empty_state("No bets logged yet. Select a match, inspect the recommendation, then add a DS or best-market bet from Match Detail.")
        else:
            st.dataframe(format_bet_log_table(df), width="stretch", hide_index=True)

    with tab_pending:
        if pending.empty:
            empty_state("No pending bets. Adding a bet does not change bankroll until you settle it.")
        else:
            st.dataframe(format_bet_log_table(pending), width="stretch", hide_index=True)

    with tab_settled:
        if settled.empty:
            empty_state("No settled bets yet. Settled wins add profit only, losses subtract stake, and void bets change nothing.")
        else:
            st.dataframe(format_bet_log_table(settled), width="stretch", hide_index=True)

    with tab_manual:
        with st.form("manual_bet_form"):
            c1, c2, c3 = st.columns(3)
            match_id = c1.text_input("match_id")
            match = c2.text_input("match")
            bookmaker = c3.text_input("bookmaker", value=st.session_state.preferred_bookmaker)
            outcome = c1.selectbox("outcome", ["Home", "Draw", "Away"])
            odds = c2.number_input("odds", min_value=1.01, value=2.0, step=0.01)
            model_probability = c3.number_input("model_probability", min_value=0.0, max_value=1.0, value=0.5)
            edge = c1.number_input("edge", value=0.0, step=0.01)
            full_kelly = c2.number_input("full_kelly", value=0.0, step=0.01)
            fractional_kelly = c3.number_input("fractional_kelly", value=0.0, step=0.01)
            stake_dkk = c1.number_input("stake_dkk", min_value=0.0, value=0.0, step=10.0)
            if st.form_submit_button("Add manual bet"):
                try:
                    add_bet(
                        match_id=match_id,
                        match=match,
                        bookmaker=bookmaker,
                        outcome=outcome,
                        odds=odds,
                        model_probability=model_probability,
                        edge=edge,
                        full_kelly=full_kelly,
                        fractional_kelly=fractional_kelly,
                        stake_dkk=stake_dkk,
                    )
                    st.success("Bet tilføjet. Bankroll ændres først ved settlement.")
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))

    with tab_settlement:
        st.warning("Settling a bet updates bankroll. This action cannot be automatically reversed. Use manual correction if needed.")
        if pending.empty:
            empty_state("No pending bets to settle.")
        else:
            bet_id = st.selectbox("Pending bet_id", pending["bet_id"].tolist())
            selected = pending[pending["bet_id"] == bet_id].iloc[0]
            st.write(
                {
                    "match": selected["match"],
                    "kickoff_time_dk": selected.get("kickoff_time_dk", ""),
                    "bookmaker": selected["bookmaker"],
                    "outcome": selected["outcome"],
                    "odds": selected["odds"],
                    "stake_dkk": selected["stake_dkk"],
                }
            )
            result = st.selectbox("Result", ["won", "lost", "void"])
            if st.button("Settle bet"):
                try:
                    settle_bet(bet_id, result)
                    st.success("Bet afregnet. Bankroll er opdateret præcis én gang for dette bet.")
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))

        if not df.empty:
            st.divider()
            st.caption("Reset settlement status only. Bankroll is not automatically reversed.")
            reset_id = st.selectbox("bet_id", df["bet_id"].tolist(), key="reset_bet_id")
            if st.button("Reset settlement"):
                reset_bet_settlement(reset_id)
                st.warning("Settlement er nulstillet. Bankroll er ikke automatisk reverseret.")
                st.rerun()


def page_analytics() -> None:
    st.title("Analytics")
    summary = calculate_bet_summary()
    cols = st.columns(5)
    with cols[0]:
        metric_card("Total staked", format_dkk(summary["total_staked"]))
    with cols[1]:
        metric_card("Total P/L", format_dkk(summary["total_profit_loss"]))
    with cols[2]:
        metric_card("ROI", format_percentage(summary["roi"]))
    with cols[3]:
        metric_card("Win rate", format_percentage(summary["win_rate"]))
    with cols[4]:
        metric_card("Average odds", format_odds(summary["average_odds"]))
    df = load_bet_log()
    settled = df[df["result"].isin(["won", "lost", "void"])] if not df.empty else df
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Profit/loss by bookmaker")
        render_chart(profit_loss_by_bookmaker_chart(settled))
    with c2:
        st.subheader("Profit/loss by outcome")
        render_chart(profit_loss_by_outcome_chart(settled))
    history = load_bankroll_history()
    st.subheader("Bankroll over time")
    if history.empty:
        empty_state("No bankroll history yet.")
    else:
        render_chart(bankroll_history_chart(history))
    st.subheader("Draw bet performance")
    draw_bets = settled[settled["outcome"] == "Draw"]
    if draw_bets.empty:
        empty_state("No draw bets logged yet.")
    else:
        st.dataframe(draw_bets, width="stretch", hide_index=True)
    st.subheader("Model backtest snapshot")
    backtest_status = get_latest_backtest_status()
    if not backtest_status["backtest_exists"]:
        empty_state("No backtest results yet. Run a walk-forward backtest from Backtest & Metrics.")
    else:
        metric_row(
            [
                ("Accuracy", "-" if pd.isna(backtest_status["overall_accuracy"]) else format_percentage(backtest_status["overall_accuracy"])),
                ("Log loss", "-" if pd.isna(backtest_status["overall_log_loss"]) else f"{backtest_status['overall_log_loss']:.3f}"),
                ("Brier", "-" if pd.isna(backtest_status["overall_brier_score"]) else f"{backtest_status['overall_brier_score']:.3f}"),
                ("ECE", "-" if pd.isna(backtest_status["overall_ece"]) else f"{backtest_status['overall_ece']:.3f}"),
                ("Predictions", str(backtest_status["prediction_count"])),
            ]
        )


def page_my_bets() -> None:
    st.title("My Bets")
    st.caption("Pending bets, settled bets, bankroll and performance. Model metrics live in Model & Data.")
    summary = calculate_bet_summary()
    state = load_bankroll_state()
    bet_df = load_bet_log()
    pending_exposure = float(bet_df.loc[bet_df["result"] == "pending", "stake_dkk"].sum()) if not bet_df.empty else 0.0
    cols = st.columns(5)
    cols[0].metric("Current bankroll", format_dkk(state["current_bankroll"]))
    cols[1].metric("Pending bets", str(summary["pending_bets"]))
    cols[2].metric("Pending exposure", format_dkk(pending_exposure))
    cols[3].metric("P/L", format_dkk(summary["total_profit_loss"]))
    cols[4].metric("ROI", format_percentage(summary["roi"]))

    bet_tab, bankroll_tab, performance_tab = st.tabs(["Pending & settled", "Bankroll", "Performance"])
    with bet_tab:
        page_bet_log()
    with bankroll_tab:
        page_bankroll()
    with performance_tab:
        df = load_bet_log()
        settled = df[df["result"].isin(["won", "lost", "void"])] if not df.empty else df
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Profit/loss by bookmaker")
            render_chart(profit_loss_by_bookmaker_chart(settled))
        with c2:
            st.subheader("Profit/loss by outcome")
            render_chart(profit_loss_by_outcome_chart(settled))


def page_model_data(df: pd.DataFrame) -> None:
    st.title("Model & Data")
    st.caption("Teknisk transparens om prediction engine, data readiness, backtest, ensemble og fixtures.")
    engine_tab, readiness_tab, model_tab, backtest_tab, ensemble_tab, draw_tab, fixture_tab = st.tabs(
        ["Prediction engine", "Data readiness", "Model status", "Backtest", "Ensemble", "Draw-context", "Fixture data"]
    )
    with engine_tab:
        st.subheader("Current prediction setup")
        metric_row(
            [
                ("Data mode", st.session_state.active_data_mode.title()),
                ("Matches loaded", str(len(df))),
                ("Probability source", PROBABILITY_SOURCE_LABELS.get(st.session_state.probability_source, st.session_state.probability_source)),
                ("Model source", st.session_state.active_model_source.replace("_", " ").title()),
            ]
        )
        st.dataframe(app_health_rows(df), width="stretch", hide_index=True)
    with readiness_tab:
        st.subheader("Data readiness")
        st.caption(fixture_provenance_text(st.session_state.active_data_mode, df))
        st.dataframe(app_health_rows(df), width="stretch", hide_index=True)
    with model_tab:
        status = get_active_model_status()
        metric_row(
            [
                ("Model available", "Yes" if status["model_exists"] else "No"),
                ("Training rows", str(status["number_of_training_rows"])),
                ("Accuracy", "-" if status["accuracy"] is None else format_percentage(status["accuracy"])),
                ("Log loss", "-" if status["log_loss"] is None else f"{status['log_loss']:.3f}"),
                ("Brier", "-" if status["brier_score"] is None else f"{status['brier_score']:.3f}"),
            ]
        )
        st.caption(f"Trained at: {status['trained_at'] or '-'}")
    with backtest_tab:
        page_backtest_metrics()
    with ensemble_tab:
        page_ensemble(df)
    with draw_tab:
        page_draw_hypothesis(df)
    with fixture_tab:
        fixtures = load_fixture_dataset()
        valid, messages = validate_fixture_dataset(fixtures)
        metric_row(
            [
                ("Fixture rows", str(len(fixtures))),
                ("Expected", "104"),
                ("Valid complete set", "Yes" if valid else "No"),
                ("Last checked", fixtures["source_last_checked"].max() if "source_last_checked" in fixtures.columns and not fixtures.empty else "-"),
            ]
        )
        for message in messages:
            st.warning(message)
        st.dataframe(fixtures, width="stretch", hide_index=True)


def _status_icon(status: str) -> str:
    return {"Complete": "OK", "Recommended": "Recommended", "Missing": "Missing", "Not available": "Not available"}.get(status, status)


def _source_label(source_name: str) -> str:
    text = str(source_name)
    if text == "market":
        return "Market odds"
    if text == "historical_model":
        return "ML model"
    if text.startswith("ensemble"):
        return "Ensemble / Best available"
    return text.replace("_", " ").title()


def _metric_card(label: str, metric_name: str, value) -> None:
    st.metric(label, display_metric_value(metric_name, value), help=metric_tooltip(metric_name))
    st.caption(f"{metric_status(metric_name, value)} · {metric_interpretation(metric_name, value)}")


def _render_quality_card(quality: dict) -> None:
    color = {
        "green": ("#166534", "#dcfce7", "#86efac"),
        "amber": ("#92400e", "#fef3c7", "#fcd34d"),
        "red": ("#991b1b", "#fee2e2", "#fecaca"),
        "gray": ("#374151", "#f3f4f6", "#d1d5db"),
    }.get(quality["status_color"], ("#374151", "#f3f4f6", "#d1d5db"))
    st.markdown(
        f"""
        <div style="border:1px solid {color[2]}; background:{color[1]}; border-radius:8px; padding:16px 18px; margin: 4px 0 14px 0;">
          <div style="font-size:0.82rem; color:{color[0]}; font-weight:700; text-transform:uppercase;">{html.escape(quality['quality_label'])}</div>
          <div style="font-size:1.2rem; color:#111827; font-weight:700; margin-top:4px;">{html.escape(quality['headline'])}</div>
          <div style="color:#374151; margin-top:6px;">{html.escape(quality['summary_text'])}</div>
          <div style="color:#111827; margin-top:8px;">Betting use: <strong>{html.escape(quality.get('betting_use', 'Use cautiously'))}</strong></div>
          <div style="color:#111827; margin-top:10px; font-weight:600;">{html.escape(quality['user_conclusion'])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_market_comparison(comparison_df: pd.DataFrame, best_source: Optional[dict]) -> None:
    st.subheader("Model vs market")
    if comparison_df is None or comparison_df.empty:
        empty_state("Market comparison cannot be calculated because historical market odds are not available.")
        st.write(
            "The current metrics show how the model performed on historical matches, but they do not yet tell "
            "whether it beats bookmaker-implied probabilities."
        )
        st.info("Recommended action: add historical market odds, then run full model validation in Advanced / Admin.")
        if st.button("Go to Advanced / Admin", key="model_perf_go_admin_market"):
            st.session_state.current_page = "Advanced / Admin"
            st.session_state.page = "Advanced / Admin"
            st.rerun()
        return

    rows = []
    for _, row in comparison_df.iterrows():
        source = str(row.get("source_name", ""))
        rows.append(
            {
                "Source": _source_label(source),
                "Accuracy": display_metric_value("accuracy", row.get("accuracy")),
                "Probability quality": display_metric_value("log_loss", row.get("log_loss")),
                "Prediction error": display_metric_value("brier_score", row.get("brier_score")),
                "Probability realism": display_metric_value("ece", row.get("ece")),
                "Matches": display_metric_value("match_count", row.get("match_count")),
                "Best": "Yes" if best_source and source == best_source.get("source_name") else "",
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    if best_source:
        st.success(f"The app currently uses {best_source['label']} because it performed best on probability quality metrics.")


def _render_betting_implication(summary: dict, quality: dict) -> None:
    st.subheader("What this means for betting")
    status = summary.get("model_status")
    if status == "demo_model":
        st.write("The model is not suitable for real predictions. The app falls back to market odds.")
        return
    if status != "production_ready":
        st.write("ML predictions are unavailable, so recommendations should be treated as market-based only.")
        return
    if summary.get("market_comparison_available"):
        st.write(
            "The app uses the prediction source that performed best in validation. Value bets are calculated by "
            "comparing that probability against available odds."
        )
    else:
        st.write(
            "We do not yet know if the model is better than the betting market. Market odds remain an important reference point, "
            "and suggested stakes should be interpreted conservatively."
        )
    if not summary.get("calibration_available"):
        st.warning("Because calibration is not yet calculated, stake suggestions should be interpreted conservatively.")
    st.caption(quality["recommended_next_action"])


def page_model_performance() -> None:
    st.title("Model Performance")
    st.caption("Can I trust the model?")

    model_status = get_active_model_status()
    readiness = get_model_readiness(predictions_exist=MODEL_PREDICTIONS_PATH.exists() or LIVE_PREDICTIONS_WITH_MODEL_PATH.exists())
    backtest_status = get_latest_backtest_status()
    ensemble_df = _load_optional_csv(FULL_BACKTEST_MARKET_COMPARISON_PATH)
    if ensemble_df.empty:
        ensemble_df = _load_optional_csv(ENSEMBLE_COMPARISON_PATH)
    performance = load_model_performance_summary(
        readiness=readiness,
        model_status=model_status,
        backtest_status=backtest_status,
        comparison_df=ensemble_df,
    )
    quality = build_model_quality_summary(readiness.get("metadata", {}), performance, ensemble_df)

    _render_quality_card(quality)

    st.subheader("Key numbers")
    metric_columns = st.columns(5)
    metrics = [
        ("Prediction accuracy", "accuracy", performance.get("accuracy")),
        ("Probability quality", "log_loss", performance.get("log_loss")),
        ("Prediction error", "brier_score", performance.get("brier_score")),
        ("Probability realism", "ece", performance.get("ece")),
        ("Matches tested", "match_count", performance.get("match_count")),
    ]
    for column, (label, metric_name, value) in zip(metric_columns, metrics):
        with column:
            _metric_card(label, metric_name, value)
    if performance["metrics_source"] == "holdout_metadata":
        st.caption("These results come from the pre-trained model's basic test set.")
        st.caption("Basic test results are useful, but full historical validation and market comparison are needed to fully validate betting performance.")
    elif performance["metrics_source"] == "full_backtest":
        st.caption("These results come from the realistic historical test.")
    else:
        st.info("Model performance has not been calculated yet.")

    _render_market_comparison(performance["comparison_df"], performance["best_comparison_source"])

    st.subheader("Validation status")
    checklist = validation_checklist(performance)
    checklist_df = pd.DataFrame([{**item, "Status": _status_icon(item["Status"])} for item in checklist])
    st.dataframe(checklist_df, width="stretch", hide_index=True)

    _render_betting_implication(performance, quality)

    with st.expander("Advanced model metrics"):
        st.caption("Technical details and diagnostics are kept here so the main page stays readable.")
        metric_row(
            [
                ("Prediction source", "Best available prediction"),
                ("Currently using", "ML model" if readiness["is_usable_as_best_available"] else "Market fallback"),
                ("Model version", str(readiness.get("model_version") or "Not available")),
                ("Feature count", str(readiness.get("feature_count") or "Not available")),
                ("Training rows", str(readiness.get("training_rows") or "Not available")),
                ("Test rows", str(readiness.get("test_rows") or "Not available")),
            ]
        )
        st.write(
            f"Training period: {readiness.get('training_data_start_date', 'Not available')} "
            f"to {readiness.get('training_data_end_date', 'Not available')}"
        )
        st.write(
            "Feature groups: "
            f"Elo={'yes' if readiness.get('includes_elo_features') else 'no'}, "
            f"FIFA ranking={'yes' if readiness.get('includes_fifa_ranking_features') else 'no'}, "
            f"form={'yes' if readiness.get('includes_form_features') else 'no'}, "
            f"tournament context={'yes' if readiness.get('includes_tournament_features') else 'no'}, "
            f"neutral venue={'yes' if readiness.get('includes_neutral_venue') else 'no'}."
        )
        if readiness.get("warnings"):
            st.write("Readiness warnings")
            for warning in readiness["warnings"]:
                st.warning(warning)
        comparison_df = _load_optional_csv(DRAW_FEATURE_COMPARISON_PATH)
        fifa_comparison_df = _load_optional_csv(MODEL_VARIANT_COMPARISON_PATH)
        if not fifa_comparison_df.empty:
            st.subheader("Elo / FIFA ranking ablation")
            st.dataframe(fifa_comparison_df, width="stretch", hide_index=True)
        if not comparison_df.empty:
            st.subheader("Historical vs draw-context")
            st.dataframe(comparison_df, width="stretch", hide_index=True)
        if not ensemble_df.empty:
            st.subheader("Market/model/ensemble")
            st.dataframe(ensemble_df, width="stretch", hide_index=True)
        full_summary_df = _load_optional_csv(FULL_BACKTEST_SUMMARY_PATH)
        full_fold_df = _load_optional_csv(FULL_BACKTEST_BY_FOLD_PATH)
        if not full_summary_df.empty:
            st.subheader("Full validation summary")
            st.dataframe(full_summary_df, width="stretch", hide_index=True)
        if not full_fold_df.empty:
            st.subheader("Fold metrics")
            st.dataframe(full_fold_df, width="stretch", hide_index=True)
        with st.expander("Raw model metadata"):
            st.json(readiness.get("metadata", {}))


def page_advanced_admin(df: pd.DataFrame) -> None:
    st.title("Advanced / Admin")
    st.warning("These actions are for development/admin use. Normal users do not need to run them.")
    data_tab, model_tab, backtest_tab, ensemble_tab, fixture_tab, debug_tab = st.tabs(
        ["Data readiness", "Model training", "Backtest", "Ensemble", "Fixture validation", "Debug"]
    )
    with data_tab:
        st.subheader("Data readiness")
        page_settings()
    with model_tab:
        st.subheader("Historical model")
        page_draw_hypothesis(df)
    with backtest_tab:
        page_backtest_metrics()
    with ensemble_tab:
        page_ensemble(df)
    with fixture_tab:
        fixtures = load_fixture_dataset()
        valid, messages = validate_fixture_dataset(fixtures)
        st.subheader("Fixture validation")
        st.metric("Valid complete fixture set", "Yes" if valid else "No")
        for message in messages:
            st.warning(message)
        st.dataframe(fixtures, width="stretch", hide_index=True)
    with debug_tab:
        st.subheader("App health")
        st.dataframe(app_health_rows(df), width="stretch", hide_index=True)


def page_user_settings() -> None:
    st.title("Settings")
    st.caption("Simple app settings for bankroll, risk and odds updates.")

    st.subheader("Preferred bookmaker")
    bookmaker_options = {"Danske Spil": "danske_spil", "Best market": "best_market"}
    reverse_bookmaker_options = {value: key for key, value in bookmaker_options.items()}
    selected_bookmaker_label = st.radio(
        "Where do you usually bet?",
        list(bookmaker_options.keys()),
        index=list(bookmaker_options.keys()).index(
            reverse_bookmaker_options.get(st.session_state.preferred_bookmaker_mode, "Danske Spil")
        ),
        horizontal=True,
        help="Choose where you normally place bets. Recommendations will be based on this bookmaker first.",
        key="user_preferred_bookmaker_mode",
    )
    selected_bookmaker_mode = bookmaker_options[selected_bookmaker_label]
    if st.session_state.preferred_bookmaker_mode != selected_bookmaker_mode:
        st.session_state.preferred_bookmaker_mode = selected_bookmaker_mode
        st.rerun()
    st.caption("Best market odds are still shown as comparison.")

    st.subheader("Bankroll")
    st.caption("Used to calculate suggested stake.")
    state = load_bankroll_state()
    net = state["current_bankroll"] - state["starting_bankroll"]
    metric_row(
        [
            ("Current bankroll", format_dkk(state["current_bankroll"])),
            ("Starting bankroll", format_dkk(state["starting_bankroll"])),
            ("Return", f"{format_dkk(net)} / {format_percentage(net / state['starting_bankroll'] if state['starting_bankroll'] else 0)}"),
        ]
    )
    with st.form("settings_bankroll_update"):
        c1, c2 = st.columns([1, 2])
        amount = c1.number_input("Bankroll adjustment", value=0.0, step=10.0)
        note = c2.text_input("Note", value="")
        if st.form_submit_button("Update bankroll"):
            try:
                update_bankroll(amount, "manual correction", note=note)
                st.success("Bankroll updated.")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
    with st.expander("Reset bankroll"):
        st.warning("Resetting bankroll changes the starting and current bankroll. Existing bet log entries are not deleted.")
        new_start = st.number_input("New starting bankroll", min_value=0.0, value=float(state["starting_bankroll"]), step=100.0)
        confirm = st.checkbox("I understand this resets starting and current bankroll")
        if st.button("Reset bankroll", disabled=not confirm):
            try:
                reset_bankroll(new_start)
                st.success("Bankroll reset.")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

    st.subheader("Risk profile")
    st.caption("Controls how aggressively the app sizes suggested bets.")
    profile_name = st.selectbox(
        "Betting risk profile",
        list(STAKING_PROFILES.keys()),
        index=list(STAKING_PROFILES.keys()).index(st.session_state.kelly_profile_name),
    )
    if profile_name != st.session_state.kelly_profile_name:
        st.session_state.kelly_profile_name = profile_name
        st.session_state.staking_profile = get_staking_profile(profile_name)
        st.rerun()
    profile = current_profile()
    metric_row(
        [
            ("Risk profile", profile_name),
            ("Max stake", format_percentage(profile["max_stake_pct_of_bankroll"])),
            ("Minimum edge", format_percentage(profile["min_edge_threshold"])),
        ]
    )

    st.subheader("Odds update")
    st.caption("Refreshes bookmaker odds used to calculate value.")
    odds_status = get_odds_source_status()
    live_freshness = get_data_freshness(LIVE_PREDICTIONS_PATH)
    metric_row(
        [
            ("Odds", "Missing" if odds_status["active_odds_source"] == "missing" else "Active"),
            ("Source", odds_status["active_odds_source"].replace("_", " ").title()),
            ("Last updated", live_freshness["last_modified"] or "-"),
        ]
    )
    if odds_status["active_odds_source"] == "missing":
        st.warning("Live odds are missing. Add API key or manual odds.")
    refresh_odds_from_ui("Refresh odds", key="settings_refresh_odds")

    st.subheader("Display")
    st.caption("Live data uses official fixtures, updated odds and best available predictions. Demo mode is available only in Advanced / Admin.")


def _load_optional_csv(path) -> pd.DataFrame:
    try:
        if not path.exists() or path.stat().st_size == 0:
            return pd.DataFrame()
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def render_world_cup_sanity_check_results() -> None:
    predictions_df = _load_optional_csv(WORLD_CUP_BACKTEST_PREDICTIONS_PATH)
    summary_df = _load_optional_csv(WORLD_CUP_BACKTEST_SUMMARY_PATH)
    if predictions_df.empty and summary_df.empty:
        return

    st.subheader("World Cup sanity check")
    if predictions_df.empty:
        st.warning("World Cup sanity check ran, but no prediction rows were created. See the summary table for the reason.")
    else:
        accuracy = pd.to_numeric(predictions_df.get("is_correct", pd.Series(dtype=float)), errors="coerce").mean()
        metric_row(
            [
                ("Predictions", str(len(predictions_df))),
                ("Correct", str(int(pd.to_numeric(predictions_df.get("is_correct", pd.Series(dtype=int)), errors="coerce").fillna(0).sum()))),
                ("Accuracy", "-" if pd.isna(accuracy) else format_percentage(float(accuracy))),
            ]
        )
        st.dataframe(predictions_df, width="stretch", hide_index=True)
    if not summary_df.empty:
        st.dataframe(summary_df, width="stretch", hide_index=True)


def page_backtest_metrics() -> None:
    st.title("Backtest & Metrics")
    st.caption("Time-based walk-forward evaluation for the historical model only.")
    status = get_latest_backtest_status()
    historical_exists = HISTORICAL_RESULTS_PATH.exists()
    training_exists = TRAINING_DATASET_PATH.exists()
    cols = st.columns(5)
    with cols[0]:
        metric_card("Historical data", "Yes" if historical_exists else "No")
    with cols[1]:
        metric_card("Training dataset", "Yes" if training_exists else "No")
    with cols[2]:
        metric_card("Backtest results", "Yes" if status["backtest_exists"] else "No")
    with cols[3]:
        metric_card("Predictions", str(status["prediction_count"]))
    with cols[4]:
        metric_card("Summary", "Yes" if status["summary_exists"] else "No")
    st.caption(f"Last backtest file update: {status['last_modified'] or '-'}")
    if not historical_exists:
        st.warning(
            "Historical training data is not available in this deployment. Retraining and new backtests are disabled."
        )

    st.subheader("Run backtest")
    c1, c2, c3, c4 = st.columns(4)
    initial_train_end_date = c1.date_input("Initial train end date", value=pd.Timestamp("2014-01-01").date())
    test_window = c2.text_input("Test window", value="365D")
    step_size = c3.text_input("Step size", value="365D")
    min_train_matches = c4.number_input("Min train matches", min_value=30, value=30, step=25)
    run_col, wc_col, full_col = st.columns(3)
    with run_col:
        if st.button("Run walk-forward backtest"):
            if not historical_exists:
                st.error("Retraining is unavailable because historical training data is not included in this deployment.")
            else:
                try:
                    raw = load_historical_results(HISTORICAL_RESULTS_PATH)
                    warnings, errors = validate_historical_results(raw)
                    for warning in warnings:
                        st.warning(warning)
                    if errors:
                        for error in errors:
                            st.error(error)
                    else:
                        standardized = standardize_historical_results(raw)
                        with st.spinner("Running walk-forward backtest..."):
                            result = run_walk_forward_backtest(
                                standardized,
                                initial_train_end_date=initial_train_end_date.isoformat(),
                                test_window=test_window,
                                step_size=step_size,
                                min_train_matches=int(min_train_matches),
                            )
                        st.success(f"Backtest complete. Predictions: {len(result['predictions'])}")
                        st.rerun()
                except Exception as exc:
                    st.error(f"Could not run backtest: {exc}")
    with wc_col:
        if st.button("Run World Cup sanity check"):
            if not historical_exists:
                st.error("World Cup sanity check is unavailable because historical training data is not included in this deployment.")
            else:
                try:
                    raw = load_historical_results(HISTORICAL_RESULTS_PATH)
                    standardized = standardize_historical_results(raw)
                    with st.spinner("Running World Cup sanity check..."):
                        result = run_world_cup_backtest(standardized)
                    st.success(f"World Cup check complete. Predictions: {len(result['predictions'])}")
                    if result["predictions"].empty:
                        st.warning("No World Cup predictions were created. This is expected if the historical file has too few World Cup rows.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not run World Cup sanity check: {exc}")
    with full_col:
        if st.button("Run full model validation"):
            if not historical_exists:
                st.error("Full validation is unavailable because historical training data is not included in this deployment.")
            else:
                try:
                    raw = load_historical_results(HISTORICAL_RESULTS_PATH)
                    warnings, errors = validate_historical_results(raw)
                    for warning in warnings:
                        st.warning(warning)
                    if errors:
                        for error in errors:
                            st.error(error)
                    else:
                        standardized = standardize_historical_results(raw)
                        with st.spinner("Running full walk-forward validation..."):
                            result = run_full_walk_forward_backtest(
                                standardized,
                                initial_train_end_date=None,
                                test_window_months=12,
                                step_months=12,
                                min_train_matches=int(min_train_matches),
                                model_variant="best_available",
                            )
                        if result.get("status") == "validation_error":
                            st.error(result.get("error", "Full validation could not be completed."))
                        else:
                            comparison = result.get("market_comparison", pd.DataFrame())
                            if comparison.empty:
                                st.warning("Full model validation ran, but market comparison cannot be calculated because historical market odds are not available.")
                            else:
                                st.success("Full model validation complete, including market comparison.")
                            st.caption(f"Predictions: {len(result.get('predictions', pd.DataFrame()))}")
                            st.rerun()
                except Exception as exc:
                    st.error(f"Could not run full model validation: {exc}")

    variant_label = st.radio("Model variant", ["Baseline model", "Draw-context model"], horizontal=True)
    predictions_path = BACKTEST_PREDICTIONS_WITH_DRAW_FEATURES_PATH if variant_label == "Draw-context model" else BACKTEST_PREDICTIONS_PATH
    summary_path = BACKTEST_SUMMARY_WITH_DRAW_FEATURES_PATH if variant_label == "Draw-context model" else BACKTEST_SUMMARY_PATH
    predictions_df = _load_optional_csv(predictions_path)
    summary_df = _load_optional_csv(summary_path)
    segment_df = _load_optional_csv(BACKTEST_BY_SEGMENT_PATH)
    draw_df = _load_optional_csv(BACKTEST_DRAW_CALIBRATION_PATH)
    calibration_df = _load_optional_csv(BACKTEST_CALIBRATION_BINS_PATH)
    if predictions_df.empty:
        if summary_df.empty:
            empty_state(f"No {variant_label.lower()} backtest results yet.")
        else:
            st.warning(
                "Backtest ran, but no prediction rows were created. Check the fold summary below; "
                "this usually means too little training data, no rows in the selected test windows, "
                "or invalid historical rows."
            )
            st.dataframe(summary_df, width="stretch", hide_index=True)
        render_world_cup_sanity_check_results()
        return

    overall = segment_df[(segment_df["segment_name"] == "Overall") & (segment_df["segment_value"] == "All")].head(1)
    overall_row = overall.iloc[0] if not overall.empty else pd.Series(dtype="object")
    st.subheader("Overall KPI")
    kpi_cols = st.columns(6)
    kpi_values = [
        ("Accuracy", format_percentage(overall_row.get("accuracy", 0))),
        ("Log loss", f"{overall_row.get('log_loss', 0):.3f}"),
        ("Brier score", f"{overall_row.get('brier_score', 0):.3f}"),
        ("ECE", f"{overall_row.get('ece', 0):.3f}"),
        ("Draw calibration gap", format_percentage(overall_row.get("draw_calibration_gap", 0), decimals=2)),
        ("Matches", str(int(overall_row.get("match_count", len(predictions_df))))),
    ]
    for col, (label, value) in zip(kpi_cols, kpi_values):
        with col:
            metric_card(label, value, model_metric_explanation(label))
    small_sample_warning(int(overall_row.get("match_count", len(predictions_df))), threshold=100)
    st.markdown(f"Draw gap badge: {calibration_gap_badge(overall_row.get('draw_calibration_gap', 0))}", unsafe_allow_html=True)

    st.subheader("Fold performance")
    metric = st.selectbox("Fold metric", ["accuracy", "log_loss", "brier_score", "ece"])
    render_chart(backtest_metric_by_fold_chart(summary_df, metric))
    st.dataframe(summary_df, width="stretch", hide_index=True)

    st.subheader("Segment performance")
    if segment_df.empty:
        empty_state("No segment metrics available.")
    else:
        render_chart(segment_metric_chart(segment_df, "accuracy", "tournament_category"))
        st.dataframe(segment_df, width="stretch", hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Draw calibration")
        render_chart(draw_calibration_chart(draw_df))
        st.dataframe(draw_df, width="stretch", hide_index=True)
    with c2:
        st.subheader("Confidence calibration")
        render_chart(confidence_calibration_chart(calibration_df))
        st.dataframe(calibration_df, width="stretch", hide_index=True)

    st.subheader("Report")
    if BACKTEST_REPORT_PATH.exists():
        st.caption(str(BACKTEST_REPORT_PATH))
        st.markdown(BACKTEST_REPORT_PATH.read_text())
    else:
        empty_state("No backtest report file yet.")

    comparison_df = _load_optional_csv(DRAW_FEATURE_COMPARISON_PATH)
    if not comparison_df.empty:
        st.subheader("Baseline vs draw-context comparison")
        st.caption("Improvement means lower log loss/Brier/ECE. Draw calibration gap closer to zero is better.")
        render_chart(draw_feature_comparison_chart(comparison_df, "log_loss"))
        st.dataframe(comparison_df, width="stretch", hide_index=True)

    render_world_cup_sanity_check_results()


def page_draw_hypothesis(df: pd.DataFrame) -> None:
    st.title("Draw Hypothesis")
    st.write(
        "We test whether group-stage and major tournament contexts are associated with higher draw probabilities, "
        "especially when one or both teams can live with a draw."
    )
    st.warning("Draw-context features are tested empirically. The app does not add a manual draw bonus unless model validation supports it.")

    st.subheader("Data availability")
    historical_exists = HISTORICAL_RESULTS_PATH.exists()
    raw_historical = pd.DataFrame()
    standardized = pd.DataFrame()
    warnings = []
    errors = []
    if historical_exists:
        try:
            raw_historical = load_historical_results(HISTORICAL_RESULTS_PATH)
            warnings, errors = validate_historical_results(raw_historical)
            standardized = standardize_historical_results(raw_historical)
        except Exception as exc:
            errors = [str(exc)]
    group_columns = {"group", "stage", "matchday", "group_matchday"}
    available_group_cols = group_columns.intersection(set(raw_historical.columns)) if not raw_historical.empty else set()
    group_metadata_status = "No"
    if available_group_cols:
        group_metadata_status = "Partial" if len(available_group_cols) < 2 else "Yes"
    major_count = 0
    world_cup_count = 0
    group_count = 0
    if not raw_historical.empty:
        from features import categorize_tournament

        categories = raw_historical.get("tournament", pd.Series(["Unknown"] * len(raw_historical))).map(categorize_tournament)
        major_count = int(categories.isin({"world_cup", "euro", "copa_america", "afcon", "asian_cup", "gold_cup"}).sum())
        world_cup_count = int((categories == "world_cup").sum())
        if "group" in raw_historical.columns:
            group_count = int(raw_historical["group"].notna().sum())
        elif "stage" in raw_historical.columns:
            group_count = int(raw_historical["stage"].astype(str).str.lower().str.contains("group", na=False).sum())
    cols = st.columns(5)
    cols[0].metric("Historical data", "Yes" if historical_exists else "No")
    cols[1].metric("Group metadata", group_metadata_status)
    cols[2].metric("Matches", str(len(raw_historical)))
    cols[3].metric("Major tournament", str(major_count))
    cols[4].metric("World Cup", str(world_cup_count))
    st.caption(f"Matches with group-stage metadata: {group_count}")
    for warning in warnings:
        st.warning(warning)
    for error in errors:
        st.error(error)

    st.subheader("Run draw hypothesis analysis")
    if st.button("Run draw hypothesis analysis"):
        if not historical_exists:
            st.error("Draw hypothesis analysis is unavailable because historical training data is not included in this deployment.")
        elif errors:
            st.error("Historical data must be valid before running the analysis.")
        else:
            try:
                with st.spinner("Running draw hypothesis analysis..."):
                    result = run_draw_hypothesis_analysis(standardized)
                st.success(f"Draw hypothesis analysis complete. Segments: {len(result['segments'])}")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not run draw hypothesis analysis: {exc}")

    summary_df = _load_optional_csv(DRAW_HYPOTHESIS_SUMMARY_PATH)
    segment_df = _load_optional_csv(DRAW_HYPOTHESIS_BY_SEGMENT_PATH)
    comparison_df = _load_optional_csv(DRAW_FEATURE_COMPARISON_PATH)
    if not summary_df.empty:
        draw_rate = summary_df[summary_df["metric"] == "overall_draw_rate"]["value"].head(1)
        group_rate = summary_df[summary_df["metric"] == "group_metadata_available_rate"]["value"].head(1)
        draw_hypothesis_summary_card(
            int(summary_df["match_count"].max()),
            float(draw_rate.iloc[0]) if not draw_rate.empty else 0,
            float(group_rate.iloc[0]) if not group_rate.empty else None,
        )

    st.subheader("Draw-rate by segment")
    if segment_df.empty:
        empty_state("No draw hypothesis results yet.")
    else:
        render_chart(draw_rate_by_segment_chart(segment_df))
        st.dataframe(segment_df, width="stretch", hide_index=True)
        small_sample_caveat(int(segment_df["match_count"].max()))

    st.subheader("Baseline vs draw-context model comparison")
    c1, c2, c3, c4 = st.columns(4)
    initial_train_end_date = c1.date_input("Initial train end date", value=pd.Timestamp("2014-01-01").date(), key="draw_cmp_start")
    test_window = c2.text_input("Test window", value="365D", key="draw_cmp_window")
    step_size = c3.text_input("Step size", value="365D", key="draw_cmp_step")
    min_train_matches = c4.number_input("Min train matches", min_value=30, value=1000, step=100, key="draw_cmp_min")
    if st.button("Compare baseline vs draw-context model"):
        if not historical_exists:
            st.error("Model comparison is unavailable because historical training data is not included in this deployment.")
        elif errors:
            st.error("Historical data must be valid before running model comparison.")
        else:
            try:
                with st.spinner("Running baseline and draw-context backtests..."):
                    result = compare_baseline_vs_draw_context_model(
                        standardized,
                        {
                            "initial_train_end_date": initial_train_end_date.isoformat(),
                            "test_window": test_window,
                            "step_size": step_size,
                            "min_train_matches": int(min_train_matches),
                        },
                    )
                st.success(f"Comparison complete. Rows: {len(result['comparison'])}")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not compare models: {exc}")
    if comparison_df.empty:
        empty_state("No draw-feature comparison yet.")
    else:
        metric = st.selectbox("Comparison metric", ["log_loss", "brier_score", "draw_calibration_gap", "ece"], key="draw_cmp_metric")
        render_chart(draw_feature_comparison_chart(comparison_df, metric))
        st.dataframe(comparison_df, width="stretch", hide_index=True)
        draw_context_decision_card(get_latest_draw_context_status())

    st.subheader("Draw-context examples")
    example_df = df.copy()
    if "draw_context_score" not in example_df.columns:
        example_df = add_draw_context_features(example_df)
    render_chart(draw_context_score_distribution_chart(example_df))
    columns = [
        "home_team",
        "away_team",
        "draw_context_score",
        "draw_context_label",
        "both_teams_draw_satisfied",
        "one_team_must_win",
        "model_draw_prob",
        "market_draw_prob",
    ]
    available_columns = [column for column in columns if column in example_df.columns]
    display_examples = example_df[available_columns].copy()
    if "draw_context_score" in display_examples.columns and "draw_context_label" in display_examples.columns:
        display_examples["draw_context"] = display_examples.apply(
            lambda row: draw_context_score_badge(row["draw_context_score"], row["draw_context_label"]),
            axis=1,
        )
    st.dataframe(display_examples, width="stretch", hide_index=True)
    if DRAW_HYPOTHESIS_REPORT_PATH.exists():
        st.subheader("Report")
        st.caption(str(DRAW_HYPOTHESIS_REPORT_PATH))
        st.markdown(DRAW_HYPOTHESIS_REPORT_PATH.read_text())


def page_ensemble(df: pd.DataFrame) -> None:
    st.title("Ensemble")
    st.write(
        "The ensemble combines market probabilities with model probabilities. Bookmaker markets are strong baselines, "
        "while the model may add small contextual edges."
    )
    st.warning(
        "Do not use ensemble weights blindly. If historical market probabilities are unavailable, the ensemble cannot "
        "be fully validated and should be treated as experimental."
    )

    active_config = load_active_probability_source()
    comparison_df = _load_optional_csv(ENSEMBLE_COMPARISON_PATH)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Market probs", "Yes" if {"market_home_prob", "market_draw_prob", "market_away_prob"}.issubset(df.columns) else "No")
    c2.metric("Historical model", "Yes" if {"model_home_prob", "model_draw_prob", "model_away_prob"}.issubset(df.columns) else "No")
    c3.metric("Draw-context model", "Yes" if {"draw_model_home_prob", "draw_model_draw_prob", "draw_model_away_prob"}.issubset(df.columns) else "No")
    c4.metric("Backtest predictions", "Yes" if BACKTEST_PREDICTIONS_PATH.exists() else "No")
    historical_market = False
    if BACKTEST_PREDICTIONS_PATH.exists() and BACKTEST_PREDICTIONS_PATH.stat().st_size > 0:
        try:
            historical_market = {"market_home_prob", "market_draw_prob", "market_away_prob"}.issubset(pd.read_csv(BACKTEST_PREDICTIONS_PATH, nrows=1).columns)
        except Exception:
            historical_market = False
    c5.metric("Historical market probs", "Yes" if historical_market else "No")

    st.subheader("Current active probability source")
    cols = st.columns(4)
    cols[0].markdown(probability_source_badge(active_config.get("resolved_source", "market")), unsafe_allow_html=True)
    cols[1].metric("Market weight", format_percentage(active_config.get("w_market", 1.0)))
    cols[2].metric("Model weight", format_percentage(active_config.get("w_model", 0.0)))
    cols[3].caption(active_config.get("reason", "-"))

    st.subheader("Run ensemble comparison")
    if st.button("Run ensemble comparison"):
        result = run_ensemble_backtest_from_saved_predictions(BACKTEST_PREDICTIONS_PATH, PROCESSED_DATA_DIR)
        if result["status"] == "market_probabilities_missing":
            st.warning("Historical market probabilities are unavailable. Ensemble backtest cannot be fully evaluated against market. Use live/current ensemble only or add historical odds data later.")
        elif result["status"] != "ok":
            st.warning(f"Comparison status: {result['status']}")
        else:
            st.success("Ensemble comparison complete.")
        st.rerun()

    if comparison_df.empty:
        empty_state("No ensemble comparison results yet.")
    else:
        recommendation = select_best_probability_source(comparison_df)
        best_source_card(recommendation)
        if st.button("Use recommended probability source"):
            save_active_probability_source(
                {
                    "source": "best_validated",
                    "resolved_source": recommendation["recommended_source"],
                    "w_market": recommendation["w_market"],
                    "w_model": recommendation["w_model"],
                    "reason": recommendation["reason"],
                    "last_validated_at": pd.Timestamp.utcnow().isoformat(),
                }
            )
            st.session_state.probability_source = "best_validated"
            st.success("Recommended probability source saved.")
            st.rerun()
        metric = st.selectbox("Ensemble metric", ["log_loss", "brier_score", "ece", "draw_calibration_gap"])
        render_chart(ensemble_weight_metric_chart(comparison_df, metric))
        render_chart(probability_source_comparison_chart(comparison_df))
        st.dataframe(comparison_df, width="stretch", hide_index=True)

    st.subheader("Manual ensemble weights")
    w_market = st.slider("Market weight", min_value=0.0, max_value=1.0, value=float(st.session_state.ensemble_w_market), step=0.05)
    model_variant = st.selectbox("Model variant", ["historical_model", "draw_context_model"], index=0 if st.session_state.ensemble_model_variant == "historical_model" else 1)
    st.caption(ensemble_weight_badge(w_market, 1 - w_market))
    if st.button("Apply manual ensemble to current matches"):
        try:
            result = apply_ensemble_to_upcoming_matches(df, w_market=w_market, model_variant=model_variant)
            for warning in result.attrs.get("warnings", []):
                st.warning(warning)
            save_active_probability_source(
                {
                    "source": "ensemble",
                    "resolved_source": "ensemble",
                    "w_market": float(result["ensemble_w_market"].iloc[0]),
                    "w_model": float(result["ensemble_w_model"].iloc[0]),
                    "reason": "Manual ensemble weights applied to current matches.",
                }
            )
            st.session_state.probability_source = "ensemble"
            st.session_state.ensemble_w_market = w_market
            st.session_state.ensemble_model_variant = model_variant
            st.success("Manual ensemble probabilities saved and selected for Kelly recommendations.")
            st.rerun()
        except Exception as exc:
            st.error(f"Could not apply manual ensemble: {exc}")

    if ENSEMBLE_REPORT_PATH.exists():
        st.subheader("Report")
        st.caption(str(ENSEMBLE_REPORT_PATH))
        st.markdown(ENSEMBLE_REPORT_PATH.read_text())


def page_settings() -> None:
    st.title("Admin / Settings")
    st.info(
        "Normal users do not need this page. The app automatically uses the best available predictions and falls back to market probabilities when model files are unavailable."
    )
    st.subheader("Preferred bookmaker")
    bookmaker_options = {"Danske Spil": "danske_spil", "Best market": "best_market"}
    reverse_bookmaker_options = {value: key for key, value in bookmaker_options.items()}
    selected_bookmaker_label = st.radio(
        "Where do you usually bet?",
        list(bookmaker_options.keys()),
        index=list(bookmaker_options.keys()).index(
            reverse_bookmaker_options.get(st.session_state.preferred_bookmaker_mode, "Danske Spil")
        ),
        horizontal=True,
        help="Choose where you normally place bets. Recommendations will be based on this bookmaker first.",
        key="admin_preferred_bookmaker_mode",
    )
    selected_bookmaker_mode = bookmaker_options[selected_bookmaker_label]
    if st.session_state.preferred_bookmaker_mode != selected_bookmaker_mode:
        st.session_state.preferred_bookmaker_mode = selected_bookmaker_mode
        st.rerun()
    st.caption("Best market odds are still shown as comparison.")

    st.subheader("Data mode")
    mode_labels = {
        "Official fixtures": "official",
        "Sample/demo data": "sample",
        "Live odds data": "live",
    }
    current_label = next(
        label for label, value in mode_labels.items()
        if value == st.session_state.data_mode
    )
    selected_mode_label = st.radio(
        "Choose data source",
        list(mode_labels.keys()),
        index=list(mode_labels.keys()).index(current_label),
        horizontal=True,
    )
    st.session_state.data_mode = mode_labels[selected_mode_label]
    live_freshness = get_data_freshness(LIVE_PREDICTIONS_PATH)
    fixture_freshness = get_data_freshness(REFERENCE_FIXTURES_PATH)
    manual_odds_freshness = get_data_freshness(MANUAL_ODDS_PATH)
    processed_odds_freshness = get_data_freshness(PROCESSED_ODDS_PATH)
    odds_status = get_odds_source_status()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Current mode", st.session_state.data_mode.title())
    with c2:
        metric_card("Fixture rows", str(fixture_freshness["row_count"]))
    with c3:
        metric_card("Odds source", odds_status["active_odds_source"].title())
    with c4:
        metric_card("Latest odds rows", str(processed_odds_freshness["row_count"]))
    st.caption(fixture_provenance_text(st.session_state.data_mode))
    if st.session_state.data_mode == "sample":
        st.warning("Sample/demo data is for testing UI only and is not official World Cup fixture data.")
    if st.session_state.data_mode == "live" and not live_freshness["file_exists"]:
        st.warning("Live predictions are missing. The app will show no live matches until fixtures/odds are fetched; sample fallback is disabled.")
    st.subheader("Odds data")
    o1, o2, o3, o4 = st.columns(4)
    o1.metric("Active source", odds_status["active_odds_source"].title())
    o2.metric("API key", "Configured" if odds_status["has_api_key"] else "Missing")
    o3.metric("Manual CSV", "Valid" if odds_status["manual_odds_valid"] else ("Present" if odds_status["manual_odds_exists"] else "Missing"))
    o4.metric("Cached snapshot", "Available" if odds_status["cached_odds_exists"] else "Missing")
    st.caption(f"Sport key: {ODDS_API_SPORT_KEY} · regions: {ODDS_API_REGIONS} · markets: {ODDS_API_MARKETS}")
    st.caption(f"Manual odds file: {MANUAL_ODDS_PATH}")
    if odds_status["warning"]:
        st.warning(odds_status["warning"])
    elif odds_status["message"]:
        st.info(odds_status["message"])
    if odds_status["last_error"]:
        st.caption(f"Status detail: {odds_status['last_error']}")

    st.subheader("FIFA ranking diagnostics")
    rankings_df, ranking_warnings = load_fifa_rankings(FIFA_RANKINGS_PATH)
    coverage_df = _load_optional_csv(FIFA_FEATURE_COVERAGE_PATH)
    variant_df = _load_optional_csv(MODEL_VARIANT_COMPARISON_PATH)
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Ranking file", "Found" if FIFA_RANKINGS_PATH.exists() else "Missing")
    r2.metric("Ranking rows", str(len(rankings_df)))
    r3.metric("Teams covered", str(rankings_df["team_normalized"].nunique()) if not rankings_df.empty else "0")
    latest_ranking_date = "-" if rankings_df.empty else str(rankings_df["ranking_date"].max().date())
    r4.metric("Latest ranking", latest_ranking_date)
    st.caption(f"Ranking file: {FIFA_RANKINGS_PATH}")
    for warning in ranking_warnings[:5]:
        st.warning(warning)
    if not coverage_df.empty:
        missing_rate = (
            coverage_df["matches_missing_ranking"].sum()
            / (coverage_df["matches_missing_ranking"].sum() + coverage_df["matches_with_ranking"].sum())
        )
        st.caption(f"FIFA ranking feature missing rate in latest run: {missing_rate:.1%}")
    if not variant_df.empty:
        selected = variant_df[variant_df["selected"].fillna(False).astype(bool)]
        selected_label = "-" if selected.empty else str(selected.iloc[0]["model_variant"])
        st.caption(f"Latest ablation selected: {selected_label}")
        with st.expander("FIFA / Elo variant comparison"):
            st.dataframe(variant_df, width="stretch", hide_index=True)

    refresh_odds_from_ui("Refresh odds now", key="admin_refresh_odds_now")
    if st.session_state.get("last_odds_refresh_result"):
        with st.expander("Provider metadata", expanded=True):
            st.json(st.session_state["last_odds_refresh_result"])

    st.divider()
    st.subheader("Model source")
    model_source_label = st.radio(
        "Choose model probability source",
        ["Market only", "Historical model", "Historical model if available"],
        index={"market_only": 0, "historical_model": 1, "historical_model_if_available": 2}.get(
            st.session_state.model_source, 2
        ),
        horizontal=True,
    )
    st.session_state.model_source = {
        "Market only": "market_only",
        "Historical model": "historical_model",
        "Historical model if available": "historical_model_if_available",
    }[model_source_label]
    if st.session_state.model_source == "market_only":
        st.info("Using market-implied probabilities as model probabilities.")

    st.subheader("Model feature settings")
    st.session_state.use_draw_context_features = st.checkbox(
        "Use draw-context features for future training/apply model runs",
        value=bool(st.session_state.use_draw_context_features),
    )
    draw_status = get_latest_draw_context_status()
    if draw_status["comparison_exists"]:
        if draw_status["recommended"]:
            st.success("Draw-context model appears beneficial based on latest comparison.")
        else:
            st.warning("Draw-context model is not currently recommended based on latest comparison.")
        st.caption(draw_status["reason"])
    else:
        st.info("Run the Draw Hypothesis comparison before enabling draw-context features for model training.")

    st.subheader("Probability source for recommendations")
    source_labels = {
        "Best validated source": "best_validated",
        "Market only": "market",
        "Historical model": "historical_model",
        "Draw-context model": "draw_context_model",
        "Ensemble": "ensemble",
    }
    reverse_source_labels = {value: key for key, value in source_labels.items()}
    selected_probability_source = st.selectbox(
        "Kelly probability source",
        list(source_labels.keys()),
        index=list(source_labels.keys()).index(reverse_source_labels.get(st.session_state.probability_source, "Best validated source")),
    )
    st.session_state.probability_source = source_labels[selected_probability_source]
    if st.session_state.probability_source == "ensemble":
        st.session_state.ensemble_w_market = st.slider(
            "Market weight",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state.ensemble_w_market),
            step=0.05,
            key="settings_ensemble_w_market",
        )
        st.caption(ensemble_weight_badge(st.session_state.ensemble_w_market, 1 - st.session_state.ensemble_w_market))
    st.caption("Changing source recalculates active probabilities and Kelly recommendations on the next app run.")

    status = get_active_model_status()
    readiness = get_model_readiness(predictions_exist=MODEL_PREDICTIONS_PATH.exists() or LIVE_PREDICTIONS_WITH_MODEL_PATH.exists())
    cols = st.columns(4)
    with cols[0]:
        metric_card("Model status", readiness["status"].replace("_", " ").title())
    with cols[1]:
        metric_card("Production ready", "Yes" if readiness["is_usable_as_best_available"] else "No")
    with cols[2]:
        metric_card("Demo model", "Yes" if readiness["status"] == "demo_model" else "No")
    with cols[3]:
        metric_card("Accuracy", "-" if status["accuracy"] is None else format_percentage(status["accuracy"]))
    log_loss_text = "-" if status["log_loss"] is None else f"{status['log_loss']:.3f}"
    brier_text = "-" if status["brier_score"] is None else f"{status['brier_score']:.3f}"
    draw_actual = "-" if status["draw_rate_actual"] is None else format_percentage(status["draw_rate_actual"])
    draw_predicted = "-" if status["draw_rate_predicted"] is None else format_percentage(status["draw_rate_predicted"])
    st.info(readiness["normal_user_message"])
    if not readiness["historical_csv_exists"]:
        st.warning(readiness["admin_training_message"])
    diagnostic_rows = [
        ("Model artifact", "Found" if readiness["model_file_exists"] else "Missing"),
        ("Metadata", "Found" if readiness["metadata_exists"] else "Missing"),
        ("Feature columns", "Found" if readiness["feature_columns_exists"] else "Missing"),
        ("Training rows", str(readiness["training_rows"])),
        ("Test rows", str(readiness["test_rows"])),
        ("Feature count", str(readiness["feature_count"])),
        ("Training years", f"{readiness.get('training_year_span', 0):.1f}"),
        ("Training data source", str(readiness["training_data_source"] or "-")),
        ("Qualifiers", "Yes" if readiness.get("includes_qualifiers") else "Missing"),
        ("World Cup/major tournaments", "Yes" if readiness.get("includes_world_cup_or_major_tournaments") else "Missing"),
        ("Elo features", "Yes" if readiness.get("includes_elo_features") else "Missing"),
        ("Form features", "Yes" if readiness.get("includes_form_features") else "Missing"),
        ("Neutral venue", "Yes" if readiness.get("includes_neutral_venue") else "Missing"),
        ("Production-ready", "Yes" if readiness["is_usable_as_best_available"] else "No"),
        ("Demo model", "Yes" if readiness["status"] == "demo_model" else "No"),
        ("Historical CSV", "Found" if readiness["historical_csv_exists"] else "Missing"),
        ("Retraining", "Available" if readiness["retraining_available"] else "Disabled"),
    ]
    st.dataframe(pd.DataFrame(diagnostic_rows, columns=["Check", "Status"]), width="stretch", hide_index=True)
    for warning in readiness["warnings"]:
        st.warning(warning)
    st.caption(
        f"Model version: {status['model_version'] or '-'} | Trained at: {status['trained_at'] or '-'} | "
        f"Training rows: {status['number_of_training_rows']} | Test rows: {status['number_of_test_rows']} | "
        f"Log loss: {log_loss_text} | "
        f"Brier: {brier_text} | Draw actual/predicted: {draw_actual} / {draw_predicted}"
    )

    train_col, apply_col = st.columns(2)
    with train_col:
        if st.button("Train/update historical model", disabled=not readiness["retraining_available"]):
            try:
                raw = load_historical_results(HISTORICAL_RESULTS_PATH)
                hist_warnings, hist_errors = validate_historical_results(raw)
                for warning in hist_warnings:
                    st.warning(warning)
                if hist_errors:
                    for error in hist_errors:
                        st.error(error)
                else:
                    standardized = standardize_historical_results(raw)
                    training_df = build_training_dataset(
                        standardized,
                        include_draw_context_features=st.session_state.use_draw_context_features,
                    )
                    metadata = train_historical_model(
                        training_df,
                        include_draw_context_features=st.session_state.use_draw_context_features,
                    )
                    st.success(
                        f"Model trained. Accuracy: {format_percentage(metadata['metrics']['accuracy'])}, "
                        f"log loss: {metadata['metrics']['log_loss']:.3f}"
                    )
            except FileNotFoundError:
                st.error("Retraining is unavailable because historical training data is not included in this deployment.")
            except ValueError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Could not train model: {exc}")
    with apply_col:
        can_apply_model = readiness["is_usable_as_best_available"] or st.session_state.data_mode == "sample"
        if st.button("Apply pre-trained model to current matches", disabled=not can_apply_model):
            try:
                base_df, _, actual_mode = load_predictions_by_mode(st.session_state.data_mode, model_source="market_only")
                if HISTORICAL_RESULTS_PATH.exists():
                    raw = load_historical_results(HISTORICAL_RESULTS_PATH)
                    standardized = standardize_historical_results(raw)
                else:
                    standardized = _empty_historical_results()
                output_path = LIVE_PREDICTIONS_WITH_MODEL_PATH if actual_mode == "live" else MODEL_PREDICTIONS_PATH
                _, model_warnings = predict_upcoming_matches(
                    base_df,
                    standardized,
                    output_path=output_path,
                    include_draw_context_features=st.session_state.use_draw_context_features,
                )
                for warning in model_warnings:
                    st.warning(warning)
                st.session_state.model_source = "historical_model"
                st.success("Historical model probabilities applied to current matches.")
                st.rerun()
            except FileNotFoundError:
                st.error("Pre-trained model artifacts are missing. The app will use market probabilities as fallback.")
            except Exception as exc:
                st.error(f"Could not apply model: {exc}")

    odds_snapshot = load_odds_snapshot()
    st.subheader("Odds data")
    if odds_snapshot.empty:
        empty_state("No odds snapshots stored yet.")
    else:
        complete_events = 0
        for _, event_df in odds_snapshot.groupby("event_id"):
            outcomes = set(event_df["outcome_name"].astype(str).str.lower())
            if "draw" in outcomes or "tie" in outcomes:
                complete_events += 1
        preferred_count = odds_snapshot[
            odds_snapshot["bookmaker_title"].isin(PREFERRED_BOOKMAKER_NAMES)
            | odds_snapshot["bookmaker_key"].isin(PREFERRED_BOOKMAKER_NAMES)
        ]["event_id"].nunique()
        cols = st.columns(5)
        with cols[0]:
            metric_card("Odds rows", str(len(odds_snapshot)))
        with cols[1]:
            metric_card("Bookmakers", str(odds_snapshot["bookmaker_title"].nunique()))
        with cols[2]:
            metric_card("Events", str(odds_snapshot["event_id"].nunique()))
        with cols[3]:
            metric_card("DS events", str(preferred_count))
        with cols[4]:
            metric_card("Draw odds events", str(complete_events), odds_snapshot["fetched_at"].max())

    st.divider()
    st.subheader("Kelly")
    profile_name = st.selectbox(
        "Kelly profile",
        list(STAKING_PROFILES.keys()),
        index=list(STAKING_PROFILES.keys()).index(st.session_state.kelly_profile_name),
    )
    if profile_name != st.session_state.kelly_profile_name:
        st.session_state.kelly_profile_name = profile_name
        st.session_state.staking_profile = get_staking_profile(profile_name)
        st.rerun()

    profile = current_profile()
    st.subheader("Manual override")
    profile["fractional_kelly_multiplier"] = st.number_input(
        "Fractional Kelly", min_value=0.0, max_value=1.0, value=float(profile["fractional_kelly_multiplier"]), step=0.01
    )
    profile["max_stake_pct_of_bankroll"] = st.number_input(
        "Max stake %", min_value=0.001, max_value=0.20, value=float(profile["max_stake_pct_of_bankroll"]), step=0.005
    )
    profile["min_edge_threshold"] = st.number_input(
        "Minimum edge %", min_value=0.0, max_value=0.50, value=float(profile["min_edge_threshold"]), step=0.005
    )
    profile["min_stake_pct_threshold"] = st.number_input(
        "Minimum stake %", min_value=0.0, max_value=0.20, value=float(profile["min_stake_pct_threshold"]), step=0.001
    )
    setting_errors = validate_staking_profile(profile)
    if setting_errors:
        for error in setting_errors:
            st.error(error)
    else:
        st.session_state.staking_profile = profile
    st.session_state.preferred_bookmaker = st.text_input("Preferred bookmaker", st.session_state.preferred_bookmaker)
    st.info("Default Standard profile uses 0.25 Kelly, max stake 2.5%, minimum edge 2.5%, and minimum stake 0.25%.")


def page_about(df: pd.DataFrame) -> None:
    st.title("About")
    st.subheader("What this app does")
    st.write("A World Cup prediction and staking dashboard for comparing model/market probabilities, Danske Spil odds, best market odds, edge and Kelly-based stake suggestions.")
    st.subheader("Current MVP/live odds limitations")
    st.write("Sample mode uses static data. Live mode can ingest odds when an API key is configured. If no applied model predictions are available, model probabilities fall back to market-implied probabilities.")
    st.subheader("Kelly and edge")
    st.write("Edge is `active_probability * odds - 1`. Kelly stake uses current bankroll, fractional Kelly and stake caps to reduce risk.")
    st.subheader("Danske Spil vs best market")
    st.write("The app keeps Danske Spil recommendations separate from best-market recommendations, so value elsewhere is visible even if DS is not playable.")
    st.subheader("Draw-context")
    st.write("Draw-context is a contextual signal only. It is not draw probability, not a recommendation and should never trigger a draw bet by itself.")
    st.subheader("What comes next")
    st.write("Historical model, backtest, draw hypothesis modelling and a market-aware ensemble.")
    st.subheader("Responsible staking")
    st.write("This tool is for analysis and tracking. It does not guarantee profit.")
    st.warning(
        "Bankroll and bet log are stored locally as CSV/JSON in this MVP. On Streamlit Community Cloud "
        "this is suitable for testing only. For production, use persistent storage such as Supabase, "
        "Postgres, SQLite with mounted storage, or another database."
    )
    st.subheader("Health check")
    st.caption("App status: data mode, matches loaded, active probability source, model, ensemble, bankroll, bet log, odds and backtest freshness.")
    st.dataframe(app_health_rows(df), width="stretch", hide_index=True)
    with st.expander("If the app feels incomplete"):
        st.write(
            "Check whether official, sample or live mode is active, whether the fixture reference is complete, "
            "whether live odds have been fetched, whether the historical model has been trained and applied, "
            "and whether an ensemble has been selected as the active probability source."
        )


init_session_state()
show_sidebar()
df, validation_warnings, validation_errors = load_enriched_predictions()
show_validation_messages(validation_warnings, validation_errors)

current_page = st.session_state.current_page

if current_page == "Match Overview":
    page_overview(df)
elif current_page == "Match Archive":
    page_match_archive(df)
elif current_page == "Betting Center":
    page_betting_center(df)
elif current_page == "My Bets":
    page_my_bets()
elif current_page == "Match Detail":
    page_match_detail(df)
elif current_page == "Model Performance":
    page_model_performance()
elif current_page == "Model & Data":
    page_model_data(df)
elif current_page == "Settings":
    page_user_settings()
elif current_page == "Advanced / Admin":
    page_advanced_admin(df)
else:
    page_overview(df)
