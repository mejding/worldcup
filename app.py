import pandas as pd
import streamlit as st

from backtest import compare_baseline_vs_draw_context_model, run_walk_forward_backtest, run_world_cup_backtest
from bankroll import load_bankroll_history, load_bankroll_state, reset_bankroll, update_bankroll
from bet_log import add_bet, calculate_bet_summary, load_bet_log, reset_bet_settlement, settle_bet
from charts import (
    backtest_metric_by_fold_chart,
    bankroll_history_chart,
    confidence_calibration_chart,
    draw_calibration_chart,
    draw_context_score_distribution_chart,
    draw_feature_comparison_chart,
    draw_rate_by_segment_chart,
    probability_comparison_chart,
    profit_loss_by_bookmaker_chart,
    profit_loss_by_outcome_chart,
    render_chart,
    segment_metric_chart,
)
from components import (
    calibration_gap_badge,
    draw_context_badge,
    draw_context_card,
    draw_context_decision_card,
    draw_context_score_badge,
    draw_hypothesis_summary_card,
    empty_state,
    format_dkk,
    format_odds,
    format_percentage,
    metric_card,
    metric_row,
    model_metric_explanation,
    odds_comparison_table,
    recommendation_card,
    small_sample_caveat,
    small_sample_warning,
    status_badge,
)
from backtest_paths import (
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
)
from config import (
    DEFAULT_PROFILE_NAME,
    HISTORICAL_RESULTS_PATH,
    LIVE_PREDICTIONS_PATH,
    LIVE_PREDICTIONS_WITH_MODEL_PATH,
    MODEL_PREDICTIONS_PATH,
    MODEL_SOURCE,
    ODDS_API_MARKET,
    ODDS_API_ODDS_FORMAT,
    ODDS_API_REGION,
    ODDS_API_SPORT_KEY,
    ODDS_SNAPSHOT_PATH,
    PREFERRED_BOOKMAKER,
    PREFERRED_BOOKMAKER_NAMES,
    SAMPLE_PREDICTIONS_PATH,
    STAKING_PROFILES,
    TRAINING_DATASET_PATH,
    get_staking_profile,
    get_secret_or_env,
    validate_staking_profile,
)
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
from fetch_fixtures import fetch_worldcup_fixtures
from fetch_odds import append_odds_snapshot, fetch_odds_from_api
from features import build_training_dataset
from historical_data import load_historical_results, standardize_historical_results, validate_historical_results
from kelly import calculate_final_stake_fraction, calculate_suggested_stake
from live_data_pipeline import build_live_predictions
from model_registry import get_active_model_status, get_latest_backtest_status, get_latest_draw_context_status
from odds_utils import calculate_edge
from predict_model import predict_upcoming_matches
from recommendations import add_recommendations
from train_model import train_historical_model


st.set_page_config(page_title="VM 2026 Prediction & Kelly", page_icon="⚽", layout="wide")
ensure_runtime_data_files()


def init_session_state() -> None:
    if "page" not in st.session_state:
        st.session_state.page = "Overview"
    if "kelly_profile_name" not in st.session_state:
        st.session_state.kelly_profile_name = DEFAULT_PROFILE_NAME
    if "staking_profile" not in st.session_state:
        st.session_state.staking_profile = get_staking_profile(DEFAULT_PROFILE_NAME)
    if "preferred_bookmaker" not in st.session_state:
        st.session_state.preferred_bookmaker = PREFERRED_BOOKMAKER
    if "selected_match_id" not in st.session_state:
        st.session_state.selected_match_id = None
    if "data_mode" not in st.session_state:
        st.session_state.data_mode = "sample"
    if "active_data_mode" not in st.session_state:
        st.session_state.active_data_mode = "sample"
    if "model_source" not in st.session_state:
        st.session_state.model_source = MODEL_SOURCE
    if "active_model_source" not in st.session_state:
        st.session_state.active_model_source = "market_only"
    if "use_draw_context_features" not in st.session_state:
        st.session_state.use_draw_context_features = False


def current_profile() -> dict:
    return st.session_state.staking_profile.copy()


def load_enriched_predictions() -> tuple[pd.DataFrame, list[str], list[str]]:
    try:
        predictions, mode_warnings, actual_mode = load_predictions_by_mode(
            st.session_state.data_mode,
            model_source=st.session_state.model_source,
        )
        st.session_state.active_data_mode = actual_mode
        st.session_state.active_model_source = (
            "market_only"
            if st.session_state.model_source == "market_only"
            or any("market probabilities" in warning.lower() for warning in mode_warnings)
            else "historical_model"
        )
    except FileNotFoundError as exc:
        return pd.DataFrame(), [], [str(exc)]
    except pd.errors.EmptyDataError:
        return pd.DataFrame(), [], ["Sample predictions file is empty or malformed."]

    warnings, errors = validate_predictions(predictions)
    warnings = mode_warnings + warnings
    if errors:
        return predictions, warnings, errors
    bankroll = load_bankroll_state()["current_bankroll"]
    enriched = add_recommendations(predictions, bankroll, current_profile())
    return enriched.rename(columns={"status": "recommendation_status"}), warnings, errors


def probability_for_outcome(row, outcome: str) -> float:
    return float(row[f"model_{outcome.lower()}_prob"])


def match_label(row) -> str:
    return f"{row['home_team']} vs {row['away_team']}"


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
            bookmaker=bookmaker,
            outcome=outcome,
            odds=row[f"recommended_odds_{market}"],
            model_probability=probability_for_outcome(row, outcome_key),
            edge=row[f"recommended_edge_{market}"],
            full_kelly=row[f"recommended_full_kelly_{market}"],
            fractional_kelly=row[f"recommended_fractional_kelly_{market}"],
            stake_dkk=row[f"recommended_stake_{market}"],
        )
        st.success(f"Bet tilføjet. Bet ID: {bet['bet_id']}")
    except ValueError as exc:
        st.error(str(exc))


def outcome_kelly_table(row) -> pd.DataFrame:
    rows = []
    profile = current_profile()
    bankroll = load_bankroll_state()["current_bankroll"]
    for outcome_key, outcome_name in [("home", "Home"), ("draw", "Draw"), ("away", "Away")]:
        model_probability = float(row[f"model_{outcome_key}_prob"])
        ds_odds = float(row[f"ds_{outcome_key}_odds"])
        best_odds = float(row[f"best_{outcome_key}_odds"])
        ds_kelly = calculate_final_stake_fraction(
            model_probability,
            ds_odds,
            profile["fractional_kelly_multiplier"],
            profile["max_stake_pct_of_bankroll"],
        )
        best_kelly = calculate_final_stake_fraction(
            model_probability,
            best_odds,
            profile["fractional_kelly_multiplier"],
            profile["max_stake_pct_of_bankroll"],
        )
        rows.append(
            {
                "Outcome": outcome_name,
                "Model probability": model_probability,
                "DS odds": ds_odds,
                "DS edge": calculate_edge(model_probability, ds_odds),
                "DS full Kelly": ds_kelly["full_kelly"],
                "DS fractional Kelly": ds_kelly["fractional_kelly"],
                "DS suggested stake": calculate_suggested_stake(bankroll, ds_kelly["final_stake_fraction"]),
                "Best odds": best_odds,
                "Best bookmaker": row[f"best_{outcome_key}_bookmaker"],
                "Best edge": calculate_edge(model_probability, best_odds),
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
        color = "green" if float(value) > 0 else "red"
        return f"color: {color}"

    return df.style.format(
        {
            "Model probability": "{:.1%}",
            "DS odds": "{:.2f}",
            "DS edge": "{:.2%}",
            "DS full Kelly": "{:.2%}",
            "DS fractional Kelly": "{:.2%}",
            "DS suggested stake": "{:.2f} DKK",
            "Best odds": "{:.2f}",
            "Best edge": "{:.2%}",
            "Best full Kelly": "{:.2%}",
            "Best fractional Kelly": "{:.2%}",
            "Best suggested stake": "{:.2f} DKK",
        }
    ).map(color_edge, subset=edge_columns)


def format_overview_table(df: pd.DataFrame) -> pd.DataFrame:
    display_df = df.copy()
    display_df["match"] = display_df["home_team"] + " vs " + display_df["away_team"]
    display_df = display_df.rename(
        columns={
            "model_home_prob": "Model H",
            "model_draw_prob": "Model U",
            "model_away_prob": "Model A",
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
    for column in ["Model H", "Model U", "Model A"]:
        display_df[column] = display_df[column].map(format_percentage)
    for column in ["DS H", "DS U", "DS A", "Best H", "Best U", "Best A"]:
        display_df[column] = display_df[column].map(format_odds)
    for column in ["Stake DS", "Stake Best"]:
        display_df[column] = display_df[column].map(format_dkk)
    return display_df[
        [
            "kickoff_time",
            "group",
            "matchday",
            "match",
            "Model H",
            "Model U",
            "Model A",
            "DS H",
            "DS U",
            "DS A",
            "Best H",
            "Best U",
            "Best A",
            "DS rec",
            "Best rec",
            "Stake DS",
            "Stake Best",
            "Status",
            "Draw context",
        ]
    ]


def show_sidebar() -> None:
    state = load_bankroll_state()
    net = state["current_bankroll"] - state["starting_bankroll"]
    ret = net / state["starting_bankroll"] if state["starting_bankroll"] else 0
    st.sidebar.title("Navigation")
    st.session_state.page = st.sidebar.radio(
        "Side",
        ["Overview", "Match Detail", "Bankroll", "Bet Log", "Analytics", "Backtest & Metrics", "Draw Hypothesis", "Settings", "About"],
        index=["Overview", "Match Detail", "Bankroll", "Bet Log", "Analytics", "Backtest & Metrics", "Draw Hypothesis", "Settings", "About"].index(
            st.session_state.page
        ) if st.session_state.page in ["Overview", "Match Detail", "Bankroll", "Bet Log", "Analytics", "Backtest & Metrics", "Draw Hypothesis", "Settings", "About"] else 0,
    )
    st.sidebar.divider()
    st.sidebar.markdown("**Bankroll**")
    st.sidebar.metric("Current bankroll", format_dkk(state["current_bankroll"]))
    st.sidebar.caption(f"Return: {format_dkk(net)} / {format_percentage(ret)}")
    st.sidebar.divider()
    st.sidebar.markdown("**Staking profile**")
    st.sidebar.caption(f"Kelly profile: {st.session_state.kelly_profile_name}")
    profile = current_profile()
    st.sidebar.caption(f"Fractional Kelly: {format_percentage(profile['fractional_kelly_multiplier'])}")
    st.sidebar.caption(f"Max stake: {format_percentage(profile['max_stake_pct_of_bankroll'])}")
    st.sidebar.caption(f"Minimum edge: {format_percentage(profile['min_edge_threshold'])}")
    st.sidebar.caption(f"Minimum stake: {format_percentage(profile['min_stake_pct_threshold'])}")
    st.sidebar.divider()
    st.sidebar.markdown("**Data mode**")
    mode_path = LIVE_PREDICTIONS_PATH if st.session_state.active_data_mode == "live" else SAMPLE_PREDICTIONS_PATH
    freshness = get_data_freshness(mode_path)
    st.sidebar.caption(f"Mode: {st.session_state.active_data_mode.title()}")
    st.sidebar.caption(f"Model: {st.session_state.active_model_source.replace('_', ' ').title()}")
    st.sidebar.caption(f"Rows: {freshness['row_count']}")
    st.sidebar.caption(f"Updated: {freshness['last_modified'] or '-'}")
    if st.session_state.active_data_mode == "live":
        st.sidebar.caption(f"API key: {'configured' if get_secret_or_env('ODDS_API_KEY') else 'missing'}")


def show_validation_messages(warnings: list[str], errors: list[str]) -> None:
    for warning in warnings:
        st.warning(warning)
    for error in errors:
        st.error(error)
    if errors:
        st.stop()


def page_overview(df: pd.DataFrame) -> None:
    st.title("VM 2026 Prediction & Kelly")
    st.caption("Kommende kampe, odds, edge og anbefalet indsats")
    counts = df["recommendation_status"].value_counts()
    kpi_cols = st.columns(6)
    with kpi_cols[0]:
        metric_card("Current bankroll", format_dkk(load_bankroll_state()["current_bankroll"]))
    with kpi_cols[1]:
        metric_card("Matches loaded", str(len(df)), st.session_state.active_data_mode.title())
    with kpi_cols[2]:
        metric_card("Playable at DS", str(counts.get("Playable at Danske Spil", 0)))
    with kpi_cols[3]:
        metric_card("Better elsewhere", str(counts.get("Better elsewhere", 0)))
    with kpi_cols[4]:
        metric_card("No bet", str(counts.get("No bet", 0)))
    with kpi_cols[5]:
        metric_card("High draw-context", str((df["draw_context_label"] == "High").sum()))
    freshness = get_data_freshness(
        LIVE_PREDICTIONS_PATH if st.session_state.active_data_mode == "live" else SAMPLE_PREDICTIONS_PATH
    )
    st.caption(f"Last updated: {freshness['last_modified'] or 'not available'} | Rows loaded: {len(df)}")
    if st.session_state.active_data_mode == "live":
        st.info(
            "Live mode currently uses market-implied probabilities as model probabilities. "
            "Historical ML model probabilities will be added later."
        )
    st.caption(f"Model source: {st.session_state.active_model_source.replace('_', ' ').title()}")

    with st.expander("Filtre", expanded=True):
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
        filtered = filtered[filtered["recommendation_status"] != "No bet"]
    if positive_edge_only:
        filtered = filtered[(filtered["recommended_edge_ds"] > 0) | (filtered["recommended_edge_best"] > 0)]
    if draw_value_only:
        filtered = filtered[
            (filtered["recommended_outcome_ds"] == "Draw") | (filtered["recommended_outcome_best"] == "Draw")
        ]
    if high_draw_only:
        filtered = filtered[filtered["draw_context_label"] == "High"]
    if playable_ds_only:
        filtered = filtered[filtered["recommendation_status"] == "Playable at Danske Spil"]
    if better_elsewhere_only:
        filtered = filtered[filtered["recommendation_status"] == "Better elsewhere"]

    for _, row in filtered.iterrows():
        with st.container(border=True):
            cols = st.columns([2.2, 1.1, 1.1, 1.2, 1.2, 1.4])
            cols[0].markdown(f"**{match_label(row)}**")
            cols[0].caption(f"{row['kickoff_time']} | Group {row['group']} | Matchday {row['matchday']}")
            cols[0].caption(
                "Model: "
                f"H {format_percentage(row['model_home_prob'])} | "
                f"U {format_percentage(row['model_draw_prob'])} | "
                f"A {format_percentage(row['model_away_prob'])}"
            )
            cols[1].markdown(status_badge(row["recommendation_status"]), unsafe_allow_html=True)
            cols[2].markdown(draw_context_badge(row["draw_context_label"]), unsafe_allow_html=True)
            cols[3].metric("DS", row["recommended_outcome_ds"], format_dkk(row["recommended_stake_ds"]))
            cols[4].metric("Best", row["recommended_outcome_best"], format_dkk(row["recommended_stake_best"]))
            if cols[5].button("Select match", key=f"select_{row['match_id']}"):
                st.session_state.selected_match_id = row["match_id"]
                st.session_state.page = "Match Detail"
                st.rerun()
            b1, b2 = st.columns(2)
            b1.button(
                "Add DS recommendation",
                key=f"add_ds_{row['match_id']}",
                disabled=row["recommended_outcome_ds"] == "No bet",
                on_click=add_recommended_bet,
                args=(row, "ds"),
            )
            b2.button(
                "Add best market recommendation",
                key=f"add_best_{row['match_id']}",
                disabled=row["recommended_outcome_best"] == "No bet",
                on_click=add_recommended_bet,
                args=(row, "best"),
            )

    table_columns = [
        "kickoff_time",
        "group",
        "matchday",
        "home_team",
        "away_team",
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
        "recommended_stake_ds",
        "recommended_outcome_best",
        "recommended_bookmaker_best",
        "recommended_stake_best",
        "recommendation_status",
        "draw_context_label",
    ]
    st.dataframe(format_overview_table(filtered[table_columns]), width="stretch", hide_index=True)


def page_match_detail(df: pd.DataFrame) -> None:
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
    st.caption(
        f"{row['kickoff_time']} | Group {row['group']} | Matchday {row['matchday']} | "
        f"Data mode: {st.session_state.active_data_mode.title()} | "
        f"Model source: {st.session_state.active_model_source.replace('_', ' ').title()}"
    )

    h1, h2, h3 = st.columns(3)
    with h1:
        metric_card("Model probabilities", f"H {format_percentage(row['model_home_prob'])}", f"U {format_percentage(row['model_draw_prob'])} | A {format_percentage(row['model_away_prob'])}")
    with h2:
        metric_card("Danske Spil odds", f"H {format_odds(row['ds_home_odds'])}", f"U {format_odds(row['ds_draw_odds'])} | A {format_odds(row['ds_away_odds'])}")
    with h3:
        metric_card("Best market odds", f"H {format_odds(row['best_home_odds'])}", f"U {format_odds(row['best_draw_odds'])} | A {format_odds(row['best_away_odds'])}")

    prob_df = pd.DataFrame(
        {
            "Outcome": ["Home", "Draw", "Away"],
            "Model": [row["model_home_prob"], row["model_draw_prob"], row["model_away_prob"]],
            "Market": [row["market_home_prob"], row["market_draw_prob"], row["market_away_prob"]],
        }
    )
    c1, c2 = st.columns([1, 1.2])
    c1.subheader("Probability comparison")
    c1.caption("Market probabilities are derived from bookmaker odds. Model probabilities use the selected model source.")
    c1.dataframe(
        prob_df.style.format({"Model": "{:.1%}", "Market": "{:.1%}"}),
        width="stretch",
        hide_index=True,
    )
    c2.plotly_chart(probability_comparison_chart(row), width="stretch")

    st.subheader("Odds comparison")
    st.dataframe(odds_comparison_table(row), width="stretch", hide_index=True)

    st.subheader("Edge and Kelly")
    st.dataframe(style_edge_table(outcome_kelly_table(row)), width="stretch", hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        recommendation_card(
            "Danske Spil",
            row["recommended_outcome_ds"],
            row["recommended_odds_ds"],
            "Danske Spil",
            row["recommended_edge_ds"],
            row["recommended_fractional_kelly_ds"],
            row["recommended_stake_ds"],
            "Playable at Danske Spil" if row["recommended_outcome_ds"] != "No bet" else "No bet",
        )
        st.button(
            "Add Danske Spil recommendation to bet log",
            disabled=row["recommended_outcome_ds"] == "No bet",
            on_click=add_recommended_bet,
            args=(row, "ds"),
        )
    with c2:
        recommendation_card(
            "Best market",
            row["recommended_outcome_best"],
            row["recommended_odds_best"],
            row["recommended_bookmaker_best"],
            row["recommended_edge_best"],
            row["recommended_fractional_kelly_best"],
            row["recommended_stake_best"],
            row["recommendation_status"] if row["recommended_outcome_best"] != "No bet" else "No bet",
        )
        st.button(
            "Add Best Market recommendation to bet log",
            disabled=row["recommended_outcome_best"] == "No bet",
            on_click=add_recommended_bet,
            args=(row, "best"),
        )

    draw_context_card(row)


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
            empty_state("No bets logged yet. Add a recommendation from Match Detail or enter a manual bet.")
        else:
            st.dataframe(df, width="stretch", hide_index=True)

    with tab_pending:
        if pending.empty:
            empty_state("No pending bets.")
        else:
            st.dataframe(pending, width="stretch", hide_index=True)

    with tab_settled:
        if settled.empty:
            empty_state("No settled bets yet.")
        else:
            st.dataframe(settled, width="stretch", hide_index=True)

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
                        match_id,
                        match,
                        bookmaker,
                        outcome,
                        odds,
                        model_probability,
                        edge,
                        full_kelly,
                        fractional_kelly,
                        stake_dkk,
                    )
                    st.success("Bet tilføjet.")
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
                    st.success("Bet afregnet, og bankroll er opdateret.")
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


def _load_optional_csv(path) -> pd.DataFrame:
    try:
        if not path.exists() or path.stat().st_size == 0:
            return pd.DataFrame()
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


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

    st.subheader("Run backtest")
    c1, c2, c3, c4 = st.columns(4)
    initial_train_end_date = c1.date_input("Initial train end date", value=pd.Timestamp("2014-01-01").date())
    test_window = c2.text_input("Test window", value="365D")
    step_size = c3.text_input("Step size", value="365D")
    min_train_matches = c4.number_input("Min train matches", min_value=30, value=1000, step=100)
    run_col, wc_col = st.columns(2)
    with run_col:
        if st.button("Run walk-forward backtest"):
            if not historical_exists:
                st.error("No historical data file found. Add data/historical/international_results.csv first.")
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
                st.error("No historical data file found.")
            else:
                try:
                    raw = load_historical_results(HISTORICAL_RESULTS_PATH)
                    standardized = standardize_historical_results(raw)
                    with st.spinner("Running World Cup sanity check..."):
                        result = run_world_cup_backtest(standardized)
                    st.success(f"World Cup check complete. Predictions: {len(result['predictions'])}")
                    if result["predictions"].empty:
                        st.warning("No World Cup predictions were created. This is expected if the historical file has too few World Cup rows.")
                except Exception as exc:
                    st.error(f"Could not run World Cup sanity check: {exc}")

    variant_label = st.radio("Model variant", ["Baseline model", "Draw-context model"], horizontal=True)
    predictions_path = BACKTEST_PREDICTIONS_WITH_DRAW_FEATURES_PATH if variant_label == "Draw-context model" else BACKTEST_PREDICTIONS_PATH
    summary_path = BACKTEST_SUMMARY_WITH_DRAW_FEATURES_PATH if variant_label == "Draw-context model" else BACKTEST_SUMMARY_PATH
    predictions_df = _load_optional_csv(predictions_path)
    summary_df = _load_optional_csv(summary_path)
    segment_df = _load_optional_csv(BACKTEST_BY_SEGMENT_PATH)
    draw_df = _load_optional_csv(BACKTEST_DRAW_CALIBRATION_PATH)
    calibration_df = _load_optional_csv(BACKTEST_CALIBRATION_BINS_PATH)
    if predictions_df.empty:
        empty_state(f"No {variant_label.lower()} backtest results yet.")
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
            st.error("No historical data file found. Add data/historical/international_results.csv first.")
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
            st.error("No historical data file found.")
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


def page_settings() -> None:
    st.title("Settings")
    st.subheader("Data mode")
    selected_mode_label = st.radio(
        "Choose data source",
        ["Sample data", "Live odds data"],
        index=0 if st.session_state.data_mode == "sample" else 1,
        horizontal=True,
    )
    st.session_state.data_mode = "sample" if selected_mode_label == "Sample data" else "live"
    api_key_configured = bool(get_secret_or_env("ODDS_API_KEY"))
    live_freshness = get_data_freshness(LIVE_PREDICTIONS_PATH)
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Current mode", st.session_state.data_mode.title())
    with c2:
        metric_card("Live rows", str(live_freshness["row_count"]))
    with c3:
        metric_card("API key", "Configured" if api_key_configured else "Missing")
    if st.session_state.data_mode == "live" and not live_freshness["file_exists"]:
        st.warning("Live predictions are missing. The app will fall back to sample data until odds are fetched.")
    if st.button("Fetch latest odds"):
        api_key = get_secret_or_env("ODDS_API_KEY")
        if not api_key:
            st.error("ODDS_API_KEY is not configured. Add it via environment variable or Streamlit secrets.")
        else:
            try:
                odds_df = fetch_odds_from_api(
                    api_key=api_key,
                    sport_key=ODDS_API_SPORT_KEY,
                    region=ODDS_API_REGION,
                    market=ODDS_API_MARKET,
                    odds_format=ODDS_API_ODDS_FORMAT,
                )
                if odds_df.empty:
                    st.warning("Odds API returned no odds. Keeping current data mode.")
                else:
                    append_odds_snapshot(odds_df, ODDS_SNAPSHOT_PATH)
                    fixtures_df = fetch_worldcup_fixtures(odds_df=odds_df)
                    live_df = build_live_predictions(
                        odds_df,
                        fixtures_df=fixtures_df,
                        preferred_bookmaker_names=PREFERRED_BOOKMAKER_NAMES,
                    )
                    live_warnings, live_errors = validate_predictions(live_df)
                    if live_errors:
                        for error in live_errors:
                            st.error(error)
                    else:
                        for warning in live_warnings:
                            st.warning(warning)
                        st.session_state.data_mode = "live"
                        st.success(f"Live odds fetched. Built {len(live_df)} app-ready matches.")
                        st.rerun()
            except ValueError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Could not fetch live odds: {exc}")

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

    status = get_active_model_status()
    cols = st.columns(4)
    with cols[0]:
        metric_card("Model available", "Yes" if status["model_exists"] else "No")
    with cols[1]:
        metric_card("Training rows", str(status["number_of_training_rows"]))
    with cols[2]:
        metric_card("Accuracy", "-" if status["accuracy"] is None else format_percentage(status["accuracy"]))
    with cols[3]:
        metric_card("Log loss", "-" if status["log_loss"] is None else f"{status['log_loss']:.3f}")
    brier_text = "-" if status["brier_score"] is None else f"{status['brier_score']:.3f}"
    draw_actual = "-" if status["draw_rate_actual"] is None else format_percentage(status["draw_rate_actual"])
    draw_predicted = "-" if status["draw_rate_predicted"] is None else format_percentage(status["draw_rate_predicted"])
    st.caption(
        f"Trained at: {status['trained_at'] or '-'} | Test rows: {status['number_of_test_rows']} | "
        f"Brier: {brier_text} | Draw actual/predicted: {draw_actual} / {draw_predicted}"
    )

    train_col, apply_col = st.columns(2)
    with train_col:
        if st.button("Train/update historical model"):
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
                st.error("No historical data file found. Add data/historical/international_results.csv to train the model.")
            except ValueError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Could not train model: {exc}")
    with apply_col:
        if st.button("Apply model to current matches"):
            try:
                base_df, _, actual_mode = load_predictions_by_mode(st.session_state.data_mode, model_source="market_only")
                raw = load_historical_results(HISTORICAL_RESULTS_PATH)
                standardized = standardize_historical_results(raw)
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
                st.error("No historical data file or trained model found. Train the model first.")
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


def page_about() -> None:
    st.title("About")
    st.subheader("What this app does")
    st.write("A World Cup prediction and staking dashboard for comparing model/market probabilities, Danske Spil odds, best market odds, edge and Kelly-based stake suggestions.")
    st.subheader("Current MVP/live odds limitations")
    st.write("Sample mode uses static data. Live mode can ingest odds, but model probabilities currently equal market-implied probabilities until the historical model is added.")
    st.subheader("Kelly and edge")
    st.write("Edge is `model_probability * odds - 1`. Kelly stake depends heavily on probability quality, so fractional Kelly and stake caps are used to reduce risk.")
    st.subheader("Danske Spil vs best market")
    st.write("The app keeps Danske Spil recommendations separate from best-market recommendations, so value elsewhere is visible even if DS is not playable.")
    st.subheader("Draw-context")
    st.write("Draw-context is explanatory only. It is not a manual draw bonus and does not change probabilities in the MVP.")
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
    try:
        health_predictions = load_predictions()
        predictions_ok = "yes"
        matches_loaded = len(health_predictions)
    except Exception:
        predictions_ok = "no"
        matches_loaded = 0
    try:
        load_bankroll_state()
        bankroll_ok = "yes"
    except Exception:
        bankroll_ok = "no"
    try:
        health_bets = load_bet_log()
        bet_log_ok = "yes"
        bets_logged = len(health_bets)
    except Exception:
        bet_log_ok = "no"
        bets_logged = 0
    st.write(
        {
            "sample_predictions_loaded": predictions_ok,
            "bankroll_state_loaded": bankroll_ok,
            "bet_log_loaded": bet_log_ok,
            "data_mode": st.session_state.active_data_mode,
            "matches_loaded": matches_loaded,
            "bets_logged": bets_logged,
        }
    )


init_session_state()
show_sidebar()
df, validation_warnings, validation_errors = load_enriched_predictions()
show_validation_messages(validation_warnings, validation_errors)

if st.session_state.page == "Overview":
    page_overview(df)
elif st.session_state.page == "Match Detail":
    page_match_detail(df)
elif st.session_state.page == "Bankroll":
    page_bankroll()
elif st.session_state.page == "Bet Log":
    page_bet_log()
elif st.session_state.page == "Analytics":
    page_analytics()
elif st.session_state.page == "Backtest & Metrics":
    page_backtest_metrics()
elif st.session_state.page == "Draw Hypothesis":
    page_draw_hypothesis(df)
elif st.session_state.page == "Settings":
    page_settings()
else:
    page_about()
