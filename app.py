import pandas as pd
import streamlit as st

from bankroll import load_bankroll_history, load_bankroll_state, reset_bankroll, update_bankroll
from bet_log import add_bet, calculate_bet_summary, load_bet_log, reset_bet_settlement, settle_bet
from charts import (
    bankroll_history_chart,
    probability_comparison_chart,
    profit_loss_by_bookmaker_chart,
    profit_loss_by_outcome_chart,
    render_chart,
)
from components import (
    draw_context_badge,
    draw_context_card,
    empty_state,
    format_dkk,
    format_odds,
    format_percentage,
    metric_card,
    metric_row,
    odds_comparison_table,
    recommendation_card,
    status_badge,
)
from config import (
    DEFAULT_PROFILE_NAME,
    LIVE_PREDICTIONS_PATH,
    ODDS_API_MARKET,
    ODDS_API_ODDS_FORMAT,
    ODDS_API_REGION,
    ODDS_API_SPORT_KEY,
    ODDS_SNAPSHOT_PATH,
    PREFERRED_BOOKMAKER,
    PREFERRED_BOOKMAKER_NAMES,
    SAMPLE_PREDICTIONS_PATH,
    STAKING_PROFILES,
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
from fetch_fixtures import fetch_worldcup_fixtures
from fetch_odds import append_odds_snapshot, fetch_odds_from_api
from kelly import calculate_final_stake_fraction, calculate_suggested_stake
from live_data_pipeline import build_live_predictions
from odds_utils import calculate_edge
from recommendations import add_recommendations


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


def current_profile() -> dict:
    return st.session_state.staking_profile.copy()


def load_enriched_predictions() -> tuple[pd.DataFrame, list[str], list[str]]:
    try:
        predictions, mode_warnings, actual_mode = load_predictions_by_mode(st.session_state.data_mode)
        st.session_state.active_data_mode = actual_mode
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
        ["Overview", "Match Detail", "Bankroll", "Bet Log", "Analytics", "Settings", "About"],
        index=["Overview", "Match Detail", "Bankroll", "Bet Log", "Analytics", "Settings", "About"].index(
            st.session_state.page
        ),
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
        f"Data mode: {st.session_state.active_data_mode.title()}"
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
    st.subheader("Coming later")
    future_cols = st.columns(5)
    for col, label in zip(
        future_cols,
        ["Accuracy", "Log loss", "Brier score", "Calibration", "Draw calibration"],
    ):
        with col:
            metric_card(label, "-", "After model backtest")
    st.info("These metrics will become available after the historical model and backtest module are added.")


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
elif st.session_state.page == "Settings":
    page_settings()
else:
    page_about()
