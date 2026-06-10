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
    format_dkk,
    format_odds,
    format_percentage,
    metric_row,
    recommendation_card,
    status_badge,
)
from config import DEFAULT_PROFILE_NAME, PREFERRED_BOOKMAKER, STAKING_PROFILES, get_staking_profile
from data_loader import ensure_runtime_data_files, load_predictions, validate_predictions
from kelly import calculate_final_stake_fraction, calculate_suggested_stake
from odds_utils import calculate_edge
from recommendations import add_recommendations


st.set_page_config(page_title="VM 2026 Prediction & Kelly", layout="wide")
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


def current_profile() -> dict:
    return st.session_state.staking_profile.copy()


def load_enriched_predictions() -> tuple[pd.DataFrame, list[str]]:
    predictions = load_predictions()
    warnings = validate_predictions(predictions)
    if any(warning.startswith("CRITICAL:") for warning in warnings):
        return predictions, warnings
    bankroll = load_bankroll_state()["current_bankroll"]
    enriched = add_recommendations(predictions, bankroll, current_profile())
    return enriched.rename(columns={"status": "recommendation_status"}), warnings


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
    for column in ["model_home_prob", "model_draw_prob", "model_away_prob"]:
        display_df[column] = display_df[column].map(format_percentage)
    for column in [
        "ds_home_odds",
        "ds_draw_odds",
        "ds_away_odds",
        "best_home_odds",
        "best_draw_odds",
        "best_away_odds",
    ]:
        display_df[column] = display_df[column].map(format_odds)
    for column in ["recommended_stake_ds", "recommended_stake_best"]:
        display_df[column] = display_df[column].map(format_dkk)
    return display_df


def show_sidebar() -> None:
    state = load_bankroll_state()
    st.sidebar.title("Navigation")
    st.session_state.page = st.sidebar.radio(
        "Side",
        ["Overview", "Match Detail", "Bankroll", "Bet Log", "Analytics", "Settings", "About"],
        index=["Overview", "Match Detail", "Bankroll", "Bet Log", "Analytics", "Settings", "About"].index(
            st.session_state.page
        ),
    )
    st.sidebar.divider()
    st.sidebar.metric("Current bankroll", format_dkk(state["current_bankroll"]))
    st.sidebar.caption(f"Kelly profile: {st.session_state.kelly_profile_name}")
    profile = current_profile()
    st.sidebar.caption(f"Fractional Kelly: {format_percentage(profile['fractional_kelly_multiplier'])}")
    st.sidebar.caption(f"Max stake: {format_percentage(profile['max_stake_pct_of_bankroll'])}")
    st.sidebar.caption(f"Minimum edge: {format_percentage(profile['min_edge_threshold'])}")
    st.sidebar.caption(f"Minimum stake: {format_percentage(profile['min_stake_pct_threshold'])}")


def show_warnings(warnings: list[str]) -> None:
    for warning in warnings:
        if warning.startswith("CRITICAL:"):
            st.error(warning)
            st.stop()
        st.warning(warning)


def page_overview(df: pd.DataFrame) -> None:
    st.title("VM 2026 Prediction & Kelly")
    st.caption("Kommende kampe, odds, edge og anbefalet indsats")
    counts = df["recommendation_status"].value_counts()
    metric_row(
        [
            ("Current bankroll", format_dkk(load_bankroll_state()["current_bankroll"])),
            ("Matches", str(len(df))),
            ("DS recommendations", str(counts.get("Playable at Danske Spil", 0))),
            ("Better elsewhere", str(counts.get("Better elsewhere", 0))),
            ("No bet", str(counts.get("No bet", 0))),
        ]
    )

    with st.expander("Filtre", expanded=True):
        c1, c2, c3 = st.columns(3)
        groups = c1.multiselect("Group", sorted(df["group"].unique()), default=sorted(df["group"].unique()))
        matchdays = c2.multiselect(
            "Matchday", sorted(df["matchday"].unique()), default=sorted(df["matchday"].unique())
        )
        recommended_only = c3.checkbox("Show recommended bets only")
        positive_edge_only = c1.checkbox("Show positive edge only")
        draw_value_only = c2.checkbox("Show draw-value opportunities only")
        high_draw_only = c3.checkbox("Show high draw-context only")
        playable_ds_only = c1.checkbox("Show bets playable at Danske Spil")
        better_elsewhere_only = c2.checkbox("Show bets better elsewhere")

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
    st.caption(f"{row['kickoff_time']} | Group {row['group']} | Matchday {row['matchday']}")

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
    c2.plotly_chart(probability_comparison_chart(prob_df), width="stretch")

    st.subheader("Odds comparison")
    odds_df = pd.DataFrame(
        {
            "Outcome": ["Home", "Draw", "Away"],
            "Danske Spil odds": [row["ds_home_odds"], row["ds_draw_odds"], row["ds_away_odds"]],
            "Best market odds": [row["best_home_odds"], row["best_draw_odds"], row["best_away_odds"]],
            "Best bookmaker": [
                row["best_home_bookmaker"],
                row["best_draw_bookmaker"],
                row["best_away_bookmaker"],
            ],
        }
    )
    st.dataframe(
        odds_df.style.format({"Danske Spil odds": "{:.2f}", "Best market odds": "{:.2f}"}),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Edge and Kelly")
    st.dataframe(style_edge_table(outcome_kelly_table(row)), width="stretch", hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        recommendation_card(
            "Danske Spil",
            row["recommended_outcome_ds"],
            row["recommended_odds_ds"],
            row["recommended_stake_ds"],
            row["recommended_edge_ds"],
            "Danske Spil",
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
            row["recommended_stake_best"],
            row["recommended_edge_best"],
            row["recommended_bookmaker_best"],
        )
        st.button(
            "Add Best Market recommendation to bet log",
            disabled=row["recommended_outcome_best"] == "No bet",
            on_click=add_recommended_bet,
            args=(row, "best"),
        )

    with st.container(border=True):
        st.subheader("Draw-context")
        st.markdown(draw_context_badge(row["draw_context_label"]), unsafe_allow_html=True)
        st.write(
            {
                "draw_context_score": row["draw_context_score"],
                "home_draw_utility": row["home_draw_utility"],
                "away_draw_utility": row["away_draw_utility"],
                "mutual_draw_acceptance": row["mutual_draw_acceptance"],
                "one_team_must_win": row["one_team_must_win"],
                "both_teams_draw_satisfied": row["both_teams_draw_satisfied"],
            }
        )
        st.info(
            "Draw-context is not a manual draw bonus. It is a contextual indicator that may later "
            "be used by the model to assess whether a draw is strategically acceptable for one or both teams."
        )


def page_bankroll() -> None:
    state = load_bankroll_state()
    net = state["current_bankroll"] - state["starting_bankroll"]
    ret = net / state["starting_bankroll"] if state["starting_bankroll"] else 0
    st.title("Bankroll")
    metric_row(
        [
            ("Starting bankroll", format_dkk(state["starting_bankroll"])),
            ("Current bankroll", format_dkk(state["current_bankroll"])),
            ("Net P/L", format_dkk(net)),
            ("Return", format_percentage(ret)),
        ]
    )

    history = load_bankroll_history()
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
            update_bankroll(signed_amount, transaction_type, note=note)
            st.success("Bankroll opdateret.")
            st.rerun()

    with st.expander("Reset bankroll"):
        new_start = st.number_input("New starting bankroll", min_value=0.0, value=1000.0, step=100.0)
        confirm = st.checkbox("I understand this resets starting and current bankroll")
        if st.button("Reset bankroll", disabled=not confirm):
            reset_bankroll(new_start)
            st.success("Bankroll nulstillet.")
            st.rerun()


def page_bet_log() -> None:
    st.title("Bet Log")
    summary = calculate_bet_summary()
    metric_row(
        [
            ("Total bets", str(summary["total_bets"])),
            ("Pending", str(summary["pending_bets"])),
            ("Settled", str(summary["settled_bets"])),
            ("Win rate", format_percentage(summary["win_rate"])),
            ("Total staked", format_dkk(summary["total_staked"])),
            ("P/L", format_dkk(summary["total_profit_loss"])),
            ("ROI", format_percentage(summary["roi"])),
            ("Avg odds", format_odds(summary["average_odds"])),
        ]
    )
    df = load_bet_log()
    st.dataframe(df, width="stretch", hide_index=True)

    with st.expander("Add manual bet"):
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

    pending = df[df["result"] == "pending"] if not df.empty else df
    with st.expander("Settle pending bet", expanded=True):
        if pending.empty:
            st.info("Ingen pending bets.")
        else:
            bet_id = st.selectbox("Pending bet_id", pending["bet_id"].tolist())
            result = st.selectbox("Result", ["won", "lost", "void"])
            if st.button("Settle bet"):
                try:
                    settle_bet(bet_id, result)
                    st.success("Bet afregnet, og bankroll er opdateret.")
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))

    with st.expander("Reset settlement"):
        if df.empty:
            st.info("Ingen bets.")
        else:
            reset_id = st.selectbox("bet_id", df["bet_id"].tolist(), key="reset_bet_id")
            if st.button("Reset settlement"):
                reset_bet_settlement(reset_id)
                st.warning("Settlement er nulstillet. Bankroll er ikke automatisk reverseret.")
                st.rerun()


def page_analytics() -> None:
    st.title("Analytics")
    summary = calculate_bet_summary()
    metric_row(
        [
            ("Total bets", str(summary["total_bets"])),
            ("Settled", str(summary["settled_bets"])),
            ("Pending", str(summary["pending_bets"])),
            ("Win rate", format_percentage(summary["win_rate"])),
            ("Total staked", format_dkk(summary["total_staked"])),
            ("P/L", format_dkk(summary["total_profit_loss"])),
            ("ROI", format_percentage(summary["roi"])),
            ("Avg edge", format_percentage(summary["average_edge"], decimals=2)),
        ]
    )
    df = load_bet_log()
    settled = df[df["result"].isin(["won", "lost", "void"])] if not df.empty else df
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Profit/loss by bookmaker")
        render_chart(profit_loss_by_bookmaker_chart(settled))
    with c2:
        st.subheader("Profit/loss by outcome")
        render_chart(profit_loss_by_outcome_chart(settled))
    st.subheader("Draw bet performance")
    st.dataframe(settled[settled["outcome"] == "Draw"], width="stretch", hide_index=True)
    st.info("Accuracy, Log loss, Brier score, Calibration and Draw calibration: Coming later when historical results and model backtest are added.")


def page_settings() -> None:
    st.title("Settings")
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
        "Max stake %", min_value=0.0, max_value=1.0, value=float(profile["max_stake_pct_of_bankroll"]), step=0.005
    )
    profile["min_edge_threshold"] = st.number_input(
        "Minimum edge %", min_value=0.0, max_value=1.0, value=float(profile["min_edge_threshold"]), step=0.005
    )
    profile["min_stake_pct_threshold"] = st.number_input(
        "Minimum stake %", min_value=0.0, max_value=1.0, value=float(profile["min_stake_pct_threshold"]), step=0.001
    )
    st.session_state.staking_profile = profile
    st.session_state.preferred_bookmaker = st.text_input("Preferred bookmaker", st.session_state.preferred_bookmaker)
    st.selectbox("Data mode", ["sample data only for now", "live odds later disabled", "manual CSV later disabled"], disabled=True)
    st.info("Default Standard profile uses 0.25 Kelly, max stake 2.5%, minimum edge 2.5%, and minimum stake 0.25%.")


def page_about() -> None:
    st.title("About")
    st.write(
        """
        This is an analytical World Cup prediction and staking dashboard. The MVP uses sample data only:
        model probabilities, Danske Spil odds and best market odds are assumed inputs in this version.

        Edge is calculated as `model_probability * odds - 1`. Kelly stake depends heavily on the
        accuracy of model probabilities, so fractional Kelly and stake caps reduce risk. Current
        bankroll is used for all stake calculations. Draw-context is explanatory only in the MVP.

        The tool is for analysis and tracking. It does not guarantee profit.
        """
    )
    st.warning(
        "Bankroll and bet log are stored locally as CSV/JSON in this MVP. On Streamlit Community Cloud "
        "this is suitable for testing only. For production, use persistent storage such as Supabase, "
        "Postgres, SQLite with mounted storage, or another database."
    )


init_session_state()
show_sidebar()
df, validation_warnings = load_enriched_predictions()
show_warnings(validation_warnings)

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
