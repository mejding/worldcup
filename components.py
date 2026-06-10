import pandas as pd
import streamlit as st


def format_percentage(value, decimals: int = 1) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value) * 100:.{decimals}f}%"


def format_dkk(value) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):,.2f} DKK"


def format_odds(value) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):.2f}"


def status_badge(status: str) -> str:
    colors = {
        "Playable at Danske Spil": "#15803d",
        "Better elsewhere": "#c2410c",
        "No bet": "#6b7280",
    }
    color = colors.get(status, "#6b7280")
    return (
        f"<span style='background:{color}; color:white; padding:0.2rem 0.5rem; "
        f"border-radius:0.35rem; font-size:0.85rem'>{status}</span>"
    )


def draw_context_badge(label: str) -> str:
    colors = {
        "High": "#7c2d12",
        "Medium": "#a16207",
        "Low": "#475569",
    }
    color = colors.get(label, "#475569")
    return (
        f"<span style='border:1px solid {color}; color:{color}; padding:0.15rem 0.45rem; "
        f"border-radius:0.35rem; font-size:0.85rem'>{label}</span>"
    )


def recommendation_card(title: str, outcome, odds, stake, edge=None, bookmaker=None) -> None:
    with st.container(border=True):
        st.subheader(title)
        if outcome == "No bet" or pd.isna(outcome):
            st.caption("Ingen gyldig anbefaling med de nuværende indstillinger.")
            return
        st.metric("Outcome", outcome)
        cols = st.columns(3)
        cols[0].metric("Odds", format_odds(odds))
        cols[1].metric("Stake", format_dkk(stake))
        cols[2].metric("Edge", format_percentage(edge, decimals=2) if edge is not None else "-")
        if bookmaker:
            st.caption(f"Bookmaker: {bookmaker}")


def metric_row(metrics: list[tuple[str, str]]) -> None:
    columns = st.columns(len(metrics))
    for column, (label, value) in zip(columns, metrics):
        column.metric(label, value)

