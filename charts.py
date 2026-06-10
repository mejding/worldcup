import pandas as pd
import plotly.express as px
import streamlit as st


def probability_comparison_chart(df: pd.DataFrame):
    chart_df = df.melt(
        id_vars=["Outcome"],
        value_vars=["Model", "Market"],
        var_name="Source",
        value_name="Probability",
    )
    fig = px.bar(
        chart_df,
        x="Outcome",
        y="Probability",
        color="Source",
        barmode="group",
        text=chart_df["Probability"].map(lambda value: f"{value:.1%}"),
    )
    fig.update_layout(yaxis_tickformat=".0%", height=340, margin=dict(l=10, r=10, t=30, b=10))
    return fig


def bankroll_history_chart(df: pd.DataFrame):
    if df.empty:
        return None
    chart_df = df.copy()
    chart_df["timestamp"] = pd.to_datetime(chart_df["timestamp"], errors="coerce")
    fig = px.line(chart_df, x="timestamp", y="bankroll_after", markers=True)
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="Bankroll")
    return fig


def profit_loss_by_bookmaker_chart(df: pd.DataFrame):
    if df.empty:
        return None
    grouped = df.groupby("bookmaker", as_index=False)["profit_loss_dkk"].sum()
    fig = px.bar(grouped, x="bookmaker", y="profit_loss_dkk", text_auto=".2f")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="Bookmaker")
    return fig


def profit_loss_by_outcome_chart(df: pd.DataFrame):
    if df.empty:
        return None
    grouped = df.groupby("outcome", as_index=False)["profit_loss_dkk"].sum()
    fig = px.bar(grouped, x="outcome", y="profit_loss_dkk", text_auto=".2f")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="Outcome")
    return fig


def render_chart(fig) -> None:
    if fig is None:
        st.info("Ingen data at vise endnu.")
    else:
        st.plotly_chart(fig, width="stretch")

