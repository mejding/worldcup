import pandas as pd
import plotly.express as px
import streamlit as st


def probability_comparison_chart(df: pd.DataFrame):
    if not isinstance(df, pd.DataFrame):
        row = df
        df = pd.DataFrame(
            {
                "Outcome": ["Home", "Draw", "Away"],
                "Model": [row["model_home_prob"], row["model_draw_prob"], row["model_away_prob"]],
                "Market": [row["market_home_prob"], row["market_draw_prob"], row["market_away_prob"]],
            }
        )
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
        color_discrete_map={"Model": "#2563eb", "Market": "#64748b"},
    )
    fig.update_layout(
        yaxis_tickformat=".0%",
        height=320,
        margin=dict(l=10, r=10, t=20, b=10),
        legend_title_text="",
    )
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


def backtest_metric_by_fold_chart(summary_df: pd.DataFrame, metric: str):
    if summary_df.empty or metric not in summary_df.columns:
        return None
    fig = px.line(summary_df, x="fold_id", y=metric, markers=True)
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="Fold", yaxis_title=metric)
    return fig


def draw_calibration_chart(draw_calibration_df: pd.DataFrame):
    if draw_calibration_df.empty:
        return None
    chart_df = draw_calibration_df.melt(
        id_vars=["bin_label"],
        value_vars=["avg_predicted_draw_probability", "actual_draw_rate"],
        var_name="Metric",
        value_name="Rate",
    )
    fig = px.bar(chart_df, x="bin_label", y="Rate", color="Metric", barmode="group", text_auto=".1%")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), yaxis_tickformat=".0%", xaxis_title="Draw probability bin")
    return fig


def confidence_calibration_chart(calibration_df: pd.DataFrame):
    if calibration_df.empty:
        return None
    chart_df = calibration_df.melt(
        id_vars=["bin_lower", "bin_upper"],
        value_vars=["avg_confidence", "accuracy"],
        var_name="Metric",
        value_name="Value",
    )
    chart_df["bin"] = chart_df["bin_lower"].map(lambda value: f"{value:.1f}") + "-" + chart_df["bin_upper"].map(lambda value: f"{value:.1f}")
    fig = px.bar(chart_df, x="bin", y="Value", color="Metric", barmode="group", text_auto=".1%")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), yaxis_tickformat=".0%", xaxis_title="Confidence bin")
    return fig


def segment_metric_chart(segment_df: pd.DataFrame, metric: str, segment_name: str):
    if segment_df.empty or metric not in segment_df.columns:
        return None
    chart_df = segment_df[segment_df["segment_name"] == segment_name].copy()
    if chart_df.empty:
        return None
    fig = px.bar(chart_df, x="segment_value", y=metric, text_auto=".3f")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), xaxis_title=segment_name, yaxis_title=metric)
    return fig


def render_chart(fig) -> None:
    if fig is None:
        st.info("Ingen data at vise endnu.")
    else:
        st.plotly_chart(fig, width="stretch")
