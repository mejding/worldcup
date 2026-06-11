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


def draw_rate_by_segment_chart(segment_df: pd.DataFrame):
    if segment_df.empty or "draw_rate" not in segment_df.columns:
        return None
    chart_df = segment_df[segment_df["segment_name"] != "overall"].copy()
    if chart_df.empty:
        return None
    chart_df["label"] = chart_df["segment_name"] + ": " + chart_df["segment_value"].astype(str)
    chart_df = chart_df.sort_values("match_count", ascending=False).head(20)
    fig = px.bar(chart_df, x="label", y="draw_rate", text_auto=".1%")
    if "baseline_draw_rate" in chart_df.columns:
        baseline = float(chart_df["baseline_draw_rate"].iloc[0])
        fig.add_hline(y=baseline, line_dash="dash", line_color="#64748b")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), yaxis_tickformat=".0%", xaxis_title="")
    return fig


def draw_feature_comparison_chart(comparison_df: pd.DataFrame, metric: str):
    if comparison_df.empty or metric not in comparison_df.columns:
        return None
    fig = px.bar(comparison_df, x="segment", y=metric, color="model_variant", barmode="group", text_auto=".3f")
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="Segment", yaxis_title=metric)
    return fig


def draw_context_score_distribution_chart(df: pd.DataFrame):
    if df.empty or "draw_context_score" not in df.columns:
        return None
    fig = px.histogram(df, x="draw_context_score", nbins=20, color="draw_context_label" if "draw_context_label" in df.columns else None)
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="Draw-context score")
    return fig


def draw_calibration_comparison_chart(baseline_df: pd.DataFrame, draw_context_df: pd.DataFrame):
    if baseline_df.empty or draw_context_df.empty:
        return None
    base = baseline_df.copy()
    base["model_variant"] = "baseline"
    draw = draw_context_df.copy()
    draw["model_variant"] = "draw_context"
    combined = pd.concat([base, draw], ignore_index=True)
    if "calibration_gap" not in combined.columns:
        return None
    fig = px.bar(combined, x="bin_label", y="calibration_gap", color="model_variant", barmode="group", text_auto=".2f")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="Draw probability bin")
    return fig


def ensemble_weight_metric_chart(comparison_df: pd.DataFrame, metric: str):
    if comparison_df.empty or metric not in comparison_df.columns or "w_market" not in comparison_df.columns:
        return None
    chart_df = comparison_df[comparison_df["source_name"].astype(str).str.startswith("ensemble")].copy()
    if chart_df.empty:
        return None
    fig = px.line(chart_df.sort_values("w_market"), x="w_market", y=metric, markers=True)
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="Market weight", yaxis_title=metric)
    return fig


def probability_source_comparison_chart(comparison_df: pd.DataFrame):
    if comparison_df.empty or "log_loss" not in comparison_df.columns:
        return None
    chart_df = comparison_df.copy()
    fig = px.bar(chart_df, x="source_name", y="log_loss", color="source_name", text_auto=".3f")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="", showlegend=False)
    return fig


def active_vs_market_model_chart(match_row):
    sources = [
        ("Market", "market"),
        ("Historical model", "model"),
        ("Draw-context model", "draw_model"),
        ("Ensemble", "ensemble"),
        ("Active", "active"),
    ]
    rows = []
    for label, prefix in sources:
        columns = [f"{prefix}_home_prob", f"{prefix}_draw_prob", f"{prefix}_away_prob"]
        if all(column in match_row.index and not pd.isna(match_row[column]) for column in columns):
            for outcome, column in zip(["Home", "Draw", "Away"], columns):
                rows.append({"Source": label, "Outcome": outcome, "Probability": float(match_row[column])})
    if not rows:
        return None
    fig = px.bar(pd.DataFrame(rows), x="Outcome", y="Probability", color="Source", barmode="group", text_auto=".1%")
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), yaxis_tickformat=".0%")
    return fig


def render_chart(fig) -> None:
    if fig is None:
        st.info("Ingen data at vise endnu.")
    else:
        st.plotly_chart(fig, width="stretch")
