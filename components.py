import pandas as pd
import streamlit as st

from time_utils import format_danish_kickoff
from tooltip_definitions import TOOLTIPS


def format_percentage(value, decimals: int = 1) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value) * 100:.{decimals}f}%"


def format_probability(value) -> str:
    return format_percentage(value, decimals=1)


def format_edge(value) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value) * 100:+.1f}%"


def format_kelly(value) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value) * 100:.2f}%"


def format_dkk(value) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):,.2f} DKK"


def format_odds(value) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):.2f}"


def format_danish_datetime(value) -> str:
    return format_danish_kickoff(value)


def outcome_label(outcome, home_team: str, away_team: str, language: str = "en") -> str:
    if outcome == "Home":
        return f"{home_team} win" if language == "en" else f"{home_team} vinder"
    if outcome == "Away":
        return f"{away_team} win" if language == "en" else f"{away_team} vinder"
    if outcome == "Draw":
        return "Draw" if language == "en" else "Uafgjort"
    return "No bet"


def status_badge(status: str) -> str:
    colors = {
        "Playable at Danske Spil": "#15803d",
        "Better elsewhere": "#c2410c",
        "No bet": "#6b7280",
    }
    color = colors.get(status, "#6b7280")
    return (
        f"<span style='background:{color}; color:white; padding:0.22rem 0.55rem; "
        f"border-radius:0.35rem; font-size:0.8rem; font-weight:600; white-space:nowrap'>{status}</span>"
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
        f"border-radius:0.35rem; font-size:0.8rem; font-weight:600'>{label}</span>"
    )


def metric_card(label: str, value: str, help_text: str = None) -> None:
    with st.container(border=True):
        st.caption(label)
        st.markdown(f"### {value}")
        if help_text:
            st.caption(help_text)


def recommendation_card(
    title: str,
    outcome,
    odds,
    bookmaker=None,
    edge=None,
    fractional_kelly=None,
    stake_dkk=None,
    status: str = "No bet",
) -> None:
    with st.container(border=True):
        c1, c2 = st.columns([1.5, 1])
        c1.subheader(title)
        c2.markdown(status_badge(status), unsafe_allow_html=True)
        if outcome == "No bet" or pd.isna(outcome):
            st.markdown("**No bet**")
            st.caption("No outcome passes the minimum edge and Kelly thresholds.")
            return

        st.markdown(f"**{outcome}**")
        if bookmaker:
            st.caption(f"Bookmaker: {bookmaker}")
        cols = st.columns(4)
        cols[0].metric("Odds", format_odds(odds))
        cols[1].metric("Edge", format_percentage(edge, decimals=2) if edge is not None else "-")
        cols[2].metric("Kelly", format_percentage(fractional_kelly, decimals=2) if fractional_kelly is not None else "-")
        cols[3].metric("Stake", format_dkk(stake_dkk))
        if status == "Better elsewhere":
            st.caption("Value exists at best market odds, but not at Danske Spil.")


def recommendation_card_v2(
    title,
    status,
    outcome_label,
    odds,
    bookmaker,
    edge,
    kelly_pct,
    stake_dkk,
    probability_source,
    reason=None,
    compact=True,
) -> None:
    status_class = {
        "Playable at Danske Spil": "wc-status-green",
        "Play at Danske Spil": "wc-status-green",
        "Better elsewhere": "wc-status-amber",
        "Better odds available elsewhere": "wc-status-amber",
        "Odds missing at Danske Spil": "wc-status-amber",
        "Best market odds missing": "wc-status-amber",
        "No bet at Danske Spil": "wc-status-muted",
        "No better market value found": "wc-status-muted",
        "Same or similar odds": "wc-status-green",
        "No bet": "wc-status-muted",
        "Playable": "wc-status-green",
    }.get(status, "wc-status-muted")
    status_text = "Playable" if status == "Playable at Danske Spil" else status
    if outcome_label == "No bet" or pd.isna(outcome_label):
        body = f"""
        <div class="wc-card">
          <div class="wc-line"><span class="wc-card-title">{title}</span><span class="{status_class}">{status_text}</span></div>
          <div class="wc-rec-main">No bet</div>
          <div class="wc-rec-details">{reason or TOOLTIPS['no_bet']}</div>
          <div class="wc-muted">Based on: {probability_source}</div>
        </div>
        """
    else:
        bookmaker_text = f" · {bookmaker}" if bookmaker else ""
        body = f"""
        <div class="wc-card">
          <div class="wc-line"><span class="wc-card-title">{title}</span><span class="{status_class}">{status_text}</span></div>
          <div class="wc-rec-main">{outcome_label} @ {format_odds(odds)}{bookmaker_text}</div>
          <div class="wc-rec-details">Edge {format_edge(edge)} · Kelly {format_kelly(kelly_pct)} · Stake {format_dkk(stake_dkk)}</div>
          <div class="wc-muted">Based on: {probability_source}</div>
        </div>
        """
    st.markdown(body, unsafe_allow_html=True)


def metric_row(metrics: list[tuple[str, str]]) -> None:
    columns = st.columns(len(metrics))
    for column, (label, value) in zip(columns, metrics):
        column.metric(label, value)


def empty_state(message: str) -> None:
    st.info(message)


def small_sample_warning(match_count: int, threshold: int = 100) -> None:
    if int(match_count or 0) < threshold:
        st.warning(f"Small sample: {int(match_count or 0)} matches. Treat these metrics as directional only.")


def calibration_gap_badge(gap) -> str:
    if pd.isna(gap):
        return "-"
    gap = float(gap)
    color = "#15803d" if abs(gap) < 0.03 else "#a16207" if abs(gap) < 0.08 else "#b91c1c"
    label = f"{gap:+.1%}"
    return (
        f"<span style='background:{color}; color:white; padding:0.18rem 0.48rem; "
        f"border-radius:0.35rem; font-size:0.8rem; font-weight:600'>{label}</span>"
    )


def model_metric_explanation(metric_name: str) -> str:
    explanations = {
        "Accuracy": TOOLTIPS["accuracy"],
        "Log loss": TOOLTIPS["log_loss"],
        "Brier score": TOOLTIPS["brier_score"],
        "ECE": TOOLTIPS["ece"],
        "Draw calibration gap": "Predicted draw rate minus actual draw rate.",
    }
    return explanations.get(metric_name, "")


def draw_hypothesis_summary_card(match_count: int, draw_rate, group_metadata_rate=None) -> None:
    with st.container(border=True):
        st.subheader("Draw hypothesis summary")
        cols = st.columns(3)
        cols[0].metric("Matches", str(int(match_count or 0)))
        cols[1].metric("Draw rate", format_percentage(draw_rate or 0))
        cols[2].metric(
            "Group metadata",
            "-" if group_metadata_rate is None or pd.isna(group_metadata_rate) else format_percentage(group_metadata_rate),
        )


def draw_context_decision_card(recommendation_dict: dict) -> None:
    recommended = bool(recommendation_dict.get("recommended", False))
    with st.container(border=True):
        st.subheader("Draw-context model decision")
        st.metric("Recommended", "Yes" if recommended else "No")
        st.caption(recommendation_dict.get("reason", "No recommendation available."))
        for caveat in recommendation_dict.get("caveats", []):
            st.warning(caveat)


def draw_context_score_badge(score, label) -> str:
    if pd.isna(score):
        return "-"
    color = {"Low": "#64748b", "Medium": "#a16207", "High": "#15803d"}.get(str(label), "#64748b")
    return (
        f"<span style='background:{color}; color:white; padding:0.18rem 0.48rem; "
        f"border-radius:0.35rem; font-size:0.8rem; font-weight:600'>{float(score):.0f} / {label}</span>"
    )


def small_sample_caveat(match_count: int) -> None:
    if int(match_count or 0) < 100:
        st.warning("Small sample caveat: fewer than 100 matches. Interpret results carefully.")


def probability_source_badge(source: str) -> str:
    labels = {
        "market": "Market only",
        "historical_model": "Historical model",
        "draw_context_model": "Draw-context model",
        "ensemble": "Market-aware ensemble",
        "best_validated": "Best validated source",
    }
    color = {"market": "#64748b", "historical_model": "#2563eb", "draw_context_model": "#15803d", "ensemble": "#7c3aed"}.get(source, "#64748b")
    return (
        f"<span style='background:{color}; color:white; padding:0.18rem 0.48rem; "
        f"border-radius:0.35rem; font-size:0.8rem; font-weight:600'>{labels.get(source, source)}</span>"
    )


def ensemble_weight_badge(w_market, w_model) -> str:
    return f"Market {float(w_market):.0%} / Model {float(w_model):.0%}"


def best_source_card(recommendation_dict: dict) -> None:
    with st.container(border=True):
        st.subheader("Best validated probability source")
        st.metric("Recommended source", recommendation_dict.get("recommended_source", "market"))
        st.caption(ensemble_weight_badge(recommendation_dict.get("w_market", 1), recommendation_dict.get("w_model", 0)))
        st.caption(recommendation_dict.get("reason", "No recommendation available."))
        for caveat in recommendation_dict.get("caveats", []):
            st.warning(caveat)


def metric_improvement_badge(delta, lower_is_better: bool = True) -> str:
    if pd.isna(delta):
        return "-"
    delta = float(delta)
    improved = delta < 0 if lower_is_better else delta > 0
    color = "#15803d" if improved else "#b91c1c"
    return (
        f"<span style='background:{color}; color:white; padding:0.15rem 0.42rem; "
        f"border-radius:0.35rem; font-size:0.8rem; font-weight:600'>{delta:+.4f}</span>"
    )


def odds_comparison_table(row) -> pd.DataFrame:
    rows = []
    for key, label in [("home", "H"), ("draw", "U"), ("away", "A")]:
        ds_odds = row.get(f"ds_{key}_odds")
        best_odds = row.get(f"best_{key}_odds")
        materially_higher = (
            not pd.isna(ds_odds)
            and not pd.isna(best_odds)
            and float(best_odds) - float(ds_odds) >= 0.05
        )
        rows.append(
            {
                "Outcome": label,
                "Danske Spil odds": "Unavailable" if pd.isna(ds_odds) else format_odds(ds_odds),
                "Best market odds": format_odds(best_odds),
                "Best bookmaker": row.get(f"best_{key}_bookmaker"),
                "Best higher": "Yes" if materially_higher else "",
            }
        )
    return pd.DataFrame(rows)


def draw_context_card(row) -> None:
    with st.container(border=True):
        st.subheader("Draw-context")
        st.markdown(draw_context_badge(row["draw_context_label"]), unsafe_allow_html=True)
        score = int(row["draw_context_score"])
        st.progress(max(0, min(score, 100)) / 100)
        cols = st.columns(3)
        cols[0].metric("Score", f"{score}/100")
        cols[1].metric("Mutual acceptance", format_percentage(row["mutual_draw_acceptance"], decimals=1))
        cols[2].metric("One must win", "Yes" if bool(row["one_team_must_win"]) else "No")
        st.caption(
            "Draw-context is not a manual draw bonus. It is a contextual indicator that may later be used "
            "by the model to assess whether a draw is strategically acceptable for one or both teams."
        )


def draw_context_card_v2(
    score,
    label,
    mutual_draw_acceptance,
    both_teams_draw_satisfied,
    one_team_must_win,
    compact=True,
) -> None:
    score = 0 if pd.isna(score) else int(score)
    label = label if not pd.isna(label) else "Low"
    if label == "High":
        explanation = "High draw-context. Look closer at draw probability and draw edge."
    elif label == "Medium":
        explanation = "Moderate contextual signs that a draw may be strategically plausible. The signal is not strong by itself."
    else:
        explanation = "Low draw-context. Match context does not especially point toward a strategic draw."
    with st.container(border=True):
        c1, c2 = st.columns([1.1, 2.4])
        c1.metric("Draw-context", f"{score}/100", label, help=TOOLTIPS["draw_context_score"])
        c2.caption(explanation)
        c2.caption("Draw-context is not draw probability, not a recommendation, and never triggers a bet alone.")
        with st.expander("Draw-context details"):
            st.metric("Mutual acceptance", format_probability(mutual_draw_acceptance), help=TOOLTIPS["mutual_acceptance"])
            st.caption(f"Both teams draw satisfied: {'Yes' if bool(both_teams_draw_satisfied) else 'No'}")
            st.caption(f"One team must win: {'Yes' if bool(one_team_must_win) else 'No'}")
            st.caption("Never bet draw based only on draw-context. Use it as a prompt to inspect draw probability, odds and edge.")
