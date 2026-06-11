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
        "Accuracy": "Share of matches where highest probability outcome matched result.",
        "Log loss": "Probability quality metric; lower is better.",
        "Brier score": "Squared probability error; lower is better.",
        "ECE": "Calibration error; lower is better.",
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
