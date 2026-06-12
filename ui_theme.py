import streamlit as st


def apply_custom_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --wc-bg: #f5f7fb;
            --wc-card: #ffffff;
            --wc-border: #d9e2ef;
            --wc-muted: #64748b;
            --wc-text: #0f172a;
            --wc-green: #15803d;
            --wc-amber: #b45309;
            --wc-blue: #1d4ed8;
        }
        .stApp { background: var(--wc-bg); color: var(--wc-text); }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
            color: #e5e7eb;
        }
        [data-testid="stSidebar"] * { color: #e5e7eb; }
        [data-testid="stSidebar"] [data-testid="stMetricValue"] { font-size: 1.05rem; }
        .block-container { padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1450px; }
        div[data-testid="stMetric"] {
            background: var(--wc-card);
            border: 1px solid var(--wc-border);
            border-radius: 8px;
            padding: 0.65rem 0.75rem;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }
        div[data-testid="stMetricLabel"] { color: var(--wc-muted); font-size: 0.78rem; }
        div[data-testid="stMetricValue"] { font-size: 1.35rem; font-weight: 750; }
        .wc-card {
            background: var(--wc-card);
            border: 1px solid var(--wc-border);
            border-radius: 8px;
            padding: 0.75rem 0.85rem;
            margin-bottom: 0.55rem;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }
        .wc-card-title { font-weight: 750; font-size: 0.98rem; margin-bottom: 0.15rem; }
        .wc-muted { color: var(--wc-muted); font-size: 0.84rem; }
        .wc-line { display: flex; flex-wrap: wrap; gap: 0.45rem 0.75rem; align-items: center; }
        .wc-pill {
            display: inline-flex; align-items: center;
            border: 1px solid var(--wc-border);
            border-radius: 999px;
            padding: 0.12rem 0.48rem;
            font-size: 0.78rem;
            background: #f8fafc;
            color: #334155;
            white-space: nowrap;
        }
        .wc-status-green { color: var(--wc-green); font-weight: 700; }
        .wc-status-amber { color: var(--wc-amber); font-weight: 700; }
        .wc-status-muted { color: var(--wc-muted); font-weight: 700; }
        .wc-rec-main { font-size: 1.02rem; font-weight: 800; }
        .wc-rec-details { color: #334155; font-size: 0.88rem; }
        .wc-hero-title { font-size: 1.55rem; font-weight: 850; margin-bottom: 0.1rem; }
        .wc-hero-subtitle { color: var(--wc-muted); font-size: 0.93rem; margin-bottom: 0.55rem; }
        .wc-match-compact {
            background: var(--wc-card);
            border: 1px solid var(--wc-border);
            border-radius: 7px;
            padding: 0.42rem 0.58rem;
            margin: 0.28rem 0 0.16rem 0;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
        }
        .wc-match-main {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 0.75rem;
        }
        .wc-match-title { font-size: 0.96rem; font-weight: 780; line-height: 1.15; }
        .wc-match-meta, .wc-match-line, .wc-match-reason {
            color: #475569;
            font-size: 0.79rem;
            line-height: 1.22;
            margin-top: 0.08rem;
        }
        .wc-match-reason { color: var(--wc-muted); }
        .wc-match-status { white-space: nowrap; font-size: 0.8rem; line-height: 1.2; }
        section.main h1, section.main h2, section.main h3 { letter-spacing: 0; }
        button[kind="primary"], .stButton button { border-radius: 7px; }
        </style>
        """,
        unsafe_allow_html=True,
    )
