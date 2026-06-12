import html

import streamlit as st


PAGES = [
    {"key": "Match Overview", "label": "Match Overview", "icon": "⚽", "group": "main"},
    {"key": "Betting Center", "label": "Betting Center", "icon": "🎯", "group": "main"},
    {"key": "My Bets", "label": "My Bets", "icon": "🧾", "group": "main"},
    {"key": "Model Performance", "label": "Model Performance", "icon": "📊", "group": "main"},
    {"key": "Settings", "label": "Settings", "icon": "⚙️", "group": "main"},
    {"key": "Advanced / Admin", "label": "Advanced / Admin", "icon": "🛠", "group": "advanced"},
]


def apply_navigation_css() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] .wc-sidebar-brand {
            margin: 0.15rem 0 0.85rem;
        }
        [data-testid="stSidebar"] .wc-sidebar-brand-main {
            color: #f8fafc;
            font-size: 1.18rem;
            font-weight: 850;
            letter-spacing: 0;
            line-height: 1.1;
        }
        [data-testid="stSidebar"] .wc-sidebar-brand-sub {
            color: #93c5fd;
            font-size: 0.82rem;
            font-weight: 650;
            margin-top: 0.08rem;
        }
        [data-testid="stSidebar"] .wc-nav-section-label {
            color: #94a3b8;
            font-size: 0.68rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            margin: 0.7rem 0 0.25rem;
            text-transform: uppercase;
        }
        [data-testid="stSidebar"] .stButton {
            margin-bottom: 0.22rem;
        }
        [data-testid="stSidebar"] .stButton > button {
            justify-content: flex-start;
            min-height: 2.15rem;
            padding: 0.35rem 0.58rem;
            border-radius: 8px;
            border: 1px solid rgba(148, 163, 184, 0.18);
            background: rgba(15, 23, 42, 0.16);
            color: #dbeafe;
            font-weight: 650;
            text-align: left;
            transition: background 120ms ease, border-color 120ms ease, transform 120ms ease;
        }
        [data-testid="stSidebar"] .stButton > button:hover {
            background: rgba(59, 130, 246, 0.16);
            border-color: rgba(147, 197, 253, 0.34);
            color: #ffffff;
            transform: translateX(1px);
        }
        [data-testid="stSidebar"] .stButton > button[kind="primary"] {
            background: linear-gradient(90deg, rgba(37, 99, 235, 0.42), rgba(14, 165, 233, 0.16));
            border-color: rgba(147, 197, 253, 0.52);
            border-left: 3px solid #60a5fa;
            color: #ffffff;
            font-weight: 800;
            box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.04);
        }
        [data-testid="stSidebar"] .wc-sidebar-status-title {
            color: #e5e7eb;
            font-size: 0.72rem;
            font-weight: 850;
            letter-spacing: 0.08em;
            margin: 0.85rem 0 0.35rem;
            text-transform: uppercase;
        }
        [data-testid="stSidebar"] .wc-sidebar-status-card {
            background: rgba(15, 23, 42, 0.24);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 8px;
            margin-bottom: 0.32rem;
            padding: 0.42rem 0.52rem;
        }
        [data-testid="stSidebar"] .wc-sidebar-status-label {
            color: #94a3b8;
            font-size: 0.68rem;
            font-weight: 750;
            line-height: 1.15;
        }
        [data-testid="stSidebar"] .wc-sidebar-status-value {
            color: #f8fafc;
            font-size: 0.82rem;
            font-weight: 760;
            line-height: 1.25;
            margin-top: 0.08rem;
            overflow-wrap: anywhere;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_navigation(pages: list[dict], active_page: str) -> str:
    selected_page = active_page
    for page in [page for page in pages if page["group"] == "main"]:
        if st.sidebar.button(
            f"{page['icon']}  {page['label']}",
            key=f"nav_{page['key']}",
            type="primary" if page["key"] == active_page else "secondary",
            use_container_width=True,
        ):
            selected_page = page["key"]

    st.sidebar.markdown('<div class="wc-nav-section-label">Advanced</div>', unsafe_allow_html=True)
    for page in [page for page in pages if page["group"] == "advanced"]:
        if st.sidebar.button(
            f"{page['icon']}  {page['label']}",
            key=f"nav_{page['key']}",
            type="primary" if page["key"] == active_page else "secondary",
            use_container_width=True,
        ):
            selected_page = page["key"]

    return selected_page


def render_sidebar_status_card(label: str, value: str) -> None:
    st.sidebar.markdown(
        f"""
        <div class="wc-sidebar-status-card">
          <div class="wc-sidebar-status-label">{html.escape(str(label))}</div>
          <div class="wc-sidebar-status-value">{html.escape(str(value))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
