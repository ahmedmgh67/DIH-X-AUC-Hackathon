"""
FreshFlow - Reusable UI Components
Centralized components for consistent UI across all pages.
"""

import streamlit as st
import pandas as pd
from typing import Optional, List, Dict, Any, Tuple


def setup_page(title: str, icon: str, layout: str = "wide"):
    """Standard page configuration."""
    st.set_page_config(page_title=f"{title} - FreshFlow", page_icon=icon, layout=layout)

    # Add custom CSS for better styling
    st.markdown("""
    <style>
    /* Better metric cards */
    div[data-testid="metric-container"] {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border-left: 4px solid #4CAF50;
    }

    /* Improved sidebar */
    section[data-testid="stSidebar"] {
        background-color: #f1f3f4;
    }

    /* Better tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 8px;
        padding: 10px 20px;
    }

    /* Better expanders */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 8px;
    }

    /* Data tables */
    .dataframe {
        font-size: 14px;
    }

    /* Status badges */
    .status-critical { color: #dc3545; font-weight: bold; }
    .status-warning { color: #fd7e14; font-weight: bold; }
    .status-success { color: #28a745; font-weight: bold; }
    .status-info { color: #17a2b8; font-weight: bold; }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)


def location_selector(data_processor, key: str = "location") -> Tuple[str, Optional[int]]:
    """
    Reusable location selector component.

    Returns:
        Tuple of (selected_location_name, place_id or None)
    """
    locations = data_processor.get_locations()
    location_options = ["All Locations"] + locations["place_name"].tolist()

    selected_location = st.selectbox(
        "📍 Location",
        location_options,
        key=key
    )

    if selected_location == "All Locations":
        place_id = None
    else:
        place_id = locations[locations["place_name"] == selected_location]["place_id"].values[0]

    return selected_location, place_id


def sidebar_nav():
    """Standard sidebar navigation."""
    st.divider()
    st.page_link("app.py", label="← Back to Dashboard", icon="🏠")


def metric_row(metrics: List[Dict[str, Any]], cols: int = 4):
    """
    Display a row of metrics.

    Args:
        metrics: List of dicts with 'label', 'value', and optional 'delta', 'help'
        cols: Number of columns
    """
    columns = st.columns(cols)

    for i, metric in enumerate(metrics):
        with columns[i % cols]:
            st.metric(
                label=metric.get('label', ''),
                value=metric.get('value', ''),
                delta=metric.get('delta'),
                help=metric.get('help')
            )


def kpi_card(title: str, value: str, subtitle: str = "", icon: str = "", delta: str = None):
    """Display a styled KPI card."""
    delta_html = f'<span style="color: {"#28a745" if delta and delta.startswith("+") else "#dc3545"}">{delta}</span>' if delta else ""

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px; border-radius: 12px; color: white; margin-bottom: 10px;">
        <div style="font-size: 14px; opacity: 0.9;">{icon} {title}</div>
        <div style="font-size: 28px; font-weight: bold; margin: 8px 0;">{value}</div>
        <div style="font-size: 12px; opacity: 0.8;">{subtitle} {delta_html}</div>
    </div>
    """, unsafe_allow_html=True)


def alert_box(message: str, type: str = "info"):
    """
    Display styled alert box.

    Args:
        message: Alert message
        type: 'success', 'warning', 'error', 'info'
    """
    colors = {
        'success': ('#d4edda', '#155724', '✅'),
        'warning': ('#fff3cd', '#856404', '⚠️'),
        'error': ('#f8d7da', '#721c24', '❌'),
        'info': ('#d1ecf1', '#0c5460', 'ℹ️')
    }
    bg, text, icon = colors.get(type, colors['info'])

    st.markdown(f"""
    <div style="background-color: {bg}; color: {text}; padding: 15px;
                border-radius: 8px; margin: 10px 0;">
        {icon} {message}
    </div>
    """, unsafe_allow_html=True)


def progress_indicator(current: float, target: float, label: str = "Progress"):
    """Display a progress indicator with percentage."""
    pct = min(100, (current / target * 100)) if target > 0 else 0
    color = "#28a745" if pct >= 80 else "#fd7e14" if pct >= 50 else "#dc3545"

    st.markdown(f"**{label}**: {current:.0f} / {target:.0f} ({pct:.1f}%)")
    st.progress(pct / 100)


def data_table(df: pd.DataFrame, title: str = None, max_rows: int = 20):
    """Display a styled data table with optional title."""
    if title:
        st.markdown(f"### {title}")

    if len(df) > max_rows:
        st.caption(f"Showing {max_rows} of {len(df)} rows")
        df = df.head(max_rows)

    st.dataframe(df, use_container_width=True, hide_index=True)


def empty_state(message: str, icon: str = "📭"):
    """Display an empty state placeholder."""
    st.markdown(f"""
    <div style="text-align: center; padding: 40px; color: #6c757d;">
        <div style="font-size: 48px;">{icon}</div>
        <div style="font-size: 18px; margin-top: 10px;">{message}</div>
    </div>
    """, unsafe_allow_html=True)


def loading_placeholder(message: str = "Loading..."):
    """Display a loading placeholder."""
    return st.empty()


def format_currency(amount: float, currency: str = "DKK") -> str:
    """Format amount as currency with proper formatting."""
    if amount >= 1_000_000:
        return f"{amount/1_000_000:.1f}M {currency}"
    elif amount >= 1_000:
        return f"{amount/1_000:.1f}K {currency}"
    else:
        return f"{amount:,.0f} {currency}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format value as percentage."""
    return f"{value:.{decimals}f}%"


def format_number(value: float, decimals: int = 0) -> str:
    """Format number with thousand separators."""
    if decimals == 0:
        return f"{value:,.0f}"
    return f"{value:,.{decimals}f}"


def trend_indicator(current: float, previous: float) -> str:
    """Return trend arrow based on change."""
    if previous == 0:
        return "→"
    change = (current - previous) / previous
    if change > 0.05:
        return "↑"
    elif change < -0.05:
        return "↓"
    return "→"


def status_badge(status: str) -> str:
    """Return colored status badge HTML."""
    colors = {
        'critical': '#dc3545',
        'high': '#fd7e14',
        'medium': '#ffc107',
        'low': '#28a745',
        'adequate': '#28a745',
        'overstocked': '#17a2b8'
    }
    color = colors.get(status.lower(), '#6c757d')
    return f'<span style="background-color: {color}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px;">{status.upper()}</span>'
