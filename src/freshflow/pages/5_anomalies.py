"""
FreshFlow - Anomaly Detection & Alerts Page
Detect unusual sales patterns with automatic alerts.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from freshflow.services.data_processor import DataProcessor
from freshflow.services.anomaly_service import AnomalyDetector, AnomalyType, AlertSeverity
from freshflow.utils.helpers import format_currency

st.set_page_config(page_title="Anomalies - FreshFlow", page_icon="🚨", layout="wide")


@st.cache_resource
def get_services():
    dp = DataProcessor()
    ad = AnomalyDetector(dp)
    return dp, ad


def severity_color(severity):
    colors = {
        AlertSeverity.CRITICAL: "#dc3545",
        AlertSeverity.HIGH: "#fd7e14",
        AlertSeverity.MEDIUM: "#ffc107",
        AlertSeverity.LOW: "#28a745"
    }
    return colors.get(severity, "#6c757d")


def severity_icon(severity):
    icons = {
        AlertSeverity.CRITICAL: "🔴",
        AlertSeverity.HIGH: "🟠",
        AlertSeverity.MEDIUM: "🟡",
        AlertSeverity.LOW: "🟢"
    }
    return icons.get(severity, "⚪")


def main():
    dp, anomaly_detector = get_services()

    st.markdown("# 🚨 Anomaly Detection & Alerts")
    st.markdown("Automatically detect unusual sales patterns and get actionable insights")
    st.divider()

    # Sidebar
    with st.sidebar:
        st.markdown("### Settings")
        locations = dp.get_locations()
        location_options = ["All Locations"] + locations["place_name"].tolist()
        selected_location = st.selectbox("📍 Location", location_options)

        place_id = None if selected_location == "All Locations" else \
            locations[locations["place_name"] == selected_location]["place_id"].values[0]

        lookback = st.slider("Analysis Period (days)", 30, 180, 90)
        sensitivity = st.select_slider(
            "Sensitivity",
            options=["Low", "Medium", "High"],
            value="Medium"
        )

        z_threshold = {"Low": 3.0, "Medium": 2.5, "High": 2.0}[sensitivity]

        st.divider()
        st.page_link("app.py", label="← Back to Dashboard", icon="🏠")

    # Get anomaly summary
    summary = anomaly_detector.get_anomaly_summary(place_id, days=lookback)
    anomalies = anomaly_detector.detect_anomalies(place_id, lookback, z_threshold)

    # Alert summary cards
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Anomalies", summary['total_anomalies'])
    with col2:
        st.metric("🔴 Critical", summary['critical'])
    with col3:
        st.metric("🟠 High", summary['high'])
    with col4:
        st.metric("🟡 Medium", summary['medium'])
    with col5:
        st.metric("🟢 Low", summary['low'])

    st.divider()

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Anomaly Timeline", "🚨 Alert List", "📈 Analysis"])

    with tab1:
        st.subheader("📊 Sales with Anomaly Highlights")

        daily_sales = dp.get_daily_sales(place_id)
        daily_sales['date'] = pd.to_datetime(daily_sales['date'])
        recent = daily_sales.tail(lookback).copy()

        # Mark anomalies on the chart
        anomaly_dates = [a.date for a in anomalies]
        recent['is_anomaly'] = recent['date'].isin(anomaly_dates)

        # Create figure
        fig = go.Figure()

        # Normal days
        normal = recent[~recent['is_anomaly']]
        fig.add_trace(go.Scatter(
            x=normal['date'],
            y=normal['total_revenue'],
            mode='lines',
            name='Normal',
            line=dict(color='#3498db', width=2)
        ))

        # Anomaly days
        anomaly_data = recent[recent['is_anomaly']]
        if len(anomaly_data) > 0:
            fig.add_trace(go.Scatter(
                x=anomaly_data['date'],
                y=anomaly_data['total_revenue'],
                mode='markers',
                name='Anomalies',
                marker=dict(color='#e74c3c', size=12, symbol='x')
            ))

        # Add average line
        avg = recent['total_revenue'].mean()
        fig.add_hline(y=avg, line_dash="dash", line_color="gray",
                     annotation_text=f"Avg: {format_currency(avg)}")

        # Add std bands
        std = recent['total_revenue'].std()
        fig.add_hrect(y0=avg - 2*std, y1=avg + 2*std,
                     fillcolor="green", opacity=0.1,
                     annotation_text="Normal range")

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Revenue (DKK)",
            hovermode="x unified",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("🚨 Detected Anomalies")

        if anomalies:
            for anomaly in anomalies[:20]:
                severity_emoji = severity_icon(anomaly.severity)

                with st.expander(
                    f"{severity_emoji} {anomaly.date.strftime('%Y-%m-%d')} - "
                    f"{anomaly.type.value.replace('_', ' ').title()} "
                    f"({anomaly.deviation_pct:+.1f}%)",
                    expanded=(anomaly.severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH])
                ):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Actual", format_currency(anomaly.actual_value))
                    with col2:
                        st.metric("Expected", format_currency(anomaly.expected_value))
                    with col3:
                        st.metric("Deviation", f"{anomaly.deviation_pct:+.1f}%")

                    st.markdown("**Possible Causes:**")
                    for cause in anomaly.possible_causes:
                        st.markdown(f"- {cause}")

                    st.markdown("**Recommended Actions:**")
                    for action in anomaly.recommended_actions:
                        st.markdown(f"- {action}")
        else:
            st.success("✅ No anomalies detected in the selected period!")

    with tab3:
        st.subheader("📈 Anomaly Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Anomaly Types")

            type_counts = {
                'Spikes': summary['spikes'],
                'Drops': summary['drops'],
                'Trend Changes': summary['trend_changes']
            }

            fig_types = px.pie(
                values=list(type_counts.values()),
                names=list(type_counts.keys()),
                color_discrete_sequence=['#2ecc71', '#e74c3c', '#f39c12']
            )
            fig_types.update_layout(height=300)
            st.plotly_chart(fig_types, use_container_width=True)

        with col2:
            st.markdown("### Severity Distribution")

            severity_counts = {
                'Critical': summary['critical'],
                'High': summary['high'],
                'Medium': summary['medium'],
                'Low': summary['low']
            }

            fig_severity = px.bar(
                x=list(severity_counts.keys()),
                y=list(severity_counts.values()),
                color=list(severity_counts.keys()),
                color_discrete_map={
                    'Critical': '#dc3545',
                    'High': '#fd7e14',
                    'Medium': '#ffc107',
                    'Low': '#28a745'
                }
            )
            fig_severity.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_severity, use_container_width=True)

        # Day of week analysis
        st.markdown("### Anomalies by Day of Week")

        if anomalies:
            dow_data = pd.DataFrame([
                {'day': a.date.strftime('%A'), 'count': 1}
                for a in anomalies
            ])
            dow_counts = dow_data.groupby('day')['count'].sum().reindex([
                'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
            ]).fillna(0)

            fig_dow = px.bar(x=dow_counts.index, y=dow_counts.values)
            fig_dow.update_layout(
                xaxis_title="Day of Week",
                yaxis_title="Number of Anomalies",
                height=300
            )
            st.plotly_chart(fig_dow, use_container_width=True)


if __name__ == "__main__":
    main()
