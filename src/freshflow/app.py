"""
FreshFlow - Intelligent Demand Forecasting Dashboard
Main Streamlit application entry point.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from freshflow.services.data_processor import DataProcessor
from freshflow.services.weather_service import WeatherService
from freshflow.services.feature_engineer import FeatureEngineer
from freshflow.models.predictor import ForecastService
from freshflow.utils.helpers import format_currency, format_percentage, get_trend_indicator

# Page configuration
st.set_page_config(
    page_title="FreshFlow - Demand Forecasting",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E5631;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-top: 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1E5631;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
    }
    .alert-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 10px 15px;
        margin: 5px 0;
        border-radius: 0 5px 5px 0;
    }
    .alert-box-info {
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
    }
    .stMetric > div {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_data_processor():
    """Initialize and cache data processor."""
    return DataProcessor()


@st.cache_resource
def get_weather_service():
    """Initialize and cache weather service."""
    return WeatherService()


@st.cache_resource
def get_forecast_service(_data_processor, _weather_service):
    """Initialize and cache forecast service."""
    return ForecastService(_data_processor, _weather_service)


def main():
    """Main application entry point."""
    # Initialize services
    data_processor = get_data_processor()
    weather_service = get_weather_service()
    forecast_service = get_forecast_service(data_processor, weather_service)

    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/leaf.png", width=60)
        st.markdown("## FreshFlow")
        st.markdown("*Intelligent Demand Forecasting*")
        st.divider()

        # Location selector
        locations = data_processor.get_locations()
        location_options = ["All Locations"] + locations["place_name"].tolist()
        selected_location = st.selectbox(
            "📍 Select Location",
            location_options,
            index=0
        )

        if selected_location == "All Locations":
            place_id = None
        else:
            place_id = locations[locations["place_name"] == selected_location]["place_id"].values[0]

        st.divider()

        # Date range info
        min_date, max_date = data_processor.get_date_range()
        st.markdown("**Data Range**")
        st.caption(f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")

        st.divider()

        # Navigation
        st.markdown("**Quick Links**")
        st.page_link("pages/1_forecasting.py", label="📈 Forecasting Details", icon="📈")
        st.page_link("pages/2_kitchen_prep.py", label="🍳 Kitchen Prep", icon="🍳")
        st.page_link("pages/3_external_factors.py", label="🌤️ External Factors", icon="🌤️")

    # Main content
    st.markdown('<p class="main-header">🌿 FreshFlow Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Intelligent Demand Forecasting for Fresh Flow Markets</p>', unsafe_allow_html=True)
    st.divider()

    # KPI Row
    col1, col2, col3, col4 = st.columns(4)

    # Get data for KPIs
    daily_sales = data_processor.get_daily_sales(place_id)
    accuracy_metrics = forecast_service.get_accuracy_metrics(place_id)

    # Today's metrics (using most recent data as proxy)
    recent_revenue = daily_sales.tail(1)["total_revenue"].values[0] if len(daily_sales) > 0 else 0
    prev_revenue = daily_sales.tail(2).head(1)["total_revenue"].values[0] if len(daily_sales) > 1 else recent_revenue

    # Weekly trend
    last_week = daily_sales.tail(7)["total_revenue"].sum() if len(daily_sales) >= 7 else 0
    prev_week = daily_sales.tail(14).head(7)["total_revenue"].sum() if len(daily_sales) >= 14 else last_week
    weekly_change = (last_week - prev_week) / prev_week if prev_week > 0 else 0

    with col1:
        st.metric(
            label="💰 Latest Day Revenue",
            value=format_currency(recent_revenue),
            delta=f"{((recent_revenue - prev_revenue) / prev_revenue * 100):.1f}%" if prev_revenue > 0 else "N/A"
        )

    with col2:
        trend_arrow, trend_color = get_trend_indicator(last_week, prev_week)
        st.metric(
            label="📈 Weekly Trend",
            value=f"{weekly_change * 100:+.1f}%",
            delta=f"{trend_arrow} vs prev week"
        )

    with col3:
        accuracy = accuracy_metrics.get("accuracy", 0.90)
        st.metric(
            label="🎯 Forecast Accuracy",
            value=f"{accuracy * 100:.1f}%",
            delta="Based on patterns"
        )

    with col4:
        waste_risk = accuracy_metrics.get("stockout_count", 0)
        st.metric(
            label="⚠️ Stockout Risk",
            value=f"{waste_risk} items",
            delta="Monitor closely" if waste_risk > 0 else "All clear"
        )

    st.divider()

    # Main content area - two columns
    left_col, right_col = st.columns([2, 1])

    with left_col:
        st.subheader("📈 7-Day Demand Forecast")

        # Get forecast
        forecast_df = forecast_service.forecast_revenue(place_id, days=7)

        # Create forecast chart
        fig = go.Figure()

        # Confidence band
        if "confidence_lower" in forecast_df.columns:
            fig.add_trace(go.Scatter(
                x=forecast_df["date"],
                y=forecast_df["confidence_upper"],
                fill=None,
                mode="lines",
                line=dict(color="rgba(30, 86, 49, 0.1)"),
                showlegend=False,
                name="Upper Bound"
            ))
            fig.add_trace(go.Scatter(
                x=forecast_df["date"],
                y=forecast_df["confidence_lower"],
                fill="tonexty",
                mode="lines",
                line=dict(color="rgba(30, 86, 49, 0.1)"),
                fillcolor="rgba(30, 86, 49, 0.1)",
                showlegend=False,
                name="Lower Bound"
            ))

        # Forecast line
        fig.add_trace(go.Scatter(
            x=forecast_df["date"],
            y=forecast_df["predicted_revenue"],
            mode="lines+markers",
            line=dict(color="#1E5631", width=3),
            marker=dict(size=10),
            name="Predicted Revenue"
        ))

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Revenue (DKK)",
            hovermode="x unified",
            showlegend=True,
            height=350,
            margin=dict(l=0, r=0, t=30, b=0)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Forecast table
        with st.expander("View Forecast Details"):
            display_df = forecast_df[["date", "day_name", "predicted_revenue"]].copy()
            display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
            display_df["predicted_revenue"] = display_df["predicted_revenue"].apply(
                lambda x: format_currency(x)
            )
            display_df.columns = ["Date", "Day", "Predicted Revenue"]
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    with right_col:
        # Top Items Today
        st.subheader("🔥 Top Selling Items")
        top_items = data_processor.get_top_items(place_id, top_n=5)

        for idx, row in top_items.iterrows():
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.markdown(f"**{row['item_title'][:25]}**" + ("..." if len(row['item_title']) > 25 else ""))
            with col_b:
                st.markdown(f"`{int(row['total_quantity'])}`")

        st.divider()

        # Alerts Section
        st.subheader("⚡ Smart Alerts")

        # Weather alert
        weather_forecast = weather_service.get_forecast(2)
        if len(weather_forecast) > 0:
            tomorrow_weather = weather_service.get_weather_impact_summary(weather_forecast.iloc[[1]])

            weather_icon = "☀️" if tomorrow_weather["icon"] == "sunny" else (
                "🌧️" if tomorrow_weather["icon"] == "rainy" else (
                    "❄️" if tomorrow_weather["icon"] == "snowy" else "☁️"
                )
            )

            if tomorrow_weather.get("is_rainy"):
                st.markdown(f"""
                <div class="alert-box">
                    {weather_icon} <strong>Rain expected tomorrow</strong><br>
                    <small>Expect +10-15% in hot meals & soups</small>
                </div>
                """, unsafe_allow_html=True)
            elif tomorrow_weather.get("is_cold"):
                st.markdown(f"""
                <div class="alert-box alert-box-info">
                    {weather_icon} <strong>Cold weather ({tomorrow_weather['temp']}°C)</strong><br>
                    <small>Hot drinks demand likely to increase</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="alert-box alert-box-info">
                    {weather_icon} <strong>{tomorrow_weather['condition']}</strong><br>
                    <small>Temperature: {tomorrow_weather['temp']}°C</small>
                </div>
                """, unsafe_allow_html=True)

        # Weekend alert
        tomorrow = datetime.now() + timedelta(days=1)
        if tomorrow.weekday() >= 4:  # Friday, Saturday, Sunday
            day_name = tomorrow.strftime("%A")
            st.markdown(f"""
            <div class="alert-box">
                📅 <strong>{day_name} approaching</strong><br>
                <small>Expect higher demand - prepare extra inventory</small>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # Bottom section - Day of Week Pattern
    st.subheader("📊 Weekly Demand Pattern")

    dow_pattern = data_processor.get_day_of_week_pattern(place_id)

    fig_dow = px.bar(
        dow_pattern,
        x="day_name",
        y="avg_revenue",
        color="relative_demand",
        color_continuous_scale=["#ff6b6b", "#ffd93d", "#6bcf6b"],
        labels={"day_name": "Day", "avg_revenue": "Avg Revenue (DKK)", "relative_demand": "vs Average"}
    )

    fig_dow.update_layout(
        xaxis_title="",
        yaxis_title="Average Daily Revenue (DKK)",
        showlegend=False,
        height=300,
        margin=dict(l=0, r=0, t=30, b=0)
    )

    st.plotly_chart(fig_dow, use_container_width=True)

    # Footer
    st.divider()
    st.caption("🌿 FreshFlow - Powered by LSTM Deep Learning | Deloitte x AUC Hackathon 2024")


if __name__ == "__main__":
    main()
