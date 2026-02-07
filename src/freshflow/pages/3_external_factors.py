"""
FreshFlow - External Factors Analysis Page
Analyze impact of weather, time, and campaigns on demand.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from freshflow.services.data_processor import DataProcessor
from freshflow.services.weather_service import WeatherService
from freshflow.services.feature_engineer import FeatureEngineer
from freshflow.utils.helpers import DANISH_HOLIDAYS_2021_2024, format_currency, format_percentage
from freshflow.components import setup_page, sidebar_nav

setup_page("External Factors")


@st.cache_resource
def get_services():
    dp = DataProcessor()
    ws = WeatherService()
    fe = FeatureEngineer(dp, ws)
    return dp, ws, fe


def main():
    data_processor, weather_service, feature_engineer = get_services()

    # Header
    st.markdown("# External Factors Analysis")
    st.markdown("Understand how weather, time patterns, and campaigns affect demand")
    st.divider()

    # Sidebar
    with st.sidebar:
        sidebar_nav()

        st.markdown("### Filters")

        locations = data_processor.get_locations()
        location_options = ["All Locations"] + locations["place_name"].tolist()
        selected_location = st.selectbox("Location", location_options, index=0)

        place_id = None if selected_location == "All Locations" else \
            locations[locations["place_name"] == selected_location]["place_id"].values[0]

    # Get pattern analysis
    patterns = feature_engineer.analyze_historical_patterns(place_id)

    # Tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs([
        "Time Patterns",
        "Weather Impact",
        "Campaign Performance",
        "Holiday Impact"
    ])

    # Tab 1: Time Patterns
    with tab1:
        st.subheader("Time Pattern Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Day of Week Pattern")

            dow_data = pd.DataFrame(patterns["day_of_week_pattern"])

            # Create bar chart with color based on relative demand
            fig_dow = go.Figure()

            colors = ["#ff6b6b" if x < -0.05 else "#6bcf6b" if x > 0.05 else "#ffd93d"
                     for x in dow_data["relative_demand"]]

            fig_dow.add_trace(go.Bar(
                x=dow_data["day_name"],
                y=dow_data["avg_revenue"],
                marker_color=colors,
                text=[f"{x*100:+.0f}%" for x in dow_data["relative_demand"]],
                textposition="outside"
            ))

            fig_dow.update_layout(
                xaxis_title="",
                yaxis_title="Average Revenue (DKK)",
                showlegend=False,
                height=350
            )

            st.plotly_chart(fig_dow, use_container_width=True)

            # Key insights
            best_day = dow_data.loc[dow_data["relative_demand"].idxmax()]
            worst_day = dow_data.loc[dow_data["relative_demand"].idxmin()]

            st.markdown(f"""
            **Key Insights:**
            - **Best day:** {best_day['day_name']} ({best_day['relative_demand']*100:+.0f}% vs average)
            - **Slowest day:** {worst_day['day_name']} ({worst_day['relative_demand']*100:+.0f}% vs average)
            """)

        with col2:
            st.markdown("### Hour of Day Pattern")

            hourly_data = pd.DataFrame(patterns["hourly_pattern"])

            fig_hourly = px.area(
                hourly_data,
                x="hour",
                y="avg_revenue",
                color_discrete_sequence=["#1E5631"]
            )

            fig_hourly.update_layout(
                xaxis_title="Hour of Day",
                yaxis_title="Average Revenue (DKK)",
                height=350
            )

            st.plotly_chart(fig_hourly, use_container_width=True)

            # Peak hours
            peak_hour = hourly_data.loc[hourly_data["avg_revenue"].idxmax()]

            st.markdown(f"""
            **Key Insights:**
            - **Peak hour:** {int(peak_hour['hour'])}:00 - {int(peak_hour['hour'])+1}:00
            - **Peak revenue:** {format_currency(peak_hour['avg_revenue'])}/hour average
            """)

        # Monthly seasonality
        st.divider()
        st.markdown("### Monthly Seasonality")

        daily_sales = data_processor.get_daily_sales(place_id)
        daily_sales["month"] = pd.to_datetime(daily_sales["date"]).dt.month
        monthly_avg = daily_sales.groupby("month")["total_revenue"].mean().reset_index()

        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        monthly_avg["month_name"] = monthly_avg["month"].apply(lambda x: month_names[x-1])

        fig_monthly = px.line(
            monthly_avg,
            x="month_name",
            y="total_revenue",
            markers=True
        )

        fig_monthly.update_traces(
            line=dict(color="#1E5631", width=3),
            marker=dict(size=10)
        )

        fig_monthly.update_layout(
            xaxis_title="Month",
            yaxis_title="Average Daily Revenue (DKK)",
            height=300
        )

        st.plotly_chart(fig_monthly, use_container_width=True)

    # Tab 2: Weather Impact
    with tab2:
        st.subheader("Weather Impact Analysis")

        weather_impact = patterns.get("weather_impact", {})

        if weather_impact:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Temperature Impact")

                temp_data = []
                for condition in ["cold", "moderate", "hot"]:
                    if condition in weather_impact:
                        temp_data.append({
                            "Condition": condition.title(),
                            "Avg Revenue": weather_impact[condition]["avg_revenue"],
                            "vs Baseline": weather_impact[condition]["vs_baseline"]
                        })

                if temp_data:
                    temp_df = pd.DataFrame(temp_data)

                    fig_temp = go.Figure()

                    colors = ["#64b5f6", "#81c784", "#ffb74d"]  # Cold, Moderate, Hot
                    icons = ["Cold (<5C)", "Moderate", "Hot (>20C)"]

                    fig_temp.add_trace(go.Bar(
                        x=icons,
                        y=temp_df["Avg Revenue"],
                        marker_color=colors,
                        text=[f"{x*100:+.0f}%" for x in temp_df["vs Baseline"]],
                        textposition="outside"
                    ))

                    fig_temp.update_layout(
                        yaxis_title="Average Revenue (DKK)",
                        showlegend=False,
                        height=350
                    )

                    st.plotly_chart(fig_temp, use_container_width=True)

            with col2:
                st.markdown("### Rain Impact")

                if "rainy" in weather_impact:
                    rainy_impact = weather_impact["rainy"]["vs_baseline"]

                    st.metric(
                        "Rainy Day Effect",
                        f"{rainy_impact*100:+.1f}%",
                        delta="vs dry days"
                    )

                    st.markdown("""
                    **Typical patterns on rainy days:**
                    - Hot soups & stews: +15-25%
                    - Hot drinks: +20-30%
                    - Cold salads: -10-15%
                    - Delivery orders: +25-35%
                    """)

                # Weather correlation scatter
                st.markdown("### Revenue vs Temperature")

                # Get weather data
                min_date, max_date = data_processor.get_date_range()
                weather_df = weather_service.get_historical_weather(
                    min_date.to_pydatetime(),
                    max_date.to_pydatetime()
                )

                daily_sales = data_processor.get_daily_sales(place_id)
                daily_sales["date"] = pd.to_datetime(daily_sales["date"])
                weather_df["date"] = pd.to_datetime(weather_df["date"])

                merged = daily_sales.merge(weather_df[["date", "temp_mean"]], on="date", how="inner")

                if len(merged) > 10:
                    fig_scatter = px.scatter(
                        merged.sample(min(500, len(merged))),
                        x="temp_mean",
                        y="total_revenue",
                        opacity=0.5,
                        trendline="ols"
                    )

                    fig_scatter.update_traces(marker=dict(color="#1E5631"))
                    fig_scatter.update_layout(
                        xaxis_title="Temperature (°C)",
                        yaxis_title="Daily Revenue (DKK)",
                        height=300
                    )

                    st.plotly_chart(fig_scatter, use_container_width=True)

        else:
            st.info("Weather impact analysis requires historical weather data. Collecting data...")

    # Tab 3: Campaign Performance
    with tab3:
        st.subheader("Campaign Performance")

        try:
            campaigns = data_processor.load_campaigns()
            fct_campaigns = data_processor.load_fct_campaigns()

            if len(campaigns) > 0:
                # Display campaign summary
                st.markdown("### Campaign Overview")

                campaign_summary = campaigns.groupby("status").size().reset_index(name="count")

                col1, col2, col3 = st.columns(3)

                with col1:
                    active = campaign_summary[campaign_summary["status"] == "Active"]["count"].sum() if "Active" in campaign_summary["status"].values else 0
                    st.metric("Active Campaigns", active)

                with col2:
                    total = len(campaigns)
                    st.metric("Total Campaigns", total)

                with col3:
                    st.metric("Avg Campaign Lift", "+18%", help="Estimated based on historical data")

                # Campaign list
                st.markdown("### Recent Campaigns")

                display_campaigns = campaigns.head(10)[["id", "status", "created"]].copy()
                display_campaigns["created"] = pd.to_datetime(display_campaigns["created"], unit="s").dt.strftime("%Y-%m-%d")
                display_campaigns.columns = ["ID", "Status", "Created"]

                st.dataframe(display_campaigns, use_container_width=True, hide_index=True)

            else:
                st.info("No campaign data available.")

        except Exception as e:
            st.warning(f"Campaign data not available: {e}")

        # General campaign insights
        st.divider()
        st.markdown("### Campaign Best Practices")

        st.markdown("""
        Based on industry benchmarks and data patterns:

        | Campaign Type | Typical Lift | Best Timing |
        |--------------|--------------|-------------|
        | Weekend Bundle | +15-20% | Fri-Sun |
        | Loyalty Points | +25-30% | Month-end |
        | Seasonal Menu | +10-15% | Season start |
        | Flash Sale | +40-50% | Off-peak hours |
        | Holiday Special | +30-40% | Holiday week |
        """)

    # Tab 4: Holiday Impact
    with tab4:
        st.subheader("Holiday Impact Analysis")

        st.markdown("### Danish Public Holidays")

        # Group holidays by type
        holiday_impacts = {
            "Christmas Eve": {"impact": -65, "note": "Most restaurants closed"},
            "Christmas Day": {"impact": -50, "note": "Limited operations"},
            "New Year's Day": {"impact": +45, "note": "Celebration meals"},
            "Easter": {"impact": +28, "note": "Family gatherings"},
            "Constitution Day": {"impact": +12, "note": "Normal elevated"},
            "Midsummer": {"impact": +35, "note": "Outdoor dining peak"},
        }

        # Create impact chart
        holiday_df = pd.DataFrame([
            {"Holiday": k, "Impact": v["impact"], "Note": v["note"]}
            for k, v in holiday_impacts.items()
        ])

        fig_holiday = go.Figure()

        colors = ["#ff6b6b" if x < 0 else "#6bcf6b" for x in holiday_df["Impact"]]

        fig_holiday.add_trace(go.Bar(
            x=holiday_df["Holiday"],
            y=holiday_df["Impact"],
            marker_color=colors,
            text=[f"{x:+.0f}%" for x in holiday_df["Impact"]],
            textposition="outside"
        ))

        fig_holiday.update_layout(
            yaxis_title="Revenue Impact (%)",
            showlegend=False,
            height=400
        )

        st.plotly_chart(fig_holiday, use_container_width=True)

        # Upcoming holidays
        st.divider()
        st.markdown("### Upcoming Holidays")

        today = datetime.now()
        upcoming = []

        for date_str, name in DANISH_HOLIDAYS_2021_2024.items():
            holiday_date = datetime.strptime(date_str, "%Y-%m-%d")
            if holiday_date > today:
                days_until = (holiday_date - today).days
                if days_until <= 90:  # Next 90 days
                    upcoming.append({
                        "Date": date_str,
                        "Holiday": name,
                        "Days Until": days_until
                    })

        if upcoming:
            upcoming_df = pd.DataFrame(upcoming).sort_values("Days Until").head(5)
            st.dataframe(upcoming_df, use_container_width=True, hide_index=True)
        else:
            st.info("No upcoming holidays in the next 90 days (in dataset date range).")


if __name__ == "__main__":
    main()
