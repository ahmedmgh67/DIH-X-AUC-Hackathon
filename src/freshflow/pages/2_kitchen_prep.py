"""
FreshFlow - Kitchen Prep Optimizer Page
Calculate optimal prep quantities to minimize waste.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from freshflow.services.data_processor import DataProcessor
from freshflow.services.weather_service import WeatherService
from freshflow.services.feature_engineer import FeatureEngineer
from freshflow.models.predictor import ForecastService
from freshflow.utils.helpers import format_currency
from freshflow.components import setup_page, sidebar_nav

setup_page("Kitchen Prep")


@st.cache_resource
def get_services():
    dp = DataProcessor()
    ws = WeatherService()
    fs = ForecastService(dp, ws)
    fe = FeatureEngineer(dp, ws)
    return dp, ws, fs, fe


def main():
    data_processor, weather_service, forecast_service, feature_engineer = get_services()

    # Header
    st.markdown("# Kitchen Prep Optimizer")
    st.markdown("Calculate optimal prep quantities to minimize waste and stockouts")
    st.divider()

    # Sidebar settings
    with st.sidebar:
        sidebar_nav()

        st.markdown("### Settings")

        # Location
        locations = data_processor.get_locations()
        location_options = ["All Locations"] + locations["place_name"].tolist()
        selected_location = st.selectbox("Location", location_options, index=0)

        place_id = None if selected_location == "All Locations" else \
            locations[locations["place_name"] == selected_location]["place_id"].values[0]

        # Prep date
        st.markdown("### Prep Date")
        date_options = {
            "Tomorrow": datetime.now() + timedelta(days=1),
            "Day After Tomorrow": datetime.now() + timedelta(days=2),
            "This Weekend (Sat)": datetime.now() + timedelta(days=(5 - datetime.now().weekday()) % 7),
        }
        selected_date_label = st.selectbox("Prep For", list(date_options.keys()))
        prep_date = date_options[selected_date_label]

        # Buffer percentage
        buffer_pct = st.slider(
            "Safety Buffer",
            min_value=5,
            max_value=25,
            value=10,
            step=5,
            help="Extra percentage to add for safety margin"
        ) / 100

    # Context Banner
    weather_forecast = weather_service.get_forecast(7)
    prep_date_weather = weather_forecast[weather_forecast["date"].dt.date == prep_date.date()]

    if len(prep_date_weather) > 0:
        weather_summary = weather_service.get_weather_impact_summary(prep_date_weather)
    else:
        weather_summary = {"condition": "Unknown", "temp": 10, "icon": "cloudy"}

    day_adj, day_reason = feature_engineer.get_day_adjustment(prep_date)
    weather_adj = feature_engineer.get_weather_adjustment(weather_summary)

    st.markdown(f"""
    <div class="context-banner">
        <h4>Prep List for: {prep_date.strftime("%A, %B %d, %Y")}</h4>
        <p>
            <strong>Weather:</strong> {weather_summary['condition']}, {weather_summary['temp']}°C |
            <strong>Day Type:</strong> {day_reason} |
            <strong>Buffer:</strong> {buffer_pct*100:.0f}%
        </p>
        <p><strong>Demand Adjustment:</strong> {(day_adj * weather_adj - 1) * 100:+.0f}% (weather + day effects)</p>
    </div>
    """, unsafe_allow_html=True)

    # Get prep quantities
    prep_data = forecast_service.get_prep_quantities(
        place_id=place_id,
        date=prep_date,
        buffer_pct=buffer_pct,
        top_n=20
    )

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Items", prep_data["total_items"])

    with col2:
        st.metric("Total Prep Qty", prep_data["total_prep_qty"])

    with col3:
        total_value = sum(
            item["prep_quantity"] * item["avg_price"]
            for item in prep_data["prep_list"]
        )
        st.metric("Est. Value", format_currency(total_value))

    with col4:
        st.metric("Buffer Applied", f"{buffer_pct*100:.0f}%")

    st.divider()

    # Prep list table
    st.subheader("Prep Quantities")

    prep_df = pd.DataFrame(prep_data["prep_list"])

    if len(prep_df) > 0:
        # Add confidence indicator
        def get_confidence_badge(conf):
            if conf == "high":
                return "High"
            elif conf == "medium":
                return "Medium"
            else:
                return "Low"

        prep_df["confidence_badge"] = prep_df["confidence"].apply(get_confidence_badge)

        # Format display
        display_df = prep_df[[
            "item_title", "predicted_quantity", "buffer_qty",
            "prep_quantity", "confidence_badge", "avg_price"
        ]].copy()

        display_df["avg_price"] = display_df["avg_price"].apply(lambda x: format_currency(x))
        display_df.columns = [
            "Item", "Forecast Qty", "Buffer", "Prep Qty", "Confidence", "Unit Price"
        ]

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Summary by confidence
        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Historical Accuracy")

            accuracy_metrics = forecast_service.get_accuracy_metrics(place_id)

            metrics_col1, metrics_col2 = st.columns(2)
            with metrics_col1:
                st.metric("Forecast Accuracy", f"{accuracy_metrics['accuracy']*100:.1f}%")
                st.metric("Avg Waste Rate", f"{accuracy_metrics['avg_waste_pct']*100:.1f}%")

            with metrics_col2:
                st.metric("Recent Stockouts", f"{accuracy_metrics['stockout_count']} items")
                st.metric("Target Waste", "< 5%")

            # Progress bar for waste
            waste_pct = accuracy_metrics['avg_waste_pct']
            st.progress(min(waste_pct / 0.10, 1.0), text=f"Waste: {waste_pct*100:.1f}% (Target: <5%)")

        with col2:
            st.subheader("Recommendations")

            # Smart recommendations based on data
            recommendations = []

            if day_adj > 1.15:
                recommendations.append(f"**High demand day** - {day_reason}. Consider extra prep for popular items.")

            if weather_summary.get("is_rainy"):
                recommendations.append("**Rain expected** - Hot soups and comfort foods typically see +15-20% demand.")

            if weather_summary.get("is_cold"):
                recommendations.append("**Cold weather** - Hot drinks and warm meals will be in higher demand.")

            if buffer_pct < 0.10:
                recommendations.append("**Low buffer** - Consider increasing buffer to 10% to reduce stockout risk.")

            if not recommendations:
                recommendations.append("**Normal conditions** - Standard prep quantities should be sufficient.")

            for rec in recommendations:
                st.markdown(rec)

        st.divider()

        # Export options
        st.subheader("Export Options")

        col1, col2, col3 = st.columns(3)

        with col1:
            csv = prep_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"prep_list_{prep_date.strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col2:
            # Create printable version
            print_content = f"""
KITCHEN PREP LIST
=================
Date: {prep_date.strftime('%A, %B %d, %Y')}
Location: {selected_location}
Weather: {weather_summary['condition']}, {weather_summary['temp']}°C
Buffer: {buffer_pct*100:.0f}%

ITEMS TO PREP:
--------------
"""
            for _, item in prep_df.iterrows():
                print_content += f"{item['item_title']}: {item['prep_quantity']} units\n"

            print_content += f"""
--------------
Total Items: {prep_data['total_items']}
Total Quantity: {prep_data['total_prep_qty']}

Generated by FreshFlow
            """

            st.download_button(
                label="Print-Ready Format",
                data=print_content,
                file_name=f"prep_list_{prep_date.strftime('%Y%m%d')}.txt",
                mime="text/plain",
                use_container_width=True
            )

        with col3:
            st.button("Email to Kitchen", disabled=True, use_container_width=True,
                     help="Email integration coming soon")

    else:
        st.warning("No prep data available for the selected criteria.")


if __name__ == "__main__":
    main()
