"""
FreshFlow - What-If Scenario Planner Page
Simulate different scenarios and see their impact on demand.
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
from freshflow.services.weather_service import WeatherService
from freshflow.services.scenario_service import ScenarioPlanner, ScenarioType
from freshflow.utils.helpers import format_currency

st.set_page_config(page_title="Scenarios - FreshFlow", page_icon="🔮", layout="wide")


@st.cache_resource
def get_services():
    dp = DataProcessor()
    ws = WeatherService()
    sp = ScenarioPlanner(dp, ws)
    return dp, ws, sp


def main():
    dp, ws, scenario_planner = get_services()

    st.markdown("# 🔮 What-If Scenario Planner")
    st.markdown("Simulate different scenarios and see how they impact your forecast")
    st.divider()

    # Sidebar
    with st.sidebar:
        st.markdown("### Settings")
        locations = dp.get_locations()
        location_options = ["All Locations"] + locations["place_name"].tolist()
        selected_location = st.selectbox("📍 Location", location_options)

        place_id = None if selected_location == "All Locations" else \
            locations[locations["place_name"] == selected_location]["place_id"].values[0]

        # Clear scenario results when location changes
        if 'last_location' not in st.session_state:
            st.session_state['last_location'] = selected_location
        elif st.session_state['last_location'] != selected_location:
            # Clear all scenario results
            for key in ['weather_result', 'promo_result', 'event_result', 'holiday_result']:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state['last_location'] = selected_location

        forecast_days = st.slider("Forecast Days", 3, 14, 7)

        st.divider()
        st.page_link("app.py", label="← Back to Dashboard", icon="🏠")

    # Get available scenarios
    available = scenario_planner.get_available_scenarios()

    # Tabs for different scenario types
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🌧️ Weather", "🎯 Promotions", "🎉 Events", "🎄 Holidays", "🔧 Custom"
    ])

    with tab1:
        st.subheader("🌧️ Weather Impact Simulation")

        col1, col2 = st.columns([1, 2])

        with col1:
            weather_options = {
                'sunny': '☀️ Sunny',
                'rain': '🌧️ Rain',
                'heavy_rain': '⛈️ Heavy Rain',
                'snow': '❄️ Snow',
                'extreme_cold': '🥶 Extreme Cold',
                'extreme_heat': '🥵 Extreme Heat'
            }

            selected_weather = st.radio(
                "Select Weather Condition",
                list(weather_options.keys()),
                format_func=lambda x: weather_options[x]
            )

            if st.button("Simulate Weather", type="primary"):
                with st.spinner("Simulating..."):
                    result = scenario_planner.simulate_weather_scenario(
                        place_id, selected_weather, forecast_days
                    )
                    st.session_state['weather_result'] = result

        with col2:
            if 'weather_result' in st.session_state:
                result = st.session_state['weather_result']

                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Baseline Revenue", format_currency(result.baseline_revenue))
                with col_b:
                    st.metric("Simulated Revenue", format_currency(result.simulated_revenue),
                             delta=f"{result.revenue_change_pct:+.1f}%")
                with col_c:
                    st.metric("Confidence", result.confidence)

                # Daily breakdown chart
                fig = go.Figure()
                breakdown = result.daily_breakdown

                fig.add_trace(go.Bar(
                    x=breakdown['day_name'],
                    y=breakdown['revenue'],
                    name='Baseline',
                    marker_color='#3498db'
                ))
                fig.add_trace(go.Bar(
                    x=breakdown['day_name'],
                    y=breakdown['simulated_revenue'],
                    name='Simulated',
                    marker_color='#2ecc71' if result.revenue_change > 0 else '#e74c3c'
                ))

                fig.update_layout(barmode='group', height=300)
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("**Insights:**")
                for insight in result.insights:
                    st.markdown(f"- {insight}")

    with tab2:
        st.subheader("🎯 Promotion Impact Simulation")

        col1, col2 = st.columns([1, 2])

        with col1:
            promo_options = {
                '10_percent_off': '10% Off Everything',
                '15_percent_off': '15% Off Everything',
                '20_percent_off': '20% Off Everything',
                '25_percent_off': '25% Off Everything',
                'bogo': 'Buy One Get One Free',
                'free_delivery': 'Free Delivery',
                'loyalty_double_points': 'Double Loyalty Points'
            }

            selected_promo = st.radio(
                "Select Promotion Type",
                list(promo_options.keys()),
                format_func=lambda x: promo_options[x]
            )

            if st.button("Simulate Promotion", type="primary"):
                with st.spinner("Simulating..."):
                    result = scenario_planner.simulate_promotion_scenario(
                        place_id, selected_promo, forecast_days
                    )
                    st.session_state['promo_result'] = result

        with col2:
            if 'promo_result' in st.session_state:
                result = st.session_state['promo_result']

                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Baseline Revenue", format_currency(result.baseline_revenue))
                with col_b:
                    st.metric("Net Revenue", format_currency(result.simulated_revenue),
                             delta=f"{result.revenue_change_pct:+.1f}%",
                             help="After accounting for discount cost")
                with col_c:
                    st.metric("Order Increase", f"+{((result.simulated_orders/result.baseline_orders)-1)*100:.0f}%")

                # Chart
                fig = go.Figure()
                breakdown = result.daily_breakdown

                fig.add_trace(go.Scatter(
                    x=breakdown['day_name'],
                    y=breakdown['revenue'],
                    name='Baseline',
                    line=dict(dash='dash')
                ))
                fig.add_trace(go.Scatter(
                    x=breakdown['day_name'],
                    y=breakdown['simulated_revenue'],
                    name='With Promotion',
                    fill='tonexty'
                ))

                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("**Insights:**")
                for insight in result.insights:
                    st.markdown(f"- {insight}")

    with tab3:
        st.subheader("🎉 Local Event Impact Simulation")

        col1, col2 = st.columns([1, 2])

        with col1:
            event_options = {
                'local_festival': '🎪 Local Festival',
                'sports_event': '⚽ Sports Event',
                'concert_nearby': '🎵 Concert Nearby',
                'competitor_closed': '🏪 Competitor Closed',
                'road_construction': '🚧 Road Construction',
                'power_outage': '⚡ Power Outage'
            }

            selected_event = st.radio(
                "Select Event Type",
                list(event_options.keys()),
                format_func=lambda x: event_options[x]
            )

            event_days = st.slider("Event Duration (days)", 1, 7, 3)

            if st.button("Simulate Event", type="primary"):
                with st.spinner("Simulating..."):
                    result = scenario_planner.simulate_event_scenario(
                        place_id, selected_event, event_days
                    )
                    st.session_state['event_result'] = result

        with col2:
            if 'event_result' in st.session_state:
                result = st.session_state['event_result']

                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Baseline", format_currency(result.baseline_revenue))
                with col_b:
                    st.metric("With Event", format_currency(result.simulated_revenue),
                             delta=f"{result.revenue_change_pct:+.1f}%")
                with col_c:
                    st.metric("Confidence", result.confidence)

                st.markdown("**Insights:**")
                for insight in result.insights:
                    st.markdown(f"- {insight}")

    with tab4:
        st.subheader("🎄 Holiday Impact Simulation")

        col1, col2 = st.columns([1, 2])

        with col1:
            holiday_options = {
                'christmas_eve': '🎄 Christmas Eve',
                'christmas_day': '🎁 Christmas Day',
                'new_years_eve': '🥂 New Year\'s Eve',
                'new_years_day': '🎊 New Year\'s Day',
                'easter': '🐰 Easter',
                'midsummer': '☀️ Midsummer',
                'constitution_day': '🇩🇰 Constitution Day'
            }

            selected_holiday = st.radio(
                "Select Holiday",
                list(holiday_options.keys()),
                format_func=lambda x: holiday_options[x]
            )

            if st.button("Simulate Holiday", type="primary"):
                with st.spinner("Simulating..."):
                    result = scenario_planner.simulate_holiday_scenario(
                        place_id, selected_holiday
                    )
                    st.session_state['holiday_result'] = result

        with col2:
            if 'holiday_result' in st.session_state:
                result = st.session_state['holiday_result']

                if result.revenue_change_pct < 0:
                    st.warning(f"⚠️ {selected_holiday.replace('_', ' ').title()} typically sees reduced operations")
                else:
                    st.success(f"📈 {selected_holiday.replace('_', ' ').title()} brings increased demand!")

                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Normal Day Revenue", format_currency(result.baseline_revenue))
                with col_b:
                    st.metric("Holiday Revenue", format_currency(result.simulated_revenue),
                             delta=f"{result.revenue_change_pct:+.1f}%")

                st.markdown("**Planning Tips:**")
                for insight in result.insights:
                    st.markdown(f"- {insight}")

    with tab5:
        st.subheader("🔧 Custom Scenario Builder")

        st.markdown("Create your own scenario with custom parameters")

        col1, col2 = st.columns(2)

        with col1:
            custom_name = st.text_input("Scenario Name", "My Custom Scenario")
            revenue_mult = st.slider("Revenue Multiplier", 0.5, 2.0, 1.0, 0.05)
            order_mult = st.slider("Order Multiplier", 0.5, 2.0, 1.0, 0.05)
            custom_days = st.slider("Duration (days)", 1, 14, 7)

        with col2:
            st.markdown("### Preview")
            st.markdown(f"**Scenario:** {custom_name}")
            st.markdown(f"**Revenue Impact:** {(revenue_mult-1)*100:+.0f}%")
            st.markdown(f"**Order Impact:** {(order_mult-1)*100:+.0f}%")
            st.markdown(f"**Duration:** {custom_days} days")

        if st.button("Run Custom Simulation", type="primary"):
            with st.spinner("Simulating..."):
                result = scenario_planner.simulate_custom_scenario(
                    place_id, custom_name, revenue_mult, order_mult, custom_days
                )

                st.divider()

                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Baseline Revenue", format_currency(result.baseline_revenue))
                with col_b:
                    st.metric("Simulated Revenue", format_currency(result.simulated_revenue),
                             delta=f"{result.revenue_change_pct:+.1f}%")
                with col_c:
                    st.metric("Revenue Change", format_currency(result.revenue_change))


if __name__ == "__main__":
    main()
