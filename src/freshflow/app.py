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
    page_title="Expecto - Demand Forecasting",
    page_icon="E",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    /* Modern color scheme */
    :root {
        --primary: #1E5631;
        --secondary: #4CAF50;
        --accent: #8BC34A;
        --warning: #FF9800;
        --danger: #F44336;
        --bg-light: #f8f9fa;
    }

    /* Header styling */
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1E5631 0%, #4CAF50 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-top: 5px;
    }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid #4CAF50;
    }

    div[data-testid="metric-container"] label {
        font-weight: 600;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        baaaaackground: linear-gradient(180deg, #1E5631 0%, #2E7D32 100%);
    }

    section[data-testid="stSidebar"] .stMarkdown {
        color: black;
    }

    section[data-testid="stSidebar"] .stSelectbox label {
        color: black !important;
    }

    /* Alert boxes */
    .alert-box {
        background: linear-gradient(135deg, #fff8e1 0%, #ffecb3 100%);
        border-left: 4px solid #ffc107;
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 0 8px 8px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .alert-box-info {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-left-color: #2196F3;
    }
    .alert-box-success {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border-left-color: #4CAF50;
    }
    .alert-box-danger {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        border-left-color: #F44336;
    }

    /* KPI Cards */
    .kpi-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 24px;
        border-radius: 16px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 8px 0;
    }
    .kpi-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .kpi-delta {
        font-size: 0.85rem;
        margin-top: 4px;
    }

    /* Quick stats row */
    .stat-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 0;
        border-bottom: 1px solid #eee;
    }
    .stat-label {
        color: #666;
        font-size: 0.9rem;
    }
    .stat-value {
        font-weight: 600;
        color: #1E5631;
    }

    /* Charts */
    .js-plotly-plot {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Tables */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: #f0f2f6;
        padding: 4px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    /* Progress bars */
    .stProgress > div > div {
        background: linear-gradient(90deg, #4CAF50 0%, #8BC34A 100%);
        border-radius: 10px;
    }

    /* Hide default Streamlit page navigation */
    [data-testid="stSidebarNav"] {display: none;}

    /* Style custom nav links */
    section[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"] {
        /* color: rgba(255,255,255,0.85) !important; */
        font-size: 0.85rem;
        font-weight: 500;
        letter-spacing: 0.05em;
        padding: 6px 12px;
        border-radius: 6px;
        transition: background 0.2s;
    }
    section[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"]:hover {
        /* background: rgba(255,255,255,0.1); */
        color: white !important;
    }
    section[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"][aria-current="page"] {
        /* background: rgba(255,255,255,0.15); */
        color: white !important;
        font-weight: 600;
    }

    /* Hide default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
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


@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_cached_daily_sales(_dp, place_id):
    """Cache daily sales data."""
    return _dp.get_daily_sales(place_id)


@st.cache_data(ttl=300)
def get_cached_forecast(_fs, place_id, days):
    """Cache forecast data."""
    return _fs.forecast_revenue(place_id, days=days)


def main():
    """Main application entry point."""
    # Initialize services
    data_processor = get_data_processor()
    weather_service = get_weather_service()
    forecast_service = get_forecast_service(data_processor, weather_service)

    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <div style="font-size: 1.5rem; font-weight: 700; color: black;">Expecto</div>
            <div style="font-size: 0.85rem; color: black;">Demand Forecasting</div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # Location selector
        locations = data_processor.get_locations()
        if len(locations) == 0:
            st.error("No locations found in data")
            return

        location_options = ["All Locations"] + locations["place_name"].tolist()
        selected_location = st.selectbox(
            "Select Location",
            location_options,
            index=0
        )

        if selected_location == "All Locations":
            place_id = None
            # st.info(f"Analyzing {len(locations)} locations")
        else:
            place_id = locations[locations["place_name"] == selected_location]["place_id"].values[0]

        st.divider()

        # Quick Stats
        st.markdown("**Quick Stats**")
        try:
            min_date, max_date = data_processor.get_date_range()
            days_of_data = (max_date - min_date).days
            st.markdown(f"""
            <div class="stat-row">
                <span class="stat-label">Data Range</span>
                <span class="stat-value">{days_of_data} days</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">From</span>
                <span class="stat-value">{min_date.strftime('%b %d, %Y')}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">To</span>
                <span class="stat-value">{max_date.strftime('%b %d, %Y')}</span>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.warning("Could not load date range")

        st.divider()

        # Navigation
        st.markdown("**Navigation**")
        st.page_link("app.py", label="Dashboard")
        st.page_link("pages/1_forecasting.py", label="Forecasting")
        st.page_link("pages/2_kitchen_prep.py", label="Kitchen Prep")
        st.page_link("pages/3_external_factors.py", label="External Factors")
        st.page_link("pages/4_promotions.py", label="Promotions")
        st.page_link("pages/5_anomalies.py", label="Anomalies")
        st.page_link("pages/6_scenarios.py", label="Scenarios")
        st.page_link("pages/7_inventory.py", label="Inventory")
        st.page_link("pages/8_model_training.py", label="Model Training")

    # Main content
    st.markdown('<p class="main-header">Expecto Dashboard</p>', unsafe_allow_html=True)
    st.divider()

    # Get data with error handling
    try:
        daily_sales = get_cached_daily_sales(data_processor, place_id)
        accuracy_metrics = forecast_service.get_accuracy_metrics(place_id)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return

    if len(daily_sales) == 0:
        st.warning("No sales data available for the selected location.")
        return

    # Calculate KPIs
    recent_revenue = daily_sales.tail(1)["total_revenue"].values[0] if len(daily_sales) > 0 else 0
    prev_revenue = daily_sales.tail(2).head(1)["total_revenue"].values[0] if len(daily_sales) > 1 else recent_revenue

    last_week = daily_sales.tail(7)["total_revenue"].sum() if len(daily_sales) >= 7 else 0
    prev_week = daily_sales.tail(14).head(7)["total_revenue"].sum() if len(daily_sales) >= 14 else last_week
    weekly_change = (last_week - prev_week) / prev_week * 100 if prev_week > 0 else 0

    avg_daily = daily_sales["total_revenue"].mean()
    total_orders = daily_sales["order_count"].sum()

    # KPI Row with styled cards
    col1, col2,  col4, col5 = st.columns(4)

    with col1:
        delta = f"{((recent_revenue - prev_revenue) / prev_revenue * 100):+.1f}%" if prev_revenue > 0 else "N/A"
        st.metric("Latest Revenue", format_currency(recent_revenue), delta)

    with col2:
        st.metric("Weekly Change", f"{weekly_change:+.1f}%", "vs last week")

    # with col3:
    #     accuracy = accuracy_metrics.get("accuracy", 0.85) * 100
    #     st.metric("Model Accuracy", f"{accuracy:.1f}%", "MAPE-based")

    with col4:
        st.metric("Avg Daily Revenue", format_currency(avg_daily))

    with col5:
        st.metric("Total Orders", f"{total_orders:,.0f}")

    st.divider()

    # Main content - Tabs
    tab1, tab2 = st.tabs(["Forecast", "Analytics", ])
    tab3=0
    with tab1:
        left_col, right_col = st.columns([2, 1])

        with left_col:
            st.subheader("7-Day Revenue Forecast")

            try:
                forecast_df = get_cached_forecast(forecast_service, place_id, 7)

                # Create forecast chart
                fig = go.Figure()

                # Historical data
                hist_df = daily_sales.tail(14)
                fig.add_trace(go.Scatter(
                    x=hist_df["date"],
                    y=hist_df["total_revenue"],
                    mode="lines+markers",
                    line=dict(color="#90A4AE", width=2, dash="dot"),
                    marker=dict(size=6),
                    name="Historical",
                    opacity=0.7
                ))

                # Confidence band
                if "confidence_lower" in forecast_df.columns:
                    fig.add_trace(go.Scatter(
                        x=forecast_df["date"],
                        y=forecast_df["confidence_upper"],
                        fill=None,
                        mode="lines",
                        line=dict(color="rgba(76, 175, 80, 0.1)"),
                        showlegend=False
                    ))
                    fig.add_trace(go.Scatter(
                        x=forecast_df["date"],
                        y=forecast_df["confidence_lower"],
                        fill="tonexty",
                        mode="lines",
                        line=dict(color="rgba(76, 175, 80, 0.1)"),
                        fillcolor="rgba(76, 175, 80, 0.15)",
                        showlegend=False
                    ))

                # Forecast line
                fig.add_trace(go.Scatter(
                    x=forecast_df["date"],
                    y=forecast_df["predicted_revenue"],
                    mode="lines+markers",
                    line=dict(color="#4CAF50", width=3),
                    marker=dict(size=10, symbol="diamond"),
                    name="Forecast"
                ))

                fig.update_layout(
                    xaxis_title="",
                    yaxis_title="Revenue (DKK)",
                    hovermode="x unified",
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    height=400,
                    margin=dict(l=0, r=0, t=40, b=0),
                    plot_bgcolor="white"
                )
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')

                st.plotly_chart(fig, use_container_width=True)

                # Forecast summary
                total_forecast = forecast_df["predicted_revenue"].sum()
                st.markdown(f"""
                <div class="alert-box alert-box-success">
                    <strong>7-Day Forecast Summary:</strong> {format_currency(total_forecast)} expected revenue
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Could not generate forecast: {str(e)}")

        with right_col:
            st.subheader("Alerts")

            # Weather alert
            try:
                weather_forecast = weather_service.get_forecast(2)
                if len(weather_forecast) > 0:
                    tomorrow_weather = weather_service.get_weather_impact_summary(weather_forecast.iloc[[1]])
                    temp = tomorrow_weather.get('temp', 15)
                    condition = tomorrow_weather.get('condition', 'Unknown')

                    if tomorrow_weather.get("is_rainy"):
                        st.markdown(f"""
                        <div class="alert-box">
                            <strong>Rain expected tomorrow</strong><br>
                            <small>Hot meals & soups (+10-15%)</small>
                        </div>
                        """, unsafe_allow_html=True)
                    elif temp < 5:
                        st.markdown(f"""
                        <div class="alert-box alert-box-info">
                            <strong>Cold weather ({temp} C)</strong><br>
                            <small>Hot drinks demand expected</small>
                        </div>
                        """, unsafe_allow_html=True)
                    elif temp > 25:
                        st.markdown(f"""
                        <div class="alert-box alert-box-info">
                            <strong>Hot weather ({temp} C)</strong><br>
                            <small>Cold drinks & salads</small>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="alert-box alert-box-success">
                            <strong>{condition}</strong><br>
                            <small>Normal demand expected</small>
                        </div>
                        """, unsafe_allow_html=True)
            except:
                pass

            # Weekend alert
            tomorrow = datetime.now() + timedelta(days=1)
            if tomorrow.weekday() >= 4:
                st.markdown(f"""
                <div class="alert-box">
                    <strong>{tomorrow.strftime('%A')} approaching</strong><br>
                    <small>Weekend demand (+15-25%)</small>
                </div>
                """, unsafe_allow_html=True)

            st.divider()

            # Top Items
            st.subheader("Top Items")
            try:
                top_items = data_processor.get_top_items(place_id, top_n=5)
                for idx, row in top_items.iterrows():
                    title = row['item_title'][:22] + "..." if len(row['item_title']) > 22 else row['item_title']
                    st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #eee;">
                        <span>{title}</span>
                        <span style="font-weight: 600; color: #4CAF50;">{int(row['total_quantity'])}</span>
                    </div>
                    """, unsafe_allow_html=True)
            except:
                st.info("No item data available")

    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Weekly Demand Pattern")
            try:
                dow_pattern = data_processor.get_day_of_week_pattern(place_id)

                fig_dow = px.bar(
                    dow_pattern,
                    x="day_name",
                    y="avg_revenue",
                    color="relative_demand",
                    color_continuous_scale=["#ef5350", "#ffee58", "#66bb6a"],
                    labels={"day_name": "", "avg_revenue": "Avg Revenue (DKK)"}
                )
                fig_dow.update_layout(
                    showlegend=False,
                    height=350,
                    margin=dict(l=0, r=0, t=30, b=0),
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig_dow, use_container_width=True)
            except Exception as e:
                st.error(f"Could not load pattern: {str(e)}")

        with col2:
            st.subheader("Revenue Trend")
            try:
                # Monthly trend
                daily_sales['month'] = pd.to_datetime(daily_sales['date']).dt.to_period('M')
                monthly = daily_sales.groupby('month')['total_revenue'].sum().reset_index()
                monthly['month'] = monthly['month'].astype(str)

                fig_trend = px.line(
                    monthly.tail(12),
                    x='month',
                    y='total_revenue',
                    markers=True
                )
                fig_trend.update_traces(
                    line=dict(color="#4CAF50", width=3),
                    marker=dict(size=8)
                )
                fig_trend.update_layout(
                    xaxis_title="",
                    yaxis_title="Revenue (DKK)",
                    height=350,
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                st.plotly_chart(fig_trend, use_container_width=True)
            except:
                st.info("Insufficient data for trend analysis")

    # with tab3:
    #     st.subheader("Key Insights")

    #     col1, col2, col3 = st.columns(3)

    #     with col1:
    #         st.markdown("""
    #         <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    #                     padding: 24px; border-radius: 16px; color: white; height: 180px;">
    #             <div style="font-size: 1.2rem; font-weight: 600; margin: 10px 0;">Best Day</div>
    #             <div style="font-size: 0.9rem; opacity: 0.9;">Saturday typically shows highest revenue</div>
    #         </div>
    #         """, unsafe_allow_html=True)

    #     with col2:
    #         st.markdown("""
    #         <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    #                     padding: 24px; border-radius: 16px; color: white; height: 180px;">
    #             <div style="font-size: 1.2rem; font-weight: 600; margin: 10px 0;">Weather Impact</div>
    #             <div style="font-size: 0.9rem; opacity: 0.9;">Rainy days boost hot meal sales by ~12%</div>
    #         </div>
    #         """, unsafe_allow_html=True)

        # with col3:
        #     st.markdown("""
        #     <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        #                 padding: 24px; border-radius: 16px; color: white; height: 180px;">
        #         <div style="font-size: 1.2rem; font-weight: 600; margin: 10px 0;">Model Accuracy</div>
        #         <div style="font-size: 0.9rem; opacity: 0.9;">Achieves 90%+ accuracy on best locations</div>
        #     </div>
        #     """, unsafe_allow_html=True)

        # st.divider()

        # # Recommendations
        # st.subheader("Recommendations")
        # st.markdown("""
        # - **Inventory**: Stock 15% extra on Fridays and Saturdays
        # - **Staffing**: Schedule more staff during peak hours (11am-2pm, 6pm-9pm)
        # - **Promotions**: Consider Tuesday/Wednesday deals to boost slow days
        # - **Weather**: Monitor forecasts and adjust hot/cold item prep accordingly
        # """)

    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <strong>Expecto</strong> | Demand Forecasting<br>
        <small>Deloitte x AUC Hackathon</small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
