"""
FreshFlow - Smart Promotions Engine Page
AI-powered promotion suggestions and ROI analysis.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from freshflow.services.data_processor import DataProcessor
from freshflow.services.promotions_service import PromotionsEngine, PromotionType
from freshflow.utils.helpers import format_currency

st.set_page_config(page_title="Promotions - FreshFlow", page_icon="🎯", layout="wide")


@st.cache_resource
def get_services():
    dp = DataProcessor()
    pe = PromotionsEngine(dp)
    return dp, pe


def main():
    dp, promotions_engine = get_services()

    st.markdown("# 🎯 Smart Promotions Engine")
    st.markdown("AI-powered promotion suggestions to boost revenue and reduce waste")
    st.divider()

    # Sidebar
    with st.sidebar:
        st.markdown("### Settings")
        locations = dp.get_locations()
        location_options = ["All Locations"] + locations["place_name"].tolist()
        selected_location = st.selectbox("📍 Location", location_options)

        place_id = None if selected_location == "All Locations" else \
            locations[locations["place_name"] == selected_location]["place_id"].values[0]

        st.divider()
        st.page_link("app.py", label="← Back to Dashboard", icon="🏠")

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Suggestions", "📊 Item Analysis", "🔗 Bundle Ideas", "💰 ROI Calculator"
    ])

    with tab1:
        st.subheader("🎯 AI-Generated Promotion Suggestions")

        suggestions = promotions_engine.generate_promotion_suggestions(place_id)

        if suggestions:
            for i, suggestion in enumerate(suggestions):
                with st.expander(
                    f"{'🏆' if i == 0 else '💡'} {suggestion.title} - Expected ROI: {suggestion.expected_roi}x",
                    expanded=(i == 0)
                ):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Expected Lift", f"+{suggestion.expected_lift}%")
                    with col2:
                        st.metric("Discount", f"{suggestion.discount_pct}%")
                    with col3:
                        st.metric("Confidence", suggestion.confidence)

                    st.markdown(f"**Description:** {suggestion.description}")
                    st.markdown(f"**Best Timing:** {suggestion.best_timing}")
                    st.markdown(f"**Target Items:** {', '.join(suggestion.target_items[:3])}")

                    st.info(f"💡 **Why this works:** {suggestion.reasoning}")

                    if st.button(f"Calculate ROI", key=f"roi_{i}"):
                        daily_sales = dp.get_daily_sales(place_id)
                        avg_revenue = daily_sales['total_revenue'].mean()
                        roi = promotions_engine.calculate_promotion_roi(suggestion, avg_revenue, 7)

                        st.success(f"""
                        **ROI Analysis (7 days):**
                        - Discount Cost: {format_currency(roi['discount_cost'])}
                        - Expected Lift: {format_currency(roi['expected_lift_revenue'])}
                        - Net Gain: {format_currency(roi['net_gain'])}
                        - ROI: {roi['roi']:.2f}x
                        - Recommendation: **{roi['recommendation']}**
                        """)
        else:
            st.info("No promotion suggestions available for this location.")

    with tab2:
        st.subheader("📊 Item Performance Matrix")

        item_performance = promotions_engine.analyze_item_performance(place_id)

        if len(item_performance) > 0:
            # BCG Matrix visualization
            fig = px.scatter(
                item_performance,
                x='market_share',
                y='avg_price',
                size='total_quantity',
                color='classification',
                hover_name='item_title',
                color_discrete_map={
                    'Star': '#2ecc71',
                    'Cash Cow': '#3498db',
                    'Question Mark': '#f39c12',
                    'Dog': '#e74c3c'
                },
                title="Item Performance Matrix (BCG Style)"
            )

            fig.update_layout(
                xaxis_title="Market Share",
                yaxis_title="Average Price (DKK)",
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

            # Classification breakdown
            col1, col2, col3, col4 = st.columns(4)

            stars = item_performance[item_performance['classification'] == 'Star']
            cows = item_performance[item_performance['classification'] == 'Cash Cow']
            questions = item_performance[item_performance['classification'] == 'Question Mark']
            dogs = item_performance[item_performance['classification'] == 'Dog']

            with col1:
                st.metric("⭐ Stars", len(stars), help="High volume, high margin - Invest!")
            with col2:
                st.metric("🐄 Cash Cows", len(cows), help="High volume, low margin - Maintain")
            with col3:
                st.metric("❓ Question Marks", len(questions), help="Low volume, high margin - Promote!")
            with col4:
                st.metric("🐕 Dogs", len(dogs), help="Low volume, low margin - Review")

            # Detailed table
            st.markdown("### Item Details")
            display_df = item_performance[['item_title', 'classification', 'total_quantity', 'avg_price', 'revenue']].copy()
            display_df['avg_price'] = display_df['avg_price'].apply(lambda x: f"{x:.0f} DKK")
            display_df['revenue'] = display_df['revenue'].apply(lambda x: f"{x:,.0f} DKK")
            display_df.columns = ['Item', 'Category', 'Qty Sold', 'Avg Price', 'Revenue']
            st.dataframe(display_df.head(20), use_container_width=True, hide_index=True)

    with tab3:
        st.subheader("🔗 Bundle Opportunities")

        bundles = promotions_engine.find_bundle_opportunities(place_id)

        if bundles:
            st.markdown("Items frequently purchased together - perfect for bundle deals!")

            for i, bundle in enumerate(bundles[:5]):
                items = bundle['items']
                lift = bundle['lift']
                support = bundle['support']

                with st.container():
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        st.markdown(f"""
                        **Bundle {i+1}:** {items[0][:30]} + {items[1][:30]}
                        """)
                        st.caption(f"Bought together {bundle['co_occurrence']} times | Lift: {lift:.1f}x")

                    with col2:
                        if lift > 2:
                            st.success(f"🔥 High potential")
                        elif lift > 1.5:
                            st.info(f"👍 Good potential")
                        else:
                            st.warning(f"📊 Moderate")

                    st.progress(min(lift / 3, 1.0))
                    st.divider()
        else:
            st.info("Not enough data to identify bundle opportunities.")

    with tab4:
        st.subheader("💰 Promotion ROI Calculator")

        col1, col2 = st.columns(2)

        with col1:
            promo_name = st.text_input("Promotion Name", "Weekend Special")
            discount_pct = st.slider("Discount %", 5, 50, 15)
            expected_lift = st.slider("Expected Order Lift %", 5, 100, 25)
            duration_days = st.slider("Duration (days)", 1, 30, 7)

        with col2:
            daily_sales = dp.get_daily_sales(place_id)
            avg_revenue = daily_sales['total_revenue'].mean() if len(daily_sales) > 0 else 10000

            st.metric("Avg Daily Revenue", format_currency(avg_revenue))

            # Calculate ROI
            discount_cost = avg_revenue * (discount_pct / 100) * duration_days
            revenue_lift = avg_revenue * (expected_lift / 100) * duration_days
            net_gain = revenue_lift - discount_cost
            roi = net_gain / discount_cost if discount_cost > 0 else 0

            st.metric("Discount Cost", format_currency(discount_cost))
            st.metric("Expected Lift", format_currency(revenue_lift))

        st.divider()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Net Gain/Loss", format_currency(net_gain),
                     delta="Profitable" if net_gain > 0 else "Loss")
        with col2:
            st.metric("ROI", f"{roi:.2f}x")
        with col3:
            if roi > 2:
                st.success("✅ Highly Recommended")
            elif roi > 1:
                st.info("👍 Recommended")
            else:
                st.warning("⚠️ Review needed")


if __name__ == "__main__":
    main()
