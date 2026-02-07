"""
FreshFlow - Inventory Reorder System Page
Automatic reorder points and stock management.
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
from freshflow.services.inventory_service import InventoryManager, StockStatus, AlertPriority
from freshflow.utils.helpers import format_currency
from freshflow.components import setup_page, sidebar_nav

setup_page("Inventory")


@st.cache_resource
def get_services():
    dp = DataProcessor()
    im = InventoryManager(dp)
    return dp, im


def status_color(status):
    colors = {
        StockStatus.CRITICAL: "#dc3545",
        StockStatus.LOW: "#fd7e14",
        StockStatus.ADEQUATE: "#28a745",
        StockStatus.OVERSTOCKED: "#17a2b8"
    }
    return colors.get(status, "#6c757d")


def status_icon(status):
    icons = {
        StockStatus.CRITICAL: "CRITICAL",
        StockStatus.LOW: "LOW",
        StockStatus.ADEQUATE: "OK",
        StockStatus.OVERSTOCKED: "OVER"
    }
    return icons.get(status, "UNKNOWN")


def main():
    dp, inventory_manager = get_services()

    st.markdown("# Inventory Reorder System")
    st.markdown("Smart inventory management with automatic reorder suggestions")
    st.divider()

    # Sidebar
    with st.sidebar:
        sidebar_nav()

        st.markdown("### Settings")
        locations = dp.get_locations()
        location_options = ["All Locations"] + locations["place_name"].tolist()
        selected_location = st.selectbox("Location", location_options)

        place_id = None if selected_location == "All Locations" else \
            locations[locations["place_name"] == selected_location]["place_id"].values[0]

        st.divider()

        st.markdown("### Parameters")
        lead_time = st.number_input("Lead Time (days)", 1, 14, 3)
        service_level = st.select_slider(
            "Service Level",
            options=[0.90, 0.95, 0.99],
            value=0.95,
            format_func=lambda x: f"{x*100:.0f}%"
        )

    # Update inventory manager with user parameters
    inventory_manager.set_parameters(lead_time_days=lead_time, service_level=service_level)

    # Get inventory summary with user parameters
    summary = inventory_manager.get_inventory_summary(
        place_id,
        lead_time_days=lead_time,
        service_level=service_level
    )

    # Summary cards
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Items", summary['total_items'])
    with col2:
        st.metric("Critical", summary['critical'])
    with col3:
        st.metric("Low Stock", summary['low'])
    with col4:
        st.metric("Adequate", summary['adequate'])
    with col5:
        st.metric("Overstocked", summary['overstocked'])

    if summary['critical'] > 0 or summary['low'] > 0:
        st.error(f"{summary['reorder_needed']} items need reordering! "
                f"Estimated cost: {format_currency(summary['estimated_reorder_cost'])}")

    st.divider()

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Stock Status", "Reorder Suggestions", "Demand Analysis", "Order Optimizer"
    ])

    with tab1:
        st.subheader("Current Stock Status")

        inventory = inventory_manager.get_inventory_status(
            place_id,
            lead_time_days=lead_time,
            service_level=service_level
        )

        if inventory:
            # Status distribution chart
            col1, col2 = st.columns([1, 2])

            with col1:
                status_counts = {
                    'Critical': len([i for i in inventory if i.status == StockStatus.CRITICAL]),
                    'Low': len([i for i in inventory if i.status == StockStatus.LOW]),
                    'Adequate': len([i for i in inventory if i.status == StockStatus.ADEQUATE]),
                    'Overstocked': len([i for i in inventory if i.status == StockStatus.OVERSTOCKED])
                }

                fig = px.pie(
                    values=list(status_counts.values()),
                    names=list(status_counts.keys()),
                    color=list(status_counts.keys()),
                    color_discrete_map={
                        'Critical': '#dc3545',
                        'Low': '#fd7e14',
                        'Adequate': '#28a745',
                        'Overstocked': '#17a2b8'
                    }
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Stock levels table
                st.markdown("### Inventory Items")

                inv_df = pd.DataFrame([
                    {
                        'Status': f"{status_icon(i.status)} {i.status.value.title()}",
                        'Item': i.item_name[:30],
                        'Current': f"{i.current_stock:.0f}",
                        'Reorder Point': f"{i.reorder_point:.0f}",
                        'Days of Stock': f"{i.days_of_stock:.1f}",
                        'Action': i.recommended_action
                    }
                    for i in inventory[:20]
                ])

                st.dataframe(inv_df, use_container_width=True, hide_index=True)
        else:
            st.info("No inventory data available.")

    with tab2:
        st.subheader("Smart Reorder Suggestions")

        suggestions = inventory_manager.generate_reorder_suggestions(
            place_id,
            lead_time_days=lead_time,
            service_level=service_level
        )

        if suggestions:
            # Urgency summary
            urgent = [s for s in suggestions if s.urgency == AlertPriority.URGENT]
            high = [s for s in suggestions if s.urgency == AlertPriority.HIGH]

            if urgent:
                st.error(f"{len(urgent)} items need URGENT ordering!")
            if high:
                st.warning(f"{len(high)} items need ordering soon")

            # Suggestions list
            total_cost = 0

            for suggestion in suggestions:
                urgency_labels = {
                    AlertPriority.URGENT: "URGENT",
                    AlertPriority.HIGH: "HIGH",
                    AlertPriority.MEDIUM: "MEDIUM",
                    AlertPriority.LOW: "LOW"
                }

                with st.expander(
                    f"[{urgency_labels[suggestion.urgency]}] {suggestion.item_name[:40]} - "
                    f"Order {suggestion.suggested_quantity:.0f} units",
                    expanded=(suggestion.urgency in [AlertPriority.URGENT, AlertPriority.HIGH])
                ):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Current Stock", f"{suggestion.current_stock:.0f}")
                    with col2:
                        st.metric("Order Quantity", f"{suggestion.suggested_quantity:.0f}")
                    with col3:
                        st.metric("Est. Cost", format_currency(suggestion.estimated_cost))

                    st.markdown(f"**Stockout Risk:** {suggestion.days_until_stockout} days until stockout")
                    st.markdown(f"**Reason:** {suggestion.reason}")

                    total_cost += suggestion.estimated_cost

            st.divider()
            st.markdown(f"### Total Estimated Order Cost: {format_currency(total_cost)}")
        else:
            st.success("No reorder suggestions - stock levels are adequate!")

    with tab3:
        st.subheader("Item Demand Analysis")

        analysis = inventory_manager.analyze_item_demand(
            place_id,
            lead_time_days=lead_time,
            service_level=service_level
        )

        if len(analysis) > 0:
            # Top items by demand
            top_demand = analysis.nlargest(15, 'avg_daily_demand')

            fig = px.bar(
                top_demand,
                x='item_name',
                y='avg_daily_demand',
                title="Top Items by Daily Demand",
                color='avg_daily_demand',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(xaxis_tickangle=-45, height=400)
            st.plotly_chart(fig, use_container_width=True)

            # Demand variability
            st.markdown("### Demand Variability")
            st.caption("Items with high variability need larger safety stock")

            analysis['cv'] = analysis['daily_demand_std'] / analysis['avg_daily_demand']
            analysis['cv'] = analysis['cv'].fillna(0)

            high_variability = analysis.nlargest(10, 'cv')

            var_df = high_variability[['item_name', 'avg_daily_demand', 'daily_demand_std', 'cv']].copy()
            var_df.columns = ['Item', 'Avg Daily Demand', 'Std Dev', 'Coefficient of Variation']
            var_df['Coefficient of Variation'] = var_df['Coefficient of Variation'].apply(lambda x: f"{x:.2f}")

            st.dataframe(var_df, use_container_width=True, hide_index=True)

    with tab4:
        st.subheader("Order Optimizer")

        st.markdown("Optimize your order based on budget and minimum order requirements")

        col1, col2 = st.columns(2)

        with col1:
            budget = st.number_input("Budget (DKK)", 0, 100000, 10000, step=1000)
            min_order = st.number_input("Minimum Order Value (DKK)", 0, 5000, 500, step=100)

        suggestions = inventory_manager.generate_reorder_suggestions(
            place_id,
            lead_time_days=lead_time,
            service_level=service_level
        )

        if suggestions and st.button("Optimize Order", type="primary"):
            optimized = inventory_manager.optimize_order(
                suggestions,
                budget=budget if budget > 0 else None,
                min_order_value=min_order
            )

            with col2:
                st.markdown("### Optimized Order")
                st.metric("Total Cost", format_currency(optimized['total_cost']))

                if not optimized.get('meets_minimum', True):
                    st.warning(optimized.get('suggestion', 'Add more items to meet minimum'))

            if optimized['items']:
                st.markdown("### Items to Order")

                order_df = pd.DataFrame(optimized['items'])
                # Select and rename columns for display
                order_df = order_df[['item', 'quantity', 'cost', 'urgency']]
                order_df['cost'] = order_df['cost'].apply(lambda x: format_currency(x))
                order_df.columns = ['Item', 'Quantity', 'Cost', 'Urgency']

                st.dataframe(order_df, use_container_width=True, hide_index=True)

                # Export order
                csv = pd.DataFrame(optimized['items']).to_csv(index=False)
                st.download_button(
                    "Download Order List",
                    csv,
                    f"order_{datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv"
                )


if __name__ == "__main__":
    main()
