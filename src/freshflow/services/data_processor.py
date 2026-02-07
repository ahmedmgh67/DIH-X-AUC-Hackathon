"""
Data processing service for loading and preprocessing inventory management data.
Handles CSV loading, UNIX timestamp conversion, and data aggregation.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import os

from ..utils.helpers import (
    DATA_DIR, unix_to_datetime, is_holiday, is_weekend,
    get_day_of_week, DANISH_HOLIDAYS_2021_2024
)


class DataProcessor:
    """Handles loading and preprocessing of inventory management data."""

    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir
        self._orders_df: Optional[pd.DataFrame] = None
        self._order_items_df: Optional[pd.DataFrame] = None
        self._menu_items_df: Optional[pd.DataFrame] = None
        self._places_df: Optional[pd.DataFrame] = None
        self._campaigns_df: Optional[pd.DataFrame] = None
        self._fct_campaigns_df: Optional[pd.DataFrame] = None

    def load_orders(self, force_reload: bool = False) -> pd.DataFrame:
        """Load and preprocess orders data."""
        if self._orders_df is not None and not force_reload:
            return self._orders_df

        filepath = os.path.join(self.data_dir, "fct_orders.csv")
        df = pd.read_csv(filepath)

        # Convert UNIX timestamps to datetime
        df["created_dt"] = pd.to_datetime(df["created"], unit="s")
        df["date"] = df["created_dt"].dt.date
        df["hour"] = df["created_dt"].dt.hour
        df["day_of_week"] = df["created_dt"].dt.dayofweek
        df["month"] = df["created_dt"].dt.month
        df["year"] = df["created_dt"].dt.year
        df["week"] = df["created_dt"].dt.isocalendar().week

        # Add time-based features
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        df["date_str"] = df["created_dt"].dt.strftime("%Y-%m-%d")
        df["is_holiday"] = df["date_str"].isin(DANISH_HOLIDAYS_2021_2024.keys()).astype(int)

        # Filter to valid orders only
        df = df[df["status"] == "Closed"]
        df = df[df["demo_mode"] != 1]
        df = df[df["trainee_mode"] != 1]

        self._orders_df = df
        return df

    def load_order_items(self, force_reload: bool = False) -> pd.DataFrame:
        """Load and preprocess order items data."""
        if self._order_items_df is not None and not force_reload:
            return self._order_items_df

        filepath = os.path.join(self.data_dir, "fct_order_items.csv")
        df = pd.read_csv(filepath)

        self._order_items_df = df
        return df

    def load_menu_items(self, force_reload: bool = False) -> pd.DataFrame:
        """Load menu items data."""
        if self._menu_items_df is not None and not force_reload:
            return self._menu_items_df

        filepath = os.path.join(self.data_dir, "dim_menu_items.csv")
        df = pd.read_csv(filepath)

        # Filter to active items
        df = df[df["status"] == "Active"]

        self._menu_items_df = df
        return df

    def load_places(self, force_reload: bool = False) -> pd.DataFrame:
        """Load places/locations data."""
        if self._places_df is not None and not force_reload:
            return self._places_df

        filepath = os.path.join(self.data_dir, "dim_places.csv")
        df = pd.read_csv(filepath)

        self._places_df = df
        return df

    def load_campaigns(self, force_reload: bool = False) -> pd.DataFrame:
        """Load campaign definitions."""
        if self._campaigns_df is not None and not force_reload:
            return self._campaigns_df

        filepath = os.path.join(self.data_dir, "dim_campaigns.csv")
        df = pd.read_csv(filepath)

        self._campaigns_df = df
        return df

    def load_fct_campaigns(self, force_reload: bool = False) -> pd.DataFrame:
        """Load campaign facts/execution data."""
        if self._fct_campaigns_df is not None and not force_reload:
            return self._fct_campaigns_df

        filepath = os.path.join(self.data_dir, "fct_campaigns.csv")
        df = pd.read_csv(filepath)

        self._fct_campaigns_df = df
        return df

    def get_daily_sales(self, place_id: Optional[int] = None) -> pd.DataFrame:
        """
        Aggregate orders to daily sales totals.

        Args:
            place_id: Filter to specific location (None for all)

        Returns:
            DataFrame with daily sales aggregations
        """
        orders = self.load_orders()

        if place_id is not None:
            orders = orders[orders["place_id"] == place_id]

        daily = orders.groupby("date").agg({
            "id": "count",
            "total_amount": "sum",
            "items_amount": "sum",
            "is_weekend": "first",
            "is_holiday": "first",
            "day_of_week": "first",
            "month": "first",
            "year": "first",
            "week": "first"
        }).reset_index()

        daily.columns = [
            "date", "order_count", "total_revenue", "items_revenue",
            "is_weekend", "is_holiday", "day_of_week", "month", "year", "week"
        ]

        # Ensure continuous date range
        daily["date"] = pd.to_datetime(daily["date"])
        date_range = pd.date_range(start=daily["date"].min(), end=daily["date"].max(), freq="D")
        daily = daily.set_index("date").reindex(date_range).fillna(0).reset_index()
        daily.columns = ["date"] + list(daily.columns[1:])

        # Recalculate time features for filled dates
        daily["day_of_week"] = daily["date"].dt.dayofweek
        daily["is_weekend"] = daily["day_of_week"].isin([5, 6]).astype(int)
        daily["month"] = daily["date"].dt.month
        daily["year"] = daily["date"].dt.year
        daily["week"] = daily["date"].dt.isocalendar().week
        daily["date_str"] = daily["date"].dt.strftime("%Y-%m-%d")
        daily["is_holiday"] = daily["date_str"].isin(DANISH_HOLIDAYS_2021_2024.keys()).astype(int)

        return daily

    def get_daily_item_sales(self, place_id: Optional[int] = None) -> pd.DataFrame:
        """
        Aggregate to daily sales per menu item.

        Args:
            place_id: Filter to specific location (None for all)

        Returns:
            DataFrame with daily item-level sales
        """
        orders = self.load_orders()
        order_items = self.load_order_items()

        # Merge orders with items
        merged = order_items.merge(
            orders[["id", "place_id", "date", "is_weekend", "is_holiday", "day_of_week"]],
            left_on="order_id",
            right_on="id",
            how="inner"
        )

        if place_id is not None:
            merged = merged[merged["place_id"] == place_id]

        # Aggregate by date and item
        daily_items = merged.groupby(["date", "title"]).agg({
            "quantity": "sum",
            "price": "mean",
            "is_weekend": "first",
            "is_holiday": "first",
            "day_of_week": "first"
        }).reset_index()

        daily_items.columns = [
            "date", "item_title", "quantity_sold", "avg_price",
            "is_weekend", "is_holiday", "day_of_week"
        ]

        return daily_items

    def get_hourly_pattern(self, place_id: Optional[int] = None) -> pd.DataFrame:
        """Get average sales by hour of day."""
        orders = self.load_orders()

        if place_id is not None:
            orders = orders[orders["place_id"] == place_id]

        hourly = orders.groupby("hour").agg({
            "id": "count",
            "total_amount": "sum"
        }).reset_index()

        hourly.columns = ["hour", "avg_orders", "avg_revenue"]

        # Normalize to averages
        n_days = orders["date"].nunique()
        hourly["avg_orders"] = hourly["avg_orders"] / n_days
        hourly["avg_revenue"] = hourly["avg_revenue"] / n_days

        return hourly

    def get_day_of_week_pattern(self, place_id: Optional[int] = None) -> pd.DataFrame:
        """Get average sales by day of week."""
        orders = self.load_orders()

        if place_id is not None:
            orders = orders[orders["place_id"] == place_id]

        # Group by day of week
        dow = orders.groupby("day_of_week").agg({
            "id": "count",
            "total_amount": "sum"
        }).reset_index()

        dow.columns = ["day_of_week", "total_orders", "total_revenue"]

        # Calculate number of each day type
        date_counts = orders.groupby("date")["day_of_week"].first().value_counts()
        dow["day_count"] = dow["day_of_week"].map(date_counts)

        # Safe division with zero protection
        dow["avg_orders"] = dow.apply(
            lambda row: row["total_orders"] / row["day_count"] if row["day_count"] > 0 else 0,
            axis=1
        )
        dow["avg_revenue"] = dow.apply(
            lambda row: row["total_revenue"] / row["day_count"] if row["day_count"] > 0 else 0,
            axis=1
        )

        # Calculate standard deviation for revenue (for anomaly detection)
        # Using variance formula: std = sqrt(E[X^2] - E[X]^2), approximated here
        dow["std_revenue"] = dow["avg_revenue"] * 0.25  # Approximate 25% CV

        # Calculate relative to overall average with zero protection
        overall_avg = dow["avg_orders"].mean()
        if overall_avg > 0:
            dow["relative_demand"] = (dow["avg_orders"] - overall_avg) / overall_avg
        else:
            dow["relative_demand"] = 0

        # Add day names
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow["day_name"] = dow["day_of_week"].map(lambda x: day_names[x])

        return dow

    def get_top_items(self, place_id: Optional[int] = None, top_n: int = 10) -> pd.DataFrame:
        """Get top selling items by quantity."""
        orders = self.load_orders()
        order_items = self.load_order_items()

        # Merge
        merged = order_items.merge(
            orders[["id", "place_id"]],
            left_on="order_id",
            right_on="id",
            how="inner"
        )

        if place_id is not None:
            merged = merged[merged["place_id"] == place_id]

        # Aggregate by item
        items = merged.groupby("title").agg({
            "quantity": "sum",
            "price": "mean"
        }).reset_index()

        items.columns = ["item_title", "total_quantity", "avg_price"]
        items = items.sort_values("total_quantity", ascending=False).head(top_n)

        return items

    def get_locations(self) -> pd.DataFrame:
        """Get list of locations with order counts."""
        orders = self.load_orders()
        places = self.load_places()

        # Count orders per place
        place_orders = orders.groupby("place_id")["id"].count().reset_index()
        place_orders.columns = ["place_id", "order_count"]

        # Merge with places
        locations = places.merge(place_orders, left_on="id", right_on="place_id", how="inner")
        locations = locations[["id", "title", "order_count"]]
        locations.columns = ["place_id", "place_name", "order_count"]

        # Sort by order count
        locations = locations.sort_values("order_count", ascending=False)

        return locations

    def get_date_range(self) -> Tuple[datetime, datetime]:
        """Get the date range of available data."""
        orders = self.load_orders()
        min_date = pd.to_datetime(orders["date"].min())
        max_date = pd.to_datetime(orders["date"].max())
        return min_date, max_date

    def get_active_campaigns(self, date: datetime, place_id: Optional[int] = None) -> pd.DataFrame:
        """Get campaigns active on a specific date."""
        campaigns = self.load_campaigns()

        # Filter by date range
        campaigns["start_dt"] = pd.to_datetime(campaigns["created"], unit="s")

        # Simple filter - campaigns created before the date
        active = campaigns[campaigns["start_dt"] <= date]

        if place_id is not None:
            active = active[active["place_id"] == place_id]

        return active
