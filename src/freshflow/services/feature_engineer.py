"""
Feature engineering service for preparing ML model inputs.
Combines sales data with external factors (time, weather, campaigns).
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from ..utils.helpers import DANISH_HOLIDAYS_2021_2024
from .data_processor import DataProcessor
from .weather_service import WeatherService


class FeatureEngineer:
    """
    Creates feature matrices for demand forecasting model.
    Combines historical sales with external factors.
    """

    def __init__(
        self,
        data_processor: DataProcessor,
        weather_service: WeatherService
    ):
        self.data_processor = data_processor
        self.weather_service = weather_service

    def create_training_features(
        self,
        place_id: Optional[int] = None,
        target_column: str = "total_revenue"
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Create feature matrix for model training.

        Args:
            place_id: Specific location or None for all
            target_column: Column to predict ('total_revenue' or 'order_count')

        Returns:
            Tuple of (feature_dataframe, feature_names)
        """
        # Get daily sales data
        daily_sales = self.data_processor.get_daily_sales(place_id)

        # Get date range for weather
        min_date = daily_sales["date"].min()
        max_date = daily_sales["date"].max()

        # Fetch historical weather
        weather_df = self.weather_service.get_historical_weather(
            min_date.to_pydatetime(),
            max_date.to_pydatetime()
        )

        # Merge sales with weather
        daily_sales["date"] = pd.to_datetime(daily_sales["date"])
        weather_df["date"] = pd.to_datetime(weather_df["date"])

        merged = daily_sales.merge(weather_df, on="date", how="left")

        # Fill missing weather data
        weather_cols = ["temp_mean", "precipitation", "is_rainy", "is_cold", "is_hot"]
        for col in weather_cols:
            if col in merged.columns:
                merged[col] = merged[col].fillna(merged[col].median())

        # Create feature matrix
        features_df = self._create_feature_matrix(merged, target_column)

        feature_names = [
            target_column,  # Target (first column)
            "order_count" if target_column == "total_revenue" else "total_revenue",
            "day_mon", "day_tue", "day_wed", "day_thu", "day_fri", "day_sat", "day_sun",
            "is_weekend", "is_holiday",
            "temp_mean", "precipitation",
            "month_sin", "month_cos"
        ]

        return features_df, feature_names

    def _create_feature_matrix(
        self,
        df: pd.DataFrame,
        target_column: str
    ) -> pd.DataFrame:
        """Create the actual feature matrix from merged data."""
        features = pd.DataFrame()

        # Target variable (first column)
        features[target_column] = df[target_column].values

        # Secondary sales metric
        if target_column == "total_revenue":
            features["order_count"] = df["order_count"].values
        else:
            features["total_revenue"] = df["total_revenue"].values

        # Day of week one-hot encoding
        for i, day in enumerate(["mon", "tue", "wed", "thu", "fri", "sat", "sun"]):
            features[f"day_{day}"] = (df["day_of_week"] == i).astype(int).values

        # Weekend and holiday flags
        features["is_weekend"] = df["is_weekend"].values
        features["is_holiday"] = df["is_holiday"].values

        # Weather features
        features["temp_mean"] = df.get("temp_mean", pd.Series([10] * len(df))).values
        features["precipitation"] = df.get("precipitation", pd.Series([0] * len(df))).values

        # Cyclical encoding of month
        month = df["month"].values
        features["month_sin"] = np.sin(2 * np.pi * month / 12)
        features["month_cos"] = np.cos(2 * np.pi * month / 12)

        return features

    def create_forecast_features(
        self,
        start_date: datetime,
        days: int = 7,
        historical_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Create feature matrix for forecasting future dates.

        Args:
            start_date: First date to forecast
            days: Number of days to forecast
            historical_data: Historical feature data for context

        Returns:
            Feature matrix for forecast dates
        """
        # Get weather forecast
        weather_forecast = self.weather_service.get_forecast(days)

        # Create date range
        dates = pd.date_range(start=start_date, periods=days, freq="D")

        features = pd.DataFrame()
        features["date"] = dates

        # Day of week one-hot
        dow = dates.dayofweek
        for i, day in enumerate(["mon", "tue", "wed", "thu", "fri", "sat", "sun"]):
            features[f"day_{day}"] = (dow == i).astype(int)

        # Weekend and holiday
        features["is_weekend"] = dow.isin([5, 6]).astype(int)
        date_strs = dates.strftime("%Y-%m-%d")
        features["is_holiday"] = date_strs.isin(DANISH_HOLIDAYS_2021_2024.keys()).astype(int)

        # Weather features
        if len(weather_forecast) >= days:
            features["temp_mean"] = weather_forecast["temp_mean"].values[:days]
            features["precipitation"] = weather_forecast["precipitation"].values[:days]
        else:
            # Fallback to averages
            features["temp_mean"] = 10.0
            features["precipitation"] = 2.0

        # Cyclical month encoding
        month = dates.month
        features["month_sin"] = np.sin(2 * np.pi * month / 12)
        features["month_cos"] = np.cos(2 * np.pi * month / 12)

        return features

    def get_weather_adjustment(self, weather_summary: Dict) -> float:
        """
        Calculate demand adjustment factor based on weather.

        Args:
            weather_summary: Weather summary dict from WeatherService

        Returns:
            Adjustment multiplier (e.g., 1.15 for +15%)
        """
        adjustment = 1.0

        # Rain increases indoor dining / delivery
        if weather_summary.get("is_rainy", 0):
            adjustment *= 1.10

        # Cold weather increases hot food demand
        if weather_summary.get("is_cold", 0):
            adjustment *= 1.08

        # Hot weather can reduce appetite but increase drinks
        if weather_summary.get("is_hot", 0):
            adjustment *= 0.95

        return adjustment

    def get_day_adjustment(self, date: datetime) -> Tuple[float, str]:
        """
        Calculate demand adjustment factor based on day/date.

        Args:
            date: Date to check

        Returns:
            Tuple of (adjustment_multiplier, reason_string)
        """
        adjustment = 1.0
        reasons = []

        # Weekend boost
        if date.weekday() >= 5:
            if date.weekday() == 5:  # Saturday
                adjustment *= 1.32
                reasons.append("Saturday (+32%)")
            else:  # Sunday
                adjustment *= 1.08
                reasons.append("Sunday (+8%)")

        # Friday boost
        if date.weekday() == 4:
            adjustment *= 1.18
            reasons.append("Friday (+18%)")

        # Holiday check
        date_str = date.strftime("%Y-%m-%d")
        if date_str in DANISH_HOLIDAYS_2021_2024:
            holiday_name = DANISH_HOLIDAYS_2021_2024[date_str]

            # Some holidays reduce demand (Christmas Eve)
            if "Christmas Eve" in holiday_name:
                adjustment *= 0.35
                reasons.append(f"{holiday_name} (-65%)")
            elif "Christmas" in holiday_name:
                adjustment *= 0.50
                reasons.append(f"{holiday_name} (-50%)")
            else:
                adjustment *= 1.25
                reasons.append(f"{holiday_name} (+25%)")

        reason_str = ", ".join(reasons) if reasons else "Normal day"
        return adjustment, reason_str

    def analyze_historical_patterns(
        self,
        place_id: Optional[int] = None
    ) -> Dict:
        """
        Analyze historical data to extract patterns.

        Returns:
            Dictionary with pattern analysis
        """
        daily_sales = self.data_processor.get_daily_sales(place_id)
        dow_pattern = self.data_processor.get_day_of_week_pattern(place_id)
        hourly_pattern = self.data_processor.get_hourly_pattern(place_id)

        # Calculate weather correlations if we have weather data
        min_date = daily_sales["date"].min()
        max_date = daily_sales["date"].max()

        weather_df = self.weather_service.get_historical_weather(
            min_date.to_pydatetime(),
            max_date.to_pydatetime()
        )

        daily_sales["date"] = pd.to_datetime(daily_sales["date"])
        weather_df["date"] = pd.to_datetime(weather_df["date"])
        merged = daily_sales.merge(weather_df, on="date", how="left")

        # Weather impact analysis
        weather_impact = {}
        if "temp_mean" in merged.columns:
            # Segment by temperature
            cold_days = merged[merged["temp_mean"] < 5]["total_revenue"].mean()
            moderate_days = merged[(merged["temp_mean"] >= 5) & (merged["temp_mean"] <= 20)]["total_revenue"].mean()
            hot_days = merged[merged["temp_mean"] > 20]["total_revenue"].mean()

            baseline = moderate_days if moderate_days > 0 else 1

            weather_impact["cold"] = {
                "avg_revenue": cold_days,
                "vs_baseline": (cold_days - baseline) / baseline if baseline > 0 else 0
            }
            weather_impact["moderate"] = {
                "avg_revenue": moderate_days,
                "vs_baseline": 0
            }
            weather_impact["hot"] = {
                "avg_revenue": hot_days,
                "vs_baseline": (hot_days - baseline) / baseline if baseline > 0 else 0
            }

        if "is_rainy" in merged.columns:
            rainy_days = merged[merged["is_rainy"] == 1]["total_revenue"].mean()
            dry_days = merged[merged["is_rainy"] == 0]["total_revenue"].mean()

            weather_impact["rainy"] = {
                "avg_revenue": rainy_days,
                "vs_baseline": (rainy_days - dry_days) / dry_days if dry_days > 0 else 0
            }

        return {
            "day_of_week_pattern": dow_pattern.to_dict("records"),
            "hourly_pattern": hourly_pattern.to_dict("records"),
            "weather_impact": weather_impact,
            "total_days": len(daily_sales),
            "avg_daily_revenue": daily_sales["total_revenue"].mean(),
            "avg_daily_orders": daily_sales["order_count"].mean()
        }
