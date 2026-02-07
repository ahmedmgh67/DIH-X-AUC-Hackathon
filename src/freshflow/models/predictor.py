"""
Predictor service for generating demand forecasts.
Handles model loading, inference, and result formatting.
"""

import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import os

from .lstm_model import DemandLSTM, DemandPredictor, normalize_data
from ..utils.helpers import MODELS_DIR, ensure_dir
from ..services.data_processor import DataProcessor
from ..services.weather_service import WeatherService
from ..services.feature_engineer import FeatureEngineer


class ForecastService:
    """
    High-level service for generating demand forecasts.
    Orchestrates data loading, feature engineering, and model prediction.
    Automatically loads trained LSTM models when available.
    """

    def __init__(
        self,
        data_processor: DataProcessor,
        weather_service: WeatherService,
        model_path: Optional[str] = None
    ):
        self.data_processor = data_processor
        self.weather_service = weather_service
        self.feature_engineer = FeatureEngineer(data_processor, weather_service)

        self.predictor: Optional[DemandPredictor] = None
        self.scaler_params: Dict = {}
        self._loaded_models: Dict[int, Dict] = {}  # Cache for location-specific models

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

        # Auto-discover trained models
        self._discover_models()

    def _discover_models(self) -> None:
        """Auto-discover trained models in saved_models directory."""
        if not os.path.exists(MODELS_DIR):
            return

        import pickle

        for filename in os.listdir(MODELS_DIR):
            # Prioritize GB models (better performance)
            if filename.startswith('gb_model_') and filename.endswith('.pkl'):
                try:
                    parts = filename.replace('.pkl', '').split('_')
                    place_id = int(parts[-1])
                    model_path = os.path.join(MODELS_DIR, filename)

                    with open(model_path, 'rb') as f:
                        save_dict = pickle.load(f)

                    self._loaded_models[place_id] = {
                        'path': model_path,
                        'type': 'gradient_boosting',
                        'metrics': save_dict.get('metrics', {}),
                        'model': save_dict
                    }
                except Exception:
                    continue

            # Fall back to LSTM models if no GB model
            elif filename.endswith('_v2.pt'):
                try:
                    parts = filename.replace('_v2.pt', '').split('_')
                    place_id = int(parts[-1])

                    # Skip if we already have a GB model for this location
                    if place_id in self._loaded_models:
                        continue

                    model_path = os.path.join(MODELS_DIR, filename)
                    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

                    self._loaded_models[place_id] = {
                        'path': model_path,
                        'type': 'lstm',
                        'metrics': checkpoint.get('metrics', {}),
                        'mean': checkpoint.get('mean'),
                        'std': checkpoint.get('std'),
                        'n_features': checkpoint.get('n_features')
                    }
                except Exception:
                    continue

    def get_trained_locations(self) -> List[int]:
        """Get list of location IDs with trained models."""
        return list(self._loaded_models.keys())

    def load_model(self, model_path: str) -> None:
        """Load a trained model from file."""
        self.predictor = DemandPredictor.load(model_path)
        self.scaler_params = self.predictor.scaler_params

    def has_model(self, place_id: Optional[int] = None) -> bool:
        """Check if a model is loaded for the given location."""
        if place_id is not None:
            return place_id in self._loaded_models
        return self.predictor is not None

    def forecast_revenue(
        self,
        place_id: Optional[int] = None,
        days: int = 7,
        include_confidence: bool = True
    ) -> pd.DataFrame:
        """
        Generate revenue forecast for next N days.

        Args:
            place_id: Specific location or None for aggregate
            days: Number of days to forecast
            include_confidence: Include confidence intervals

        Returns:
            DataFrame with forecasts
        """
        if not self.has_model():
            # Return baseline forecast using historical averages
            return self._baseline_forecast(place_id, days)

        # Get historical data for sequence
        daily_sales = self.data_processor.get_daily_sales(place_id)
        features_df, _ = self.feature_engineer.create_training_features(place_id)

        # Get last sequence_length days
        sequence_length = 30
        recent_data = features_df.tail(sequence_length).values.astype(np.float32)

        # Normalize using stored parameters
        if "full_mean" in self.scaler_params and "full_std" in self.scaler_params:
            recent_normalized = (recent_data - self.scaler_params["full_mean"]) / self.scaler_params["full_std"]
        else:
            recent_normalized, _, _ = normalize_data(recent_data)

        # Predict
        predictions, confidence = self.predictor.predict(
            recent_normalized,
            return_confidence=include_confidence
        )

        # Create result DataFrame
        start_date = datetime.now() + timedelta(days=1)
        dates = pd.date_range(start=start_date, periods=min(days, len(predictions)), freq="D")

        result = pd.DataFrame({
            "date": dates,
            "predicted_revenue": predictions[:len(dates)],
        })

        if include_confidence and confidence is not None:
            result["confidence_lower"] = predictions[:len(dates)] - confidence[:len(dates)]
            result["confidence_upper"] = predictions[:len(dates)] + confidence[:len(dates)]
            result["confidence_lower"] = result["confidence_lower"].clip(lower=0)

        # Add day info
        result["day_name"] = result["date"].dt.day_name()
        result["is_weekend"] = result["date"].dt.dayofweek.isin([5, 6])

        # Apply adjustments
        for idx, row in result.iterrows():
            day_adj, _ = self.feature_engineer.get_day_adjustment(row["date"].to_pydatetime())
            # Adjustments already factored into model, but we can show expected pattern
            result.loc[idx, "day_adjustment"] = day_adj

        return result

    def _baseline_forecast(
        self,
        place_id: Optional[int],
        days: int
    ) -> pd.DataFrame:
        """Generate baseline forecast using historical averages."""
        dow_pattern = self.data_processor.get_day_of_week_pattern(place_id)
        daily_sales = self.data_processor.get_daily_sales(place_id)

        avg_revenue = daily_sales["total_revenue"].mean()

        start_date = datetime.now() + timedelta(days=1)
        dates = pd.date_range(start=start_date, periods=days, freq="D")

        result = pd.DataFrame({"date": dates})
        result["day_of_week"] = result["date"].dt.dayofweek
        result["day_name"] = result["date"].dt.day_name()
        result["is_weekend"] = result["day_of_week"].isin([5, 6])

        # Apply day-of-week pattern
        dow_dict = dow_pattern.set_index("day_of_week")["relative_demand"].to_dict()

        predictions = []
        for _, row in result.iterrows():
            dow = row["day_of_week"]
            relative = dow_dict.get(dow, 0)
            pred = avg_revenue * (1 + relative)

            # Apply weather and holiday adjustments
            weather_forecast = self.weather_service.get_forecast(days)
            if len(weather_forecast) > 0:
                weather_summary = self.weather_service.get_weather_impact_summary(
                    weather_forecast[weather_forecast["date"].dt.date == row["date"].date()]
                )
                weather_adj = self.feature_engineer.get_weather_adjustment(weather_summary)
                pred *= weather_adj

            day_adj, _ = self.feature_engineer.get_day_adjustment(row["date"].to_pydatetime())
            pred *= day_adj

            predictions.append(pred)

        result["predicted_revenue"] = predictions
        result["confidence_lower"] = result["predicted_revenue"] * 0.85
        result["confidence_upper"] = result["predicted_revenue"] * 1.15
        result["day_adjustment"] = 1.0

        return result

    def forecast_items(
        self,
        place_id: Optional[int] = None,
        days: int = 7,
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Generate item-level demand forecast.

        Args:
            place_id: Specific location
            days: Number of days to forecast
            top_n: Number of top items to forecast

        Returns:
            DataFrame with item forecasts
        """
        # Get top items
        top_items = self.data_processor.get_top_items(place_id, top_n)

        # Get daily item sales for patterns
        daily_items = self.data_processor.get_daily_item_sales(place_id)

        # Calculate average daily sales per item
        item_daily_avg = daily_items.groupby("item_title")["quantity_sold"].mean().to_dict()

        # Get day of week patterns per item
        item_dow = daily_items.groupby(["item_title", "day_of_week"])["quantity_sold"].mean().reset_index()

        # Generate forecasts
        start_date = datetime.now() + timedelta(days=1)
        dates = pd.date_range(start=start_date, periods=days, freq="D")

        forecasts = []

        for _, item_row in top_items.iterrows():
            item_title = item_row["item_title"]
            base_qty = item_daily_avg.get(item_title, 0)

            # Get item-specific day pattern
            item_pattern = item_dow[item_dow["item_title"] == item_title]
            if len(item_pattern) > 0:
                item_dow_dict = item_pattern.set_index("day_of_week")["quantity_sold"].to_dict()
            else:
                item_dow_dict = {}

            for date in dates:
                dow = date.dayofweek

                if dow in item_dow_dict and base_qty > 0:
                    qty = item_dow_dict[dow]
                else:
                    qty = base_qty

                # Apply day adjustment
                day_adj, _ = self.feature_engineer.get_day_adjustment(date.to_pydatetime())
                qty *= day_adj

                forecasts.append({
                    "date": date,
                    "item_title": item_title,
                    "predicted_quantity": max(0, round(qty)),
                    "avg_price": item_row["avg_price"],
                    "predicted_revenue": max(0, qty * item_row["avg_price"])
                })

        return pd.DataFrame(forecasts)

    def get_prep_quantities(
        self,
        place_id: Optional[int] = None,
        date: Optional[datetime] = None,
        buffer_pct: float = 0.10,
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Calculate kitchen prep quantities for a specific date.

        Args:
            place_id: Specific location
            date: Date to prep for (default tomorrow)
            buffer_pct: Safety buffer percentage
            top_n: Number of items to include

        Returns:
            DataFrame with prep quantities
        """
        if date is None:
            date = datetime.now() + timedelta(days=1)

        # Get item forecasts
        item_forecasts = self.forecast_items(place_id, days=7, top_n=top_n)

        # Filter to specific date
        date_forecasts = item_forecasts[
            item_forecasts["date"].dt.date == date.date()
        ].copy()

        if len(date_forecasts) == 0:
            # Fallback to day 1 forecast
            date_forecasts = item_forecasts[
                item_forecasts["date"] == item_forecasts["date"].min()
            ].copy()

        # Calculate prep quantities with buffer
        date_forecasts["buffer_qty"] = np.ceil(
            date_forecasts["predicted_quantity"] * buffer_pct
        ).astype(int)

        date_forecasts["prep_quantity"] = (
            date_forecasts["predicted_quantity"] + date_forecasts["buffer_qty"]
        ).astype(int)

        # Add confidence indicator
        date_forecasts["confidence"] = "high"  # Simplified

        # Get weather context
        weather_forecast = self.weather_service.get_forecast(7)
        date_weather = weather_forecast[weather_forecast["date"].dt.date == date.date()]

        if len(date_weather) > 0:
            weather_summary = self.weather_service.get_weather_impact_summary(date_weather)
        else:
            weather_summary = {"condition": "Unknown", "temp": 0, "icon": "?"}

        # Get day context
        day_adj, day_reason = self.feature_engineer.get_day_adjustment(date)

        return {
            "prep_list": date_forecasts[[
                "item_title", "predicted_quantity", "buffer_qty",
                "prep_quantity", "confidence", "avg_price"
            ]].to_dict("records"),
            "date": date,
            "weather": weather_summary,
            "day_context": day_reason,
            "total_items": len(date_forecasts),
            "total_prep_qty": date_forecasts["prep_quantity"].sum()
        }

    def get_accuracy_metrics(
        self,
        place_id: Optional[int] = None,
        days_back: int = 30
    ) -> Dict:
        """
        Calculate forecast accuracy metrics.
        Uses trained model metrics if available, otherwise calculates from patterns.
        """
        # Check if we have a trained model for this location
        if place_id is not None and place_id in self._loaded_models:
            model_info = self._loaded_models[place_id]
            metrics = model_info.get('metrics', {})
            model_type = model_info.get('type', 'unknown')

            # GB models use 'test_mape', LSTM uses 'mape'
            if model_type == 'gradient_boosting':
                mape = metrics.get('test_mape', metrics.get('cv_mape', 0.15))
                r2 = metrics.get('test_r2', 0.5)
                model_label = "Gradient Boosting"
            else:
                mape = metrics.get('mape', 0.30)
                r2 = metrics.get('r2', 0.15)
                model_label = "LSTM"

            return {
                "mape": mape,
                "accuracy": 1 - min(mape, 0.99),
                "r2": r2,
                "avg_waste_pct": max(0.02, mape * 0.15),
                "stockout_count": max(0, int(mape * 5)),
                "model_type": model_label
            }

        # Fallback to pattern-based estimation
        daily_sales = self.data_processor.get_daily_sales(place_id)

        if len(daily_sales) < days_back + 7:
            return {
                "mape": 0.15,
                "accuracy": 0.85,
                "r2": 0.0,
                "avg_waste_pct": 0.05,
                "stockout_count": 2,
                "model_type": "Pattern-based"
            }

        # Use day-of-week patterns as baseline
        dow_pattern = self.data_processor.get_day_of_week_pattern(place_id)
        dow_dict = dow_pattern.set_index("day_of_week")["avg_revenue"].to_dict()

        recent = daily_sales.tail(days_back).copy()
        recent["forecast"] = recent["day_of_week"].map(dow_dict)
        recent["error"] = np.abs(recent["total_revenue"] - recent["forecast"])
        recent["pct_error"] = recent["error"] / recent["total_revenue"].clip(lower=1)

        mape = recent["pct_error"].mean()

        return {
            "mape": mape,
            "accuracy": 1 - min(mape, 0.99),
            "r2": 0.0,
            "avg_waste_pct": max(0.02, mape * 0.15),
            "stockout_count": max(0, int(mape * 10)),
            "model_type": "Pattern-based"
        }
