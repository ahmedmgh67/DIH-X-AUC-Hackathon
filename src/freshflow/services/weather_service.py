"""
Weather service for fetching historical and forecast weather data.
Uses Open-Meteo API (free, no API key required).
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np


# Open-Meteo API endpoints
HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Copenhagen coordinates (default for Denmark)
DEFAULT_LATITUDE = 55.6761
DEFAULT_LONGITUDE = 12.5683

# Weather code descriptions
WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


class WeatherService:
    """Fetches weather data from Open-Meteo API."""

    def __init__(self, latitude: float = DEFAULT_LATITUDE, longitude: float = DEFAULT_LONGITUDE):
        self.latitude = latitude
        self.longitude = longitude
        self._cache: Dict[str, pd.DataFrame] = {}

    def get_historical_weather(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Fetch historical weather data for a date range.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with daily weather data
        """
        cache_key = f"hist_{start_date.date()}_{end_date.date()}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "daily": [
                "temperature_2m_mean",
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_sum",
                "rain_sum",
                "snowfall_sum",
                "weather_code"
            ],
            "timezone": "Europe/Copenhagen"
        }

        try:
            response = requests.get(HISTORICAL_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            df = pd.DataFrame({
                "date": pd.to_datetime(data["daily"]["time"]),
                "temp_mean": data["daily"]["temperature_2m_mean"],
                "temp_max": data["daily"]["temperature_2m_max"],
                "temp_min": data["daily"]["temperature_2m_min"],
                "precipitation": data["daily"]["precipitation_sum"],
                "rain": data["daily"]["rain_sum"],
                "snow": data["daily"]["snowfall_sum"],
                "weather_code": data["daily"]["weather_code"]
            })

            # Add derived features
            df["is_rainy"] = (df["precipitation"] > 1).astype(int)
            df["is_cold"] = (df["temp_mean"] < 5).astype(int)
            df["is_hot"] = (df["temp_mean"] > 20).astype(int)
            df["weather_desc"] = df["weather_code"].map(WEATHER_CODES).fillna("Unknown")

            self._cache[cache_key] = df
            return df

        except Exception as e:
            print(f"Error fetching historical weather: {e}")
            return self._generate_synthetic_weather(start_date, end_date)

    def get_forecast(self, days: int = 7) -> pd.DataFrame:
        """
        Fetch weather forecast for next N days.

        Args:
            days: Number of days to forecast (max 16)

        Returns:
            DataFrame with forecast data
        """
        cache_key = f"forecast_{datetime.now().date()}_{days}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "daily": [
                "temperature_2m_mean",
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_sum",
                "precipitation_probability_max",
                "weather_code"
            ],
            "timezone": "Europe/Copenhagen",
            "forecast_days": min(days, 16)
        }

        try:
            response = requests.get(FORECAST_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            df = pd.DataFrame({
                "date": pd.to_datetime(data["daily"]["time"]),
                "temp_mean": data["daily"]["temperature_2m_mean"],
                "temp_max": data["daily"]["temperature_2m_max"],
                "temp_min": data["daily"]["temperature_2m_min"],
                "precipitation": data["daily"]["precipitation_sum"],
                "precip_probability": data["daily"]["precipitation_probability_max"],
                "weather_code": data["daily"]["weather_code"]
            })

            # Add derived features
            df["is_rainy"] = (df["precipitation"] > 1).astype(int)
            df["is_cold"] = (df["temp_mean"] < 5).astype(int)
            df["is_hot"] = (df["temp_mean"] > 20).astype(int)
            df["weather_desc"] = df["weather_code"].map(WEATHER_CODES).fillna("Unknown")

            self._cache[cache_key] = df
            return df

        except Exception as e:
            print(f"Error fetching forecast: {e}")
            return self._generate_synthetic_forecast(days)

    def _generate_synthetic_weather(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Generate synthetic weather data when API is unavailable."""
        dates = pd.date_range(start=start_date, end=end_date, freq="D")
        n = len(dates)

        # Seasonal temperature patterns (Denmark)
        day_of_year = dates.dayofyear
        base_temp = 8 + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        temp_noise = np.random.normal(0, 3, n)
        temp_mean = base_temp + temp_noise

        # Precipitation (more in autumn/winter)
        precip_base = 2 + 1.5 * np.sin(2 * np.pi * (day_of_year - 200) / 365)
        precipitation = np.maximum(0, precip_base + np.random.exponential(2, n) - 2)

        # Weather codes based on conditions
        weather_codes = []
        for i in range(n):
            if precipitation[i] > 5:
                weather_codes.append(63)  # Moderate rain
            elif precipitation[i] > 1:
                weather_codes.append(61)  # Slight rain
            elif temp_mean[i] < 0:
                weather_codes.append(71)  # Slight snow
            else:
                weather_codes.append(np.random.choice([0, 1, 2, 3]))

        df = pd.DataFrame({
            "date": dates,
            "temp_mean": temp_mean,
            "temp_max": temp_mean + np.random.uniform(2, 5, n),
            "temp_min": temp_mean - np.random.uniform(2, 5, n),
            "precipitation": precipitation,
            "rain": precipitation * (temp_mean > 0).astype(float),
            "snow": precipitation * (temp_mean <= 0).astype(float),
            "weather_code": weather_codes
        })

        df["is_rainy"] = (df["precipitation"] > 1).astype(int)
        df["is_cold"] = (df["temp_mean"] < 5).astype(int)
        df["is_hot"] = (df["temp_mean"] > 20).astype(int)
        df["weather_desc"] = df["weather_code"].map(WEATHER_CODES).fillna("Unknown")

        return df

    def _generate_synthetic_forecast(self, days: int) -> pd.DataFrame:
        """Generate synthetic forecast data when API is unavailable."""
        start_date = datetime.now()
        end_date = start_date + timedelta(days=days - 1)
        df = self._generate_synthetic_weather(start_date, end_date)
        df["precip_probability"] = np.random.uniform(0, 100, len(df))
        return df

    def get_weather_impact_summary(self, weather_df: pd.DataFrame) -> Dict:
        """
        Summarize weather conditions for display.

        Args:
            weather_df: DataFrame with weather data

        Returns:
            Dictionary with weather summary
        """
        if weather_df.empty:
            return {
                "condition": "Unknown",
                "temp": 0,
                "icon": "?"
            }

        row = weather_df.iloc[0] if len(weather_df) > 0 else {}

        temp = row.get("temp_mean", 0)
        weather_code = row.get("weather_code", 0)

        # Determine icon
        if weather_code in [0, 1]:
            icon = "sunny"
        elif weather_code in [2, 3]:
            icon = "cloudy"
        elif weather_code in [61, 63, 65, 80, 81, 82]:
            icon = "rainy"
        elif weather_code in [71, 73, 75, 85, 86]:
            icon = "snowy"
        else:
            icon = "cloudy"

        return {
            "condition": row.get("weather_desc", "Unknown"),
            "temp": round(temp, 1),
            "icon": icon,
            "is_rainy": row.get("is_rainy", 0),
            "is_cold": row.get("is_cold", 0),
            "is_hot": row.get("is_hot", 0)
        }
