"""
FreshFlow utility functions for data handling, formatting, and common operations.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "Inventory Management")
MODELS_DIR = os.path.join(PROJECT_ROOT, "src", "freshflow", "saved_models")

# Danish holidays (hardcoded for reliability)
DANISH_HOLIDAYS_2021_2024 = {
    # 2021
    "2021-01-01": "New Year's Day",
    "2021-04-01": "Maundy Thursday",
    "2021-04-02": "Good Friday",
    "2021-04-04": "Easter Sunday",
    "2021-04-05": "Easter Monday",
    "2021-05-13": "Ascension Day",
    "2021-05-23": "Whit Sunday",
    "2021-05-24": "Whit Monday",
    "2021-06-05": "Constitution Day",
    "2021-12-24": "Christmas Eve",
    "2021-12-25": "Christmas Day",
    "2021-12-26": "Second Christmas Day",
    "2021-12-31": "New Year's Eve",
    # 2022
    "2022-01-01": "New Year's Day",
    "2022-04-14": "Maundy Thursday",
    "2022-04-15": "Good Friday",
    "2022-04-17": "Easter Sunday",
    "2022-04-18": "Easter Monday",
    "2022-05-26": "Ascension Day",
    "2022-06-05": "Constitution Day",
    "2022-06-05": "Whit Sunday",
    "2022-06-06": "Whit Monday",
    "2022-12-24": "Christmas Eve",
    "2022-12-25": "Christmas Day",
    "2022-12-26": "Second Christmas Day",
    "2022-12-31": "New Year's Eve",
    # 2023
    "2023-01-01": "New Year's Day",
    "2023-04-06": "Maundy Thursday",
    "2023-04-07": "Good Friday",
    "2023-04-09": "Easter Sunday",
    "2023-04-10": "Easter Monday",
    "2023-05-18": "Ascension Day",
    "2023-05-28": "Whit Sunday",
    "2023-05-29": "Whit Monday",
    "2023-06-05": "Constitution Day",
    "2023-12-24": "Christmas Eve",
    "2023-12-25": "Christmas Day",
    "2023-12-26": "Second Christmas Day",
    "2023-12-31": "New Year's Eve",
    # 2024
    "2024-01-01": "New Year's Day",
    "2024-03-28": "Maundy Thursday",
    "2024-03-29": "Good Friday",
    "2024-03-31": "Easter Sunday",
    "2024-04-01": "Easter Monday",
    "2024-05-09": "Ascension Day",
    "2024-05-19": "Whit Sunday",
    "2024-05-20": "Whit Monday",
    "2024-06-05": "Constitution Day",
    "2024-12-24": "Christmas Eve",
    "2024-12-25": "Christmas Day",
    "2024-12-26": "Second Christmas Day",
    "2024-12-31": "New Year's Eve",
    # 2025
    "2025-01-01": "New Year's Day",
    "2025-04-17": "Maundy Thursday",
    "2025-04-18": "Good Friday",
    "2025-04-20": "Easter Sunday",
    "2025-04-21": "Easter Monday",
    "2025-05-29": "Ascension Day",
    "2025-06-05": "Constitution Day",
    "2025-06-08": "Whit Sunday",
    "2025-06-09": "Whit Monday",
    "2025-12-24": "Christmas Eve",
    "2025-12-25": "Christmas Day",
    "2025-12-26": "Second Christmas Day",
    "2025-12-31": "New Year's Eve",
    # 2026
    "2026-01-01": "New Year's Day",
    "2026-04-02": "Maundy Thursday",
    "2026-04-03": "Good Friday",
    "2026-04-05": "Easter Sunday",
    "2026-04-06": "Easter Monday",
    "2026-05-14": "Ascension Day",
    "2026-05-24": "Whit Sunday",
    "2026-05-25": "Whit Monday",
    "2026-06-05": "Constitution Day",
    "2026-12-24": "Christmas Eve",
    "2026-12-25": "Christmas Day",
    "2026-12-26": "Second Christmas Day",
    "2026-12-31": "New Year's Eve",
}


def unix_to_datetime(unix_ts: int) -> datetime:
    """Convert UNIX timestamp to datetime."""
    return datetime.fromtimestamp(unix_ts)


def datetime_to_unix(dt: datetime) -> int:
    """Convert datetime to UNIX timestamp."""
    return int(dt.timestamp())


def format_currency(amount: float, currency: str = "DKK") -> str:
    """Format amount as currency string."""
    return f"{amount:,.2f} {currency}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format value as percentage string."""
    return f"{value * 100:.{decimals}f}%"


def is_holiday(date: datetime) -> bool:
    """Check if date is a Danish holiday."""
    date_str = date.strftime("%Y-%m-%d")
    return date_str in DANISH_HOLIDAYS_2021_2024


def get_holiday_name(date: datetime) -> Optional[str]:
    """Get holiday name if date is a Danish holiday."""
    date_str = date.strftime("%Y-%m-%d")
    return DANISH_HOLIDAYS_2021_2024.get(date_str)


def is_weekend(date: datetime) -> bool:
    """Check if date is a weekend (Saturday or Sunday)."""
    return date.weekday() >= 5


def get_day_of_week(date: datetime) -> int:
    """Get day of week (0=Monday, 6=Sunday)."""
    return date.weekday()


def get_day_name(date: datetime) -> str:
    """Get day name."""
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    return days[date.weekday()]


def calculate_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error."""
    mask = actual != 0
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask]))


def calculate_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Root Mean Square Error."""
    return np.sqrt(np.mean((actual - predicted) ** 2))


def calculate_r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate R-squared score."""
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0


def get_trend_indicator(current: float, previous: float) -> Tuple[str, str]:
    """Get trend indicator (arrow and color) based on change."""
    if previous == 0:
        return "→", "gray"

    change = (current - previous) / previous
    if change > 0.05:
        return "↑", "green"
    elif change < -0.05:
        return "↓", "red"
    else:
        return "→", "gray"


def get_confidence_color(confidence: float) -> str:
    """Get color based on confidence level."""
    if confidence >= 0.8:
        return "green"
    elif confidence >= 0.5:
        return "orange"
    else:
        return "red"


def create_date_range(start_date: datetime, end_date: datetime) -> List[datetime]:
    """Create list of dates between start and end (inclusive)."""
    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        current += timedelta(days=1)
    return dates


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
