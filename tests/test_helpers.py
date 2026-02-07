"""
Tests for utility helper functions.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from freshflow.utils.helpers import (
    unix_to_datetime,
    format_currency,
    format_percentage,
    is_holiday,
    is_weekend,
    get_day_name,
    calculate_mape,
    calculate_rmse,
    calculate_r2,
    get_trend_indicator,
    create_date_range,
)


class TestUnixTimestamp:
    def test_known_timestamp(self):
        result = unix_to_datetime(1609459200)
        assert result.year == 2021
        assert result.month == 1
        assert result.day == 1

    def test_returns_datetime(self):
        result = unix_to_datetime(0)
        assert isinstance(result, datetime)


class TestFormatCurrency:
    def test_default_currency(self):
        result = format_currency(1250.50)
        assert "1,250.50" in result
        assert "DKK" in result

    def test_custom_currency(self):
        result = format_currency(99.99, "USD")
        assert "USD" in result

    def test_zero(self):
        result = format_currency(0)
        assert "0.00" in result


class TestFormatPercentage:
    def test_basic(self):
        assert format_percentage(0.5) == "50.0%"

    def test_decimals(self):
        assert format_percentage(0.1234, 2) == "12.34%"


class TestHolidays:
    def test_christmas_2024_is_holiday(self):
        assert is_holiday(datetime(2024, 12, 25)) is True

    def test_regular_day_is_not_holiday(self):
        assert is_holiday(datetime(2024, 3, 15)) is False

    def test_new_years_2025(self):
        assert is_holiday(datetime(2025, 1, 1)) is True


class TestWeekend:
    def test_saturday(self):
        assert is_weekend(datetime(2025, 2, 1)) is True  # Saturday

    def test_sunday(self):
        assert is_weekend(datetime(2025, 2, 2)) is True  # Sunday

    def test_monday(self):
        assert is_weekend(datetime(2025, 2, 3)) is False  # Monday


class TestDayName:
    def test_monday(self):
        assert get_day_name(datetime(2025, 2, 3)) == "Monday"

    def test_friday(self):
        assert get_day_name(datetime(2025, 2, 7)) == "Friday"


class TestMetrics:
    def test_mape(self):
        actual = np.array([100, 200, 300])
        predicted = np.array([110, 190, 310])
        mape = calculate_mape(actual, predicted)
        assert 0 < mape < 0.1  # Should be around 5-6%

    def test_rmse(self):
        actual = np.array([100.0, 200.0, 300.0])
        predicted = np.array([100.0, 200.0, 300.0])
        assert calculate_rmse(actual, predicted) == 0.0

    def test_rmse_with_error(self):
        actual = np.array([100.0, 200.0])
        predicted = np.array([110.0, 190.0])
        rmse = calculate_rmse(actual, predicted)
        assert rmse == 10.0

    def test_r2_perfect(self):
        actual = np.array([1.0, 2.0, 3.0])
        predicted = np.array([1.0, 2.0, 3.0])
        assert calculate_r2(actual, predicted) == 1.0

    def test_r2_imperfect(self):
        actual = np.array([1.0, 2.0, 3.0])
        predicted = np.array([1.1, 2.2, 2.8])
        r2 = calculate_r2(actual, predicted)
        assert 0 < r2 < 1


class TestTrendIndicator:
    def test_upward(self):
        arrow, color = get_trend_indicator(120, 100)
        assert arrow == "↑"
        assert color == "green"

    def test_downward(self):
        arrow, color = get_trend_indicator(80, 100)
        assert arrow == "↓"
        assert color == "red"

    def test_stable(self):
        arrow, color = get_trend_indicator(101, 100)
        assert arrow == "→"
        assert color == "gray"

    def test_zero_previous(self):
        arrow, color = get_trend_indicator(100, 0)
        assert arrow == "→"


class TestDateRange:
    def test_single_day(self):
        start = datetime(2025, 1, 1)
        result = create_date_range(start, start)
        assert len(result) == 1

    def test_week(self):
        start = datetime(2025, 1, 1)
        end = datetime(2025, 1, 7)
        result = create_date_range(start, end)
        assert len(result) == 7
