"""
Tests for inventory management calculations.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from freshflow.services.inventory_service import InventoryManager, StockStatus


@pytest.fixture
def inventory_manager():
    mock_dp = MagicMock()
    manager = InventoryManager(mock_dp)
    return manager


class TestReorderPoint:
    def test_basic_calculation(self, inventory_manager):
        rop = inventory_manager.calculate_reorder_point(
            avg_daily_demand=10,
            demand_std=2,
            lead_time_days=3,
            service_level=0.95
        )
        # ROP = (10 * 3) + (1.65 * 2 * sqrt(3)) = 30 + 5.72 = 35.72
        assert 35 < rop < 37

    def test_zero_demand(self, inventory_manager):
        rop = inventory_manager.calculate_reorder_point(
            avg_daily_demand=0,
            demand_std=0,
            lead_time_days=3,
            service_level=0.95
        )
        assert rop == 0

    def test_higher_service_level_means_higher_rop(self, inventory_manager):
        rop_95 = inventory_manager.calculate_reorder_point(10, 2, 3, 0.95)
        rop_99 = inventory_manager.calculate_reorder_point(10, 2, 3, 0.99)
        assert rop_99 > rop_95

    def test_longer_lead_time_increases_rop(self, inventory_manager):
        rop_short = inventory_manager.calculate_reorder_point(10, 2, 2, 0.95)
        rop_long = inventory_manager.calculate_reorder_point(10, 2, 7, 0.95)
        assert rop_long > rop_short


class TestEOQ:
    def test_basic_calculation(self, inventory_manager):
        eoq = inventory_manager.calculate_economic_order_quantity(
            annual_demand=1000,
            unit_cost=10,
            order_cost=50,
            holding_cost_pct=0.25
        )
        # EOQ = sqrt(2 * 1000 * 50 / (10 * 0.25)) = sqrt(100000/2.5) = sqrt(40000) = 200
        assert abs(eoq - 200) < 1

    def test_zero_demand_returns_zero(self, inventory_manager):
        eoq = inventory_manager.calculate_economic_order_quantity(0, 10)
        assert eoq == 0

    def test_zero_cost_returns_zero(self, inventory_manager):
        eoq = inventory_manager.calculate_economic_order_quantity(1000, 0)
        assert eoq == 0

    def test_higher_demand_means_larger_order(self, inventory_manager):
        eoq_low = inventory_manager.calculate_economic_order_quantity(500, 10)
        eoq_high = inventory_manager.calculate_economic_order_quantity(2000, 10)
        assert eoq_high > eoq_low


class TestSetParameters:
    def test_set_lead_time(self, inventory_manager):
        inventory_manager.set_parameters(lead_time_days=5)
        assert inventory_manager.default_lead_time_days == 5

    def test_set_service_level(self, inventory_manager):
        inventory_manager.set_parameters(service_level=0.99)
        assert inventory_manager.default_service_level == 0.99

    def test_defaults_unchanged_when_none(self, inventory_manager):
        original_lt = inventory_manager.default_lead_time_days
        original_sl = inventory_manager.default_service_level
        inventory_manager.set_parameters()
        assert inventory_manager.default_lead_time_days == original_lt
        assert inventory_manager.default_service_level == original_sl


class TestZScores:
    def test_service_levels_exist(self):
        assert 0.90 in InventoryManager.Z_SCORES
        assert 0.95 in InventoryManager.Z_SCORES
        assert 0.99 in InventoryManager.Z_SCORES

    def test_z_scores_increase_with_service_level(self):
        z = InventoryManager.Z_SCORES
        assert z[0.90] < z[0.95] < z[0.99]
