"""
FreshFlow - Inventory Reorder System
Automatic reorder points, safety stock, and stock-out risk management.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class StockStatus(Enum):
    CRITICAL = "critical"      # Below safety stock
    LOW = "low"                # Below reorder point
    ADEQUATE = "adequate"      # Normal levels
    OVERSTOCKED = "overstocked"  # Above max level


class AlertPriority(Enum):
    URGENT = "urgent"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class InventoryItem:
    """Inventory item with stock levels and recommendations."""
    item_id: int
    item_name: str
    current_stock: float
    unit: str
    safety_stock: float
    reorder_point: float
    reorder_quantity: float
    max_stock: float
    avg_daily_usage: float
    days_of_stock: float
    status: StockStatus
    last_order_date: Optional[datetime]
    recommended_action: str
    avg_price: float  # Added for cost calculations


@dataclass
class ReorderSuggestion:
    """Suggested reorder for an item."""
    item_name: str
    current_stock: float
    suggested_quantity: float
    urgency: AlertPriority
    estimated_stockout_date: datetime
    days_until_stockout: int
    estimated_cost: float
    unit_cost: float  # Added actual unit cost
    reason: str


class InventoryManager:
    """
    Manages inventory levels, calculates reorder points,
    and generates stock alerts.
    """

    # Z-scores for different service levels
    Z_SCORES = {
        0.90: 1.28,
        0.95: 1.65,
        0.99: 2.33
    }

    def __init__(self, data_processor, forecast_service=None):
        self.data_processor = data_processor
        self.forecast_service = forecast_service

        # Default parameters (can be customized)
        self.default_lead_time_days = 3
        self.default_service_level = 0.95

    def set_parameters(self, lead_time_days: int = None, service_level: float = None):
        """Update default parameters."""
        if lead_time_days is not None:
            self.default_lead_time_days = lead_time_days
        if service_level is not None:
            self.default_service_level = service_level

    def calculate_reorder_point(
        self,
        avg_daily_demand: float,
        demand_std: float,
        lead_time_days: int = None,
        service_level: float = None
    ) -> float:
        """
        Calculate reorder point using safety stock formula.

        ROP = (Average Daily Demand x Lead Time) + Safety Stock
        Safety Stock = Z x sigma_daily x sqrt(Lead Time)
        """
        lead_time = lead_time_days if lead_time_days is not None else self.default_lead_time_days
        service = service_level if service_level is not None else self.default_service_level

        # Get Z-score, default to 95% if not found
        z = self.Z_SCORES.get(service, 1.65)

        # Safety stock calculation
        safety_stock = z * demand_std * np.sqrt(lead_time)

        # Reorder point
        rop = (avg_daily_demand * lead_time) + safety_stock

        return max(0, rop)

    def calculate_economic_order_quantity(
        self,
        annual_demand: float,
        unit_cost: float,
        order_cost: float = 50,  # Fixed cost per order
        holding_cost_pct: float = 0.25  # 25% of item value per year
    ) -> float:
        """
        Calculate Economic Order Quantity (EOQ).

        EOQ = sqrt(2 x D x S / H)
        D = Annual demand
        S = Order cost
        H = Holding cost per unit per year
        """
        if annual_demand <= 0 or unit_cost <= 0:
            return 0

        holding_cost = unit_cost * holding_cost_pct

        if holding_cost <= 0:
            return 0

        eoq = np.sqrt((2 * annual_demand * order_cost) / holding_cost)

        return max(1, round(eoq))

    def analyze_item_demand(
        self,
        place_id: Optional[int] = None,
        days: int = 90,
        lead_time_days: int = None,
        service_level: float = None
    ) -> pd.DataFrame:
        """
        Analyze demand patterns for inventory items.
        Returns DataFrame with demand statistics and reorder parameters.
        """
        # Get order items data
        orders = self.data_processor.load_orders()
        order_items = self.data_processor.load_order_items()

        if len(orders) == 0 or len(order_items) == 0:
            return pd.DataFrame()

        if place_id is not None:
            orders = orders[orders['place_id'] == place_id]

        if len(orders) == 0:
            return pd.DataFrame()

        # Filter to recent orders
        orders['date'] = pd.to_datetime(orders['created'], unit='s')
        max_date = orders['date'].max()
        cutoff = max_date - timedelta(days=days)
        recent_orders = orders[orders['date'] >= cutoff]

        if len(recent_orders) == 0:
            return pd.DataFrame()

        # Merge with items
        merged = order_items.merge(
            recent_orders[['id', 'date']],
            left_on='order_id',
            right_on='id',
            how='inner'
        )

        if len(merged) == 0:
            return pd.DataFrame()

        # Calculate daily demand per item
        daily_demand = merged.groupby(['title', merged['date'].dt.date]).agg({
            'quantity': 'sum',
            'price': 'mean'
        }).reset_index()
        daily_demand.columns = ['item_name', 'date', 'daily_qty', 'price']

        # Aggregate statistics per item
        item_stats = daily_demand.groupby('item_name').agg({
            'daily_qty': ['sum', 'mean', 'std', 'count'],
            'price': 'mean'
        }).reset_index()
        item_stats.columns = ['item_name', 'total_qty', 'avg_daily_demand',
                             'daily_demand_std', 'days_with_sales', 'avg_price']

        # Calculate actual number of days in period
        n_days = max(1, (max_date - cutoff).days)

        # Adjust avg_daily_demand to account for days without sales
        item_stats['avg_daily_demand'] = item_stats['total_qty'] / n_days

        # Fill NaN std with 30% of mean (coefficient of variation)
        item_stats['daily_demand_std'] = item_stats['daily_demand_std'].fillna(
            item_stats['avg_daily_demand'] * 0.3
        )

        # Ensure std is at least 10% of mean for safety
        item_stats['daily_demand_std'] = item_stats['daily_demand_std'].clip(
            lower=item_stats['avg_daily_demand'] * 0.1
        )

        # Calculate reorder points with actual parameters
        lead_time = lead_time_days if lead_time_days is not None else self.default_lead_time_days
        service = service_level if service_level is not None else self.default_service_level

        item_stats['reorder_point'] = item_stats.apply(
            lambda row: self.calculate_reorder_point(
                row['avg_daily_demand'],
                row['daily_demand_std'],
                lead_time,
                service
            ),
            axis=1
        )

        # Calculate EOQ with actual prices
        item_stats['eoq'] = item_stats.apply(
            lambda row: self.calculate_economic_order_quantity(
                row['avg_daily_demand'] * 365,
                row['avg_price']
            ),
            axis=1
        )

        # Safety stock based on service level
        z = self.Z_SCORES.get(service, 1.65)
        item_stats['safety_stock'] = z * item_stats['daily_demand_std'] * np.sqrt(lead_time)

        return item_stats

    def estimate_current_stock(
        self,
        item_name: str,
        avg_daily_demand: float,
        reorder_point: float,
        days_since_last_order: int = 7
    ) -> float:
        """
        Estimate current stock level based on demand patterns.

        In production, this would query actual inventory database.
        For demo, we estimate based on typical restock patterns.
        """
        # Assume typical restock brings inventory to 2x reorder point
        typical_restock_level = reorder_point * 2

        # Estimate consumption since last restock
        consumed = avg_daily_demand * days_since_last_order

        # Estimated current stock
        estimated_stock = max(0, typical_restock_level - consumed)

        return estimated_stock

    def get_inventory_status(
        self,
        place_id: Optional[int] = None,
        current_stock: Optional[Dict[str, float]] = None,
        lead_time_days: int = None,
        service_level: float = None
    ) -> List[InventoryItem]:
        """
        Get inventory status for all items.

        Args:
            place_id: Filter by location
            current_stock: Dict of actual stock levels (item_name -> quantity)
            lead_time_days: Lead time for reorder calculations
            service_level: Target service level (0.90, 0.95, 0.99)
        """
        item_analysis = self.analyze_item_demand(
            place_id,
            lead_time_days=lead_time_days,
            service_level=service_level
        )

        if len(item_analysis) == 0:
            return []

        inventory_items = []

        # Use a deterministic seed based on item count for consistent demo data
        np.random.seed(len(item_analysis))

        for idx, row in item_analysis.iterrows():
            # Get actual stock if provided, otherwise estimate
            if current_stock and row['item_name'] in current_stock:
                stock = current_stock[row['item_name']]
            else:
                # Estimate stock based on demand pattern
                # Use deterministic variation based on item index
                days_since_restock = (idx % 14) + 1  # 1-14 days
                stock = self.estimate_current_stock(
                    row['item_name'],
                    row['avg_daily_demand'],
                    row['reorder_point'],
                    days_since_restock
                )

            # Calculate days of stock remaining
            if row['avg_daily_demand'] > 0:
                days_of_stock = stock / row['avg_daily_demand']
            else:
                days_of_stock = 999

            # Determine status based on thresholds
            if stock <= row['safety_stock']:
                status = StockStatus.CRITICAL
                action = "ORDER IMMEDIATELY - Below safety stock"
            elif stock <= row['reorder_point']:
                status = StockStatus.LOW
                action = "Place order - Below reorder point"
            elif stock > row['avg_daily_demand'] * 14:
                status = StockStatus.OVERSTOCKED
                action = "Consider reducing next order"
            else:
                status = StockStatus.ADEQUATE
                action = "Stock levels adequate"

            inventory_items.append(InventoryItem(
                item_id=hash(row['item_name']) % 10000,
                item_name=row['item_name'],
                current_stock=round(stock, 1),
                unit="units",
                safety_stock=round(row['safety_stock'], 1),
                reorder_point=round(row['reorder_point'], 1),
                reorder_quantity=round(row['eoq'], 0),
                max_stock=round(row['avg_daily_demand'] * 14, 1),
                avg_daily_usage=round(row['avg_daily_demand'], 2),
                days_of_stock=round(days_of_stock, 1),
                status=status,
                last_order_date=None,
                recommended_action=action,
                avg_price=round(row['avg_price'], 2)
            ))

        # Sort by status severity
        status_order = {
            StockStatus.CRITICAL: 0,
            StockStatus.LOW: 1,
            StockStatus.ADEQUATE: 2,
            StockStatus.OVERSTOCKED: 3
        }
        inventory_items.sort(key=lambda x: status_order[x.status])

        return inventory_items

    def generate_reorder_suggestions(
        self,
        place_id: Optional[int] = None,
        forecast_days: int = 14,
        lead_time_days: int = None,
        service_level: float = None
    ) -> List[ReorderSuggestion]:
        """
        Generate smart reorder suggestions based on forecasted demand.
        """
        inventory = self.get_inventory_status(
            place_id,
            lead_time_days=lead_time_days,
            service_level=service_level
        )
        suggestions = []

        for item in inventory:
            if item.status in [StockStatus.CRITICAL, StockStatus.LOW]:
                # Calculate days until stockout
                days_until_stockout = max(0, item.days_of_stock)

                # Determine urgency based on days until stockout
                if days_until_stockout <= 1:
                    urgency = AlertPriority.URGENT
                elif days_until_stockout <= 3:
                    urgency = AlertPriority.HIGH
                elif days_until_stockout <= 7:
                    urgency = AlertPriority.MEDIUM
                else:
                    urgency = AlertPriority.LOW

                # Calculate suggested quantity
                # Order enough for forecast period + safety stock - current stock
                needed = (item.avg_daily_usage * forecast_days) + item.safety_stock - item.current_stock
                suggested_qty = max(item.reorder_quantity, needed)

                # Estimate stockout date
                stockout_date = datetime.now() + timedelta(days=days_until_stockout)

                # Calculate cost using actual item price
                unit_cost = item.avg_price
                estimated_cost = suggested_qty * unit_cost

                suggestions.append(ReorderSuggestion(
                    item_name=item.item_name,
                    current_stock=item.current_stock,
                    suggested_quantity=round(suggested_qty, 0),
                    urgency=urgency,
                    estimated_stockout_date=stockout_date,
                    days_until_stockout=int(days_until_stockout),
                    estimated_cost=round(estimated_cost, 2),
                    unit_cost=unit_cost,
                    reason=item.recommended_action
                ))

        # Sort by urgency
        urgency_order = {
            AlertPriority.URGENT: 0,
            AlertPriority.HIGH: 1,
            AlertPriority.MEDIUM: 2,
            AlertPriority.LOW: 3
        }
        suggestions.sort(key=lambda x: urgency_order[x.urgency])

        return suggestions

    def get_inventory_summary(
        self,
        place_id: Optional[int] = None,
        lead_time_days: int = None,
        service_level: float = None
    ) -> Dict:
        """
        Get summary of inventory status.
        """
        inventory = self.get_inventory_status(
            place_id,
            lead_time_days=lead_time_days,
            service_level=service_level
        )

        if not inventory:
            return {
                'total_items': 0,
                'critical': 0,
                'low': 0,
                'adequate': 0,
                'overstocked': 0,
                'reorder_needed': 0,
                'estimated_reorder_cost': 0
            }

        critical = [i for i in inventory if i.status == StockStatus.CRITICAL]
        low = [i for i in inventory if i.status == StockStatus.LOW]
        adequate = [i for i in inventory if i.status == StockStatus.ADEQUATE]
        overstocked = [i for i in inventory if i.status == StockStatus.OVERSTOCKED]

        # Calculate reorder costs using actual prices
        reorder_items = critical + low
        estimated_cost = sum(i.reorder_quantity * i.avg_price for i in reorder_items)

        return {
            'total_items': len(inventory),
            'critical': len(critical),
            'low': len(low),
            'adequate': len(adequate),
            'overstocked': len(overstocked),
            'reorder_needed': len(reorder_items),
            'estimated_reorder_cost': round(estimated_cost, 2),
            'critical_items': [i.item_name for i in critical[:5]],
            'low_items': [i.item_name for i in low[:5]]
        }

    def optimize_order(
        self,
        suggestions: List[ReorderSuggestion],
        budget: float = None,
        min_order_value: float = 500
    ) -> Dict:
        """
        Optimize order based on budget constraints and minimum order values.
        """
        if not suggestions:
            return {'items': [], 'total_cost': 0, 'savings': 0}

        # Sort by urgency (most urgent first)
        sorted_suggestions = sorted(
            suggestions,
            key=lambda x: (
                {'urgent': 0, 'high': 1, 'medium': 2, 'low': 3}[x.urgency.value],
                -x.estimated_cost  # Higher cost items first within same urgency
            )
        )

        selected_items = []
        total_cost = 0

        for suggestion in sorted_suggestions:
            item_cost = suggestion.estimated_cost

            if budget is None or total_cost + item_cost <= budget:
                selected_items.append({
                    'item': suggestion.item_name,
                    'quantity': suggestion.suggested_quantity,
                    'cost': round(item_cost, 2),
                    'unit_cost': suggestion.unit_cost,
                    'urgency': suggestion.urgency.value
                })
                total_cost += item_cost

        # Check if we meet minimum order value
        if total_cost < min_order_value and selected_items:
            shortfall = min_order_value - total_cost

            return {
                'items': selected_items,
                'total_cost': round(total_cost, 2),
                'meets_minimum': False,
                'minimum_shortfall': round(shortfall, 2),
                'suggestion': f"Add {shortfall:.0f} DKK more to meet minimum order value"
            }

        return {
            'items': selected_items,
            'total_cost': round(total_cost, 2),
            'meets_minimum': True,
            'item_count': len(selected_items)
        }
