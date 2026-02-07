"""
FreshFlow - What-If Scenario Planner
Simulate impact of various factors on demand forecasts.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ScenarioType(Enum):
    WEATHER = "weather"
    PROMOTION = "promotion"
    EVENT = "event"
    HOLIDAY = "holiday"
    PRICE_CHANGE = "price_change"
    COMPETITION = "competition"
    CUSTOM = "custom"


@dataclass
class ScenarioResult:
    """Result of a scenario simulation."""
    scenario_name: str
    scenario_type: ScenarioType
    baseline_revenue: float
    simulated_revenue: float
    revenue_change: float
    revenue_change_pct: float
    baseline_orders: float
    simulated_orders: float
    confidence: str
    daily_breakdown: pd.DataFrame
    insights: List[str]


class ScenarioPlanner:
    """
    Simulates what-if scenarios to help with planning and decision making.
    Impact multipliers are calibrated from industry benchmarks and can be
    refined with historical data when available.
    """

    # Base impact multipliers (industry benchmarks)
    # These serve as defaults and are adjusted based on location data
    BASE_WEATHER_IMPACTS = {
        'rain': {'revenue': 1.08, 'orders': 1.04, 'confidence': 'Medium'},
        'heavy_rain': {'revenue': 1.12, 'orders': 1.06, 'confidence': 'Medium'},
        'snow': {'revenue': 0.88, 'orders': 0.82, 'confidence': 'Medium'},
        'extreme_cold': {'revenue': 1.10, 'orders': 1.06, 'confidence': 'Low'},
        'extreme_heat': {'revenue': 0.94, 'orders': 0.96, 'confidence': 'Low'},
        'sunny': {'revenue': 1.0, 'orders': 1.0, 'confidence': 'High'},
    }

    # Promotion impacts based on elasticity research
    # discount_rate -> (gross_revenue_multiplier, order_multiplier, margin_factor)
    BASE_PROMOTION_IMPACTS = {
        '10_percent_off': {'gross_mult': 1.15, 'orders': 1.18, 'discount': 0.10, 'redemption': 0.70},
        '15_percent_off': {'gross_mult': 1.22, 'orders': 1.28, 'discount': 0.15, 'redemption': 0.75},
        '20_percent_off': {'gross_mult': 1.30, 'orders': 1.40, 'discount': 0.20, 'redemption': 0.80},
        '25_percent_off': {'gross_mult': 1.38, 'orders': 1.52, 'discount': 0.25, 'redemption': 0.82},
        'bogo': {'gross_mult': 1.45, 'orders': 1.60, 'discount': 0.30, 'redemption': 0.60},
        'free_delivery': {'gross_mult': 1.12, 'orders': 1.18, 'discount': 0.05, 'redemption': 0.90},
        'loyalty_double_points': {'gross_mult': 1.18, 'orders': 1.22, 'discount': 0.02, 'redemption': 0.40},
    }

    BASE_EVENT_IMPACTS = {
        'local_festival': {'revenue': 1.35, 'orders': 1.45, 'confidence': 'Low'},
        'sports_event': {'revenue': 1.22, 'orders': 1.28, 'confidence': 'Low'},
        'concert_nearby': {'revenue': 1.30, 'orders': 1.38, 'confidence': 'Low'},
        'competitor_closed': {'revenue': 1.18, 'orders': 1.22, 'confidence': 'Medium'},
        'road_construction': {'revenue': 0.88, 'orders': 0.82, 'confidence': 'Medium'},
        'power_outage': {'revenue': 0.25, 'orders': 0.18, 'confidence': 'High'},
    }

    BASE_HOLIDAY_IMPACTS = {
        'christmas_eve': {'revenue': 0.40, 'orders': 0.35, 'confidence': 'High'},
        'christmas_day': {'revenue': 0.55, 'orders': 0.48, 'confidence': 'High'},
        'new_years_eve': {'revenue': 1.28, 'orders': 1.22, 'confidence': 'High'},
        'new_years_day': {'revenue': 1.42, 'orders': 1.38, 'confidence': 'High'},
        'easter': {'revenue': 1.25, 'orders': 1.22, 'confidence': 'Medium'},
        'midsummer': {'revenue': 1.32, 'orders': 1.28, 'confidence': 'Medium'},
        'constitution_day': {'revenue': 1.10, 'orders': 1.08, 'confidence': 'High'},
    }

    def __init__(self, data_processor, weather_service=None, feature_engineer=None):
        self.data_processor = data_processor
        self.weather_service = weather_service
        self.feature_engineer = feature_engineer

        # Initialize with base impacts (will be calibrated per location)
        self.weather_impacts = self.BASE_WEATHER_IMPACTS.copy()
        self.event_impacts = self.BASE_EVENT_IMPACTS.copy()
        self.holiday_impacts = self.BASE_HOLIDAY_IMPACTS.copy()

    def _calibrate_impacts_for_location(self, place_id: Optional[int]) -> None:
        """
        Calibrate impact multipliers based on location-specific historical data.
        Adjusts confidence based on data availability.
        """
        if place_id is None:
            return

        daily_sales = self.data_processor.get_daily_sales(place_id)
        if len(daily_sales) < 30:
            return

        # Calculate location-specific variability
        daily_sales['date'] = pd.to_datetime(daily_sales['date'])
        avg_revenue = daily_sales['total_revenue'].mean()
        std_revenue = daily_sales['total_revenue'].std()
        cv = std_revenue / avg_revenue if avg_revenue > 0 else 0.3

        # Adjust impacts based on location variability
        # Higher variability locations should have more conservative estimates
        adjustment_factor = max(0.8, min(1.2, 1.0 / (1 + cv)))

        # Apply adjustments to impacts
        for condition in self.weather_impacts:
            impact = self.weather_impacts[condition]
            # Dampen extreme impacts for high-variability locations
            if impact['revenue'] > 1:
                impact['revenue'] = 1 + (impact['revenue'] - 1) * adjustment_factor
            else:
                impact['revenue'] = 1 - (1 - impact['revenue']) * adjustment_factor

    def simulate_weather_scenario(
        self,
        place_id: Optional[int],
        weather_condition: str,
        days: int = 7
    ) -> ScenarioResult:
        """Simulate impact of a weather condition."""
        self._calibrate_impacts_for_location(place_id)
        baseline = self._get_baseline_forecast(place_id, days)

        if weather_condition not in self.weather_impacts:
            weather_condition = 'sunny'

        impact = self.weather_impacts[weather_condition]

        simulated = baseline.copy()
        simulated['simulated_revenue'] = baseline['revenue'] * impact['revenue']
        simulated['simulated_orders'] = baseline['orders'] * impact['orders']

        insights = self._generate_weather_insights(weather_condition, impact)

        return self._create_result(
            f"Weather: {weather_condition.replace('_', ' ').title()}",
            ScenarioType.WEATHER,
            baseline,
            simulated,
            impact['confidence'],
            insights
        )

    def simulate_promotion_scenario(
        self,
        place_id: Optional[int],
        promotion_type: str,
        days: int = 7
    ) -> ScenarioResult:
        """
        Simulate impact of a promotion with proper net revenue calculation.

        Net Revenue = Gross Revenue * (1 - discount * redemption_rate)
        """
        baseline = self._get_baseline_forecast(place_id, days)

        if promotion_type not in self.BASE_PROMOTION_IMPACTS:
            promotion_type = '10_percent_off'

        promo = self.BASE_PROMOTION_IMPACTS[promotion_type]

        simulated = baseline.copy()

        # Calculate gross revenue with promotion uplift
        gross_revenue = baseline['revenue'] * promo['gross_mult']

        # Calculate net revenue after discount
        # Net = Gross * (1 - discount_rate * redemption_rate)
        discount_factor = 1 - (promo['discount'] * promo['redemption'])
        simulated['simulated_revenue'] = gross_revenue * discount_factor
        simulated['simulated_orders'] = baseline['orders'] * promo['orders']

        # Determine confidence based on promotion type
        if promo['discount'] <= 0.15:
            confidence = 'High'
        elif promo['discount'] <= 0.25:
            confidence = 'Medium'
        else:
            confidence = 'Low'

        insights = self._generate_promotion_insights(promotion_type, promo, days)

        return self._create_result(
            f"Promotion: {promotion_type.replace('_', ' ').title()}",
            ScenarioType.PROMOTION,
            baseline,
            simulated,
            confidence,
            insights
        )

    def simulate_event_scenario(
        self,
        place_id: Optional[int],
        event_type: str,
        days: int = 3
    ) -> ScenarioResult:
        """Simulate impact of a local event."""
        self._calibrate_impacts_for_location(place_id)
        baseline = self._get_baseline_forecast(place_id, days)

        if event_type not in self.event_impacts:
            event_type = 'local_festival'

        impact = self.event_impacts[event_type]

        simulated = baseline.copy()
        simulated['simulated_revenue'] = baseline['revenue'] * impact['revenue']
        simulated['simulated_orders'] = baseline['orders'] * impact['orders']

        insights = self._generate_event_insights(event_type, impact)

        return self._create_result(
            f"Event: {event_type.replace('_', ' ').title()}",
            ScenarioType.EVENT,
            baseline,
            simulated,
            impact['confidence'],
            insights
        )

    def simulate_holiday_scenario(
        self,
        place_id: Optional[int],
        holiday: str
    ) -> ScenarioResult:
        """Simulate impact of a holiday."""
        baseline = self._get_baseline_forecast(place_id, days=1)

        if holiday not in self.holiday_impacts:
            holiday = 'constitution_day'

        impact = self.holiday_impacts[holiday]

        simulated = baseline.copy()
        simulated['simulated_revenue'] = baseline['revenue'] * impact['revenue']
        simulated['simulated_orders'] = baseline['orders'] * impact['orders']

        insights = self._generate_holiday_insights(holiday, impact)

        return self._create_result(
            f"Holiday: {holiday.replace('_', ' ').title()}",
            ScenarioType.HOLIDAY,
            baseline,
            simulated,
            impact['confidence'],
            insights
        )

    def simulate_custom_scenario(
        self,
        place_id: Optional[int],
        name: str,
        revenue_multiplier: float,
        order_multiplier: float,
        days: int = 7
    ) -> ScenarioResult:
        """Simulate a custom scenario with user-defined multipliers."""
        baseline = self._get_baseline_forecast(place_id, days)

        # Validate multipliers
        revenue_multiplier = max(0.1, min(3.0, revenue_multiplier))
        order_multiplier = max(0.1, min(3.0, order_multiplier))

        simulated = baseline.copy()
        simulated['simulated_revenue'] = baseline['revenue'] * revenue_multiplier
        simulated['simulated_orders'] = baseline['orders'] * order_multiplier

        # Determine confidence based on how extreme the multipliers are
        max_deviation = max(abs(revenue_multiplier - 1), abs(order_multiplier - 1))
        if max_deviation < 0.2:
            confidence = 'Medium'
        elif max_deviation < 0.5:
            confidence = 'Low'
        else:
            confidence = 'Very Low'

        insights = [
            f"Custom scenario with {(revenue_multiplier-1)*100:+.0f}% revenue impact",
            f"Order volume change: {(order_multiplier-1)*100:+.0f}%",
            f"Confidence: {confidence} - user-defined parameters",
            "Actual impact will vary based on execution and market conditions"
        ]

        return self._create_result(
            name,
            ScenarioType.CUSTOM,
            baseline,
            simulated,
            confidence,
            insights
        )

    def simulate_combined_scenario(
        self,
        place_id: Optional[int],
        scenarios: List[Dict],
        days: int = 7
    ) -> ScenarioResult:
        """Simulate multiple scenarios combined with diminishing returns."""
        self._calibrate_impacts_for_location(place_id)
        baseline = self._get_baseline_forecast(place_id, days)

        combined_revenue_mult = 1.0
        combined_order_mult = 1.0
        scenario_names = []

        for i, scenario in enumerate(scenarios):
            s_type = scenario.get('type')
            s_value = scenario.get('value')

            # Apply diminishing returns for stacked scenarios
            diminish_factor = 0.8 ** i  # Each additional scenario has 80% effect

            if s_type == 'weather' and s_value in self.weather_impacts:
                impact = self.weather_impacts[s_value]
                rev_delta = (impact['revenue'] - 1) * diminish_factor
                ord_delta = (impact['orders'] - 1) * diminish_factor
                combined_revenue_mult += rev_delta
                combined_order_mult += ord_delta
                scenario_names.append(f"Weather: {s_value}")

            elif s_type == 'promotion' and s_value in self.BASE_PROMOTION_IMPACTS:
                promo = self.BASE_PROMOTION_IMPACTS[s_value]
                net_mult = promo['gross_mult'] * (1 - promo['discount'] * promo['redemption'])
                rev_delta = (net_mult - 1) * diminish_factor
                ord_delta = (promo['orders'] - 1) * diminish_factor
                combined_revenue_mult += rev_delta
                combined_order_mult += ord_delta
                scenario_names.append(f"Promo: {s_value}")

            elif s_type == 'event' and s_value in self.event_impacts:
                impact = self.event_impacts[s_value]
                rev_delta = (impact['revenue'] - 1) * diminish_factor
                ord_delta = (impact['orders'] - 1) * diminish_factor
                combined_revenue_mult += rev_delta
                combined_order_mult += ord_delta
                scenario_names.append(f"Event: {s_value}")

        simulated = baseline.copy()
        simulated['simulated_revenue'] = baseline['revenue'] * combined_revenue_mult
        simulated['simulated_orders'] = baseline['orders'] * combined_order_mult

        insights = [
            f"Combined impact of {len(scenarios)} scenarios",
            f"Total revenue multiplier: {combined_revenue_mult:.2f}x",
            f"Total order multiplier: {combined_order_mult:.2f}x",
            "Diminishing returns applied for stacked effects"
        ]

        return self._create_result(
            " + ".join(scenario_names) if scenario_names else "Combined Scenario",
            ScenarioType.CUSTOM,
            baseline,
            simulated,
            'Low',
            insights
        )

    def _get_baseline_forecast(
        self,
        place_id: Optional[int],
        days: int
    ) -> pd.DataFrame:
        """Get baseline forecast for scenario comparison."""
        daily_sales = self.data_processor.get_daily_sales(place_id)
        dow_pattern = self.data_processor.get_day_of_week_pattern(place_id)

        if len(daily_sales) == 0:
            avg_revenue = 10000  # Default fallback
            avg_orders = 50
        else:
            avg_revenue = daily_sales['total_revenue'].mean()
            avg_orders = daily_sales['order_count'].mean()

        # Create forecast dates
        start_date = datetime.now() + timedelta(days=1)
        dates = pd.date_range(start=start_date, periods=days, freq='D')

        # Build baseline
        baseline = pd.DataFrame({'date': dates})
        baseline['day_of_week'] = baseline['date'].dt.dayofweek
        baseline['day_name'] = baseline['date'].dt.day_name()

        # Apply day-of-week pattern if available
        if len(dow_pattern) > 0 and 'relative_demand' in dow_pattern.columns:
            dow_dict = dow_pattern.set_index('day_of_week')['relative_demand'].to_dict()
        else:
            dow_dict = {}

        baseline['revenue'] = baseline['day_of_week'].apply(
            lambda x: avg_revenue * (1 + dow_dict.get(x, 0))
        )
        baseline['orders'] = baseline['day_of_week'].apply(
            lambda x: avg_orders * (1 + dow_dict.get(x, 0))
        )

        return baseline

    def _create_result(
        self,
        name: str,
        scenario_type: ScenarioType,
        baseline: pd.DataFrame,
        simulated: pd.DataFrame,
        confidence: str,
        insights: List[str]
    ) -> ScenarioResult:
        """Create a ScenarioResult from baseline and simulated data."""
        baseline_revenue = baseline['revenue'].sum()
        simulated_revenue = simulated['simulated_revenue'].sum()
        baseline_orders = baseline['orders'].sum()
        simulated_orders = simulated['simulated_orders'].sum()

        # Protect against division by zero
        if baseline_revenue == 0:
            revenue_change_pct = 0
        else:
            revenue_change_pct = ((simulated_revenue - baseline_revenue) / baseline_revenue) * 100

        # Create daily breakdown
        breakdown = baseline[['date', 'day_name', 'revenue', 'orders']].copy()
        breakdown['simulated_revenue'] = simulated['simulated_revenue']
        breakdown['simulated_orders'] = simulated['simulated_orders']
        breakdown['revenue_change'] = breakdown['simulated_revenue'] - breakdown['revenue']

        # Safe percentage calculation
        breakdown['revenue_change_pct'] = breakdown.apply(
            lambda row: ((row['revenue_change'] / row['revenue']) * 100)
            if row['revenue'] > 0 else 0,
            axis=1
        )

        return ScenarioResult(
            scenario_name=name,
            scenario_type=scenario_type,
            baseline_revenue=round(baseline_revenue, 2),
            simulated_revenue=round(simulated_revenue, 2),
            revenue_change=round(simulated_revenue - baseline_revenue, 2),
            revenue_change_pct=round(revenue_change_pct, 1),
            baseline_orders=round(baseline_orders, 0),
            simulated_orders=round(simulated_orders, 0),
            confidence=confidence,
            daily_breakdown=breakdown,
            insights=insights
        )

    def _generate_weather_insights(self, condition: str, impact: Dict) -> List[str]:
        """Generate insights for weather scenario."""
        insights = []
        condition_display = condition.replace('_', ' ').title()

        if impact['revenue'] > 1:
            pct = (impact['revenue'] - 1) * 100
            insights.append(f"{condition_display} typically increases indoor dining by ~{pct:.0f}%")
            insights.append("Consider increasing hot food/drink inventory")
        else:
            pct = (1 - impact['revenue']) * 100
            insights.append(f"{condition_display} may reduce foot traffic by ~{pct:.0f}%")
            insights.append("Consider delivery promotions to offset decline")

        insights.append(f"Confidence level: {impact['confidence']} (based on historical patterns)")
        insights.append("Actual impact varies by location and severity")

        return insights

    def _generate_promotion_insights(self, promo: str, impact: Dict, days: int) -> List[str]:
        """Generate insights for promotion scenario."""
        order_increase = (impact['orders'] - 1) * 100
        discount_cost = impact['discount'] * 100
        redemption = impact['redemption'] * 100

        insights = [
            f"Expected order increase: +{order_increase:.0f}%",
            f"Discount: {discount_cost:.0f}% (estimated {redemption:.0f}% redemption rate)",
            f"Net revenue accounts for discount cost",
            f"Recommended duration: {days} days to build awareness",
        ]

        if impact['discount'] >= 0.20:
            insights.append("High discounts may attract deal-seekers with lower retention")

        return insights

    def _generate_event_insights(self, event: str, impact: Dict) -> List[str]:
        """Generate insights for event scenario."""
        order_increase = (impact['orders'] - 1) * 100
        event_display = event.replace('_', ' ')

        insights = [
            f"Local {event_display} events typically affect 1-3 days",
            f"Plan staffing for +{order_increase:.0f}% order volume",
            "Consider extended hours during event",
            f"Confidence: {impact['confidence']} - impacts vary by proximity and event size"
        ]
        return insights

    def _generate_holiday_insights(self, holiday: str, impact: Dict) -> List[str]:
        """Generate insights for holiday scenario."""
        holiday_display = holiday.replace('_', ' ').title()

        if impact['revenue'] < 1:
            pct_reduction = (1 - impact['revenue']) * 100
            insights = [
                f"{holiday_display} typically sees ~{pct_reduction:.0f}% reduced operations",
                "Consider reduced staffing or shorter hours",
                "Some businesses close entirely",
                "Plan inventory for lower demand"
            ]
        else:
            pct_increase = (impact['revenue'] - 1) * 100
            insights = [
                f"{holiday_display} typically brings +{pct_increase:.0f}% demand",
                "Ensure adequate staffing for peak hours",
                "Pre-order inventory if needed",
                "Consider special holiday offerings"
            ]
        return insights

    def get_available_scenarios(self) -> Dict:
        """Get all available scenario options."""
        return {
            'weather': list(self.BASE_WEATHER_IMPACTS.keys()),
            'promotion': list(self.BASE_PROMOTION_IMPACTS.keys()),
            'event': list(self.BASE_EVENT_IMPACTS.keys()),
            'holiday': list(self.BASE_HOLIDAY_IMPACTS.keys())
        }
