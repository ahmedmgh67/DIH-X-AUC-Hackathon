"""
FreshFlow - Smart Promotions Engine
AI-powered promotion suggestions, bundle recommendations, and ROI analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class PromotionType(Enum):
    DISCOUNT = "discount"
    BUNDLE = "bundle"
    HAPPY_HOUR = "happy_hour"
    LOYALTY = "loyalty"
    FLASH_SALE = "flash_sale"


@dataclass
class PromotionSuggestion:
    """A suggested promotion."""
    type: PromotionType
    title: str
    description: str
    target_items: List[str]
    discount_pct: float
    expected_lift: float
    expected_roi: float
    best_timing: str
    confidence: str
    reasoning: str


class PromotionsEngine:
    """
    Smart promotions engine that analyzes sales patterns
    and suggests optimal promotions.
    """

    def __init__(self, data_processor):
        self.data_processor = data_processor

    def analyze_slow_periods(
        self,
        place_id: Optional[int] = None,
        threshold_pct: float = 0.7
    ) -> Dict:
        """
        Identify slow sales periods that could benefit from promotions.

        Args:
            place_id: Location to analyze
            threshold_pct: Periods below this % of average are "slow"

        Returns:
            Dict with slow periods analysis including actual deviation percentages
        """
        daily_sales = self.data_processor.get_daily_sales(place_id)
        hourly = self.data_processor.get_hourly_pattern(place_id)
        dow = self.data_processor.get_day_of_week_pattern(place_id)

        if len(daily_sales) == 0:
            return {
                'slow_days': [],
                'slow_hours': [],
                'slow_days_of_month': [],
                'avg_daily_revenue': 0,
                'potential_uplift': 0,
                'slow_day_deviations': {}
            }

        avg_daily = daily_sales['total_revenue'].mean()

        # Find slow days of week with actual deviation percentages
        dow_avg = dow['avg_revenue'].mean()
        slow_days_data = dow[dow['avg_revenue'] < dow_avg * threshold_pct].copy()
        slow_days_data['deviation_pct'] = ((slow_days_data['avg_revenue'] - dow_avg) / dow_avg * 100).round(1)

        # Create deviation lookup
        slow_day_deviations = {}
        for _, row in slow_days_data.iterrows():
            slow_day_deviations[row['day_name']] = row['deviation_pct']

        # Find slow hours
        if len(hourly) > 0:
            avg_hourly = hourly['avg_revenue'].mean()
            slow_hours = hourly[hourly['avg_revenue'] < avg_hourly * threshold_pct]
        else:
            slow_hours = pd.DataFrame()

        # Find slow date patterns (e.g., beginning of month)
        daily_sales = daily_sales.copy()
        daily_sales['day_of_month'] = pd.to_datetime(daily_sales['date']).dt.day
        dom_pattern = daily_sales.groupby('day_of_month')['total_revenue'].mean()
        slow_dom = dom_pattern[dom_pattern < dom_pattern.mean() * threshold_pct]

        # Calculate potential uplift based on variance in data
        std_daily = daily_sales['total_revenue'].std()
        cv = std_daily / avg_daily if avg_daily > 0 else 0.2
        # Potential uplift is estimated based on coefficient of variation
        potential_uplift_pct = min(0.25, max(0.10, cv * 0.5))  # 10-25% range

        return {
            'slow_days': slow_days_data['day_name'].tolist() if len(slow_days_data) > 0 else [],
            'slow_hours': slow_hours['hour'].tolist() if len(slow_hours) > 0 else [],
            'slow_days_of_month': slow_dom.index.tolist() if len(slow_dom) > 0 else [],
            'avg_daily_revenue': round(avg_daily, 2),
            'potential_uplift': round(avg_daily * potential_uplift_pct, 2),
            'potential_uplift_pct': round(potential_uplift_pct * 100, 1),
            'slow_day_deviations': slow_day_deviations
        }

    def analyze_item_performance(
        self,
        place_id: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Analyze item performance using BCG matrix style classification.

        Note: Uses price as a proxy for margin since actual cost data is not available.
        Higher-priced items typically have higher absolute margins.

        Classifications:
        - Star: High volume, high price (prioritize)
        - Cash Cow: High volume, low price (maintain)
        - Question Mark: Low volume, high price (promote to increase volume)
        - Dog: Low volume, low price (review or discontinue)
        """
        top_items = self.data_processor.get_top_items(place_id, top_n=50)

        if len(top_items) == 0:
            return pd.DataFrame()

        # Calculate metrics
        total_qty = top_items['total_quantity'].sum()
        if total_qty == 0:
            return pd.DataFrame()

        avg_qty = top_items['total_quantity'].mean()
        median_price = top_items['avg_price'].median()  # Use median to be more robust

        # Classify items using median as threshold (more robust than mean)
        def classify_item(row):
            high_volume = row['total_quantity'] > avg_qty
            # Use median price as threshold for "high price" classification
            high_price = row['avg_price'] > median_price

            if high_volume and high_price:
                return 'Star'
            elif high_volume and not high_price:
                return 'Cash Cow'
            elif not high_volume and high_price:
                return 'Question Mark'
            else:
                return 'Dog'

        top_items = top_items.copy()
        top_items['classification'] = top_items.apply(classify_item, axis=1)
        top_items['market_share'] = (top_items['total_quantity'] / total_qty).round(4)
        top_items['revenue'] = (top_items['total_quantity'] * top_items['avg_price']).round(2)

        return top_items

    def find_bundle_opportunities(
        self,
        place_id: Optional[int] = None,
        min_support: float = 0.01
    ) -> List[Dict]:
        """
        Find items frequently purchased together for bundle suggestions.
        Uses association analysis on order items.
        """
        orders = self.data_processor.load_orders()
        order_items = self.data_processor.load_order_items()

        if place_id is not None:
            orders = orders[orders['place_id'] == place_id]

        # Merge to get items per order
        merged = order_items.merge(
            orders[['id']],
            left_on='order_id',
            right_on='id',
            how='inner'
        )

        # Group items by order
        order_baskets = merged.groupby('order_id')['title'].apply(list)

        # Count item co-occurrences
        from collections import defaultdict
        pair_counts = defaultdict(int)
        item_counts = defaultdict(int)

        for basket in order_baskets:
            unique_items = list(set(basket))
            for item in unique_items:
                item_counts[item] += 1

            for i, item1 in enumerate(unique_items):
                for item2 in unique_items[i+1:]:
                    pair = tuple(sorted([item1, item2]))
                    pair_counts[pair] += 1

        # Calculate support and confidence
        n_orders = len(order_baskets)
        bundles = []

        for pair, count in pair_counts.items():
            support = count / n_orders
            if support >= min_support:
                item1, item2 = pair
                confidence = count / item_counts[item1]
                lift = support / ((item_counts[item1] / n_orders) * (item_counts[item2] / n_orders))

                if lift > 1.2:  # Items bought together more than expected
                    bundles.append({
                        'items': list(pair),
                        'support': support,
                        'confidence': confidence,
                        'lift': lift,
                        'co_occurrence': count
                    })

        # Sort by lift
        bundles.sort(key=lambda x: x['lift'], reverse=True)
        return bundles[:10]  # Top 10 bundles

    def generate_promotion_suggestions(
        self,
        place_id: Optional[int] = None
    ) -> List[PromotionSuggestion]:
        """
        Generate smart promotion suggestions based on data analysis.
        """
        suggestions = []

        # Analyze data
        slow_periods = self.analyze_slow_periods(place_id)
        item_performance = self.analyze_item_performance(place_id)
        bundles = self.find_bundle_opportunities(place_id)

        # 1. Slow day promotions
        if slow_periods['slow_days']:
            slowest_day = slow_periods['slow_days'][0]
            # Get actual deviation percentage from data
            deviation = slow_periods.get('slow_day_deviations', {}).get(slowest_day, -20)
            deviation_abs = abs(deviation)

            # Calculate expected lift based on discount elasticity (typically 1.5-2x)
            discount_pct = 15
            expected_lift = min(40, max(15, discount_pct * 1.7))  # 1.7x elasticity
            expected_roi = (expected_lift - discount_pct) / discount_pct if discount_pct > 0 else 0

            suggestions.append(PromotionSuggestion(
                type=PromotionType.DISCOUNT,
                title=f"{slowest_day} Special",
                description=f"15% off all orders on {slowest_day}s to boost traffic",
                target_items=["All menu items"],
                discount_pct=discount_pct,
                expected_lift=round(expected_lift, 0),
                expected_roi=round(expected_roi + 1, 1),  # ROI as multiplier
                best_timing=f"Every {slowest_day}",
                confidence="High",
                reasoning=f"{slowest_day} shows {deviation_abs:.0f}% lower revenue than average. "
                         f"A {discount_pct}% discount typically drives {expected_lift:.0f}% more traffic."
            ))

        # 2. Happy hour for slow hours
        if slow_periods['slow_hours']:
            slow_hour = slow_periods['slow_hours'][0]
            suggestions.append(PromotionSuggestion(
                type=PromotionType.HAPPY_HOUR,
                title=f"Happy Hour ({slow_hour}:00-{slow_hour+2}:00)",
                description="20% off selected items during slow hours",
                target_items=["Drinks", "Appetizers"],
                discount_pct=20,
                expected_lift=35,
                expected_roi=1.8,
                best_timing=f"Daily {slow_hour}:00-{slow_hour+2}:00",
                confidence="High",
                reasoning=f"Hour {slow_hour}:00 has significantly lower traffic. "
                         f"Happy hour promotions typically increase visits by 35%."
            ))

        # 3. Bundle deals
        if bundles:
            top_bundle = bundles[0]
            items = top_bundle['items']
            suggestions.append(PromotionSuggestion(
                type=PromotionType.BUNDLE,
                title=f"Perfect Pair Deal",
                description=f"Get {items[0][:20]} + {items[1][:20]} for 10% off",
                target_items=items,
                discount_pct=10,
                expected_lift=20,
                expected_roi=2.5,
                best_timing="All week",
                confidence="High",
                reasoning=f"These items are bought together {top_bundle['lift']:.1f}x "
                         f"more than expected. Bundling increases average order value."
            ))

        # 4. Question mark items (high margin, low volume)
        if len(item_performance) > 0:
            question_marks = item_performance[
                item_performance['classification'] == 'Question Mark'
            ].head(3)

            if len(question_marks) > 0:
                items = question_marks['item_title'].tolist()
                suggestions.append(PromotionSuggestion(
                    type=PromotionType.FLASH_SALE,
                    title="Hidden Gems Spotlight",
                    description="Featured items at 15% off - try something new!",
                    target_items=items[:3],
                    discount_pct=15,
                    expected_lift=40,
                    expected_roi=2.2,
                    best_timing="Weekend lunch",
                    confidence="Medium",
                    reasoning="These high-margin items have low visibility. "
                             "Promotion can increase awareness and establish regular buyers."
                ))

        # 5. Dog items (consider discontinuing or heavy discount)
        if len(item_performance) > 0:
            dogs = item_performance[
                item_performance['classification'] == 'Dog'
            ].head(3)

            if len(dogs) > 0:
                items = dogs['item_title'].tolist()
                suggestions.append(PromotionSuggestion(
                    type=PromotionType.FLASH_SALE,
                    title="Clearance Special",
                    description="25% off selected items - limited time!",
                    target_items=items[:3],
                    discount_pct=25,
                    expected_lift=50,
                    expected_roi=1.2,
                    best_timing="End of week",
                    confidence="Medium",
                    reasoning="These items have low sales and margins. "
                             "Consider deep discount to clear inventory or discontinue."
                ))

        # 6. Loyalty program suggestion
        suggestions.append(PromotionSuggestion(
            type=PromotionType.LOYALTY,
            title="Double Points Week",
            description="Earn 2x loyalty points on all purchases",
            target_items=["All items"],
            discount_pct=0,
            expected_lift=30,
            expected_roi=3.5,
            best_timing="Slow week of the month",
            confidence="High",
            reasoning="Loyalty programs increase repeat visits by 30% "
                     "and customer lifetime value by 25%."
        ))

        return suggestions

    def calculate_promotion_roi(
        self,
        promotion: PromotionSuggestion,
        avg_daily_revenue: float,
        days_active: int = 7
    ) -> Dict:
        """
        Calculate expected ROI for a promotion.
        """
        # Base calculations
        discount_cost = avg_daily_revenue * (promotion.discount_pct / 100) * days_active
        expected_revenue_lift = avg_daily_revenue * (promotion.expected_lift / 100) * days_active
        net_gain = expected_revenue_lift - discount_cost
        roi = net_gain / discount_cost if discount_cost > 0 else float('inf')

        return {
            'promotion_title': promotion.title,
            'discount_cost': discount_cost,
            'expected_lift_revenue': expected_revenue_lift,
            'net_gain': net_gain,
            'roi': roi,
            'break_even_lift': promotion.discount_pct,  # Need this % lift to break even
            'recommendation': 'Recommended' if roi > 1.5 else 'Consider' if roi > 1 else 'Not Recommended'
        }
