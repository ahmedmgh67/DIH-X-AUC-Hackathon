"""
FreshFlow - Anomaly Detection Service
Detect unusual sales patterns, automatic alerts, and root cause analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from scipy import stats


class AnomalyType(Enum):
    SPIKE = "spike"
    DROP = "drop"
    TREND_CHANGE = "trend_change"
    UNUSUAL_PATTERN = "unusual_pattern"


class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Anomaly:
    """Detected anomaly."""
    date: datetime
    type: AnomalyType
    severity: AlertSeverity
    metric: str
    actual_value: float
    expected_value: float
    deviation_pct: float
    z_score: float
    possible_causes: List[str]
    recommended_actions: List[str]


class AnomalyDetector:
    """
    Detects anomalies in sales data using multiple methods:
    - Z-score based detection
    - IQR (Interquartile Range) method
    - Rolling window comparison
    - Day-of-week deviation
    """

    def __init__(self, data_processor, weather_service=None):
        self.data_processor = data_processor
        self.weather_service = weather_service

    def detect_anomalies(
        self,
        place_id: Optional[int] = None,
        lookback_days: int = 90,
        z_threshold: float = 2.5
    ) -> List[Anomaly]:
        """
        Detect anomalies in recent sales data.

        Args:
            place_id: Location to analyze
            lookback_days: Days of history to analyze
            z_threshold: Z-score threshold for anomaly detection

        Returns:
            List of detected anomalies
        """
        daily_sales = self.data_processor.get_daily_sales(place_id)

        if len(daily_sales) < 30:
            return []

        # Get recent data
        daily_sales['date'] = pd.to_datetime(daily_sales['date'])
        recent = daily_sales.tail(lookback_days).copy()

        anomalies = []

        # 1. Z-score based detection on revenue
        anomalies.extend(self._detect_zscore_anomalies(recent, 'total_revenue', z_threshold))

        # 2. Z-score on order count
        anomalies.extend(self._detect_zscore_anomalies(recent, 'order_count', z_threshold))

        # 3. Day-of-week deviation detection
        anomalies.extend(self._detect_dow_anomalies(recent, place_id))

        # 4. Trend change detection
        anomalies.extend(self._detect_trend_changes(recent))

        # Sort by date (most recent first) and severity
        severity_order = {AlertSeverity.CRITICAL: 0, AlertSeverity.HIGH: 1,
                        AlertSeverity.MEDIUM: 2, AlertSeverity.LOW: 3}
        anomalies.sort(key=lambda x: (x.date, severity_order[x.severity]), reverse=True)

        return anomalies

    def _detect_zscore_anomalies(
        self,
        data: pd.DataFrame,
        column: str,
        z_threshold: float
    ) -> List[Anomaly]:
        """Detect anomalies using z-score method."""
        anomalies = []

        values = data[column].values
        mean = np.mean(values)
        std = np.std(values)

        if std == 0:
            return anomalies

        z_scores = (values - mean) / std

        for i, (z, val) in enumerate(zip(z_scores, values)):
            if abs(z) > z_threshold:
                date = data.iloc[i]['date']
                deviation_pct = ((val - mean) / mean) * 100

                if z > 0:
                    anomaly_type = AnomalyType.SPIKE
                    causes = self._get_spike_causes(date, deviation_pct)
                else:
                    anomaly_type = AnomalyType.DROP
                    causes = self._get_drop_causes(date, deviation_pct)

                severity = self._calculate_severity(abs(z), abs(deviation_pct))

                anomalies.append(Anomaly(
                    date=date,
                    type=anomaly_type,
                    severity=severity,
                    metric=column,
                    actual_value=val,
                    expected_value=mean,
                    deviation_pct=deviation_pct,
                    z_score=z,
                    possible_causes=causes,
                    recommended_actions=self._get_recommended_actions(anomaly_type, severity)
                ))

        return anomalies

    def _detect_dow_anomalies(
        self,
        data: pd.DataFrame,
        place_id: Optional[int]
    ) -> List[Anomaly]:
        """Detect anomalies based on day-of-week patterns."""
        anomalies = []

        # Get expected pattern with std deviation
        dow_pattern = self.data_processor.get_day_of_week_pattern(place_id)
        if len(dow_pattern) == 0:
            return anomalies

        # Build lookup with mean and std for proper z-score calculation
        dow_stats = {}
        for _, row in dow_pattern.iterrows():
            dow = row['day_of_week']
            avg_rev = row['avg_revenue']
            # Estimate std as 25% of mean if not available
            std_rev = row.get('std_revenue', avg_rev * 0.25)
            dow_stats[dow] = {'mean': avg_rev, 'std': max(std_rev, avg_rev * 0.1)}

        for _, row in data.iterrows():
            dow = row['day_of_week']
            if dow not in dow_stats:
                continue

            expected = dow_stats[dow]['mean']
            std = dow_stats[dow]['std']
            actual = row['total_revenue']

            if expected > 0 and std > 0:
                deviation_pct = ((actual - expected) / expected) * 100
                # Calculate proper z-score
                z_score = (actual - expected) / std

                # Flag if z-score exceeds threshold (approximately 40% deviation)
                if abs(z_score) > 1.5:
                    anomaly_type = AnomalyType.SPIKE if z_score > 0 else AnomalyType.DROP

                    severity = self._calculate_severity(abs(z_score), abs(deviation_pct))

                    anomalies.append(Anomaly(
                        date=row['date'],
                        type=anomaly_type,
                        severity=severity,
                        metric='dow_deviation',
                        actual_value=actual,
                        expected_value=expected,
                        deviation_pct=deviation_pct,
                        z_score=z_score,
                        possible_causes=self._get_dow_causes(row, deviation_pct),
                        recommended_actions=self._get_recommended_actions(anomaly_type, severity)
                    ))

        return anomalies

    def _detect_trend_changes(self, data: pd.DataFrame) -> List[Anomaly]:
        """Detect significant trend changes."""
        anomalies = []

        if len(data) < 14:
            return anomalies

        # Calculate 7-day rolling average
        data = data.copy()
        data['rolling_7d'] = data['total_revenue'].rolling(7).mean()

        # Calculate week-over-week change
        data['wow_change'] = data['rolling_7d'].pct_change(7) * 100

        # Calculate std of wow changes for proper z-score
        wow_std = data['wow_change'].std()
        wow_mean = data['wow_change'].mean()

        if wow_std == 0 or pd.isna(wow_std):
            wow_std = 15  # Default std for stability

        # Find significant trend changes (z-score > 2)
        for _, row in data.iterrows():
            if pd.notna(row['wow_change']):
                # Calculate proper z-score
                z_score = (row['wow_change'] - wow_mean) / wow_std if wow_std > 0 else 0

                if abs(z_score) > 2 or abs(row['wow_change']) > 30:
                    anomaly_type = AnomalyType.TREND_CHANGE

                    if row['wow_change'] > 0:
                        causes = ["Positive trend emerging", "Possible seasonal uptick",
                                 "Marketing campaign effect", "Competition changes"]
                    else:
                        causes = ["Negative trend detected", "Possible seasonal decline",
                                 "External factors (weather, events)", "Competition impact"]

                    severity = self._calculate_severity(abs(z_score), abs(row['wow_change']))

                    # Safely calculate expected value
                    divisor = 1 + row['wow_change'] / 100
                    expected = row['rolling_7d'] / divisor if divisor != 0 else row['rolling_7d']

                    anomalies.append(Anomaly(
                        date=row['date'],
                        type=anomaly_type,
                        severity=severity,
                        metric='trend',
                        actual_value=row['rolling_7d'],
                        expected_value=expected,
                        deviation_pct=row['wow_change'],
                        z_score=z_score,
                        possible_causes=causes,
                        recommended_actions=[
                            "Monitor trend over next 7 days",
                            "Analyze contributing factors",
                            "Adjust forecasts if trend persists"
                        ]
                    ))

        return anomalies

    def _calculate_severity(self, z_score: float, deviation_pct: float) -> AlertSeverity:
        """Calculate alert severity based on z-score and deviation."""
        if z_score > 4 or abs(deviation_pct) > 80:
            return AlertSeverity.CRITICAL
        elif z_score > 3 or abs(deviation_pct) > 50:
            return AlertSeverity.HIGH
        elif z_score > 2.5 or abs(deviation_pct) > 30:
            return AlertSeverity.MEDIUM
        else:
            return AlertSeverity.LOW

    def _get_spike_causes(self, date: datetime, deviation_pct: float) -> List[str]:
        """Get possible causes for a sales spike."""
        causes = []

        # Check if weekend
        if hasattr(date, 'dayofweek'):
            dow = date.dayofweek
        else:
            dow = pd.to_datetime(date).dayofweek

        if dow >= 5:
            causes.append("Weekend effect")

        # Check if holiday period
        month = pd.to_datetime(date).month
        if month == 12:
            causes.append("Holiday season")
        elif month in [6, 7, 8]:
            causes.append("Summer season")

        # Generic causes
        causes.extend([
            "Possible promotion or event",
            "Weather driving indoor dining",
            "Local event or holiday",
            "Social media or viral mention"
        ])

        return causes[:4]

    def _get_drop_causes(self, date: datetime, deviation_pct: float) -> List[str]:
        """Get possible causes for a sales drop."""
        causes = []

        if hasattr(date, 'dayofweek'):
            dow = date.dayofweek
        else:
            dow = pd.to_datetime(date).dayofweek

        if dow in [0, 1]:  # Monday, Tuesday
            causes.append("Typical slow day")

        causes.extend([
            "Bad weather conditions",
            "Local competition event",
            "Staff or supply issues",
            "External event keeping customers away",
            "System or reporting issue (verify data)"
        ])

        return causes[:4]

    def _get_dow_causes(self, row, deviation_pct: float) -> List[str]:
        """Get causes for day-of-week deviation."""
        causes = []

        if row.get('is_holiday', 0):
            causes.append("Holiday effect")

        if row.get('is_weekend', 0):
            if deviation_pct > 0:
                causes.append("Stronger than usual weekend")
            else:
                causes.append("Weaker than usual weekend")

        causes.extend([
            "Weather impact",
            "Local event effect",
            "Promotional activity"
        ])

        return causes[:4]

    def _get_recommended_actions(
        self,
        anomaly_type: AnomalyType,
        severity: AlertSeverity
    ) -> List[str]:
        """Get recommended actions based on anomaly type and severity."""
        actions = []

        if anomaly_type == AnomalyType.SPIKE:
            actions = [
                "Verify data accuracy",
                "Document cause for future reference",
                "Consider if this can be replicated",
                "Update forecast models if pattern repeats"
            ]
        elif anomaly_type == AnomalyType.DROP:
            if severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
                actions = [
                    "Investigate immediately",
                    "Check for operational issues",
                    "Review staff scheduling",
                    "Consider promotional response"
                ]
            else:
                actions = [
                    "Monitor for pattern continuation",
                    "Review potential causes",
                    "Compare with similar past periods"
                ]
        else:
            actions = [
                "Continue monitoring",
                "Analyze contributing factors",
                "Update forecasts as needed"
            ]

        return actions

    def get_anomaly_summary(
        self,
        place_id: Optional[int] = None,
        days: int = 30
    ) -> Dict:
        """Get summary of recent anomalies."""
        anomalies = self.detect_anomalies(place_id, lookback_days=days)

        summary = {
            'total_anomalies': len(anomalies),
            'critical': len([a for a in anomalies if a.severity == AlertSeverity.CRITICAL]),
            'high': len([a for a in anomalies if a.severity == AlertSeverity.HIGH]),
            'medium': len([a for a in anomalies if a.severity == AlertSeverity.MEDIUM]),
            'low': len([a for a in anomalies if a.severity == AlertSeverity.LOW]),
            'spikes': len([a for a in anomalies if a.type == AnomalyType.SPIKE]),
            'drops': len([a for a in anomalies if a.type == AnomalyType.DROP]),
            'trend_changes': len([a for a in anomalies if a.type == AnomalyType.TREND_CHANGE]),
            'recent_anomalies': anomalies[:5]  # Most recent 5
        }

        return summary
