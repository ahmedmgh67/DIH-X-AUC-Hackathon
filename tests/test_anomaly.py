"""
Tests for anomaly detection data structures and classification.
"""

import pytest
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from freshflow.services.anomaly_service import (
    AnomalyType,
    AlertSeverity,
    Anomaly,
)


class TestAnomalyTypes:
    def test_spike_type(self):
        assert AnomalyType.SPIKE.value == "spike"

    def test_drop_type(self):
        assert AnomalyType.DROP.value == "drop"

    def test_all_types_exist(self):
        types = [t.value for t in AnomalyType]
        assert "spike" in types
        assert "drop" in types
        assert "trend_change" in types
        assert "unusual_pattern" in types


class TestAlertSeverity:
    def test_severity_values(self):
        assert AlertSeverity.LOW.value == "low"
        assert AlertSeverity.MEDIUM.value == "medium"
        assert AlertSeverity.HIGH.value == "high"
        assert AlertSeverity.CRITICAL.value == "critical"


class TestAnomalyDataclass:
    def test_create_anomaly(self):
        anomaly = Anomaly(
            date=datetime(2025, 1, 15),
            type=AnomalyType.SPIKE,
            severity=AlertSeverity.HIGH,
            metric="revenue",
            actual_value=5000,
            expected_value=3000,
            deviation_pct=66.7,
            z_score=3.2,
            possible_causes=["Holiday weekend", "Local event"],
            recommended_actions=["Increase stock for next occurrence"],
        )
        assert anomaly.actual_value == 5000
        assert anomaly.type == AnomalyType.SPIKE
        assert anomaly.severity == AlertSeverity.HIGH
        assert len(anomaly.possible_causes) == 2

    def test_anomaly_deviation(self):
        anomaly = Anomaly(
            date=datetime(2025, 1, 15),
            type=AnomalyType.DROP,
            severity=AlertSeverity.CRITICAL,
            metric="order_count",
            actual_value=10,
            expected_value=50,
            deviation_pct=-80.0,
            z_score=-4.1,
            possible_causes=["System outage"],
            recommended_actions=["Investigate POS system"],
        )
        assert anomaly.deviation_pct < 0
        assert anomaly.z_score < -3
        assert anomaly.actual_value < anomaly.expected_value
