"""
Unit tests for anomaly detection types.

Tests for AnomalyType enum.
"""

import pytest
from am_qadf.anomaly_detection.core.types import AnomalyType


class TestAnomalyType:
    """Test suite for AnomalyType enum."""

    @pytest.mark.unit
    def test_anomaly_type_enum_values(self):
        """Test that all AnomalyType enum values exist."""
        assert AnomalyType.POINT == AnomalyType.POINT
        assert AnomalyType.CONTEXTUAL == AnomalyType.CONTEXTUAL
        assert AnomalyType.COLLECTIVE == AnomalyType.COLLECTIVE
        assert AnomalyType.SPATIAL == AnomalyType.SPATIAL
        assert AnomalyType.TEMPORAL == AnomalyType.TEMPORAL

    @pytest.mark.unit
    def test_anomaly_type_string_values(self):
        """Test AnomalyType string representations."""
        assert AnomalyType.POINT.value == "point"
        assert AnomalyType.CONTEXTUAL.value == "contextual"
        assert AnomalyType.COLLECTIVE.value == "collective"
        assert AnomalyType.SPATIAL.value == "spatial"
        assert AnomalyType.TEMPORAL.value == "temporal"

    @pytest.mark.unit
    def test_anomaly_type_comparison(self):
        """Test AnomalyType comparison operations."""
        assert AnomalyType.POINT == AnomalyType.POINT
        assert AnomalyType.POINT != AnomalyType.SPATIAL
        assert AnomalyType.SPATIAL != AnomalyType.TEMPORAL

    @pytest.mark.unit
    def test_anomaly_type_from_string(self):
        """Test creating AnomalyType from string value."""
        assert AnomalyType("point") == AnomalyType.POINT
        assert AnomalyType("spatial") == AnomalyType.SPATIAL
        assert AnomalyType("temporal") == AnomalyType.TEMPORAL
        assert AnomalyType("contextual") == AnomalyType.CONTEXTUAL
        assert AnomalyType("collective") == AnomalyType.COLLECTIVE

    @pytest.mark.unit
    def test_anomaly_type_invalid_string(self):
        """Test that invalid string raises ValueError."""
        with pytest.raises(ValueError):
            AnomalyType("invalid_type")

    @pytest.mark.unit
    def test_anomaly_type_list_all(self):
        """Test listing all AnomalyType values."""
        all_types = list(AnomalyType)
        assert len(all_types) == 5
        assert AnomalyType.POINT in all_types
        assert AnomalyType.CONTEXTUAL in all_types
        assert AnomalyType.COLLECTIVE in all_types
        assert AnomalyType.SPATIAL in all_types
        assert AnomalyType.TEMPORAL in all_types

    @pytest.mark.unit
    def test_anomaly_type_str_representation(self):
        """Test string representation of AnomalyType."""
        assert str(AnomalyType.POINT) == "AnomalyType.POINT"
        assert str(AnomalyType.SPATIAL) == "AnomalyType.SPATIAL"

    @pytest.mark.unit
    def test_anomaly_type_repr(self):
        """Test repr representation of AnomalyType."""
        assert repr(AnomalyType.POINT) == "<AnomalyType.POINT: 'point'>"
        assert repr(AnomalyType.SPATIAL) == "<AnomalyType.SPATIAL: 'spatial'>"
