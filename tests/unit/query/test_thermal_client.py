"""
Unit tests for ThermalClient.

Tests for thermal field data query functionality.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from am_qadf.query.thermal_client import ThermalClient
from am_qadf.query.base_query_client import (
    SpatialQuery,
    TemporalQuery,
    QueryResult,
    SignalType,
)


class TestThermalClient:
    """Test suite for ThermalClient."""

    @pytest.mark.unit
    def test_thermal_client_creation(self):
        """Test creating ThermalClient."""
        mock_laser_client = Mock()
        client = ThermalClient(laser_client=mock_laser_client)

        assert client.laser_client == mock_laser_client
        assert client.thermal_generator is not None

    @pytest.mark.unit
    def test_thermal_client_creation_with_generator(self):
        """Test creating ThermalClient with custom generator."""
        mock_laser_client = Mock()
        mock_generator = Mock()

        client = ThermalClient(laser_client=mock_laser_client, thermal_generator=mock_generator)

        assert client.thermal_generator == mock_generator

    @pytest.mark.unit
    def test_thermal_client_query_with_laser_client(self):
        """Test querying thermal data using laser client."""
        # Mock laser client
        mock_laser_client = Mock()
        mock_laser_result = QueryResult(
            points=[(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)],
            signals={"energy": [2.0, 2.5]},  # ThermalClient expects 'energy' key
            metadata={},
        )
        mock_laser_client.query.return_value = mock_laser_result

        client = ThermalClient(laser_client=mock_laser_client)

        result = client.query()

        assert isinstance(result, QueryResult)
        assert len(result.points) > 0
        # Should have thermal/temperature signals
        assert any("thermal" in k.lower() or "temperature" in k.lower() for k in result.signals.keys())

    @pytest.mark.unit
    def test_thermal_client_query_with_spatial(self):
        """Test querying with spatial constraints."""
        mock_laser_client = Mock()
        mock_laser_client.query.return_value = QueryResult(points=[], signals={}, metadata={})

        client = ThermalClient(laser_client=mock_laser_client)

        spatial = SpatialQuery(bbox_min=(0.0, 0.0, 0.0), bbox_max=(10.0, 10.0, 10.0))

        result = client.query(spatial=spatial)

        assert isinstance(result, QueryResult)
        # Verify spatial query was passed to laser client
        mock_laser_client.query.assert_called_once()

    @pytest.mark.unit
    def test_thermal_client_query_with_temporal(self):
        """Test querying with temporal constraints."""
        mock_laser_client = Mock()
        mock_laser_client.query.return_value = QueryResult(points=[], signals={}, metadata={})

        client = ThermalClient(laser_client=mock_laser_client)

        temporal = TemporalQuery(layer_start=0, layer_end=10)

        result = client.query(temporal=temporal)

        assert isinstance(result, QueryResult)

    @pytest.mark.unit
    def test_thermal_client_error_handling_no_laser_client(self):
        """Test error handling when laser client is not provided."""
        client = ThermalClient(laser_client=None)

        # Should raise ValueError when querying without laser client
        with pytest.raises(ValueError, match="LaserParameterClient required"):
            client.query()
