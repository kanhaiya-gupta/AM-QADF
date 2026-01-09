"""
Unit tests for core domain entities.

Tests for VoxelData and other domain entities.
"""

import pytest
import numpy as np
from am_qadf.core.entities import VoxelData


class TestVoxelData:
    """Test suite for VoxelData entity."""

    @pytest.mark.unit
    def test_voxel_data_creation_empty(self):
        """Test creating empty VoxelData."""
        voxel_data = VoxelData()

        assert voxel_data.signals == {}
        assert voxel_data.count == 0

    @pytest.mark.unit
    def test_voxel_data_creation_with_signals(self):
        """Test creating VoxelData with initial signals."""
        signals = {"power": 200.0, "speed": 100.0}
        voxel_data = VoxelData(signals=signals, count=2)

        assert voxel_data.signals == signals
        assert voxel_data.count == 2

    @pytest.mark.unit
    def test_add_signal_single_value(self):
        """Test adding a single signal value."""
        voxel_data = VoxelData()

        voxel_data.add_signal("power", 200.0)

        assert "power" in voxel_data.signals
        assert isinstance(voxel_data.signals["power"], list)
        assert voxel_data.signals["power"] == [200.0]
        assert voxel_data.count == 1

    @pytest.mark.unit
    def test_add_signal_multiple_values(self):
        """Test adding multiple values for the same signal."""
        voxel_data = VoxelData()

        voxel_data.add_signal("power", 200.0)
        voxel_data.add_signal("power", 250.0)
        voxel_data.add_signal("power", 300.0)

        assert "power" in voxel_data.signals
        assert isinstance(voxel_data.signals["power"], list)
        assert len(voxel_data.signals["power"]) == 3
        assert voxel_data.signals["power"] == [200.0, 250.0, 300.0]
        assert voxel_data.count == 3

    @pytest.mark.unit
    def test_add_signal_multiple_signals(self):
        """Test adding multiple different signals."""
        voxel_data = VoxelData()

        voxel_data.add_signal("power", 200.0)
        voxel_data.add_signal("speed", 100.0)
        voxel_data.add_signal("temperature", 500.0)

        assert len(voxel_data.signals) == 3
        assert "power" in voxel_data.signals
        assert "speed" in voxel_data.signals
        assert "temperature" in voxel_data.signals
        assert voxel_data.count == 3

    @pytest.mark.unit
    def test_add_signal_after_finalize(self):
        """Test adding signal after finalizing converts value to list."""
        voxel_data = VoxelData()

        # Add and finalize
        voxel_data.add_signal("power", 200.0)
        voxel_data.finalize()

        # Add another value - should convert existing value to list
        voxel_data.add_signal("power", 250.0)

        assert isinstance(voxel_data.signals["power"], list)
        assert len(voxel_data.signals["power"]) == 2
        assert 200.0 in voxel_data.signals["power"]
        assert 250.0 in voxel_data.signals["power"]

    @pytest.mark.unit
    def test_finalize_mean_aggregation(self):
        """Test finalizing with mean aggregation."""
        voxel_data = VoxelData()

        voxel_data.add_signal("power", 200.0)
        voxel_data.add_signal("power", 250.0)
        voxel_data.add_signal("power", 300.0)

        voxel_data.finalize(aggregation="mean")

        assert isinstance(voxel_data.signals["power"], (int, float, np.floating))
        assert voxel_data.signals["power"] == 250.0  # (200 + 250 + 300) / 3

    @pytest.mark.unit
    def test_finalize_max_aggregation(self):
        """Test finalizing with max aggregation."""
        voxel_data = VoxelData()

        voxel_data.add_signal("power", 200.0)
        voxel_data.add_signal("power", 250.0)
        voxel_data.add_signal("power", 300.0)

        voxel_data.finalize(aggregation="max")

        assert voxel_data.signals["power"] == 300.0

    @pytest.mark.unit
    def test_finalize_min_aggregation(self):
        """Test finalizing with min aggregation."""
        voxel_data = VoxelData()

        voxel_data.add_signal("power", 200.0)
        voxel_data.add_signal("power", 250.0)
        voxel_data.add_signal("power", 300.0)

        voxel_data.finalize(aggregation="min")

        assert voxel_data.signals["power"] == 200.0

    @pytest.mark.unit
    def test_finalize_sum_aggregation(self):
        """Test finalizing with sum aggregation."""
        voxel_data = VoxelData()

        voxel_data.add_signal("power", 200.0)
        voxel_data.add_signal("power", 250.0)
        voxel_data.add_signal("power", 300.0)

        voxel_data.finalize(aggregation="sum")

        assert voxel_data.signals["power"] == 750.0

    @pytest.mark.unit
    def test_finalize_default_aggregation(self):
        """Test finalizing with default (mean) aggregation."""
        voxel_data = VoxelData()

        voxel_data.add_signal("power", 200.0)
        voxel_data.add_signal("power", 250.0)

        voxel_data.finalize()  # Default is mean

        assert voxel_data.signals["power"] == 225.0

    @pytest.mark.unit
    def test_finalize_invalid_aggregation(self):
        """Test finalizing with invalid aggregation defaults to mean."""
        voxel_data = VoxelData()

        voxel_data.add_signal("power", 200.0)
        voxel_data.add_signal("power", 250.0)

        voxel_data.finalize(aggregation="invalid_method")

        # Should default to mean
        assert voxel_data.signals["power"] == 225.0

    @pytest.mark.unit
    def test_finalize_multiple_signals(self):
        """Test finalizing multiple signals with different aggregations."""
        voxel_data = VoxelData()

        # Add multiple signals
        voxel_data.add_signal("power", 200.0)
        voxel_data.add_signal("power", 250.0)
        voxel_data.add_signal("speed", 100.0)
        voxel_data.add_signal("speed", 150.0)
        voxel_data.add_signal("speed", 200.0)

        voxel_data.finalize(aggregation="mean")

        assert voxel_data.signals["power"] == 225.0
        assert voxel_data.signals["speed"] == 150.0

    @pytest.mark.unit
    def test_finalize_already_finalized(self):
        """Test finalizing already finalized data (no-op)."""
        voxel_data = VoxelData()

        voxel_data.add_signal("power", 200.0)
        voxel_data.finalize()

        # Finalize again - should not change anything
        original_value = voxel_data.signals["power"]
        voxel_data.finalize()

        assert voxel_data.signals["power"] == original_value

    @pytest.mark.unit
    def test_finalize_empty_signals(self):
        """Test finalizing with no signals (should not error)."""
        voxel_data = VoxelData()

        # Should not raise an error
        voxel_data.finalize()

        assert voxel_data.signals == {}
        assert voxel_data.count == 0
