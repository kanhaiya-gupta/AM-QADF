"""
Unit tests for performance monitoring utilities.

Tests for performance_monitor decorator.
"""

import pytest
import time
import numpy as np
from unittest.mock import patch, MagicMock
from am_qadf.signal_mapping.utils._performance import performance_monitor


class TestPerformanceMonitor:
    """Test suite for performance_monitor decorator."""

    @pytest.mark.unit
    def test_performance_monitor_basic(self):
        """Test performance_monitor decorator with basic function."""

        @performance_monitor
        def test_function(x, y):
            return x + y

        result = test_function(2, 3)

        assert result == 5

    @pytest.mark.unit
    def test_performance_monitor_with_points_arg(self):
        """Test performance_monitor with points as first argument."""

        @performance_monitor
        def test_interpolate(points, signals, voxel_grid):
            time.sleep(0.01)  # Simulate work
            return voxel_grid

        points = np.random.rand(1000, 3)
        signals = {"power": np.random.rand(1000)}
        voxel_grid = MagicMock()

        result = test_interpolate(points, signals, voxel_grid)

        assert result is voxel_grid

    @pytest.mark.unit
    def test_performance_monitor_with_points_kwarg(self):
        """Test performance_monitor with points as keyword argument."""

        @performance_monitor
        def test_interpolate(signals, voxel_grid, points=None):
            time.sleep(0.01)  # Simulate work
            return voxel_grid

        points = np.random.rand(1000, 3)
        signals = {"power": np.random.rand(1000)}
        voxel_grid = MagicMock()

        result = test_interpolate(signals, voxel_grid, points=points)

        assert result is voxel_grid

    @pytest.mark.unit
    def test_performance_monitor_empty_points(self):
        """Test performance_monitor with empty points array."""

        @performance_monitor
        def test_interpolate(points, signals, voxel_grid):
            return voxel_grid

        points = np.array([]).reshape(0, 3)
        signals = {}
        voxel_grid = MagicMock()

        result = test_interpolate(points, signals, voxel_grid)

        assert result is voxel_grid

    @pytest.mark.unit
    def test_performance_monitor_no_points(self):
        """Test performance_monitor with function that doesn't have points."""

        @performance_monitor
        def test_function(x, y, z):
            return x * y * z

        result = test_function(2, 3, 4)

        assert result == 24

    @pytest.mark.unit
    @patch("am_qadf.signal_mapping.utils._performance.logger")
    def test_performance_monitor_logging(self, mock_logger):
        """Test that performance_monitor logs performance information."""

        @performance_monitor
        def test_interpolate(points, signals, voxel_grid):
            time.sleep(0.01)  # Simulate work
            return voxel_grid

        points = np.random.rand(1000, 3)
        signals = {"power": np.random.rand(1000)}
        voxel_grid = MagicMock()

        test_interpolate(points, signals, voxel_grid)

        # Verify that logger.info was called
        assert mock_logger.info.called

    @pytest.mark.unit
    def test_performance_monitor_preserves_function_metadata(self):
        """Test that performance_monitor preserves function metadata."""

        @performance_monitor
        def test_function(x, y):
            """Test function docstring."""
            return x + y

        # Function should still be callable
        assert callable(test_function)
        assert test_function(2, 3) == 5
