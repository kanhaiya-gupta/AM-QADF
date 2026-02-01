"""
Unit tests for IncrementalProcessor.

Tests for incremental voxel grid processing.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
from datetime import datetime

from am_qadf.streaming.incremental_processor import IncrementalProcessor


class TestIncrementalProcessor:
    """Test suite for IncrementalProcessor class."""

    @pytest.fixture
    def mock_voxel_grid(self):
        """Create a mock VoxelGrid instance."""
        mock_grid = MagicMock()
        mock_grid.add_point = MagicMock()
        mock_grid.finalize = MagicMock()
        return mock_grid

    @pytest.fixture
    def processor(self, mock_voxel_grid):
        """Create an IncrementalProcessor instance."""
        return IncrementalProcessor(voxel_grid=mock_voxel_grid)

    @pytest.mark.unit
    def test_processor_creation(self, processor, mock_voxel_grid):
        """Test creating IncrementalProcessor."""
        assert processor is not None
        assert processor.voxel_grid == mock_voxel_grid
        assert len(processor._updated_regions) == 0
        assert processor._last_update_time is None

    @pytest.mark.unit
    def test_processor_creation_no_grid(self):
        """Test creating IncrementalProcessor without voxel grid."""
        processor = IncrementalProcessor()
        assert processor.voxel_grid is None

    @pytest.mark.unit
    def test_update_voxel_grid(self, processor, mock_voxel_grid):
        """Test updating voxel grid incrementally."""
        new_data = np.array([1.0, 2.0, 3.0])
        coordinates = np.array(
            [
                [10.0, 20.0, 30.0],
                [11.0, 21.0, 31.0],
                [12.0, 22.0, 32.0],
            ]
        )

        result = processor.update_voxel_grid(new_data, coordinates)

        assert result == mock_voxel_grid
        assert mock_voxel_grid.add_point.call_count == 3
        assert len(processor._updated_regions) == 1
        assert processor._last_update_time is not None

    @pytest.mark.unit
    def test_update_voxel_grid_2d_signals(self, processor, mock_voxel_grid):
        """Test updating voxel grid with 2D signal data."""
        new_data = np.array(
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
            ]
        )
        coordinates = np.array(
            [
                [10.0, 20.0, 30.0],
                [11.0, 21.0, 31.0],
                [12.0, 22.0, 32.0],
            ]
        )

        processor.update_voxel_grid(new_data, coordinates)

        assert mock_voxel_grid.add_point.call_count == 3

    @pytest.mark.unit
    def test_update_voxel_grid_no_grid(self):
        """Test updating voxel grid when grid not initialized."""
        processor = IncrementalProcessor()
        new_data = np.array([1.0])
        coordinates = np.array([[10.0, 20.0, 30.0]])

        with pytest.raises(ValueError, match="VoxelGrid not initialized"):
            processor.update_voxel_grid(new_data, coordinates)

    @pytest.mark.unit
    def test_update_voxel_grid_empty_coordinates(self, processor, mock_voxel_grid):
        """Test updating with empty coordinates."""
        new_data = np.array([1.0])
        coordinates = np.array([]).reshape(0, 3)

        # Should log warning and return grid
        result = processor.update_voxel_grid(new_data, coordinates)
        assert result == mock_voxel_grid
        assert mock_voxel_grid.add_point.call_count == 0

    @pytest.mark.unit
    def test_update_voxel_grid_invalid_coordinates(self, processor, mock_voxel_grid):
        """Test updating with invalid coordinate shape."""
        new_data = np.array([1.0])
        coordinates = np.array([10.0, 20.0])  # Missing z coordinate

        with pytest.raises(ValueError, match="Coordinates must be N x 3"):
            processor.update_voxel_grid(new_data, coordinates)

    @pytest.mark.unit
    def test_process_signal_batch(self, processor, mock_voxel_grid):
        """Test processing signal batch."""
        signal_data = {
            "coordinates": np.array(
                [
                    [10.0, 20.0, 30.0],
                    [11.0, 21.0, 31.0],
                ]
            ),
            "temperature": np.array([100.0, 200.0]),
            "power": np.array([50.0, 60.0]),
        }

        result = processor.process_signal_batch(signal_data)

        assert result["processed"] is True
        assert result["points_processed"] == 2
        assert result["signals_processed"] == ["temperature", "power"]
        assert result["voxel_grid_updated"] is True

    @pytest.mark.unit
    def test_process_signal_batch_no_coordinates(self, processor):
        """Test processing signal batch without coordinates."""
        signal_data = {
            "temperature": np.array([100.0]),
        }

        with pytest.raises(ValueError, match="must contain 'coordinates' key"):
            processor.process_signal_batch(signal_data)

    @pytest.mark.unit
    def test_process_signal_batch_no_signals(self, processor, mock_voxel_grid):
        """Test processing signal batch with no signals."""
        signal_data = {
            "coordinates": np.array([[10.0, 20.0, 30.0]]),
        }

        result = processor.process_signal_batch(signal_data)

        assert result["processed"] is False
        assert result["reason"] == "no_signals"

    @pytest.mark.unit
    def test_get_updated_regions(self, processor, mock_voxel_grid):
        """Test getting updated regions."""
        # Update grid
        new_data = np.array([1.0, 2.0])
        coordinates = np.array(
            [
                [10.0, 20.0, 30.0],
                [11.0, 21.0, 31.0],
            ]
        )
        processor.update_voxel_grid(new_data, coordinates)

        regions = processor.get_updated_regions()

        assert len(regions) == 1
        assert len(regions[0]) == 2  # (bbox_min, bbox_max)

    @pytest.mark.unit
    def test_get_updated_regions_clear(self, processor, mock_voxel_grid):
        """Test getting updated regions with clear option."""
        # Update grid
        new_data = np.array([1.0])
        coordinates = np.array([[10.0, 20.0, 30.0]])
        processor.update_voxel_grid(new_data, coordinates)

        assert len(processor._updated_regions) == 1

        regions = processor.get_updated_regions(clear=True)

        assert len(regions) == 1
        assert len(processor._updated_regions) == 0

    @pytest.mark.unit
    def test_get_combined_updated_region(self, processor, mock_voxel_grid):
        """Test getting combined updated region."""
        # Update grid multiple times
        processor.update_voxel_grid(np.array([1.0]), np.array([[10.0, 20.0, 30.0]]))
        processor.update_voxel_grid(np.array([2.0]), np.array([[15.0, 25.0, 35.0]]))

        combined = processor.get_combined_updated_region()

        assert combined is not None
        assert len(combined) == 2
        # Should encompass both regions
        assert combined[0][0] <= 10.0  # min_x
        assert combined[1][0] >= 15.0  # max_x

    @pytest.mark.unit
    def test_get_combined_updated_region_empty(self, processor):
        """Test getting combined region when no updates."""
        combined = processor.get_combined_updated_region()
        assert combined is None

    @pytest.mark.unit
    def test_reset(self, processor, mock_voxel_grid):
        """Test resetting processor state."""
        # Update grid
        processor.update_voxel_grid(np.array([1.0]), np.array([[10.0, 20.0, 30.0]]))

        assert len(processor._updated_regions) > 0
        assert processor._last_update_time is not None

        processor.reset()

        assert len(processor._updated_regions) == 0
        assert processor._last_update_time is None
        assert processor._stats["total_points_processed"] == 0
        assert processor._stats["total_updates"] == 0

    @pytest.mark.unit
    def test_set_voxel_grid(self, processor, mock_voxel_grid):
        """Test setting voxel grid."""
        new_grid = MagicMock()
        processor.set_voxel_grid(new_grid)

        assert processor.voxel_grid == new_grid

    @pytest.mark.unit
    def test_get_statistics(self, processor, mock_voxel_grid):
        """Test getting statistics."""
        # Update grid
        processor.update_voxel_grid(
            np.array([1.0, 2.0]),
            np.array(
                [
                    [10.0, 20.0, 30.0],
                    [11.0, 21.0, 31.0],
                ]
            ),
        )

        stats = processor.get_statistics()

        assert stats["total_points_processed"] == 2
        assert stats["total_updates"] == 1
        assert stats["regions_updated"] == 1
        assert stats["has_voxel_grid"] is True
        assert stats["last_update_time"] is not None
