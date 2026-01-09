"""
Unit tests for voxel fusion.

Tests for VoxelFusion class.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
from am_qadf.fusion.voxel_fusion import VoxelFusion
from am_qadf.synchronization.data_fusion import FusionStrategy


class MockVoxelData:
    """Mock voxel data object for testing."""

    def __init__(self, signals: dict):
        """Initialize with signal dictionary."""
        self._signals = signals

    def get_signal_array(self, signal_name: str, default: float = 0.0) -> np.ndarray:
        """Get signal array by name."""
        return self._signals.get(signal_name, np.array([default]))

    def add_signal(self, signal_name: str, signal_array: np.ndarray):
        """Add signal to voxel data."""
        self._signals[signal_name] = signal_array


class TestVoxelFusion:
    """Test suite for VoxelFusion class."""

    @pytest.fixture
    def voxel_fusion(self):
        """Create a VoxelFusion instance."""
        return VoxelFusion()

    @pytest.fixture
    def mock_voxel_data(self):
        """Create mock voxel data with signals."""
        signals = {
            "signal1": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            "signal2": np.array([2.0, 3.0, 4.0, 5.0, 6.0]),
            "signal3": np.array([3.0, 4.0, 5.0, 6.0, 7.0]),
        }
        return MockVoxelData(signals)

    @pytest.mark.unit
    def test_voxel_fusion_creation_default(self):
        """Test creating VoxelFusion with default parameters."""
        fusion = VoxelFusion()

        assert fusion.fusion_engine is not None
        assert fusion.use_quality_scores is True

    @pytest.mark.unit
    def test_voxel_fusion_creation_custom(self):
        """Test creating VoxelFusion with custom parameters."""
        fusion = VoxelFusion(default_strategy=FusionStrategy.MEDIAN, use_quality_scores=False)

        assert fusion.fusion_engine.default_strategy == FusionStrategy.MEDIAN
        assert fusion.use_quality_scores is False

    @pytest.mark.unit
    def test_fuse_voxel_signals_average(self, voxel_fusion, mock_voxel_data):
        """Test fusing voxel signals with AVERAGE strategy."""
        signals = ["signal1", "signal2", "signal3"]

        fused = voxel_fusion.fuse_voxel_signals(mock_voxel_data, signals, fusion_strategy=FusionStrategy.AVERAGE)

        assert len(fused) == 5
        # Average of [1,2,3], [2,3,4], [3,4,5] = [2,3,4]
        assert np.allclose(fused[0], 2.0, atol=0.1)

    @pytest.mark.unit
    def test_fuse_voxel_signals_median(self, voxel_fusion, mock_voxel_data):
        """Test fusing voxel signals with MEDIAN strategy."""
        signals = ["signal1", "signal2", "signal3"]

        fused = voxel_fusion.fuse_voxel_signals(mock_voxel_data, signals, fusion_strategy=FusionStrategy.MEDIAN)

        assert len(fused) == 5
        # Median of [1,2,3], [2,3,4], [3,4,5] = [2,3,4]
        assert np.allclose(fused[0], 2.0, atol=0.1)

    @pytest.mark.unit
    def test_fuse_voxel_signals_max(self, voxel_fusion, mock_voxel_data):
        """Test fusing voxel signals with MAX strategy."""
        signals = ["signal1", "signal2", "signal3"]

        fused = voxel_fusion.fuse_voxel_signals(mock_voxel_data, signals, fusion_strategy=FusionStrategy.MAX)

        assert len(fused) == 5
        # Max of [1,2,3], [2,3,4], [3,4,5] = [3,4,5]
        assert np.allclose(fused[0], 3.0, atol=0.1)

    @pytest.mark.unit
    def test_fuse_voxel_signals_min(self, voxel_fusion, mock_voxel_data):
        """Test fusing voxel signals with MIN strategy."""
        signals = ["signal1", "signal2", "signal3"]

        fused = voxel_fusion.fuse_voxel_signals(mock_voxel_data, signals, fusion_strategy=FusionStrategy.MIN)

        assert len(fused) == 5
        # Min of [1,2,3], [2,3,4], [3,4,5] = [1,2,3]
        assert np.allclose(fused[0], 1.0, atol=0.1)

    @pytest.mark.unit
    def test_fuse_voxel_signals_weighted_average(self, voxel_fusion, mock_voxel_data):
        """Test fusing voxel signals with WEIGHTED_AVERAGE strategy."""
        signals = ["signal1", "signal2", "signal3"]
        quality_scores = {"signal1": 0.9, "signal2": 0.5, "signal3": 0.3}

        fused = voxel_fusion.fuse_voxel_signals(
            mock_voxel_data,
            signals,
            fusion_strategy=FusionStrategy.WEIGHTED_AVERAGE,
            quality_scores=quality_scores,
        )

        assert len(fused) == 5
        # Should weight signal1 more heavily

    @pytest.mark.unit
    def test_fuse_voxel_signals_default_strategy(self, voxel_fusion, mock_voxel_data):
        """Test fusing voxel signals with default strategy."""
        signals = ["signal1", "signal2"]

        fused = voxel_fusion.fuse_voxel_signals(mock_voxel_data, signals)

        assert len(fused) == 5

    @pytest.mark.unit
    def test_fuse_voxel_signals_empty(self, voxel_fusion, mock_voxel_data):
        """Test fusing with empty signal list."""
        signals = []

        with pytest.raises(ValueError, match="At least one signal must be provided"):
            voxel_fusion.fuse_voxel_signals(mock_voxel_data, signals)

    @pytest.mark.unit
    def test_fuse_voxel_signals_missing_signal(self, voxel_fusion, mock_voxel_data):
        """Test fusing with missing signal (should skip with warning)."""
        signals = ["signal1", "nonexistent"]

        fused = voxel_fusion.fuse_voxel_signals(mock_voxel_data, signals)

        # Should still work with available signals
        assert len(fused) == 5

    @pytest.mark.unit
    def test_fuse_voxel_signals_all_missing(self, voxel_fusion):
        """Test fusing when all signals are missing."""
        empty_voxel_data = MockVoxelData({})
        signals = ["signal1", "signal2"]

        with pytest.raises(ValueError, match="No valid signals found"):
            voxel_fusion.fuse_voxel_signals(empty_voxel_data, signals)

    @pytest.mark.unit
    def test_fuse_voxel_signals_adds_to_voxel_data(self, voxel_fusion, mock_voxel_data):
        """Test that fused signal is added to voxel data."""
        signals = ["signal1", "signal2"]

        fused = voxel_fusion.fuse_voxel_signals(mock_voxel_data, signals, output_signal_name="fused")

        # Check if signal was added
        assert "fused" in mock_voxel_data._signals
        assert np.array_equal(mock_voxel_data._signals["fused"], fused)

    @pytest.mark.unit
    def test_fuse_with_quality_weights(self, voxel_fusion, mock_voxel_data):
        """Test fusing with quality-based weighting."""
        signals = ["signal1", "signal2", "signal3"]
        quality_scores = {"signal1": 0.9, "signal2": 0.5, "signal3": 0.3}

        fused = voxel_fusion.fuse_with_quality_weights(mock_voxel_data, signals, quality_scores)

        assert len(fused) == 5
        # Should use weighted average with quality scores

    @pytest.mark.unit
    def test_fuse_per_voxel_custom(self, voxel_fusion, mock_voxel_data):
        """Test fusing with custom per-voxel function."""
        signals = ["signal1", "signal2", "signal3"]

        def custom_fusion(values):
            """Custom fusion: sum of values."""
            return sum(values)

        fused = voxel_fusion.fuse_per_voxel(mock_voxel_data, signals, fusion_func=custom_fusion)

        assert len(fused) == 5
        # First voxel: sum of [1, 2, 3] = 6
        assert np.allclose(fused[0], 6.0, atol=0.1)

    @pytest.mark.unit
    def test_fuse_per_voxel_empty(self, voxel_fusion):
        """Test fusing per voxel with empty voxel data."""
        empty_voxel_data = MockVoxelData({})
        signals = ["signal1"]

        with pytest.raises(ValueError, match="No valid signals found"):
            voxel_fusion.fuse_per_voxel(empty_voxel_data, signals, fusion_func=lambda x: x[0])

    @pytest.mark.unit
    def test_fuse_per_voxel_with_nan(self, voxel_fusion, mock_voxel_data):
        """Test fusing per voxel with NaN values."""
        # Add signal with NaN
        mock_voxel_data._signals["signal_nan"] = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        signals = ["signal1", "signal_nan"]

        def custom_fusion(values):
            """Custom fusion: mean of values."""
            return np.mean(values)

        fused = voxel_fusion.fuse_per_voxel(mock_voxel_data, signals, fusion_func=custom_fusion)

        assert len(fused) == 5
        # NaN should be skipped

    @pytest.mark.unit
    def test_fuse_voxel_signals_2d(self, voxel_fusion):
        """Test fusing 2D signal arrays."""
        signals_2d = {
            "signal1": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "signal2": np.array([[2.0, 3.0], [4.0, 5.0]]),
        }
        voxel_data_2d = MockVoxelData(signals_2d)

        fused = voxel_fusion.fuse_voxel_signals(
            voxel_data_2d,
            ["signal1", "signal2"],
            fusion_strategy=FusionStrategy.AVERAGE,
        )

        assert fused.shape == (2, 2)

    @pytest.mark.unit
    def test_fuse_voxel_signals_3d(self, voxel_fusion):
        """Test fusing 3D signal arrays."""
        signals_3d = {
            "signal1": np.ones((3, 3, 3)) * 1.0,
            "signal2": np.ones((3, 3, 3)) * 2.0,
        }
        voxel_data_3d = MockVoxelData(signals_3d)

        fused = voxel_fusion.fuse_voxel_signals(
            voxel_data_3d,
            ["signal1", "signal2"],
            fusion_strategy=FusionStrategy.AVERAGE,
        )

        assert fused.shape == (3, 3, 3)
        # Average of 1.0 and 2.0 = 1.5
        assert np.allclose(fused, 1.5)
