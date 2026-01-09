"""
Unit tests for data fusion.

Tests for FusionStrategy and DataFusion.
"""

import pytest
import numpy as np
from am_qadf.synchronization.data_fusion import (
    FusionStrategy,
    DataFusion,
)


class TestFusionStrategy:
    """Test suite for FusionStrategy enum."""

    @pytest.mark.unit
    def test_fusion_strategy_values(self):
        """Test FusionStrategy enum values."""
        assert FusionStrategy.AVERAGE.value == "average"
        assert FusionStrategy.WEIGHTED_AVERAGE.value == "weighted_average"
        assert FusionStrategy.MEDIAN.value == "median"
        assert FusionStrategy.MAX.value == "max"
        assert FusionStrategy.MIN.value == "min"
        assert FusionStrategy.FIRST.value == "first"
        assert FusionStrategy.LAST.value == "last"
        assert FusionStrategy.QUALITY_BASED.value == "quality_based"

    @pytest.mark.unit
    def test_fusion_strategy_enumeration(self):
        """Test that FusionStrategy can be enumerated."""
        strategies = list(FusionStrategy)
        assert len(strategies) == 8
        assert FusionStrategy.AVERAGE in strategies


class TestDataFusion:
    """Test suite for DataFusion class."""

    @pytest.fixture
    def data_fusion(self):
        """Create a DataFusion instance."""
        return DataFusion()

    @pytest.mark.unit
    def test_data_fusion_creation_default(self):
        """Test creating DataFusion with default parameters."""
        fusion = DataFusion()

        assert fusion.default_strategy == FusionStrategy.WEIGHTED_AVERAGE
        assert fusion.default_weights == {}

    @pytest.mark.unit
    def test_data_fusion_creation_custom(self):
        """Test creating DataFusion with custom parameters."""
        fusion = DataFusion(
            default_strategy=FusionStrategy.MEDIAN,
            default_weights={"source1": 0.7, "source2": 0.3},
        )

        assert fusion.default_strategy == FusionStrategy.MEDIAN
        assert fusion.default_weights["source1"] == 0.7

    @pytest.mark.unit
    def test_register_source_quality(self, data_fusion):
        """Test registering source quality."""
        data_fusion.register_source_quality("source1", 0.9)
        data_fusion.register_source_quality("source2", 0.5)

        assert data_fusion._source_qualities["source1"] == 0.9
        assert data_fusion._source_qualities["source2"] == 0.5

    @pytest.mark.unit
    def test_register_source_quality_clamp(self, data_fusion):
        """Test that quality scores are clamped to [0, 1]."""
        data_fusion.register_source_quality("source1", 1.5)  # Above 1.0
        data_fusion.register_source_quality("source2", -0.5)  # Below 0.0

        assert data_fusion._source_qualities["source1"] == 1.0
        assert data_fusion._source_qualities["source2"] == 0.0

    @pytest.mark.unit
    def test_compute_weights_equal(self, data_fusion):
        """Test computing equal weights."""
        source_names = ["source1", "source2", "source3"]
        weights = data_fusion.compute_weights(source_names, use_quality=False)

        assert len(weights) == 3
        assert np.allclose(weights, 1.0 / 3.0)
        assert np.allclose(np.sum(weights), 1.0)

    @pytest.mark.unit
    def test_compute_weights_with_quality(self, data_fusion):
        """Test computing weights based on quality."""
        data_fusion.register_source_quality("source1", 0.9)
        data_fusion.register_source_quality("source2", 0.5)
        data_fusion.register_source_quality("source3", 0.1)

        source_names = ["source1", "source2", "source3"]
        weights = data_fusion.compute_weights(source_names, use_quality=True)

        assert len(weights) == 3
        assert np.allclose(np.sum(weights), 1.0)
        # source1 should have highest weight
        assert weights[0] > weights[1] > weights[2]

    @pytest.mark.unit
    def test_fuse_signals_single(self, data_fusion):
        """Test fusing single signal."""
        signals = {"source1": np.array([100.0, 200.0, 300.0])}

        result = data_fusion.fuse_signals(signals)

        assert np.array_equal(result, signals["source1"])

    @pytest.mark.unit
    def test_fuse_signals_empty(self, data_fusion):
        """Test fusing empty signals."""
        signals = {}

        with pytest.raises(ValueError, match="No signals provided"):
            data_fusion.fuse_signals(signals)

    @pytest.mark.unit
    def test_fuse_signals_average(self, data_fusion):
        """Test fusing signals with AVERAGE strategy."""
        signals = {
            "source1": np.array([100.0, 200.0, 300.0]),
            "source2": np.array([150.0, 250.0, 350.0]),
        }

        result = data_fusion.fuse_signals(signals, strategy=FusionStrategy.AVERAGE)

        expected = np.array([125.0, 225.0, 325.0])
        assert np.allclose(result, expected)

    @pytest.mark.unit
    def test_fuse_signals_median(self, data_fusion):
        """Test fusing signals with MEDIAN strategy."""
        signals = {
            "source1": np.array([100.0, 200.0, 300.0]),
            "source2": np.array([150.0, 250.0, 350.0]),
            "source3": np.array([120.0, 220.0, 320.0]),
        }

        result = data_fusion.fuse_signals(signals, strategy=FusionStrategy.MEDIAN)

        expected = np.array([120.0, 220.0, 320.0])
        assert np.allclose(result, expected)

    @pytest.mark.unit
    def test_fuse_signals_max(self, data_fusion):
        """Test fusing signals with MAX strategy."""
        signals = {
            "source1": np.array([100.0, 200.0, 300.0]),
            "source2": np.array([150.0, 250.0, 350.0]),
        }

        result = data_fusion.fuse_signals(signals, strategy=FusionStrategy.MAX)

        expected = np.array([150.0, 250.0, 350.0])
        assert np.allclose(result, expected)

    @pytest.mark.unit
    def test_fuse_signals_min(self, data_fusion):
        """Test fusing signals with MIN strategy."""
        signals = {
            "source1": np.array([100.0, 200.0, 300.0]),
            "source2": np.array([150.0, 250.0, 350.0]),
        }

        result = data_fusion.fuse_signals(signals, strategy=FusionStrategy.MIN)

        expected = np.array([100.0, 200.0, 300.0])
        assert np.allclose(result, expected)

    @pytest.mark.unit
    def test_fuse_signals_weighted_average(self, data_fusion):
        """Test fusing signals with WEIGHTED_AVERAGE strategy."""
        signals = {
            "source1": np.array([100.0, 200.0]),
            "source2": np.array([200.0, 300.0]),
        }
        weights = {"source1": 0.7, "source2": 0.3}

        result = data_fusion.fuse_signals(signals, strategy=FusionStrategy.WEIGHTED_AVERAGE, weights=weights)

        expected = np.array([130.0, 230.0])  # 0.7*100 + 0.3*200, etc.
        assert np.allclose(result, expected)

    @pytest.mark.unit
    def test_fuse_signals_first(self, data_fusion):
        """Test fusing signals with FIRST strategy."""
        signals = {
            "source1": np.array([100.0, 200.0, np.nan]),
            "source2": np.array([150.0, 250.0, 350.0]),
        }

        result = data_fusion.fuse_signals(signals, strategy=FusionStrategy.FIRST)

        # Should use source1, filling NaN with source2
        assert result[0] == 100.0
        assert result[1] == 200.0
        assert result[2] == 350.0  # Filled from source2

    @pytest.mark.unit
    def test_fuse_signals_last(self, data_fusion):
        """Test fusing signals with LAST strategy."""
        signals = {
            "source1": np.array([100.0, 200.0, np.nan]),
            "source2": np.array([150.0, 250.0, 350.0]),
        }

        result = data_fusion.fuse_signals(signals, strategy=FusionStrategy.LAST)

        # Should use source2, filling NaN with source1
        assert result[0] == 100.0  # Filled from source1
        assert result[1] == 250.0
        assert result[2] == 350.0

    @pytest.mark.unit
    def test_fuse_signals_quality_based(self, data_fusion):
        """Test fusing signals with QUALITY_BASED strategy."""
        data_fusion.register_source_quality("source1", 0.3)
        data_fusion.register_source_quality("source2", 0.9)  # Highest quality

        signals = {
            "source1": np.array([100.0, 200.0]),
            "source2": np.array([150.0, 250.0]),
        }

        result = data_fusion.fuse_signals(signals, strategy=FusionStrategy.QUALITY_BASED)

        # Should use source2 (highest quality)
        assert np.array_equal(result, signals["source2"])

    @pytest.mark.unit
    def test_fuse_signals_with_mask(self, data_fusion):
        """Test fusing signals with mask."""
        signals = {
            "source1": np.array([100.0, 200.0, 300.0]),
            "source2": np.array([150.0, 250.0, 350.0]),
        }
        mask = np.array([True, True, False])

        result = data_fusion.fuse_signals(signals, mask=mask)

        # Invalid voxels should be set to 0.0
        assert result[2] == 0.0

    @pytest.mark.unit
    def test_fuse_multiple_signals(self, data_fusion):
        """Test fusing multiple signals from multiple sources."""
        signal_dicts = [
            {"power": np.array([100.0, 200.0]), "speed": np.array([50.0, 100.0])},
            {"power": np.array([150.0, 250.0]), "speed": np.array([75.0, 125.0])},
        ]
        signal_names = ["power", "speed"]

        result = data_fusion.fuse_multiple_signals(signal_dicts, signal_names)

        assert "power" in result
        assert "speed" in result
        assert result["power"].shape == (2,)
        assert result["speed"].shape == (2,)

    @pytest.mark.unit
    def test_handle_conflicts(self, data_fusion):
        """Test handling conflicts between sources."""
        signals = {
            "source1": np.array([100.0, 200.0, 300.0]),
            "source2": np.array([150.0, 250.0, 350.0]),
        }

        fused, conflict_mask = data_fusion.handle_conflicts(signals)

        assert fused.shape == (3,)
        assert conflict_mask.shape == (3,)
        assert conflict_mask.dtype == bool

    @pytest.mark.unit
    def test_handle_conflicts_single_source(self, data_fusion):
        """Test handling conflicts with single source."""
        signals = {"source1": np.array([100.0, 200.0, 300.0])}

        fused, conflict_mask = data_fusion.handle_conflicts(signals)

        assert np.array_equal(fused, signals["source1"])
        assert np.all(~conflict_mask)  # No conflicts with single source

    @pytest.mark.unit
    def test_handle_conflicts_empty(self, data_fusion):
        """Test handling conflicts with empty signals."""
        signals = {}

        fused, conflict_mask = data_fusion.handle_conflicts(signals)

        assert len(fused) == 0
        assert len(conflict_mask) == 0

    @pytest.mark.unit
    def test_compute_fusion_quality(self, data_fusion):
        """Test computing fusion quality metrics."""
        signals = {
            "source1": np.array([100.0, 200.0, 300.0]),
            "source2": np.array([150.0, 250.0, 350.0]),
        }
        fused = np.array([125.0, 225.0, 325.0])

        metrics = data_fusion.compute_fusion_quality(signals, fused)

        assert "coefficient_of_variation" in metrics
        assert "agreement" in metrics
        assert "coverage" in metrics
        assert "mean" in metrics
        assert "std" in metrics

    @pytest.mark.unit
    def test_compute_fusion_quality_single_source(self, data_fusion):
        """Test computing fusion quality with single source."""
        signals = {"source1": np.array([100.0, 200.0, 300.0])}
        fused = np.array([100.0, 200.0, 300.0])

        metrics = data_fusion.compute_fusion_quality(signals, fused)

        assert "coverage" in metrics
        assert "mean" in metrics
        assert "std" in metrics
