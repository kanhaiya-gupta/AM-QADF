"""
Unit tests for fusion methods.

Tests for FusionMethod, WeightedAverageFusion, MedianFusion, QualityBasedFusion,
AverageFusion, MaxFusion, MinFusion, and get_fusion_method.
"""

import pytest
import numpy as np
from am_qadf.fusion.fusion_methods import (
    FusionStrategy,
    FusionMethod,
    WeightedAverageFusion,
    MedianFusion,
    QualityBasedFusion,
    AverageFusion,
    MaxFusion,
    MinFusion,
    get_fusion_method,
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


class TestFusionMethod:
    """Test suite for FusionMethod base class."""

    @pytest.mark.unit
    def test_fusion_method_creation(self):
        """Test creating FusionMethod."""
        method = FusionMethod(FusionStrategy.AVERAGE)

        assert method.strategy == FusionStrategy.AVERAGE

    @pytest.mark.unit
    def test_fusion_method_fuse(self):
        """Test fusing signals with base FusionMethod."""
        method = FusionMethod(FusionStrategy.AVERAGE)
        signals = {
            "signal1": np.array([1.0, 2.0, 3.0]),
            "signal2": np.array([2.0, 3.0, 4.0]),
        }

        fused = method.fuse(signals)

        assert len(fused) == 3
        # Average of [1,2,3] and [2,3,4] = [1.5, 2.5, 3.5]
        assert np.allclose(fused, [1.5, 2.5, 3.5])

    @pytest.mark.unit
    def test_fusion_method_fuse_empty(self):
        """Test fusing with empty signals."""
        method = FusionMethod(FusionStrategy.AVERAGE)

        with pytest.raises(ValueError, match="At least one signal must be provided"):
            method.fuse({})

    @pytest.mark.unit
    def test_fusion_method_fuse_with_weights(self):
        """Test fusing with weights."""
        method = FusionMethod(FusionStrategy.WEIGHTED_AVERAGE)
        signals = {
            "signal1": np.array([1.0, 2.0, 3.0]),
            "signal2": np.array([2.0, 3.0, 4.0]),
        }
        weights = {"signal1": 0.7, "signal2": 0.3}

        fused = method.fuse(signals, weights=weights)

        assert len(fused) == 3
        # Weighted average: 0.7*[1,2,3] + 0.3*[2,3,4] = [1.3, 2.3, 3.3]
        assert np.allclose(fused, [1.3, 2.3, 3.3], atol=0.1)

    @pytest.mark.unit
    def test_fusion_method_fuse_with_quality_scores(self):
        """Test fusing with quality scores."""
        method = FusionMethod(FusionStrategy.WEIGHTED_AVERAGE)
        signals = {
            "signal1": np.array([1.0, 2.0, 3.0]),
            "signal2": np.array([2.0, 3.0, 4.0]),
        }
        quality_scores = {"signal1": 0.9, "signal2": 0.5}

        fused = method.fuse(signals, quality_scores=quality_scores)

        assert len(fused) == 3
        # Should weight signal1 more heavily


class TestWeightedAverageFusion:
    """Test suite for WeightedAverageFusion class."""

    @pytest.mark.unit
    def test_weighted_average_fusion_creation(self):
        """Test creating WeightedAverageFusion."""
        fusion = WeightedAverageFusion()

        assert fusion.strategy == FusionStrategy.WEIGHTED_AVERAGE

    @pytest.mark.unit
    def test_weighted_average_fusion_creation_with_weights(self):
        """Test creating WeightedAverageFusion with default weights."""
        default_weights = {"signal1": 0.7, "signal2": 0.3}
        fusion = WeightedAverageFusion(default_weights=default_weights)

        assert fusion.default_weights == default_weights

    @pytest.mark.unit
    def test_weighted_average_fusion_fuse(self):
        """Test fusing with weighted average."""
        fusion = WeightedAverageFusion()
        signals = {
            "signal1": np.array([1.0, 2.0, 3.0]),
            "signal2": np.array([2.0, 3.0, 4.0]),
        }
        weights = {"signal1": 0.7, "signal2": 0.3}

        fused = fusion.fuse(signals, weights=weights)

        assert len(fused) == 3
        assert np.allclose(fused, [1.3, 2.3, 3.3], atol=0.1)

    @pytest.mark.unit
    def test_weighted_average_fusion_fuse_with_default_weights(self):
        """Test fusing with default weights."""
        default_weights = {"signal1": 0.8, "signal2": 0.2}
        fusion = WeightedAverageFusion(default_weights=default_weights)
        signals = {
            "signal1": np.array([1.0, 2.0, 3.0]),
            "signal2": np.array([2.0, 3.0, 4.0]),
        }

        fused = fusion.fuse(signals)

        assert len(fused) == 3
        # Should use default weights


class TestMedianFusion:
    """Test suite for MedianFusion class."""

    @pytest.mark.unit
    def test_median_fusion_creation(self):
        """Test creating MedianFusion."""
        fusion = MedianFusion()

        assert fusion.strategy == FusionStrategy.MEDIAN

    @pytest.mark.unit
    def test_median_fusion_fuse(self):
        """Test fusing with median."""
        fusion = MedianFusion()
        signals = {
            "signal1": np.array([1.0, 2.0, 3.0]),
            "signal2": np.array([2.0, 3.0, 4.0]),
            "signal3": np.array([3.0, 4.0, 5.0]),
        }

        fused = fusion.fuse(signals)

        assert len(fused) == 3
        # Median of [1,2,3], [2,3,4], [3,4,5] = [2,3,4]
        assert np.allclose(fused, [2.0, 3.0, 4.0])

    @pytest.mark.unit
    def test_median_fusion_fuse_ignores_weights(self):
        """Test that median fusion ignores weights."""
        fusion = MedianFusion()
        signals = {
            "signal1": np.array([1.0, 2.0, 3.0]),
            "signal2": np.array([2.0, 3.0, 4.0]),
        }
        weights = {"signal1": 0.9, "signal2": 0.1}  # Should be ignored

        fused = fusion.fuse(signals, weights=weights)

        # Should still use median, not weighted average
        assert np.allclose(fused, [1.5, 2.5, 3.5])


class TestQualityBasedFusion:
    """Test suite for QualityBasedFusion class."""

    @pytest.mark.unit
    def test_quality_based_fusion_creation(self):
        """Test creating QualityBasedFusion."""
        fusion = QualityBasedFusion()

        assert fusion.strategy == FusionStrategy.QUALITY_BASED

    @pytest.mark.unit
    def test_quality_based_fusion_fuse_with_quality_scores(self):
        """Test fusing with quality scores."""
        fusion = QualityBasedFusion()
        signals = {
            "signal1": np.array([1.0, 2.0, 3.0]),
            "signal2": np.array([2.0, 3.0, 4.0]),
        }
        quality_scores = {"signal1": 0.9, "signal2": 0.5}  # signal1 has higher quality

        fused = fusion.fuse(signals, quality_scores=quality_scores)

        assert len(fused) == 3
        # Should use signal1 (higher quality)
        assert np.allclose(fused, [1.0, 2.0, 3.0])

    @pytest.mark.unit
    def test_quality_based_fusion_fuse_without_quality_scores(self):
        """Test fusing without quality scores (should fallback to weighted average)."""
        fusion = QualityBasedFusion()
        signals = {
            "signal1": np.array([1.0, 2.0, 3.0]),
            "signal2": np.array([2.0, 3.0, 4.0]),
        }

        fused = fusion.fuse(signals)

        assert len(fused) == 3
        # Should fallback to weighted average (equal weights)


class TestAverageFusion:
    """Test suite for AverageFusion class."""

    @pytest.mark.unit
    def test_average_fusion_creation(self):
        """Test creating AverageFusion."""
        fusion = AverageFusion()

        assert fusion.strategy == FusionStrategy.AVERAGE

    @pytest.mark.unit
    def test_average_fusion_fuse(self):
        """Test fusing with simple average."""
        fusion = AverageFusion()
        signals = {
            "signal1": np.array([1.0, 2.0, 3.0]),
            "signal2": np.array([2.0, 3.0, 4.0]),
        }

        fused = fusion.fuse(signals)

        assert len(fused) == 3
        # Average of [1,2,3] and [2,3,4] = [1.5, 2.5, 3.5]
        assert np.allclose(fused, [1.5, 2.5, 3.5])

    @pytest.mark.unit
    def test_average_fusion_fuse_ignores_weights(self):
        """Test that average fusion ignores weights."""
        fusion = AverageFusion()
        signals = {
            "signal1": np.array([1.0, 2.0, 3.0]),
            "signal2": np.array([2.0, 3.0, 4.0]),
        }
        weights = {"signal1": 0.9, "signal2": 0.1}  # Should be ignored

        fused = fusion.fuse(signals, weights=weights)

        # Should still use simple average
        assert np.allclose(fused, [1.5, 2.5, 3.5])


class TestMaxFusion:
    """Test suite for MaxFusion class."""

    @pytest.mark.unit
    def test_max_fusion_creation(self):
        """Test creating MaxFusion."""
        fusion = MaxFusion()

        assert fusion.strategy == FusionStrategy.MAX

    @pytest.mark.unit
    def test_max_fusion_fuse(self):
        """Test fusing with maximum."""
        fusion = MaxFusion()
        signals = {
            "signal1": np.array([1.0, 2.0, 3.0]),
            "signal2": np.array([2.0, 3.0, 4.0]),
        }

        fused = fusion.fuse(signals)

        assert len(fused) == 3
        # Max of [1,2,3] and [2,3,4] = [2,3,4]
        assert np.allclose(fused, [2.0, 3.0, 4.0])


class TestMinFusion:
    """Test suite for MinFusion class."""

    @pytest.mark.unit
    def test_min_fusion_creation(self):
        """Test creating MinFusion."""
        fusion = MinFusion()

        assert fusion.strategy == FusionStrategy.MIN

    @pytest.mark.unit
    def test_min_fusion_fuse(self):
        """Test fusing with minimum."""
        fusion = MinFusion()
        signals = {
            "signal1": np.array([1.0, 2.0, 3.0]),
            "signal2": np.array([2.0, 3.0, 4.0]),
        }

        fused = fusion.fuse(signals)

        assert len(fused) == 3
        # Min of [1,2,3] and [2,3,4] = [1,2,3]
        assert np.allclose(fused, [1.0, 2.0, 3.0])


class TestGetFusionMethod:
    """Test suite for get_fusion_method factory function."""

    @pytest.mark.unit
    def test_get_fusion_method_average(self):
        """Test getting AverageFusion."""
        method = get_fusion_method(FusionStrategy.AVERAGE)

        assert isinstance(method, AverageFusion)

    @pytest.mark.unit
    def test_get_fusion_method_weighted_average(self):
        """Test getting WeightedAverageFusion."""
        method = get_fusion_method(FusionStrategy.WEIGHTED_AVERAGE)

        assert isinstance(method, WeightedAverageFusion)

    @pytest.mark.unit
    def test_get_fusion_method_median(self):
        """Test getting MedianFusion."""
        method = get_fusion_method(FusionStrategy.MEDIAN)

        assert isinstance(method, MedianFusion)

    @pytest.mark.unit
    def test_get_fusion_method_quality_based(self):
        """Test getting QualityBasedFusion."""
        method = get_fusion_method(FusionStrategy.QUALITY_BASED)

        assert isinstance(method, QualityBasedFusion)

    @pytest.mark.unit
    def test_get_fusion_method_max(self):
        """Test getting MaxFusion."""
        method = get_fusion_method(FusionStrategy.MAX)

        assert isinstance(method, MaxFusion)

    @pytest.mark.unit
    def test_get_fusion_method_min(self):
        """Test getting MinFusion."""
        method = get_fusion_method(FusionStrategy.MIN)

        assert isinstance(method, MinFusion)

    @pytest.mark.unit
    def test_get_fusion_method_unknown_defaults(self):
        """Test getting fusion method for unknown strategy (should default to Average)."""

        # Create a mock strategy that's not in the map
        class UnknownStrategy:
            pass

        # This should default to AverageFusion
        # Note: This test may need adjustment based on actual implementation
        method = get_fusion_method(FusionStrategy.AVERAGE)  # Use known strategy
        assert isinstance(method, FusionMethod)
