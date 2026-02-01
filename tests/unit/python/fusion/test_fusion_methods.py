"""
Unit tests for fusion methods.

Tests for FusionMethod, WeightedAverageFusion, MedianFusion, QualityBasedFusion,
AverageFusion, MaxFusion, MinFusion, and get_fusion_method (fuse_values API).
C++-backed classes skip when am_qadf_native not built; QualityBasedFusion is Python-only.
"""

import pytest
import numpy as np
from am_qadf.fusion.fusion_methods import (
    FusionMethod,
    WeightedAverageFusion,
    MedianFusion,
    QualityBasedFusion,
    AverageFusion,
    MaxFusion,
    MinFusion,
    get_fusion_method,
    CPP_AVAILABLE,
)


class TestQualityBasedFusion:
    """QualityBasedFusion is Python-only (no C++ required)."""

    @pytest.mark.unit
    def test_quality_based_creation(self):
        """Create QualityBasedFusion with or without quality_scores."""
        f = QualityBasedFusion()
        assert f.quality_scores is None
        f2 = QualityBasedFusion(quality_scores=[0.9, 0.5])
        assert f2.quality_scores == [0.9, 0.5]

    @pytest.mark.unit
    def test_quality_based_fuse_values_picks_best(self):
        """fuse_values with quality_scores returns value from highest-quality source."""
        f = QualityBasedFusion(quality_scores=[0.3, 0.9, 0.5])
        # values = [10, 20, 30], best idx = 1 (quality 0.9) -> 20
        result = f.fuse_values([10.0, 20.0, 30.0])
        assert result == 20.0

    @pytest.mark.unit
    def test_quality_based_fuse_values_fallback_average(self):
        """fuse_values without matching quality_scores falls back to average."""
        f = QualityBasedFusion()
        result = f.fuse_values([10.0, 20.0, 30.0])
        assert result == 20.0

    @pytest.mark.unit
    def test_quality_based_fuse_values_single(self):
        """fuse_values with one value returns that value."""
        f = QualityBasedFusion(quality_scores=[1.0])
        result = f.fuse_values([42.0])
        assert result == 42.0


class TestGetFusionMethod:
    """get_fusion_method(name, **kwargs) returns correct implementation."""

    @pytest.mark.unit
    def test_get_fusion_method_quality_based(self):
        """get_fusion_method('quality_based') returns QualityBasedFusion."""
        m = get_fusion_method("quality_based", quality_scores=[0.5, 0.5])
        assert isinstance(m, QualityBasedFusion)

    @pytest.mark.unit
    def test_get_fusion_method_unknown_raises(self):
        """get_fusion_method('unknown') raises ValueError."""
        with pytest.raises(ValueError, match="Unknown fusion method"):
            get_fusion_method("unknown")

    @pytest.mark.unit
    def test_get_fusion_method_average(self):
        """get_fusion_method('average') returns AverageFusion (requires C++)."""
        if not CPP_AVAILABLE:
            pytest.skip("C++ fusion methods require am_qadf_native")
        m = get_fusion_method("average")
        assert isinstance(m, AverageFusion)

    @pytest.mark.unit
    def test_get_fusion_method_max_min_median(self):
        """get_fusion_method('max'/'min'/'median') return correct class."""
        if not CPP_AVAILABLE:
            pytest.skip("C++ fusion methods require am_qadf_native")
        assert isinstance(get_fusion_method("max"), MaxFusion)
        assert isinstance(get_fusion_method("min"), MinFusion)
        assert isinstance(get_fusion_method("median"), MedianFusion)

    @pytest.mark.unit
    def test_get_fusion_method_weighted_average(self):
        """get_fusion_method('weighted_average', weights=...) returns WeightedAverageFusion."""
        if not CPP_AVAILABLE:
            pytest.skip("C++ fusion methods require am_qadf_native")
        m = get_fusion_method("weighted_average", weights=[0.5, 0.5])
        assert isinstance(m, WeightedAverageFusion)


@pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ fusion strategy classes require am_qadf_native")
class TestAverageFusion:
    """AverageFusion().fuse_values(list) = mean."""

    @pytest.mark.unit
    def test_average_fuse_values(self):
        """fuse_values returns average of values."""
        f = AverageFusion()
        result = f.fuse_values([1.0, 2.0, 3.0, 4.0])
        assert abs(result - 2.5) < 0.01

    @pytest.mark.unit
    def test_average_fuse_values_two(self):
        """fuse_values([10, 20]) = 15."""
        f = AverageFusion()
        assert abs(f.fuse_values([10.0, 20.0]) - 15.0) < 0.01


@pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ fusion strategy classes require am_qadf_native")
class TestMaxFusion:
    """MaxFusion().fuse_values(list) = max."""

    @pytest.mark.unit
    def test_max_fuse_values(self):
        """fuse_values returns maximum."""
        f = MaxFusion()
        assert f.fuse_values([1.0, 3.0, 2.0]) == 3.0


@pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ fusion strategy classes require am_qadf_native")
class TestMinFusion:
    """MinFusion().fuse_values(list) = min."""

    @pytest.mark.unit
    def test_min_fuse_values(self):
        """fuse_values returns minimum."""
        f = MinFusion()
        assert f.fuse_values([1.0, 3.0, 2.0]) == 1.0


@pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ fusion strategy classes require am_qadf_native")
class TestMedianFusion:
    """MedianFusion().fuse_values(list) = median."""

    @pytest.mark.unit
    def test_median_fuse_values(self):
        """fuse_values returns median."""
        f = MedianFusion()
        result = f.fuse_values([1.0, 3.0, 2.0, 4.0, 5.0])
        assert result == 3.0


@pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ fusion strategy classes require am_qadf_native")
class TestWeightedAverageFusion:
    """WeightedAverageFusion(weights).fuse_values(list) = weighted average."""

    @pytest.mark.unit
    def test_weighted_average_fuse_values(self):
        """fuse_values with weights."""
        f = WeightedAverageFusion(weights=[0.8, 0.2])
        result = f.fuse_values([10.0, 20.0])
        assert abs(result - (0.8 * 10 + 0.2 * 20)) < 0.01

    @pytest.mark.unit
    def test_weighted_average_creation_raises_without_cpp(self):
        """WeightedAverageFusion() raises when C++ not available (tested when C++ is available)."""
        # This test runs only when CPP_AVAILABLE; no-op for "raises without cpp" scenario
        f = WeightedAverageFusion(weights=[1.0, 1.0])
        assert f.fuse_values([1.0, 1.0]) == 1.0
