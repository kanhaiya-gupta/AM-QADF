"""
Unit tests for completeness assessment (analytics).

Tests for GapFillingStrategy, CompletenessMetrics, and CompletenessAnalyzer.
"""

import pytest
import numpy as np
from am_qadf.analytics.quality_assessment.completeness import (
    GapFillingStrategy,
    CompletenessMetrics,
    CompletenessAnalyzer,
)


class MockVoxelData:
    """Mock voxel data object for testing."""

    def __init__(
        self,
        signals: dict,
        dims: tuple = (10, 10, 10),
        bbox_min: tuple = (0, 0, 0),
        resolution: float = 1.0,
    ):
        """Initialize with signal dictionary."""
        self._signals = signals
        self.dims = dims
        self.bbox_min = bbox_min
        self.resolution = resolution
        self.available_signals = list(signals.keys())

    def get_signal_array(self, signal_name: str, default: float = 0.0) -> np.ndarray:
        """Get signal array by name."""
        return self._signals.get(signal_name, np.full(self.dims, default))


class TestGapFillingStrategy:
    """Test suite for GapFillingStrategy enum."""

    @pytest.mark.unit
    def test_gap_filling_strategy_values(self):
        """Test GapFillingStrategy enum values."""
        assert GapFillingStrategy.NONE.value == "none"
        assert GapFillingStrategy.ZERO.value == "zero"
        assert GapFillingStrategy.NEAREST.value == "nearest"
        assert GapFillingStrategy.LINEAR.value == "linear"
        assert GapFillingStrategy.MEAN.value == "mean"
        assert GapFillingStrategy.MEDIAN.value == "median"

    @pytest.mark.unit
    def test_gap_filling_strategy_enumeration(self):
        """Test that GapFillingStrategy can be enumerated."""
        strategies = list(GapFillingStrategy)
        assert len(strategies) == 6


class TestCompletenessMetrics:
    """Test suite for CompletenessMetrics dataclass."""

    @pytest.mark.unit
    def test_completeness_metrics_creation(self):
        """Test creating CompletenessMetrics."""
        metrics = CompletenessMetrics(
            completeness_ratio=0.9,
            spatial_coverage=0.85,
            temporal_coverage=0.95,
            missing_voxels_count=100,
            missing_regions_count=5,
            gap_fillable_ratio=0.8,
            missing_voxel_indices=np.array([1, 2, 3]),
            missing_regions=[((0, 0, 0), (1, 1, 1))],
        )

        assert metrics.completeness_ratio == 0.9
        assert metrics.spatial_coverage == 0.85
        assert metrics.temporal_coverage == 0.95
        assert metrics.missing_voxels_count == 100
        assert metrics.missing_regions_count == 5
        assert metrics.gap_fillable_ratio == 0.8

    @pytest.mark.unit
    def test_completeness_metrics_to_dict(self):
        """Test converting CompletenessMetrics to dictionary."""
        metrics = CompletenessMetrics(
            completeness_ratio=0.9,
            spatial_coverage=0.85,
            temporal_coverage=0.95,
            missing_voxels_count=100,
            missing_regions_count=5,
            gap_fillable_ratio=0.8,
            missing_voxel_indices=np.array([1, 2, 3]),
            missing_regions=[((0, 0, 0), (1, 1, 1))],
        )

        result = metrics.to_dict()

        assert isinstance(result, dict)
        assert result["completeness_ratio"] == 0.9
        assert result["missing_voxels_count"] == 100
        assert "missing_voxel_indices_count" in result
        assert result["missing_voxel_indices_count"] == 3


class TestCompletenessAnalyzer:
    """Test suite for CompletenessAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create a CompletenessAnalyzer instance."""
        return CompletenessAnalyzer()

    @pytest.mark.unit
    def test_completeness_analyzer_creation(self, analyzer):
        """Test creating CompletenessAnalyzer."""
        assert analyzer is not None

    @pytest.mark.unit
    def test_detect_missing_data(self, analyzer):
        """Test detecting missing data."""
        signal_array = np.array([1.0, 2.0, np.nan, 0.0, 5.0])

        missing_count, missing_indices = analyzer.detect_missing_data(signal_array)

        assert missing_count == 2  # NaN and 0.0
        assert missing_indices is not None

    @pytest.mark.unit
    def test_detect_missing_data_no_store_indices(self, analyzer):
        """Test detecting missing data without storing indices."""
        signal_array = np.array([1.0, 2.0, np.nan, 0.0, 5.0])

        missing_count, missing_indices = analyzer.detect_missing_data(signal_array, store_indices=False)

        assert missing_count == 2
        assert missing_indices is None

    @pytest.mark.unit
    def test_analyze_coverage(self, analyzer):
        """Test analyzing spatial and temporal coverage."""
        signals = {"signal1": np.ones((10, 10, 10)), "signal2": np.ones((10, 10, 10))}
        voxel_data = MockVoxelData(signals, dims=(10, 10, 10))

        spatial_coverage, temporal_coverage = analyzer.analyze_coverage(voxel_data, ["signal1", "signal2"])

        assert 0.0 <= spatial_coverage <= 1.0
        assert 0.0 <= temporal_coverage <= 1.0

    @pytest.mark.unit
    def test_fill_gaps(self, analyzer):
        """Test filling gaps."""
        signal_array = np.array([1.0, 2.0, np.nan, 0.0, 5.0])

        filled = analyzer.fill_gaps(signal_array, strategy=GapFillingStrategy.MEAN)

        assert isinstance(filled, np.ndarray)
        assert not np.isnan(filled).any()

    @pytest.mark.unit
    def test_assess_completeness(self, analyzer):
        """Test assessing overall completeness."""
        signals = {"signal1": np.ones((10, 10, 10)), "signal2": np.ones((10, 10, 10))}
        voxel_data = MockVoxelData(signals, dims=(10, 10, 10))

        metrics = analyzer.assess_completeness(voxel_data, ["signal1", "signal2"])

        assert isinstance(metrics, CompletenessMetrics)
        assert 0.0 <= metrics.completeness_ratio <= 1.0
        assert 0.0 <= metrics.spatial_coverage <= 1.0
        assert 0.0 <= metrics.temporal_coverage <= 1.0
