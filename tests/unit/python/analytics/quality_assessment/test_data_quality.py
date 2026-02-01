"""
Unit tests for data quality assessment (analytics).

Tests for DataQualityMetrics and DataQualityAnalyzer.
"""

import pytest
import numpy as np
from am_qadf.analytics.quality_assessment.data_quality import (
    DataQualityMetrics,
    DataQualityAnalyzer,
)


class MockVoxelData:
    """Mock voxel data object for testing."""

    def __init__(
        self,
        signals: dict,
        dims: tuple = (10, 10, 10),
        bbox_min: tuple = (0, 0, 0),
        bbox_max: tuple = (10, 10, 10),
        resolution: float = 1.0,
    ):
        """Initialize with signal dictionary."""
        self._signals = signals
        self.dims = dims
        self.bbox_min = bbox_min
        self.bbox_max = bbox_max
        self.resolution = resolution
        self.available_signals = list(signals.keys())

    def get_signal_array(self, signal_name: str, default: float = 0.0) -> np.ndarray:
        """Get signal array by name."""
        return self._signals.get(signal_name, np.full(self.dims, default))


class TestDataQualityMetrics:
    """Test suite for DataQualityMetrics dataclass."""

    @pytest.mark.unit
    def test_metrics_creation(self):
        """Test creating DataQualityMetrics."""
        metrics = DataQualityMetrics(
            completeness=0.9,
            coverage_spatial=0.85,
            coverage_temporal=0.95,
            consistency_score=0.8,
            accuracy_score=0.85,
            reliability_score=0.9,
            filled_voxels=900,
            total_voxels=1000,
            sources_count=3,
            missing_regions=[((0, 0, 0), (1, 1, 1))],
        )

        assert metrics.completeness == 0.9
        assert metrics.coverage_spatial == 0.85
        assert metrics.coverage_temporal == 0.95
        assert metrics.consistency_score == 0.8
        assert metrics.filled_voxels == 900
        assert metrics.total_voxels == 1000
        assert metrics.sources_count == 3

    @pytest.mark.unit
    def test_metrics_to_dict(self):
        """Test converting DataQualityMetrics to dictionary."""
        metrics = DataQualityMetrics(
            completeness=0.9,
            coverage_spatial=0.85,
            coverage_temporal=0.95,
            consistency_score=0.8,
            accuracy_score=0.85,
            reliability_score=0.9,
            filled_voxels=900,
            total_voxels=1000,
            sources_count=3,
            missing_regions=[((0, 0, 0), (1, 1, 1))],
        )

        result = metrics.to_dict()

        assert isinstance(result, dict)
        assert result["completeness"] == 0.9
        assert result["filled_voxels"] == 900
        assert result["missing_regions_count"] == 1


class TestDataQualityAnalyzer:
    """Test suite for DataQualityAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create a DataQualityAnalyzer instance."""
        return DataQualityAnalyzer()

    @pytest.mark.unit
    def test_analyzer_creation(self, analyzer):
        """Test creating DataQualityAnalyzer."""
        assert analyzer is not None

    @pytest.mark.unit
    def test_calculate_completeness(self, analyzer):
        """Test calculating completeness."""
        signals = {"signal1": np.ones((10, 10, 10)), "signal2": np.ones((10, 10, 10))}
        voxel_data = MockVoxelData(signals, dims=(10, 10, 10))

        completeness = analyzer.calculate_completeness(voxel_data, ["signal1", "signal2"])

        assert 0.0 <= completeness <= 1.0
        assert completeness > 0

    @pytest.mark.unit
    def test_calculate_spatial_coverage(self, analyzer):
        """Test calculating spatial coverage."""
        signals = {"signal1": np.ones((10, 10, 10)), "signal2": np.ones((10, 10, 10))}
        voxel_data = MockVoxelData(signals, dims=(10, 10, 10))

        coverage = analyzer.calculate_spatial_coverage(voxel_data, ["signal1", "signal2"])

        assert 0.0 <= coverage <= 1.0
        assert coverage > 0

    @pytest.mark.unit
    def test_calculate_temporal_coverage(self, analyzer):
        """Test calculating temporal coverage."""
        signals = {"signal1": np.ones((10, 10, 10)), "signal2": np.ones((10, 10, 10))}
        voxel_data = MockVoxelData(signals, dims=(10, 10, 10))

        coverage = analyzer.calculate_temporal_coverage(voxel_data)

        assert 0.0 <= coverage <= 1.0
        assert coverage > 0

    @pytest.mark.unit
    def test_calculate_consistency(self, analyzer):
        """Test calculating consistency."""
        signals = {
            "signal1": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            "signal2": np.array([1.1, 2.1, 3.1, 4.1, 5.1]),  # Similar to signal1
        }
        voxel_data = MockVoxelData(signals, dims=(5, 1, 1))

        consistency = analyzer.calculate_consistency(voxel_data, ["signal1", "signal2"])

        assert 0.0 <= consistency <= 1.0
        # Similar signals should have high consistency
        assert consistency >= 0.5

    @pytest.mark.unit
    def test_assess_quality(self, analyzer):
        """Test assessing overall data quality."""
        signals = {"signal1": np.ones((10, 10, 10)), "signal2": np.ones((10, 10, 10))}
        voxel_data = MockVoxelData(signals, dims=(10, 10, 10))

        metrics = analyzer.assess_quality(voxel_data, ["signal1", "signal2"])

        assert isinstance(metrics, DataQualityMetrics)
        assert 0.0 <= metrics.completeness <= 1.0
        assert 0.0 <= metrics.coverage_spatial <= 1.0
        assert 0.0 <= metrics.coverage_temporal <= 1.0
        assert 0.0 <= metrics.consistency_score <= 1.0
        assert metrics.filled_voxels > 0
        assert metrics.total_voxels == 1000
        assert metrics.sources_count == 2
