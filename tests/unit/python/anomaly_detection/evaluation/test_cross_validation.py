"""
Unit tests for cross-validation utilities.

Tests for CVResult, AnomalyDetectionCV, and convenience functions.
"""

import pytest
import numpy as np
from unittest.mock import Mock
from am_qadf.anomaly_detection.evaluation.cross_validation import (
    CVResult,
    AnomalyDetectionCV,
    k_fold_cv,
    time_series_cv,
    spatial_cv,
)
from am_qadf.anomaly_detection.core.base_detector import (
    BaseAnomalyDetector,
    AnomalyDetectionResult,
)


class MockDetector(BaseAnomalyDetector):
    """Mock detector for testing."""

    def __init__(self, name="MockDetector"):
        super().__init__()
        self.name = name
        self.is_fitted = False

    def fit(self, data, labels=None):
        self.is_fitted = True
        return self

    def predict(self, data):
        results = []
        for i in range(len(data)):
            results.append(
                AnomalyDetectionResult(
                    voxel_index=(i, 0, 0),
                    voxel_coordinates=(0, 0, 0),
                    is_anomaly=np.random.rand() > 0.5,
                    anomaly_score=np.random.rand(),
                    confidence=0.8,
                    detector_name=self.name,
                )
            )
        return results


class TestCVResult:
    """Test suite for CVResult dataclass."""

    @pytest.mark.unit
    def test_cv_result_creation(self):
        """Test creating CVResult."""
        detector = MockDetector()
        result = CVResult(
            fold=0,
            train_indices=np.array([0, 1, 2]),
            test_indices=np.array([3, 4]),
            metrics={"precision": 0.9, "recall": 0.8},
            detector=detector,
        )

        assert result.fold == 0
        assert len(result.train_indices) == 3
        assert len(result.test_indices) == 2
        assert result.metrics["precision"] == 0.9
        assert result.detector == detector


class TestAnomalyDetectionCV:
    """Test suite for AnomalyDetectionCV class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        return np.random.randn(50, 3) * 10 + 100

    @pytest.fixture
    def sample_labels(self):
        """Create sample labels."""
        return np.array([0, 0, 1, 1, 0] * 10)

    @pytest.fixture
    def sample_spatial_coords(self):
        """Create sample spatial coordinates."""
        return np.random.randn(50, 3) * 10

    @pytest.mark.unit
    def test_cv_creation_default(self):
        """Test creating AnomalyDetectionCV with default parameters."""
        cv = AnomalyDetectionCV()

        assert cv.n_splits == 5
        assert cv.random_state is None
        assert cv.results == []

    @pytest.mark.unit
    def test_cv_creation_custom(self):
        """Test creating AnomalyDetectionCV with custom parameters."""
        cv = AnomalyDetectionCV(n_splits=10, random_state=42)

        assert cv.n_splits == 10
        assert cv.random_state == 42

    @pytest.mark.unit
    def test_k_fold_cv(self, sample_data, sample_labels):
        """Test k-fold cross-validation."""
        detector = MockDetector()
        cv = AnomalyDetectionCV(n_splits=3)
        results = cv.k_fold_cv(detector, sample_data, sample_labels)

        assert isinstance(results, list)
        assert len(results) == 3
        assert all(isinstance(r, CVResult) for r in results)
        assert all(r.fold == i for i, r in enumerate(results))
        assert all("precision" in r.metrics for r in results)
        assert all("recall" in r.metrics for r in results)

    @pytest.mark.unit
    def test_k_fold_cv_without_labels(self, sample_data):
        """Test k-fold cross-validation without labels."""
        detector = MockDetector()
        cv = AnomalyDetectionCV(n_splits=3)
        results = cv.k_fold_cv(detector, sample_data)

        assert isinstance(results, list)
        assert len(results) == 3
        assert all(isinstance(r, CVResult) for r in results)
        # Metrics should be empty without labels
        assert all(len(r.metrics) == 0 for r in results)

    @pytest.mark.unit
    def test_k_fold_cv_no_shuffle(self, sample_data, sample_labels):
        """Test k-fold cross-validation without shuffling."""
        detector = MockDetector()
        cv = AnomalyDetectionCV(n_splits=3)
        results = cv.k_fold_cv(detector, sample_data, sample_labels, shuffle=False)

        assert isinstance(results, list)
        assert len(results) == 3

    @pytest.mark.unit
    def test_time_series_cv(self, sample_data, sample_labels):
        """Test time-series cross-validation."""
        detector = MockDetector()
        cv = AnomalyDetectionCV(n_splits=3)
        results = cv.time_series_cv(detector, sample_data, sample_labels)

        assert isinstance(results, list)
        assert len(results) == 3
        assert all(isinstance(r, CVResult) for r in results)
        # Time-series CV should respect temporal ordering
        # Earlier folds should have fewer training samples
        assert len(results[0].train_indices) < len(results[-1].train_indices)

    @pytest.mark.unit
    def test_time_series_cv_without_labels(self, sample_data):
        """Test time-series cross-validation without labels."""
        detector = MockDetector()
        cv = AnomalyDetectionCV(n_splits=3)
        results = cv.time_series_cv(detector, sample_data)

        assert isinstance(results, list)
        assert len(results) == 3

    @pytest.mark.unit
    def test_spatial_cv(self, sample_data, sample_spatial_coords, sample_labels):
        """Test spatial cross-validation."""
        detector = MockDetector()
        cv = AnomalyDetectionCV(n_splits=5)
        results = cv.spatial_cv(detector, sample_data, sample_spatial_coords, sample_labels, n_regions=5)

        assert isinstance(results, list)
        assert len(results) == 5
        assert all(isinstance(r, CVResult) for r in results)
        assert all("precision" in r.metrics for r in results)

    @pytest.mark.unit
    def test_spatial_cv_without_labels(self, sample_data, sample_spatial_coords):
        """Test spatial cross-validation without labels."""
        detector = MockDetector()
        cv = AnomalyDetectionCV(n_splits=5)
        results = cv.spatial_cv(detector, sample_data, sample_spatial_coords, n_regions=5)

        assert isinstance(results, list)
        assert len(results) == 5

    @pytest.mark.unit
    def test_get_summary_metrics(self, sample_data, sample_labels):
        """Test getting summary metrics across CV folds."""
        detector = MockDetector()
        cv = AnomalyDetectionCV(n_splits=3)
        cv.k_fold_cv(detector, sample_data, sample_labels)

        summary = cv.get_summary_metrics()

        assert isinstance(summary, dict)
        assert "precision" in summary
        assert "recall" in summary
        assert "f1_score" in summary
        assert all(isinstance(v, tuple) and len(v) == 2 for v in summary.values())
        # Each tuple should be (mean, std)
        assert all(isinstance(v[0], (int, float)) and isinstance(v[1], (int, float)) for v in summary.values())

    @pytest.mark.unit
    def test_get_summary_metrics_no_results(self):
        """Test getting summary metrics with no results."""
        cv = AnomalyDetectionCV()
        summary = cv.get_summary_metrics()

        assert isinstance(summary, dict)
        assert len(summary) == 0


class TestConvenienceFunctions:
    """Test suite for convenience functions."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        np.random.seed(42)
        return np.random.randn(30, 3) * 10 + 100

    @pytest.fixture
    def sample_labels(self):
        """Create sample labels."""
        return np.array([0, 0, 1, 1, 0] * 6)

    @pytest.mark.unit
    def test_k_fold_cv_function(self, sample_data, sample_labels):
        """Test k_fold_cv convenience function."""
        detector = MockDetector()
        results = k_fold_cv(detector, sample_data, sample_labels, n_splits=3)

        assert isinstance(results, list)
        assert len(results) == 3
        assert all(isinstance(r, CVResult) for r in results)

    @pytest.mark.unit
    def test_time_series_cv_function(self, sample_data, sample_labels):
        """Test time_series_cv convenience function."""
        detector = MockDetector()
        results = time_series_cv(detector, sample_data, sample_labels, n_splits=3)

        assert isinstance(results, list)
        assert len(results) == 3
        assert all(isinstance(r, CVResult) for r in results)

    @pytest.mark.unit
    def test_spatial_cv_function(self, sample_data, sample_labels):
        """Test spatial_cv convenience function."""
        detector = MockDetector()
        spatial_coords = np.random.randn(30, 3) * 10
        results = spatial_cv(detector, sample_data, spatial_coords, sample_labels, n_regions=3)

        assert isinstance(results, list)
        assert len(results) == 3
        assert all(isinstance(r, CVResult) for r in results)
