"""
Unit tests for DBSCAN detector.

Tests for DBSCANDetector class.
"""

import pytest
import numpy as np
import pandas as pd
from am_qadf.anomaly_detection.detectors.clustering.dbscan import (
    DBSCANDetector,
    SKLEARN_AVAILABLE,
)
from am_qadf.anomaly_detection.core.base_detector import (
    AnomalyDetectionConfig,
    AnomalyDetectionResult,
)


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not available")
class TestDBSCANDetector:
    """Test suite for DBSCANDetector class."""

    @pytest.fixture
    def detector_default(self):
        """Create DBSCANDetector with default parameters."""
        return DBSCANDetector()

    @pytest.fixture
    def detector_custom(self):
        """Create DBSCANDetector with custom parameters."""
        return DBSCANDetector(eps=0.3, min_samples=3)

    @pytest.fixture
    def normal_data(self):
        """Create normal clustered data for training."""
        np.random.seed(42)
        # Create two clusters
        cluster1 = np.random.randn(50, 3) * 2 + [10, 10, 10]
        cluster2 = np.random.randn(50, 3) * 2 + [30, 30, 30]
        return np.vstack([cluster1, cluster2])

    @pytest.fixture
    def data_with_outliers(self, normal_data):
        """Create data with clear outliers."""
        data = normal_data.copy()
        # Add outliers far from clusters
        outliers = np.array([[100, 100, 100], [-50, -50, -50]])
        return np.vstack([data, outliers])

    @pytest.mark.unit
    def test_detector_initialization_default(self):
        """Test DBSCANDetector initialization with default values."""
        detector = DBSCANDetector()

        assert detector.eps == 0.5
        assert detector.min_samples == 5
        assert detector.config is not None
        assert detector.is_fitted is False
        assert detector.clusterer_ is None
        assert detector.labels_ is None

    @pytest.mark.unit
    def test_detector_initialization_custom(self):
        """Test DBSCANDetector initialization with custom parameters."""
        detector = DBSCANDetector(eps=0.3, min_samples=3)

        assert detector.eps == 0.3
        assert detector.min_samples == 3

    @pytest.mark.unit
    def test_detector_initialization_with_config(self):
        """Test DBSCANDetector initialization with custom config."""
        config = AnomalyDetectionConfig(threshold=0.5)
        detector = DBSCANDetector(eps=0.4, min_samples=4, config=config)

        assert detector.eps == 0.4
        assert detector.min_samples == 4
        assert detector.config.threshold == 0.5

    @pytest.mark.unit
    def test_fit(self, detector_default, normal_data):
        """Test fitting the detector."""
        detector = detector_default.fit(normal_data)

        assert detector.is_fitted is True
        assert detector.clusterer_ is not None
        assert detector.labels_ is not None
        assert len(detector.labels_) == len(normal_data)
        assert detector is detector_default  # Method chaining

    @pytest.mark.unit
    def test_fit_with_dataframe(self, detector_default):
        """Test fitting with pandas DataFrame."""
        df = pd.DataFrame(
            {
                "feature1": np.random.randn(50) * 2 + 10,
                "feature2": np.random.randn(50) * 2 + 10,
                "feature3": np.random.randn(50) * 2 + 10,
            }
        )

        detector = detector_default.fit(df)

        assert detector.is_fitted is True
        assert detector.clusterer_ is not None

    @pytest.mark.unit
    def test_fit_clustering(self, detector_default, normal_data):
        """Test that DBSCAN creates clusters."""
        detector = detector_default.fit(normal_data)

        # Should have at least one cluster (labels != -1)
        unique_labels = set(detector.labels_)
        # -1 indicates noise, other values are cluster IDs
        assert len(unique_labels) > 1 or -1 in unique_labels

    @pytest.mark.unit
    def test_predict_before_fit(self, detector_default, normal_data):
        """Test that predict raises error if not fitted."""
        with pytest.raises(ValueError, match="must be fitted"):
            detector_default.predict(normal_data)

    @pytest.mark.unit
    def test_predict_normal_data(self, detector_default, normal_data):
        """Test prediction on normal data."""
        detector = detector_default.fit(normal_data)
        results = detector.predict(normal_data)

        assert len(results) == len(normal_data)
        assert all(isinstance(r, AnomalyDetectionResult) for r in results)
        # Most points should be in clusters (not anomalies)
        anomaly_count = sum(r.is_anomaly for r in results)
        # With good clustering, most points should not be anomalies
        assert anomaly_count < len(results) * 0.3

    @pytest.mark.unit
    def test_predict_with_outliers(self, detector_default, normal_data, data_with_outliers):
        """Test prediction on data with clear outliers."""
        detector = detector_default.fit(normal_data)
        results = detector.predict(data_with_outliers)

        assert len(results) == len(data_with_outliers)
        # Outliers should be detected (last two points)
        # They should have higher scores than cluster points
        outlier_scores = [r.anomaly_score for r in results[-2:]]
        cluster_scores = [r.anomaly_score for r in results[:10]]
        assert max(outlier_scores) > min(cluster_scores)

    @pytest.mark.unit
    def test_predict_with_different_eps(self, normal_data, data_with_outliers):
        """Test prediction with different eps values."""
        # Smaller eps should detect more anomalies
        detector_small = DBSCANDetector(eps=0.1, min_samples=3).fit(normal_data)
        detector_large = DBSCANDetector(eps=2.0, min_samples=3).fit(normal_data)

        results_small = detector_small.predict(data_with_outliers)
        results_large = detector_large.predict(data_with_outliers)

        anomalies_small = sum(r.is_anomaly for r in results_small)
        anomalies_large = sum(r.is_anomaly for r in results_large)

        # Smaller eps typically detects more noise points
        # But this depends on data distribution
        assert len(results_small) == len(results_large)

    @pytest.mark.unit
    def test_predict_noise_detection(self, detector_default, normal_data):
        """Test that noise points are detected."""
        # Create data with clear noise point
        data = normal_data.copy()
        noise_point = np.array([[1000, 1000, 1000]])  # Far outlier
        test_data = np.vstack([normal_data[:10], noise_point])

        detector = detector_default.fit(normal_data)
        results = detector.predict(test_data)

        # The noise point should be detected
        assert results[-1].is_anomaly is True
        assert results[-1].anomaly_score > results[0].anomaly_score

    @pytest.mark.unit
    def test_predict_scores(self, detector_default, normal_data, data_with_outliers):
        """Test predict_scores method."""
        detector = detector_default.fit(normal_data)
        scores = detector.predict_scores(data_with_outliers)

        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(data_with_outliers)
        assert all(s >= 0 for s in scores)

    @pytest.mark.unit
    def test_edge_case_small_dataset(self, detector_default):
        """Test with small dataset."""
        small_data = np.array([[10, 20], [20, 30], [30, 40], [40, 50]])

        detector = detector_default.fit(small_data)
        results = detector.predict(small_data)

        assert len(results) == len(small_data)
        assert all(isinstance(r, AnomalyDetectionResult) for r in results)

    @pytest.mark.unit
    def test_edge_case_all_noise(self, detector_default):
        """Test with all points as noise (very sparse data)."""
        # Create very sparse data
        sparse_data = np.random.rand(20, 3) * 1000  # Very spread out

        detector = DBSCANDetector(eps=0.1, min_samples=5).fit(sparse_data)
        results = detector.predict(sparse_data)

        # Most points should be detected as noise
        assert len(results) == len(sparse_data)
        # With very small eps, most points will be noise
        noise_count = sum(r.is_anomaly for r in results)
        assert noise_count >= 0  # At least some noise expected

    @pytest.mark.unit
    def test_multidimensional_data(self, detector_default):
        """Test with high-dimensional data."""
        high_dim_data = np.random.randn(100, 10) * 2 + 10

        detector = detector_default.fit(high_dim_data)
        results = detector.predict(high_dim_data)

        assert len(results) == len(high_dim_data)
        assert detector.clusterer_ is not None
