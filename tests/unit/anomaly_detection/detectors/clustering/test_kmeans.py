"""
Unit tests for K-Means detector.

Tests for KMeansDetector class.
"""

import pytest
import numpy as np
import pandas as pd
from am_qadf.anomaly_detection.detectors.clustering.kmeans import (
    KMeansDetector,
    SKLEARN_AVAILABLE,
)
from am_qadf.anomaly_detection.core.base_detector import (
    AnomalyDetectionConfig,
    AnomalyDetectionResult,
)


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not available")
class TestKMeansDetector:
    """Test suite for KMeansDetector class."""

    @pytest.fixture
    def detector_default(self):
        """Create KMeansDetector with default parameters."""
        return KMeansDetector()

    @pytest.fixture
    def detector_custom(self):
        """Create KMeansDetector with custom parameters."""
        return KMeansDetector(n_clusters=3, distance_threshold=1.5, random_state=42)

    @pytest.fixture
    def normal_data(self):
        """Create normal clustered data for training."""
        np.random.seed(42)
        # Create three clusters
        cluster1 = np.random.randn(30, 3) * 2 + [10, 10, 10]
        cluster2 = np.random.randn(30, 3) * 2 + [30, 30, 30]
        cluster3 = np.random.randn(30, 3) * 2 + [50, 50, 50]
        return np.vstack([cluster1, cluster2, cluster3])

    @pytest.fixture
    def data_with_outliers(self, normal_data):
        """Create data with clear outliers."""
        data = normal_data.copy()
        # Add outliers far from cluster centers
        outliers = np.array([[100, 100, 100], [-50, -50, -50]])
        return np.vstack([data, outliers])

    @pytest.mark.unit
    def test_detector_initialization_default(self):
        """Test KMeansDetector initialization with default values."""
        detector = KMeansDetector()

        assert detector.n_clusters == 5
        assert detector.distance_threshold == 2.0
        assert detector.random_state is None
        assert detector.config is not None
        assert detector.is_fitted is False
        assert detector.kmeans_ is None
        assert detector.cluster_centers_ is None
        assert detector.cluster_stds_ is None

    @pytest.mark.unit
    def test_detector_initialization_custom(self):
        """Test KMeansDetector initialization with custom parameters."""
        detector = KMeansDetector(n_clusters=3, distance_threshold=1.5, random_state=42)

        assert detector.n_clusters == 3
        assert detector.distance_threshold == 1.5
        assert detector.random_state == 42

    @pytest.mark.unit
    def test_detector_initialization_with_config(self):
        """Test KMeansDetector initialization with custom config."""
        config = AnomalyDetectionConfig(threshold=0.5)
        detector = KMeansDetector(n_clusters=4, config=config)

        assert detector.n_clusters == 4
        assert detector.config.threshold == 0.5

    @pytest.mark.unit
    def test_fit(self, detector_default, normal_data):
        """Test fitting the detector."""
        detector = detector_default.fit(normal_data)

        assert detector.is_fitted is True
        assert detector.kmeans_ is not None
        assert detector.cluster_centers_ is not None
        assert detector.cluster_stds_ is not None
        assert len(detector.cluster_centers_) == detector.n_clusters
        assert len(detector.cluster_stds_) == detector.n_clusters
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
        assert detector.cluster_centers_ is not None

    @pytest.mark.unit
    def test_fit_adjusts_n_clusters(self, normal_data):
        """Test that n_clusters is adjusted if data is smaller."""
        # Create small dataset
        small_data = normal_data[:5]
        detector = KMeansDetector(n_clusters=10)  # More clusters than data points

        detector.fit(small_data)

        # Should adjust to min(n_clusters, len(data))
        assert len(detector.cluster_centers_) <= len(small_data)

    @pytest.mark.unit
    def test_fit_cluster_centers(self, detector_default, normal_data):
        """Test that cluster centers are calculated."""
        detector = detector_default.fit(normal_data)

        # Cluster centers should be in reasonable range
        assert detector.cluster_centers_.shape[1] == normal_data.shape[1]
        # Centers should be within data range
        assert np.all(detector.cluster_centers_ >= normal_data.min() - 10)
        assert np.all(detector.cluster_centers_ <= normal_data.max() + 10)

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
        # Most points should be close to cluster centers (not anomalies)
        anomaly_count = sum(r.is_anomaly for r in results)
        # Allow up to 35% anomalies due to distance-based scoring
        assert anomaly_count < len(results) * 0.35

    @pytest.mark.unit
    def test_predict_with_outliers(self, detector_default, normal_data, data_with_outliers):
        """Test prediction on data with clear outliers."""
        detector = detector_default.fit(normal_data)
        results = detector.predict(data_with_outliers)

        assert len(results) == len(data_with_outliers)
        # Outliers should have higher scores
        outlier_scores = [r.anomaly_score for r in results[-2:]]
        cluster_scores = [r.anomaly_score for r in results[:10]]
        assert max(outlier_scores) > min(cluster_scores)
        # Outliers should be detected as anomalies
        assert results[-1].is_anomaly is True or results[-2].is_anomaly is True

    @pytest.mark.unit
    def test_predict_distance_calculation(self, detector_default, normal_data):
        """Test that distances to cluster centers are calculated."""
        detector = detector_default.fit(normal_data)
        test_point = normal_data[0]
        results = detector.predict(normal_data[:1])

        # Score should be based on distance to nearest cluster center
        distances = [np.linalg.norm(test_point - center) for center in detector.cluster_centers_]
        min_distance = min(distances)

        # Score should be related to distance
        assert results[0].anomaly_score >= 0
        # Points close to centers should have lower scores
        if min_distance < 5:  # Close to a center
            assert results[0].anomaly_score < 0.5

    @pytest.mark.unit
    def test_predict_with_different_threshold(self, normal_data, data_with_outliers):
        """Test prediction with different distance thresholds."""
        # Lower threshold should detect more anomalies
        detector_low = KMeansDetector(distance_threshold=1.0, random_state=42).fit(normal_data)
        detector_high = KMeansDetector(distance_threshold=5.0, random_state=42).fit(normal_data)

        results_low = detector_low.predict(data_with_outliers)
        results_high = detector_high.predict(data_with_outliers)

        anomalies_low = sum(r.is_anomaly for r in results_low)
        anomalies_high = sum(r.is_anomaly for r in results_high)

        assert anomalies_low >= anomalies_high

    @pytest.mark.unit
    def test_predict_with_different_n_clusters(self, normal_data, data_with_outliers):
        """Test prediction with different number of clusters."""
        detector_small = KMeansDetector(n_clusters=2, random_state=42).fit(normal_data)
        detector_large = KMeansDetector(n_clusters=10, random_state=42).fit(normal_data)

        results_small = detector_small.predict(data_with_outliers)
        results_large = detector_large.predict(data_with_outliers)

        # Both should work, but may produce different results
        assert len(results_small) == len(results_large)

    @pytest.mark.unit
    def test_predict_scores(self, detector_default, normal_data, data_with_outliers):
        """Test predict_scores method."""
        detector = detector_default.fit(normal_data)
        scores = detector.predict_scores(data_with_outliers)

        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(data_with_outliers)
        assert all(s >= 0 for s in scores)
        assert all(s <= 1 for s in scores)  # Normalized
        # Outliers should have higher scores
        assert scores[-1] > scores[5]
        assert scores[-2] > scores[5]

    @pytest.mark.unit
    def test_predict_score_normalization(self, detector_default, normal_data):
        """Test that scores are normalized to [0, 1]."""
        detector = detector_default.fit(normal_data)
        results = detector.predict(normal_data)

        scores = [r.anomaly_score for r in results]
        assert all(0 <= s <= 1 for s in scores)
        assert min(scores) >= 0
        assert max(scores) <= 1

    @pytest.mark.unit
    def test_edge_case_small_dataset(self, detector_default):
        """Test with small dataset."""
        small_data = np.array([[10, 20], [20, 30], [30, 40], [40, 50]])

        detector = detector_default.fit(small_data)
        results = detector.predict(small_data)

        assert len(results) == len(small_data)
        assert all(isinstance(r, AnomalyDetectionResult) for r in results)

    @pytest.mark.unit
    def test_edge_case_single_cluster(self, detector_default):
        """Test with single cluster (n_clusters=1)."""
        data = np.random.randn(50, 3) * 2 + 10

        detector = KMeansDetector(n_clusters=1, random_state=42).fit(data)
        results = detector.predict(data)

        assert len(results) == len(data)
        assert len(detector.cluster_centers_) == 1

    @pytest.mark.unit
    def test_multidimensional_data(self, detector_default):
        """Test with high-dimensional data."""
        high_dim_data = np.random.randn(100, 10) * 2 + 10

        detector = detector_default.fit(high_dim_data)
        results = detector.predict(high_dim_data)

        assert len(results) == len(high_dim_data)
        assert detector.cluster_centers_.shape[1] == 10
