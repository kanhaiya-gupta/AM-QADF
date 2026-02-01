"""
Unit tests for Mahalanobis detector.

Tests for MahalanobisDetector class.
"""

import pytest
import numpy as np
import pandas as pd
from am_qadf.anomaly_detection.detectors.statistical.mahalanobis import (
    MahalanobisDetector,
)
from am_qadf.anomaly_detection.core.base_detector import (
    AnomalyDetectionConfig,
    AnomalyDetectionResult,
)


class TestMahalanobisDetector:
    """Test suite for MahalanobisDetector class."""

    @pytest.fixture
    def detector_default(self):
        """Create MahalanobisDetector with default parameters."""
        return MahalanobisDetector()

    @pytest.fixture
    def detector_custom(self):
        """Create MahalanobisDetector with custom threshold."""
        return MahalanobisDetector(threshold=2.5)

    @pytest.fixture
    def normal_data(self):
        """Create normal multivariate data for training."""
        np.random.seed(42)
        # Create correlated data
        mean = [100, 50, 10]
        cov = [[10, 3, 1], [3, 5, 0.5], [1, 0.5, 2]]
        return np.random.multivariate_normal(mean, cov, 100)

    @pytest.fixture
    def data_with_outliers(self, normal_data):
        """Create data with clear outliers."""
        data = normal_data.copy()
        # Add outliers far from the distribution
        data[0] = [200, 200, 200]  # High outlier
        data[1] = [0, 0, 0]  # Low outlier
        return data

    @pytest.mark.unit
    def test_detector_initialization_default(self):
        """Test MahalanobisDetector initialization with default values."""
        detector = MahalanobisDetector()

        assert detector.threshold == 3.0
        assert detector.config is not None
        assert detector.config.threshold == 3.0
        assert detector.is_fitted is False
        assert detector.mean_ is None
        assert detector.cov_ is None
        assert detector.inv_cov_ is None

    @pytest.mark.unit
    def test_detector_initialization_custom(self):
        """Test MahalanobisDetector initialization with custom threshold."""
        detector = MahalanobisDetector(threshold=2.5)

        assert detector.threshold == 2.5
        assert detector.config.threshold == 2.5

    @pytest.mark.unit
    def test_detector_initialization_with_config(self):
        """Test MahalanobisDetector initialization with custom config."""
        config = AnomalyDetectionConfig(threshold=0.5)
        detector = MahalanobisDetector(threshold=2.0, config=config)

        assert detector.threshold == 2.0
        assert detector.config.threshold == 2.0

    @pytest.mark.unit
    def test_fit(self, detector_default, normal_data):
        """Test fitting the detector."""
        detector = detector_default.fit(normal_data)

        assert detector.is_fitted is True
        assert detector.mean_ is not None
        assert detector.cov_ is not None
        assert detector.inv_cov_ is not None
        assert detector.mean_.shape == (3,)
        assert detector.cov_.shape == (3, 3)
        assert detector.inv_cov_.shape == (3, 3)
        # Covariance matrix should be symmetric
        assert np.allclose(detector.cov_, detector.cov_.T)

    @pytest.mark.unit
    def test_fit_with_dataframe(self, detector_default):
        """Test fitting with pandas DataFrame."""
        df = pd.DataFrame(
            {
                "feature1": np.random.randn(50) * 10 + 100,
                "feature2": np.random.randn(50) * 5 + 50,
                "feature3": np.random.randn(50) * 2 + 10,
            }
        )

        detector = detector_default.fit(df)

        assert detector.is_fitted is True
        assert detector.mean_.shape == (3,)
        assert detector.cov_.shape == (3, 3)

    @pytest.mark.unit
    def test_fit_covariance_calculation(self, detector_default, normal_data):
        """Test that covariance is calculated correctly."""
        detector = detector_default.fit(normal_data)

        # Manual covariance calculation
        manual_mean = np.mean(normal_data, axis=0)
        manual_cov = np.cov(normal_data.T)

        assert np.allclose(detector.mean_, manual_mean)
        assert np.allclose(detector.cov_, manual_cov, rtol=1e-5)

    @pytest.mark.unit
    def test_fit_singular_covariance(self, detector_default):
        """Test handling of singular covariance matrix."""
        # Create data with perfect correlation (singular covariance)
        data = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9], [4, 8, 12]])

        # Should not raise error, should add regularization
        detector = detector_default.fit(data)

        assert detector.is_fitted is True
        assert detector.cov_ is not None
        assert detector.inv_cov_ is not None

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
        # Most points should not be anomalies with threshold=3.0
        anomaly_count = sum(r.is_anomaly for r in results)
        assert anomaly_count < len(results) * 0.1  # Less than 10% anomalies

    @pytest.mark.unit
    def test_predict_with_outliers(self, detector_default, normal_data, data_with_outliers):
        """Test prediction on data with clear outliers."""
        detector = detector_default.fit(normal_data)
        results = detector.predict(data_with_outliers)

        assert len(results) == len(data_with_outliers)
        # Outliers should have higher scores
        assert results[0].anomaly_score > results[5].anomaly_score
        assert results[1].anomaly_score > results[5].anomaly_score

    @pytest.mark.unit
    def test_predict_mahalanobis_distance(self, detector_default, normal_data):
        """Test that Mahalanobis distances are calculated correctly."""
        detector = detector_default.fit(normal_data)

        # Test point
        test_point = normal_data[0]
        results = detector.predict(normal_data[:1])

        # Manual Mahalanobis distance calculation
        diff = test_point - detector.mean_
        manual_distance = np.sqrt(diff @ detector.inv_cov_ @ diff)

        # Score should be distance / threshold
        expected_score = manual_distance / detector.threshold

        assert abs(results[0].anomaly_score - expected_score) < 1e-5

    @pytest.mark.unit
    def test_predict_with_different_threshold(self, normal_data, data_with_outliers):
        """Test prediction with different threshold values."""
        # Lower threshold should detect more anomalies
        detector_low = MahalanobisDetector(threshold=1.0).fit(normal_data)
        detector_high = MahalanobisDetector(threshold=5.0).fit(normal_data)

        results_low = detector_low.predict(data_with_outliers)
        results_high = detector_high.predict(data_with_outliers)

        anomalies_low = sum(r.is_anomaly for r in results_low)
        anomalies_high = sum(r.is_anomaly for r in results_high)

        assert anomalies_low >= anomalies_high

    @pytest.mark.unit
    def test_predict_scores(self, detector_default, normal_data, data_with_outliers):
        """Test predict_scores method."""
        detector = detector_default.fit(normal_data)
        scores = detector.predict_scores(data_with_outliers)

        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(data_with_outliers)
        assert all(s >= 0 for s in scores)
        # Outliers should have higher scores
        assert scores[0] > scores[5]
        assert scores[1] > scores[5]

    @pytest.mark.unit
    def test_multidimensional_data(self, detector_default):
        """Test with high-dimensional data."""
        np.random.seed(42)
        high_dim_data = np.random.randn(100, 10) * 10 + 100

        detector = detector_default.fit(high_dim_data)
        results = detector.predict(high_dim_data)

        assert len(results) == len(high_dim_data)
        assert detector.mean_.shape == (10,)
        assert detector.cov_.shape == (10, 10)
        assert detector.inv_cov_.shape == (10, 10)

    @pytest.mark.unit
    def test_edge_case_single_feature(self, detector_default):
        """Test with single feature (degenerate case)."""
        single_feature = np.random.randn(50, 1) * 10 + 100

        detector = detector_default.fit(single_feature)
        results = detector.predict(single_feature)

        assert len(results) == len(single_feature)
        assert detector.cov_.shape == (1, 1)

    @pytest.mark.unit
    def test_edge_case_small_dataset(self, detector_default):
        """Test with small dataset."""
        small_data = np.array([[10, 20], [20, 30], [30, 40]])

        detector = detector_default.fit(small_data)
        results = detector.predict(small_data)

        assert len(results) == len(small_data)
        assert all(isinstance(r, AnomalyDetectionResult) for r in results)
