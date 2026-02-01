"""
Unit tests for IQR detector.

Tests for IQRDetector class.
"""

import pytest
import numpy as np
import pandas as pd
from am_qadf.anomaly_detection.detectors.statistical.iqr import IQRDetector
from am_qadf.anomaly_detection.core.base_detector import (
    AnomalyDetectionConfig,
    AnomalyDetectionResult,
)


class TestIQRDetector:
    """Test suite for IQRDetector class."""

    @pytest.fixture
    def detector_default(self):
        """Create IQRDetector with default parameters."""
        return IQRDetector()

    @pytest.fixture
    def detector_custom(self):
        """Create IQRDetector with custom multiplier."""
        return IQRDetector(multiplier=2.0)

    @pytest.fixture
    def normal_data(self):
        """Create normal data for training."""
        np.random.seed(42)
        return np.random.randn(100, 3) * 10 + 100

    @pytest.fixture
    def data_with_outliers(self, normal_data):
        """Create data with clear outliers."""
        data = normal_data.copy()
        # Add outliers beyond IQR bounds
        data[0] = [200, 200, 200]  # High outlier
        data[1] = [0, 0, 0]  # Low outlier
        return data

    @pytest.mark.unit
    def test_detector_initialization_default(self):
        """Test IQRDetector initialization with default values."""
        detector = IQRDetector()

        assert detector.multiplier == 1.5
        assert detector.config is not None
        assert detector.is_fitted is False
        assert detector.Q1_ is None
        assert detector.Q3_ is None
        assert detector.IQR_ is None

    @pytest.mark.unit
    def test_detector_initialization_custom(self):
        """Test IQRDetector initialization with custom multiplier."""
        detector = IQRDetector(multiplier=2.0)

        assert detector.multiplier == 2.0

    @pytest.mark.unit
    def test_detector_initialization_with_config(self):
        """Test IQRDetector initialization with custom config."""
        config = AnomalyDetectionConfig(threshold=0.5)
        detector = IQRDetector(multiplier=1.8, config=config)

        assert detector.multiplier == 1.8
        assert detector.config.threshold == 0.5

    @pytest.mark.unit
    def test_fit(self, detector_default, normal_data):
        """Test fitting the detector."""
        detector = detector_default.fit(normal_data)

        assert detector.is_fitted is True
        assert detector.Q1_ is not None
        assert detector.Q3_ is not None
        assert detector.IQR_ is not None
        assert detector.Q1_.shape == (3,)
        assert detector.Q3_.shape == (3,)
        assert detector.IQR_.shape == (3,)
        # Q3 should be greater than Q1
        assert np.all(detector.Q3_ >= detector.Q1_)
        # IQR should be positive
        assert np.all(detector.IQR_ >= 0)

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
        assert detector.Q1_.shape == (3,)
        assert detector.Q3_.shape == (3,)

    @pytest.mark.unit
    def test_fit_quartile_calculation(self, detector_default):
        """Test that quartiles are calculated correctly."""
        data = np.array([[10, 20], [20, 30], [30, 40], [40, 50], [50, 60]])
        detector = detector_default.fit(data)

        # Q1 should be around 20, Q3 around 40
        assert detector.Q1_[0] < 30
        assert detector.Q3_[0] > 30
        assert detector.IQR_[0] == detector.Q3_[0] - detector.Q1_[0]

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
        # Most points should not be anomalies
        anomaly_count = sum(r.is_anomaly for r in results)
        assert anomaly_count < len(results) * 0.1  # Less than 10% anomalies

    @pytest.mark.unit
    def test_predict_with_outliers(self, detector_default, normal_data, data_with_outliers):
        """Test prediction on data with clear outliers."""
        detector = detector_default.fit(normal_data)
        results = detector.predict(data_with_outliers)

        assert len(results) == len(data_with_outliers)
        # Outliers should be detected
        assert results[0].is_anomaly is True or results[1].is_anomaly is True

    @pytest.mark.unit
    def test_predict_with_different_multipliers(self, normal_data, data_with_outliers):
        """Test prediction with different multiplier values."""
        # Lower multiplier should detect more anomalies
        detector_low = IQRDetector(multiplier=1.0).fit(normal_data)
        detector_high = IQRDetector(multiplier=3.0).fit(normal_data)

        results_low = detector_low.predict(data_with_outliers)
        results_high = detector_high.predict(data_with_outliers)

        anomalies_low = sum(r.is_anomaly for r in results_low)
        anomalies_high = sum(r.is_anomaly for r in results_high)

        assert anomalies_low >= anomalies_high

    @pytest.mark.unit
    def test_predict_bounds_calculation(self, detector_default, normal_data):
        """Test that bounds are calculated correctly."""
        detector = detector_default.fit(normal_data)

        lower_bound = detector.Q1_ - detector.multiplier * detector.IQR_
        upper_bound = detector.Q3_ + detector.multiplier * detector.IQR_

        # Test data should mostly be within bounds
        test_data = normal_data[:10]
        for point in test_data:
            for j in range(len(point)):
                # Most points should be within bounds
                assert (point[j] >= lower_bound[j] and point[j] <= upper_bound[j]) or (
                    point[j] < lower_bound[j] or point[j] > upper_bound[j]
                )

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
        small_data = np.array([[10, 20], [20, 30], [30, 40]])

        detector = detector_default.fit(small_data)
        results = detector.predict(small_data)

        assert len(results) == len(small_data)
        assert all(isinstance(r, AnomalyDetectionResult) for r in results)

    @pytest.mark.unit
    def test_edge_case_all_same_values(self, detector_default):
        """Test with all same values."""
        same_data = np.ones((50, 3)) * 100

        detector = detector_default.fit(same_data)
        results = detector.predict(same_data)

        # All same values should have IQR=0, so no outliers
        assert np.all(detector.IQR_ == 0)
        # All points should have same score (0 or very low)
        scores = [r.anomaly_score for r in results]
        assert all(abs(s - scores[0]) < 1e-10 for s in scores)

    @pytest.mark.unit
    def test_multidimensional_data(self, detector_default):
        """Test with high-dimensional data."""
        high_dim_data = np.random.randn(100, 10) * 10 + 100

        detector = detector_default.fit(high_dim_data)
        results = detector.predict(high_dim_data)

        assert len(results) == len(high_dim_data)
        assert detector.Q1_.shape == (10,)
        assert detector.Q3_.shape == (10,)
