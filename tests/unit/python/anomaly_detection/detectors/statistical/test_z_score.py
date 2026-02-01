"""
Unit tests for Z-Score detector.

Tests for ZScoreDetector class.
"""

import pytest
import numpy as np
import pandas as pd
from am_qadf.anomaly_detection.detectors.statistical.z_score import ZScoreDetector
from am_qadf.anomaly_detection.core.base_detector import (
    AnomalyDetectionConfig,
    AnomalyDetectionResult,
)


class TestZScoreDetector:
    """Test suite for ZScoreDetector class."""

    @pytest.fixture
    def detector_default(self):
        """Create ZScoreDetector with default parameters."""
        return ZScoreDetector()

    @pytest.fixture
    def detector_custom(self):
        """Create ZScoreDetector with custom threshold."""
        return ZScoreDetector(threshold=2.5)

    @pytest.fixture
    def normal_data(self):
        """Create normal data for training."""
        np.random.seed(42)
        return np.random.randn(100, 3) * 10 + 100

    @pytest.fixture
    def data_with_outliers(self, normal_data):
        """Create data with clear outliers."""
        data = normal_data.copy()
        # Add clear outliers
        data[0] = [200, 200, 200]  # High outlier
        data[1] = [0, 0, 0]  # Low outlier
        return data

    @pytest.mark.unit
    def test_detector_initialization_default(self):
        """Test ZScoreDetector initialization with default values."""
        detector = ZScoreDetector()

        assert detector.threshold == 3.0
        assert detector.config is not None
        assert detector.config.threshold == 3.0
        assert detector.is_fitted is False
        assert detector.mean_ is None
        assert detector.std_ is None

    @pytest.mark.unit
    def test_detector_initialization_custom(self):
        """Test ZScoreDetector initialization with custom threshold."""
        detector = ZScoreDetector(threshold=2.5)

        assert detector.threshold == 2.5
        assert detector.config.threshold == 2.5

    @pytest.mark.unit
    def test_detector_initialization_with_config(self):
        """Test ZScoreDetector initialization with custom config."""
        config = AnomalyDetectionConfig(threshold=0.5)
        detector = ZScoreDetector(threshold=2.0, config=config)

        assert detector.threshold == 2.0
        assert detector.config.threshold == 2.0

    @pytest.mark.unit
    def test_fit(self, detector_default, normal_data):
        """Test fitting the detector."""
        detector = detector_default.fit(normal_data)

        assert detector.is_fitted is True
        assert detector.mean_ is not None
        assert detector.std_ is not None
        assert detector.mean_.shape == (3,)
        assert detector.std_.shape == (3,)
        assert detector is detector_default  # Method chaining

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
        assert detector.std_.shape == (3,)

    @pytest.mark.unit
    def test_fit_with_single_feature(self, detector_default):
        """Test fitting with single feature."""
        data = np.random.randn(50, 1) * 10 + 100
        detector = detector_default.fit(data)

        assert detector.is_fitted is True
        assert detector.mean_.shape == (1,)
        assert detector.std_.shape == (1,)

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
        # First two points should be detected as anomalies
        assert results[0].is_anomaly is True
        assert results[1].is_anomaly is True
        assert results[0].anomaly_score > results[2].anomaly_score
        assert results[1].anomaly_score > results[2].anomaly_score

    @pytest.mark.unit
    def test_predict_with_different_threshold(self, normal_data, data_with_outliers):
        """Test prediction with different threshold values."""
        # Lower threshold should detect more anomalies
        detector_low = ZScoreDetector(threshold=1.0).fit(normal_data)
        detector_high = ZScoreDetector(threshold=5.0).fit(normal_data)

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
    def test_predict_with_dataframe(self, detector_default, normal_data):
        """Test prediction with pandas DataFrame."""
        detector = detector_default.fit(normal_data)
        df = pd.DataFrame(normal_data, columns=["f1", "f2", "f3"])
        results = detector.predict(df)

        assert len(results) == len(df)
        assert all(isinstance(r, AnomalyDetectionResult) for r in results)

    @pytest.mark.unit
    def test_edge_case_empty_data(self, detector_default):
        """Test with empty data."""
        empty_data = np.array([]).reshape(0, 3)

        detector = detector_default.fit(empty_data)
        assert detector.is_fitted is True

        results = detector.predict(empty_data)
        assert len(results) == 0

    @pytest.mark.unit
    def test_edge_case_single_point(self, detector_default):
        """Test with single data point."""
        single_point = np.array([[100, 100, 100]])

        detector = detector_default.fit(single_point)
        results = detector.predict(single_point)

        assert len(results) == 1
        # Single point should not be an anomaly (no variance)
        assert results[0].is_anomaly is False

    @pytest.mark.unit
    def test_edge_case_all_same_values(self, detector_default):
        """Test with all same values."""
        same_data = np.ones((50, 3)) * 100

        detector = detector_default.fit(same_data)
        results = detector.predict(same_data)

        # All same values should have zero std, so no anomalies
        assert all(not r.is_anomaly for r in results)
        assert all(r.anomaly_score == 0.0 for r in results)

    @pytest.mark.unit
    def test_multidimensional_data(self, detector_default):
        """Test with high-dimensional data."""
        high_dim_data = np.random.randn(100, 10) * 10 + 100

        detector = detector_default.fit(high_dim_data)
        results = detector.predict(high_dim_data)

        assert len(results) == len(high_dim_data)
        assert detector.mean_.shape == (10,)
        assert detector.std_.shape == (10,)

    @pytest.mark.unit
    def test_z_score_calculation(self, detector_default, normal_data):
        """Test that Z-scores are calculated correctly."""
        detector = detector_default.fit(normal_data)

        # Manually calculate Z-score for first point
        point = normal_data[0]
        manual_z = np.abs((point - detector.mean_) / detector.std_)
        manual_max_z = np.max(manual_z)

        results = detector.predict(normal_data[:1])
        detector_z = results[0].anomaly_score

        # Should match (within floating point precision)
        assert abs(manual_max_z - detector_z) < 1e-10
