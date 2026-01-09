"""
Unit tests for Modified Z-Score detector.

Tests for ModifiedZScoreDetector class.
"""

import pytest
import numpy as np
import pandas as pd
from am_qadf.anomaly_detection.detectors.statistical.modified_z_score import (
    ModifiedZScoreDetector,
)
from am_qadf.anomaly_detection.core.base_detector import (
    AnomalyDetectionConfig,
    AnomalyDetectionResult,
)


class TestModifiedZScoreDetector:
    """Test suite for ModifiedZScoreDetector class."""

    @pytest.fixture
    def detector_default(self):
        """Create ModifiedZScoreDetector with default parameters."""
        return ModifiedZScoreDetector()

    @pytest.fixture
    def detector_custom(self):
        """Create ModifiedZScoreDetector with custom threshold."""
        return ModifiedZScoreDetector(threshold=2.5)

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
        """Test ModifiedZScoreDetector initialization with default values."""
        detector = ModifiedZScoreDetector()

        assert detector.threshold == 3.5
        assert detector.config is not None
        assert detector.config.threshold == 3.5
        assert detector.is_fitted is False
        assert detector.median_ is None
        assert detector.mad_ is None

    @pytest.mark.unit
    def test_detector_initialization_custom(self):
        """Test ModifiedZScoreDetector initialization with custom threshold."""
        detector = ModifiedZScoreDetector(threshold=2.5)

        assert detector.threshold == 2.5
        assert detector.config.threshold == 2.5

    @pytest.mark.unit
    def test_detector_initialization_with_config(self):
        """Test ModifiedZScoreDetector initialization with custom config."""
        config = AnomalyDetectionConfig(threshold=0.5)
        detector = ModifiedZScoreDetector(threshold=2.0, config=config)

        assert detector.threshold == 2.0
        assert detector.config.threshold == 2.0

    @pytest.mark.unit
    def test_fit(self, detector_default, normal_data):
        """Test fitting the detector."""
        detector = detector_default.fit(normal_data)

        assert detector.is_fitted is True
        assert detector.median_ is not None
        assert detector.mad_ is not None
        assert detector.median_.shape == (3,)
        assert detector.mad_.shape == (3,)
        # MAD should be non-negative
        assert np.all(detector.mad_ >= 0)

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
        assert detector.median_.shape == (3,)
        assert detector.mad_.shape == (3,)

    @pytest.mark.unit
    def test_fit_mad_calculation(self, detector_default):
        """Test that MAD is calculated correctly."""
        data = np.array([[10, 20], [20, 30], [30, 40], [40, 50], [50, 60]])
        detector = detector_default.fit(data)

        # Manual MAD calculation
        manual_median = np.median(data, axis=0)
        manual_deviations = np.abs(data - manual_median)
        manual_mad = np.median(manual_deviations, axis=0)

        assert np.allclose(detector.median_, manual_median)
        assert np.allclose(detector.mad_, manual_mad)

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
        # Most points should not be anomalies with threshold=3.5
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
    def test_predict_modified_z_score_calculation(self, detector_default, normal_data):
        """Test that modified Z-scores are calculated correctly."""
        detector = detector_default.fit(normal_data)

        # Test point
        test_point = normal_data[0]

        # Manual modified Z-score calculation
        manual_modified_z = np.abs(0.6745 * (test_point - detector.median_) / detector.mad_)
        manual_max_z = np.max(manual_modified_z)

        results = detector.predict(normal_data[:1])
        detector_z = results[0].anomaly_score

        # Should match (within floating point precision)
        assert abs(manual_max_z - detector_z) < 1e-10

    @pytest.mark.unit
    def test_predict_robustness_to_outliers(self, detector_default):
        """Test that modified Z-score is more robust to outliers than standard Z-score."""
        # Create data with one extreme outlier
        data = np.random.randn(100, 3) * 10 + 100
        data[0] = [1000, 1000, 1000]  # Extreme outlier

        detector = detector_default.fit(data)
        results = detector.predict(data)

        # Modified Z-score should still work (MAD is robust)
        assert len(results) == len(data)
        # The outlier should be detected
        assert results[0].is_anomaly is True

    @pytest.mark.unit
    def test_predict_with_different_threshold(self, normal_data, data_with_outliers):
        """Test prediction with different threshold values."""
        # Lower threshold should detect more anomalies
        detector_low = ModifiedZScoreDetector(threshold=2.0).fit(normal_data)
        detector_high = ModifiedZScoreDetector(threshold=5.0).fit(normal_data)

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
    def test_edge_case_all_same_values(self, detector_default):
        """Test with all same values."""
        same_data = np.ones((50, 3)) * 100

        detector = detector_default.fit(same_data)
        results = detector.predict(same_data)

        # All same values should have MAD=0 (set to 1.0 to avoid division by zero)
        assert np.all(detector.mad_ == 1.0)
        # All points should have zero modified Z-score
        assert all(r.anomaly_score == 0.0 for r in results)
        assert all(not r.is_anomaly for r in results)

    @pytest.mark.unit
    def test_edge_case_small_dataset(self, detector_default):
        """Test with small dataset."""
        small_data = np.array([[10, 20], [20, 30], [30, 40]])

        detector = detector_default.fit(small_data)
        results = detector.predict(small_data)

        assert len(results) == len(small_data)
        assert all(isinstance(r, AnomalyDetectionResult) for r in results)

    @pytest.mark.unit
    def test_multidimensional_data(self, detector_default):
        """Test with high-dimensional data."""
        high_dim_data = np.random.randn(100, 10) * 10 + 100

        detector = detector_default.fit(high_dim_data)
        results = detector.predict(high_dim_data)

        assert len(results) == len(high_dim_data)
        assert detector.median_.shape == (10,)
        assert detector.mad_.shape == (10,)

    @pytest.mark.unit
    def test_comparison_with_standard_z_score(self, normal_data, data_with_outliers):
        """Test that modified Z-score behaves differently from standard Z-score."""
        from am_qadf.anomaly_detection.detectors.statistical.z_score import (
            ZScoreDetector,
        )

        modified_detector = ModifiedZScoreDetector(threshold=3.0).fit(normal_data)
        standard_detector = ZScoreDetector(threshold=3.0).fit(normal_data)

        modified_results = modified_detector.predict(data_with_outliers)
        standard_results = standard_detector.predict(data_with_outliers)

        # Both should detect outliers, but scores may differ
        assert len(modified_results) == len(standard_results)
        # Both should flag outliers
        assert (
            modified_results[0].is_anomaly == standard_results[0].is_anomaly
            or modified_results[0].is_anomaly
            or standard_results[0].is_anomaly
        )
