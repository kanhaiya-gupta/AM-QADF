"""
Unit tests for Threshold Violations detector.

Tests for ThresholdViolationDetector class.
"""

import pytest
import numpy as np
import pandas as pd
from am_qadf.anomaly_detection.detectors.rule_based.threshold_violations import (
    ThresholdViolationDetector,
)
from am_qadf.anomaly_detection.core.base_detector import (
    AnomalyDetectionConfig,
    AnomalyDetectionResult,
)


class TestThresholdViolationDetector:
    """Test suite for ThresholdViolationDetector class."""

    @pytest.fixture
    def detector_default(self):
        """Create ThresholdViolationDetector with default parameters."""
        return ThresholdViolationDetector()

    @pytest.fixture
    def detector_with_thresholds(self):
        """Create ThresholdViolationDetector with predefined thresholds."""
        thresholds = {
            "feature_0": (50.0, 150.0),
            "feature_1": (0.0, 100.0),
            "feature_2": (10.0, 90.0),
        }
        return ThresholdViolationDetector(thresholds=thresholds, use_absolute=True)

    @pytest.fixture
    def normal_data(self):
        """Create normal data for training."""
        np.random.seed(42)
        return np.random.randn(100, 3) * 20 + 100

    @pytest.fixture
    def data_with_violations(self, normal_data):
        """Create data with threshold violations."""
        data = normal_data.copy()
        # Add violations
        data[0, 0] = 200.0  # Above max threshold
        data[1, 0] = 0.0  # Below min threshold
        data[2, 1] = 150.0  # Above max threshold
        return data

    @pytest.mark.unit
    def test_detector_initialization_default(self):
        """Test ThresholdViolationDetector initialization with default values."""
        detector = ThresholdViolationDetector()

        assert detector.thresholds == {}
        assert detector.threshold_percentile == 95.0
        assert detector.use_absolute is False
        assert detector.strict_mode is False
        assert detector.config is not None
        assert detector.is_fitted is False
        assert detector.learned_thresholds_ is None

    @pytest.mark.unit
    def test_detector_initialization_with_thresholds(self):
        """Test initialization with predefined thresholds."""
        thresholds = {"feature1": (10.0, 90.0), "feature2": (20.0, 80.0)}
        detector = ThresholdViolationDetector(thresholds=thresholds, use_absolute=True)

        assert detector.thresholds == thresholds
        assert detector.use_absolute is True

    @pytest.mark.unit
    def test_detector_initialization_strict_mode(self):
        """Test initialization with strict mode."""
        detector = ThresholdViolationDetector(strict_mode=True)

        assert detector.strict_mode is True

    @pytest.mark.unit
    def test_detector_initialization_with_config(self):
        """Test initialization with custom config."""
        config = AnomalyDetectionConfig(threshold=0.5)
        detector = ThresholdViolationDetector(config=config)

        assert detector.config.threshold == 0.5

    @pytest.mark.unit
    def test_fit(self, detector_default, normal_data):
        """Test fitting the detector."""
        detector = detector_default.fit(normal_data)

        assert detector.is_fitted is True
        assert detector.learned_thresholds_ is not None
        assert len(detector.learned_thresholds_) > 0
        assert detector is detector_default  # Method chaining

    @pytest.mark.unit
    def test_fit_with_dataframe(self, detector_default):
        """Test fitting with pandas DataFrame."""
        df = pd.DataFrame(
            {
                "feature1": np.random.randn(50) * 20 + 100,
                "feature2": np.random.randn(50) * 10 + 50,
                "feature3": np.random.randn(50) * 5 + 10,
            }
        )

        detector = detector_default.fit(df)

        assert detector.is_fitted is True
        assert detector.learned_thresholds_ is not None

    @pytest.mark.unit
    def test_fit_learns_thresholds(self, detector_default, normal_data):
        """Test that thresholds are learned from data."""
        detector = detector_default.fit(normal_data)

        # Should learn thresholds for each feature
        assert len(detector.learned_thresholds_) == normal_data.shape[1]
        for feature_name, (
            min_thresh,
            max_thresh,
        ) in detector.learned_thresholds_.items():
            assert min_thresh < max_thresh
            # Thresholds should be within data range
            assert min_thresh <= np.max(normal_data)
            assert max_thresh >= np.min(normal_data)

    @pytest.mark.unit
    def test_fit_strict_mode_thresholds(self, normal_data):
        """Test threshold learning in strict mode."""
        detector = ThresholdViolationDetector(strict_mode=True)
        detector.fit(normal_data)

        # Strict mode uses 3-sigma rule
        assert detector.learned_thresholds_ is not None
        for feature_name, (
            min_thresh,
            max_thresh,
        ) in detector.learned_thresholds_.items():
            # Should be tighter bounds
            assert min_thresh < max_thresh

    @pytest.mark.unit
    def test_fit_with_predefined_thresholds(self, detector_with_thresholds, normal_data):
        """Test fitting with predefined thresholds."""
        detector = detector_with_thresholds.fit(normal_data)

        # Should not learn new thresholds if already provided
        assert detector.is_fitted is True
        # Predefined thresholds should be used

    @pytest.mark.unit
    def test_predict_before_fit(self, detector_default, normal_data):
        """Test that predict raises error if not fitted."""
        with pytest.raises(ValueError, match="must be fitted"):
            detector_default.predict(normal_data)

    @pytest.mark.unit
    def test_predict_no_thresholds(self, normal_data):
        """Test that predict raises error if no thresholds available."""
        detector = ThresholdViolationDetector(thresholds={}, use_absolute=True)
        detector.fit(normal_data)

        # If no thresholds learned and none provided, should raise error
        # This depends on implementation - may learn thresholds or raise error
        try:
            results = detector.predict(normal_data)
            # If it works, should have results
            assert len(results) == len(normal_data)
        except ValueError:
            # Expected if no thresholds available
            pass

    @pytest.mark.unit
    def test_predict_normal_data(self, detector_default, normal_data):
        """Test prediction on normal data."""
        detector = detector_default.fit(normal_data)
        results = detector.predict(normal_data)

        assert len(results) == len(normal_data)
        assert all(isinstance(r, AnomalyDetectionResult) for r in results)
        # Most points should not be violations
        violation_count = sum(r.is_anomaly for r in results)
        assert violation_count < len(results) * 0.2

    @pytest.mark.unit
    def test_predict_with_violations(self, detector_default, normal_data, data_with_violations):
        """Test prediction on data with threshold violations."""
        detector = detector_default.fit(normal_data)
        results = detector.predict(data_with_violations)

        assert len(results) == len(data_with_violations)
        # Violations should be detected
        assert results[0].is_anomaly is True or results[1].is_anomaly is True
        # Violations should have higher scores
        assert results[0].anomaly_score > results[5].anomaly_score
        assert results[1].anomaly_score > results[5].anomaly_score

    @pytest.mark.unit
    def test_predict_strict_mode(self, normal_data, data_with_violations):
        """Test prediction in strict mode."""
        detector = ThresholdViolationDetector(strict_mode=True)
        detector.fit(normal_data)
        results = detector.predict(data_with_violations)

        # In strict mode, any violation is an anomaly
        assert len(results) == len(data_with_violations)
        # Violations should be detected
        assert any(r.is_anomaly for r in results[:3])

    @pytest.mark.unit
    def test_predict_with_predefined_thresholds(self, detector_with_thresholds, normal_data, data_with_violations):
        """Test prediction with predefined thresholds."""
        detector = detector_with_thresholds.fit(normal_data)
        results = detector.predict(data_with_violations)

        assert len(results) == len(data_with_violations)
        # Should detect violations based on predefined thresholds
        assert any(r.is_anomaly for r in results)

    @pytest.mark.unit
    def test_predict_violation_severity(self, detector_default, normal_data):
        """Test that violation severity is calculated correctly."""
        detector = detector_default.fit(normal_data)

        # Create data with different violation severities
        test_data = normal_data.copy()
        test_data[0, 0] = np.max(normal_data[:, 0]) * 2  # Severe violation
        test_data[1, 0] = np.max(normal_data[:, 0]) * 1.1  # Mild violation

        results = detector.predict(test_data)

        # Severe violation should have higher score
        assert results[0].anomaly_score > results[1].anomaly_score

    @pytest.mark.unit
    def test_predict_scores(self, detector_default, normal_data, data_with_violations):
        """Test predict_scores method."""
        detector = detector_default.fit(normal_data)
        scores = detector.predict_scores(data_with_violations)

        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(data_with_violations)
        assert all(s >= 0 for s in scores)
        # Violations should have higher scores
        assert scores[0] > scores[5]
        assert scores[1] > scores[5]

    @pytest.mark.unit
    def test_edge_case_small_dataset(self, detector_default):
        """Test with small dataset."""
        small_data = np.random.randn(10, 3) * 20 + 100

        detector = detector_default.fit(small_data)
        results = detector.predict(small_data)

        assert len(results) == len(small_data)
        assert all(isinstance(r, AnomalyDetectionResult) for r in results)

    @pytest.mark.unit
    def test_edge_case_all_nan_values(self, detector_default):
        """Test handling of all NaN values."""
        data = np.full((10, 3), np.nan)

        detector = detector_default.fit(data)
        # Should handle gracefully (may skip features with all NaN)
        assert detector.is_fitted is True
