"""
Unit tests for Pattern Deviation detector.

Tests for PatternDeviationDetector class.
"""

import pytest
import numpy as np
import pandas as pd
from am_qadf.anomaly_detection.detectors.rule_based.pattern_deviation import (
    PatternDeviationDetector,
)
from am_qadf.anomaly_detection.core.base_detector import (
    AnomalyDetectionConfig,
    AnomalyDetectionResult,
)


class TestPatternDeviationDetector:
    """Test suite for PatternDeviationDetector class."""

    @pytest.fixture
    def detector_default(self):
        """Create PatternDeviationDetector with default parameters."""
        return PatternDeviationDetector()

    @pytest.fixture
    def detector_custom(self):
        """Create PatternDeviationDetector with custom parameters."""
        return PatternDeviationDetector(
            control_limit_sigma=2.0,
            pattern_window=5,
            use_trend_detection=True,
            use_cyclical_detection=False,
        )

    @pytest.fixture
    def normal_data(self):
        """Create normal data for training."""
        np.random.seed(42)
        return np.random.randn(100, 3) * 10 + 100

    @pytest.fixture
    def data_with_deviations(self, normal_data):
        """Create data with pattern deviations."""
        data = normal_data.copy()
        # Add trend deviation
        data[0:10, 0] = np.linspace(200, 250, 10)  # Strong upward trend
        # Add control limit violation
        data[20, 1] = 200  # Above UCL
        return data

    @pytest.mark.unit
    def test_detector_initialization_default(self):
        """Test PatternDeviationDetector initialization with default values."""
        detector = PatternDeviationDetector()

        assert detector.control_limit_sigma == 3.0
        assert detector.pattern_window == 10
        assert detector.use_trend_detection is True
        assert detector.use_cyclical_detection is True
        assert detector.threshold_percentile == 95.0
        assert detector.config is not None
        assert detector.is_fitted is False
        assert detector.control_limits_ is None
        assert detector.baseline_stats_ is None

    @pytest.mark.unit
    def test_detector_initialization_custom(self):
        """Test initialization with custom parameters."""
        detector = PatternDeviationDetector(control_limit_sigma=2.0, pattern_window=5, use_trend_detection=False)

        assert detector.control_limit_sigma == 2.0
        assert detector.pattern_window == 5
        assert detector.use_trend_detection is False

    @pytest.mark.unit
    def test_detector_initialization_with_config(self):
        """Test initialization with custom config."""
        config = AnomalyDetectionConfig(threshold=0.5)
        detector = PatternDeviationDetector(config=config)

        assert detector.config.threshold == 0.5

    @pytest.mark.unit
    def test_fit(self, detector_default, normal_data):
        """Test fitting the detector."""
        detector = detector_default.fit(normal_data)

        assert detector.is_fitted is True
        assert detector.control_limits_ is not None
        assert detector.baseline_stats_ is not None
        assert len(detector.control_limits_) == normal_data.shape[1]
        assert len(detector.baseline_stats_) == normal_data.shape[1]
        assert detector is detector_default  # Method chaining

    @pytest.mark.unit
    def test_fit_control_limits(self, detector_default, normal_data):
        """Test that control limits are calculated correctly."""
        detector = detector_default.fit(normal_data)

        for feature_name, limits in detector.control_limits_.items():
            assert "ucl" in limits
            assert "lcl" in limits
            assert "center" in limits
            assert limits["lcl"] < limits["center"] < limits["ucl"]
            # UCL and LCL should be 3 sigma from center
            center = limits["center"]
            ucl = limits["ucl"]
            lcl = limits["lcl"]
            assert abs((ucl - center) - (center - lcl)) < 1.0  # Approximately symmetric

    @pytest.mark.unit
    def test_fit_baseline_stats(self, detector_default, normal_data):
        """Test that baseline statistics are calculated."""
        detector = detector_default.fit(normal_data)

        for feature_name, stats in detector.baseline_stats_.items():
            assert "mean" in stats
            assert "std" in stats
            assert "median" in stats
            assert "min" in stats
            assert "max" in stats
            assert stats["min"] <= stats["median"] <= stats["max"]

    @pytest.mark.unit
    def test_detect_trend(self, detector_default):
        """Test trend detection method."""
        # Increasing trend
        increasing = np.array([10, 20, 30, 40, 50])
        trend_score = detector_default._detect_trend(increasing)
        assert trend_score > 0

        # Decreasing trend
        decreasing = np.array([50, 40, 30, 20, 10])
        trend_score = detector_default._detect_trend(decreasing)
        assert trend_score > 0

        # No trend
        no_trend = np.array([10, 10, 10, 10, 10])
        trend_score = detector_default._detect_trend(no_trend)
        assert trend_score == 0.0

    @pytest.mark.unit
    def test_detect_cyclical(self, detector_default):
        """Test cyclical pattern detection."""
        # Create cyclical pattern
        cyclical = np.sin(np.linspace(0, 4 * np.pi, 20))
        cyclical_score = detector_default._detect_cyclical(cyclical)
        assert cyclical_score >= 0

        # Random data (less cyclical)
        random = np.random.randn(20)
        random_score = detector_default._detect_cyclical(random)
        assert random_score >= 0

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
        assert anomaly_count < len(results) * 0.2

    @pytest.mark.unit
    def test_predict_with_deviations(self, detector_default, normal_data, data_with_deviations):
        """Test prediction on data with pattern deviations."""
        detector = detector_default.fit(normal_data)
        results = detector.predict(data_with_deviations)

        assert len(results) == len(data_with_deviations)
        # Deviations should be detected
        assert any(r.is_anomaly for r in results)
        # Trend deviation should have high score
        trend_scores = [r.anomaly_score for r in results[:10]]
        assert max(trend_scores) > min(trend_scores)

    @pytest.mark.unit
    def test_predict_control_limit_violations(self, detector_default, normal_data):
        """Test detection of control limit violations."""
        detector = detector_default.fit(normal_data)

        # Create data with UCL violation
        test_data = normal_data.copy()
        limits = detector.control_limits_["feature_0"]
        test_data[0, 0] = limits["ucl"] + 10  # Above UCL

        results = detector.predict(test_data)

        # Should detect violation
        assert results[0].is_anomaly is True
        assert results[0].anomaly_score > results[5].anomaly_score

    @pytest.mark.unit
    def test_predict_with_trend_detection(self, normal_data):
        """Test prediction with trend detection enabled."""
        detector = PatternDeviationDetector(use_trend_detection=True)
        detector.fit(normal_data)

        # Create data with strong trend
        test_data = normal_data.copy()
        test_data[0:10, 0] = np.linspace(200, 300, 10)  # Strong trend

        results = detector.predict(test_data)

        # Should detect trend
        assert any(r.is_anomaly for r in results[:10])

    @pytest.mark.unit
    def test_predict_without_trend_detection(self, normal_data):
        """Test prediction with trend detection disabled."""
        detector = PatternDeviationDetector(use_trend_detection=False)
        detector.fit(normal_data)

        # Create data with strong trend
        test_data = normal_data.copy()
        test_data[0:10, 0] = np.linspace(200, 300, 10)

        results = detector.predict(test_data)

        # May or may not detect without trend detection
        assert len(results) == len(test_data)

    @pytest.mark.unit
    def test_predict_scores(self, detector_default, normal_data, data_with_deviations):
        """Test predict_scores method."""
        detector = detector_default.fit(normal_data)
        scores = detector.predict_scores(data_with_deviations)

        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(data_with_deviations)
        assert all(s >= 0 for s in scores)

    @pytest.mark.unit
    def test_edge_case_small_dataset(self, detector_default):
        """Test with small dataset."""
        small_data = np.random.randn(10, 3) * 10 + 100

        detector = detector_default.fit(small_data)
        results = detector.predict(small_data)

        assert len(results) == len(small_data)
        assert all(isinstance(r, AnomalyDetectionResult) for r in results)

    @pytest.mark.unit
    def test_different_control_limit_sigma(self, normal_data):
        """Test with different control limit sigma values."""
        detector_tight = PatternDeviationDetector(control_limit_sigma=2.0).fit(normal_data)
        detector_loose = PatternDeviationDetector(control_limit_sigma=4.0).fit(normal_data)

        # Tighter limits should detect more violations
        test_data = normal_data.copy()
        test_data[0, 0] = np.max(normal_data[:, 0]) * 1.5

        results_tight = detector_tight.predict(test_data)
        results_loose = detector_loose.predict(test_data)

        # Both should work
        assert len(results_tight) == len(results_loose)
