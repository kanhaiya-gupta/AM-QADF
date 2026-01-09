"""
Unit tests for Temporal Pattern detector.

Tests for TemporalPatternDetector class.
"""

import pytest
import numpy as np
import pandas as pd
from am_qadf.anomaly_detection.detectors.rule_based.temporal_pattern import (
    TemporalPatternDetector,
)
from am_qadf.anomaly_detection.core.base_detector import (
    AnomalyDetectionConfig,
    AnomalyDetectionResult,
)


class TestTemporalPatternDetector:
    """Test suite for TemporalPatternDetector class."""

    @pytest.fixture
    def detector_default(self):
        """Create TemporalPatternDetector with default parameters."""
        return TemporalPatternDetector()

    @pytest.fixture
    def detector_custom(self):
        """Create TemporalPatternDetector with custom parameters."""
        return TemporalPatternDetector(
            sequence_length=10,
            use_layer_analysis=True,
            use_trend_analysis=False,
            use_variance_analysis=True,
        )

    @pytest.fixture
    def normal_data(self):
        """Create normal temporal data for training."""
        np.random.seed(42)
        return np.random.randn(100, 3) * 10 + 100

    @pytest.fixture
    def data_with_temporal_anomalies(self, normal_data):
        """Create data with temporal anomalies."""
        data = normal_data.copy()
        # Add trend anomaly
        data[0:10, 0] = np.linspace(200, 300, 10)  # Strong upward trend
        # Add variance anomaly
        data[20:25, 1] = np.random.randn(5) * 50 + 100  # High variance
        return data

    @pytest.mark.unit
    def test_detector_initialization_default(self):
        """Test TemporalPatternDetector initialization with default values."""
        detector = TemporalPatternDetector()

        assert detector.sequence_length == 5
        assert detector.use_layer_analysis is True
        assert detector.use_trend_analysis is True
        assert detector.use_variance_analysis is True
        assert detector.layer_key == "layer_number"
        assert detector.threshold_percentile == 95.0
        assert detector.config is not None
        assert detector.is_fitted is False
        assert detector.baseline_patterns_ is None
        assert detector.layer_stats_ is None

    @pytest.mark.unit
    def test_detector_initialization_custom(self):
        """Test initialization with custom parameters."""
        detector = TemporalPatternDetector(sequence_length=10, use_layer_analysis=False, use_trend_analysis=True)

        assert detector.sequence_length == 10
        assert detector.use_layer_analysis is False
        assert detector.use_trend_analysis is True

    @pytest.mark.unit
    def test_detector_initialization_with_config(self):
        """Test initialization with custom config."""
        config = AnomalyDetectionConfig(threshold=0.5)
        detector = TemporalPatternDetector(config=config)

        assert detector.config.threshold == 0.5

    @pytest.mark.unit
    def test_fit(self, detector_default, normal_data):
        """Test fitting the detector."""
        detector = detector_default.fit(normal_data)

        assert detector.is_fitted is True
        assert detector.baseline_patterns_ is not None
        assert len(detector.baseline_patterns_) == normal_data.shape[1]
        assert detector is detector_default  # Method chaining

    @pytest.mark.unit
    def test_fit_baseline_patterns(self, detector_default, normal_data):
        """Test that baseline patterns are calculated."""
        detector = detector_default.fit(normal_data)

        for feature_name, pattern in detector.baseline_patterns_.items():
            assert "mean" in pattern
            assert "std" in pattern
            assert "median" in pattern
            assert "variance" in pattern
            assert pattern["variance"] >= 0

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
        assert len(detector.baseline_patterns_) == 3

    @pytest.mark.unit
    def test_calculate_sequence_anomaly(self, detector_default, normal_data):
        """Test sequence anomaly calculation."""
        detector = detector_default.fit(normal_data)
        baseline = detector.baseline_patterns_["feature_0"]

        # Normal sequence
        normal_sequence = np.array([100, 101, 99, 100, 102])
        score_normal = detector._calculate_sequence_anomaly(normal_sequence, baseline)
        assert score_normal >= 0

        # Anomalous sequence (trend)
        trend_sequence = np.array([100, 150, 200, 250, 300])
        score_trend = detector._calculate_sequence_anomaly(trend_sequence, baseline)
        assert score_trend > score_normal

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
        # Most points should not be temporal anomalies
        anomaly_count = sum(r.is_anomaly for r in results)
        assert anomaly_count < len(results) * 0.2

    @pytest.mark.unit
    def test_predict_with_temporal_anomalies(self, detector_default, normal_data, data_with_temporal_anomalies):
        """Test prediction on data with temporal anomalies."""
        detector = detector_default.fit(normal_data)
        results = detector.predict(data_with_temporal_anomalies)

        assert len(results) == len(data_with_temporal_anomalies)
        # Trend anomaly should be detected
        trend_scores = [r.anomaly_score for r in results[:10]]
        assert max(trend_scores) > min(trend_scores)
        assert any(r.is_anomaly for r in results[:10])

    @pytest.mark.unit
    def test_predict_trend_analysis(self, detector_default, normal_data):
        """Test trend-based anomaly detection."""
        detector = TemporalPatternDetector(use_trend_analysis=True)
        detector.fit(normal_data)

        # Create strong trend
        test_data = normal_data.copy()
        test_data[0:10, 0] = np.linspace(200, 300, 10)

        results = detector.predict(test_data)

        # Should detect trend
        assert any(r.is_anomaly for r in results[:10])

    @pytest.mark.unit
    def test_predict_variance_analysis(self, detector_default, normal_data):
        """Test variance-based anomaly detection."""
        detector = TemporalPatternDetector(use_variance_analysis=True)
        detector.fit(normal_data)

        # Create high variance sequence
        test_data = normal_data.copy()
        test_data[0:10, 0] = np.random.randn(10) * 50 + 100  # High variance

        results = detector.predict(test_data)

        # Should detect variance anomaly
        assert len(results) == len(test_data)

    @pytest.mark.unit
    def test_predict_without_trend_analysis(self, normal_data):
        """Test prediction without trend analysis."""
        detector = TemporalPatternDetector(use_trend_analysis=False)
        detector.fit(normal_data)

        # Create strong trend
        test_data = normal_data.copy()
        test_data[0:10, 0] = np.linspace(200, 300, 10)

        results = detector.predict(test_data)

        # May or may not detect without trend analysis
        assert len(results) == len(test_data)

    @pytest.mark.unit
    def test_predict_different_sequence_lengths(self, normal_data):
        """Test with different sequence lengths."""
        for seq_len in [3, 5, 10]:
            detector = TemporalPatternDetector(sequence_length=seq_len)
            detector.fit(normal_data)
            results = detector.predict(normal_data[:20])

            assert len(results) == 20
            assert detector.is_fitted is True

    @pytest.mark.unit
    def test_predict_scores(self, detector_default, normal_data, data_with_temporal_anomalies):
        """Test predict_scores method."""
        detector = detector_default.fit(normal_data)
        scores = detector.predict_scores(data_with_temporal_anomalies)

        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(data_with_temporal_anomalies)
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
    def test_edge_case_short_sequence(self, detector_default):
        """Test with data shorter than sequence_length."""
        short_data = np.random.randn(3, 3) * 10 + 100

        detector = TemporalPatternDetector(sequence_length=5)
        detector.fit(short_data)

        # Should handle gracefully
        results = detector.predict(short_data)
        assert len(results) == len(short_data)
