"""
Unit tests for signal quality assessment (analytics).

Tests for SignalQualityMetrics and SignalQualityAnalyzer.
"""

import pytest
import numpy as np
from am_qadf.analytics.quality_assessment.signal_quality import (
    SignalQualityMetrics,
    SignalQualityAnalyzer,
)


class TestSignalQualityMetrics:
    """Test suite for SignalQualityMetrics dataclass."""

    @pytest.mark.unit
    def test_metrics_creation(self):
        """Test creating SignalQualityMetrics."""
        metrics = SignalQualityMetrics(
            signal_name="test_signal",
            snr_mean=30.0,
            snr_std=5.0,
            snr_min=20.0,
            snr_max=40.0,
            uncertainty_mean=0.05,
            confidence_mean=0.9,
            quality_score=0.85,
        )

        assert metrics.signal_name == "test_signal"
        assert metrics.snr_mean == 30.0
        assert metrics.uncertainty_mean == 0.05
        assert metrics.confidence_mean == 0.9
        assert metrics.quality_score == 0.85

    @pytest.mark.unit
    def test_metrics_to_dict(self):
        """Test converting SignalQualityMetrics to dictionary."""
        metrics = SignalQualityMetrics(
            signal_name="test_signal",
            snr_mean=30.0,
            snr_std=5.0,
            snr_min=20.0,
            snr_max=40.0,
            uncertainty_mean=0.05,
            confidence_mean=0.9,
            quality_score=0.85,
            snr_map=np.array([30.0, 35.0]),
        )

        result = metrics.to_dict()

        assert isinstance(result, dict)
        assert result["signal_name"] == "test_signal"
        assert "snr_map_shape" in result


class TestSignalQualityAnalyzer:
    """Test suite for SignalQualityAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create a SignalQualityAnalyzer instance."""
        return SignalQualityAnalyzer()

    @pytest.mark.unit
    def test_analyzer_creation_default(self):
        """Test creating SignalQualityAnalyzer with default parameters."""
        analyzer = SignalQualityAnalyzer()

        assert analyzer.noise_floor == 1e-6

    @pytest.mark.unit
    def test_analyzer_creation_custom(self):
        """Test creating SignalQualityAnalyzer with custom parameters."""
        analyzer = SignalQualityAnalyzer(noise_floor=1e-5)

        assert analyzer.noise_floor == 1e-5

    @pytest.mark.unit
    def test_calculate_snr(self, analyzer):
        """Test calculating Signal-to-Noise Ratio."""
        signal_array = np.array([100.0, 200.0, 300.0, 400.0, 500.0])

        mean_snr, std_snr, min_snr, max_snr, snr_map = analyzer.calculate_snr(signal_array)

        assert isinstance(mean_snr, float)
        assert isinstance(std_snr, float)
        assert isinstance(min_snr, float)
        assert isinstance(max_snr, float)
        assert snr_map is not None

    @pytest.mark.unit
    def test_calculate_uncertainty(self, analyzer):
        """Test calculating uncertainty."""
        signal_array = np.array([100.0, 200.0, 300.0, 400.0, 500.0])

        mean_uncertainty, uncertainty_map = analyzer.calculate_uncertainty(signal_array)

        assert isinstance(mean_uncertainty, float)
        assert uncertainty_map is not None

    @pytest.mark.unit
    def test_calculate_confidence(self, analyzer):
        """Test calculating confidence scores."""
        signal_array = np.array([100.0, 200.0, 300.0, 400.0, 500.0])

        mean_confidence, confidence_map = analyzer.calculate_confidence(signal_array)

        assert isinstance(mean_confidence, float)
        assert 0.0 <= mean_confidence <= 1.0
        assert confidence_map is not None

    @pytest.mark.unit
    def test_assess_signal_quality(self, analyzer):
        """Test assessing overall signal quality."""
        signal_array = np.array([100.0, 200.0, 300.0, 400.0, 500.0])

        metrics = analyzer.assess_signal_quality("test_signal", signal_array)

        assert isinstance(metrics, SignalQualityMetrics)
        assert metrics.signal_name == "test_signal"
        assert metrics.snr_mean >= 0
        assert 0.0 <= metrics.confidence_mean <= 1.0
        assert 0.0 <= metrics.quality_score <= 1.0
