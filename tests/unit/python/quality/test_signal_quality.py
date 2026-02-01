"""
Unit tests for signal quality assessment.

Tests for SignalQualityMetrics and SignalQualityAnalyzer.
"""

import pytest
import numpy as np
from am_qadf.quality.signal_quality import (
    SignalQualityMetrics,
    SignalQualityAnalyzer,
)


class TestSignalQualityMetrics:
    """Test suite for SignalQualityMetrics dataclass."""

    @pytest.mark.unit
    def test_signal_quality_metrics_creation(self):
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
            snr_map=np.array([30.0, 35.0, 25.0]),
            uncertainty_map=np.array([0.05, 0.04, 0.06]),
            confidence_map=np.array([0.9, 0.95, 0.85]),
        )

        assert metrics.signal_name == "test_signal"
        assert metrics.snr_mean == 30.0
        assert metrics.uncertainty_mean == 0.05
        assert metrics.confidence_mean == 0.9
        assert metrics.quality_score == 0.85

    @pytest.mark.unit
    def test_signal_quality_metrics_to_dict(self):
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
            uncertainty_map=np.array([0.05, 0.04]),
            confidence_map=np.array([0.9, 0.95]),
        )

        result = metrics.to_dict()

        assert isinstance(result, dict)
        assert result["signal_name"] == "test_signal"
        assert result["snr_mean"] == 30.0
        assert "snr_map_shape" in result
        assert "uncertainty_map_shape" in result
        assert "confidence_map_shape" in result


class TestSignalQualityAnalyzer:
    """Test suite for SignalQualityAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create a SignalQualityAnalyzer instance."""
        return SignalQualityAnalyzer()

    @pytest.mark.unit
    def test_signal_quality_analyzer_creation_default(self):
        """Test creating SignalQualityAnalyzer with default parameters."""
        analyzer = SignalQualityAnalyzer()

        assert analyzer.noise_floor == 1e-6

    @pytest.mark.unit
    def test_signal_quality_analyzer_creation_custom(self):
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
        assert len(snr_map) == len(signal_array)

    @pytest.mark.unit
    def test_calculate_snr_with_noise_estimate(self, analyzer):
        """Test calculating SNR with noise estimate."""
        signal_array = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        noise_estimate = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

        mean_snr, std_snr, min_snr, max_snr, snr_map = analyzer.calculate_snr(signal_array, noise_estimate=noise_estimate)

        assert isinstance(mean_snr, float)
        assert snr_map is not None

    @pytest.mark.unit
    def test_calculate_snr_empty(self, analyzer):
        """Test calculating SNR with empty signal."""
        signal_array = np.array([])

        mean_snr, std_snr, min_snr, max_snr, snr_map = analyzer.calculate_snr(signal_array)

        assert mean_snr == 0.0
        assert snr_map is None

    @pytest.mark.unit
    def test_calculate_snr_no_store_map(self, analyzer):
        """Test calculating SNR without storing map."""
        signal_array = np.array([100.0, 200.0, 300.0])

        mean_snr, std_snr, min_snr, max_snr, snr_map = analyzer.calculate_snr(signal_array, store_map=False)

        assert snr_map is None

    @pytest.mark.unit
    def test_calculate_uncertainty(self, analyzer):
        """Test calculating uncertainty."""
        signal_array = np.array([100.0, 200.0, 300.0, 400.0, 500.0])

        mean_uncertainty, uncertainty_map = analyzer.calculate_uncertainty(signal_array)

        assert isinstance(mean_uncertainty, float)
        assert uncertainty_map is not None
        assert len(uncertainty_map) == len(signal_array)

    @pytest.mark.unit
    def test_calculate_uncertainty_with_measurement_uncertainty(self, analyzer):
        """Test calculating uncertainty with measurement uncertainty."""
        signal_array = np.array([100.0, 200.0, 300.0])
        measurement_uncertainty = 0.1  # 10% relative

        mean_uncertainty, uncertainty_map = analyzer.calculate_uncertainty(
            signal_array, measurement_uncertainty=measurement_uncertainty
        )

        assert isinstance(mean_uncertainty, float)
        assert uncertainty_map is not None

    @pytest.mark.unit
    def test_calculate_uncertainty_absolute(self, analyzer):
        """Test calculating uncertainty with absolute uncertainty."""
        signal_array = np.array([100.0, 200.0, 300.0])
        measurement_uncertainty = 5.0  # Absolute uncertainty

        mean_uncertainty, uncertainty_map = analyzer.calculate_uncertainty(
            signal_array, measurement_uncertainty=measurement_uncertainty
        )

        assert isinstance(mean_uncertainty, float)
        # Should be approximately 5.0
        assert np.allclose(uncertainty_map[~np.isnan(uncertainty_map)], 5.0, atol=0.1)

    @pytest.mark.unit
    def test_calculate_uncertainty_with_interpolation(self, analyzer):
        """Test calculating uncertainty with interpolation uncertainty."""
        signal_array = np.array([100.0, 200.0, 300.0])
        interpolation_uncertainty = np.array([1.0, 2.0, 3.0])

        mean_uncertainty, uncertainty_map = analyzer.calculate_uncertainty(
            signal_array, interpolation_uncertainty=interpolation_uncertainty
        )

        assert isinstance(mean_uncertainty, float)
        assert uncertainty_map is not None

    @pytest.mark.unit
    def test_calculate_uncertainty_no_store_map(self, analyzer):
        """Test calculating uncertainty without storing map."""
        signal_array = np.array([100.0, 200.0, 300.0])

        mean_uncertainty, uncertainty_map = analyzer.calculate_uncertainty(signal_array, store_map=False)

        assert uncertainty_map is None

    @pytest.mark.unit
    def test_calculate_confidence(self, analyzer):
        """Test calculating confidence scores."""
        signal_array = np.array([100.0, 200.0, 300.0, 400.0, 500.0])

        mean_confidence, confidence_map = analyzer.calculate_confidence(signal_array)

        assert isinstance(mean_confidence, float)
        assert 0.0 <= mean_confidence <= 1.0
        assert confidence_map is not None
        assert len(confidence_map) == len(signal_array)

    @pytest.mark.unit
    def test_calculate_confidence_with_snr(self, analyzer):
        """Test calculating confidence with SNR map."""
        signal_array = np.array([100.0, 200.0, 300.0])
        snr_map = np.array([30.0, 35.0, 25.0])

        mean_confidence, confidence_map = analyzer.calculate_confidence(signal_array, snr_map=snr_map)

        assert isinstance(mean_confidence, float)
        assert 0.0 <= mean_confidence <= 1.0

    @pytest.mark.unit
    def test_calculate_confidence_with_uncertainty(self, analyzer):
        """Test calculating confidence with uncertainty map."""
        signal_array = np.array([100.0, 200.0, 300.0])
        uncertainty_map = np.array([5.0, 4.0, 6.0])

        mean_confidence, confidence_map = analyzer.calculate_confidence(signal_array, uncertainty_map=uncertainty_map)

        assert isinstance(mean_confidence, float)
        assert 0.0 <= mean_confidence <= 1.0

    @pytest.mark.unit
    def test_calculate_confidence_with_both(self, analyzer):
        """Test calculating confidence with both SNR and uncertainty maps."""
        signal_array = np.array([100.0, 200.0, 300.0])
        snr_map = np.array([30.0, 35.0, 25.0])
        uncertainty_map = np.array([5.0, 4.0, 6.0])

        mean_confidence, confidence_map = analyzer.calculate_confidence(
            signal_array, snr_map=snr_map, uncertainty_map=uncertainty_map
        )

        assert isinstance(mean_confidence, float)
        assert 0.0 <= mean_confidence <= 1.0

    @pytest.mark.unit
    def test_calculate_confidence_empty(self, analyzer):
        """Test calculating confidence with empty signal."""
        signal_array = np.array([])

        mean_confidence, confidence_map = analyzer.calculate_confidence(signal_array)

        assert mean_confidence == 0.0
        assert confidence_map is None

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

    @pytest.mark.unit
    def test_assess_signal_quality_with_noise_estimate(self, analyzer):
        """Test assessing signal quality with noise estimate."""
        signal_array = np.array([100.0, 200.0, 300.0])
        noise_estimate = np.array([10.0, 20.0, 30.0])

        metrics = analyzer.assess_signal_quality("test_signal", signal_array, noise_estimate=noise_estimate)

        assert isinstance(metrics, SignalQualityMetrics)

    @pytest.mark.unit
    def test_assess_signal_quality_with_uncertainty(self, analyzer):
        """Test assessing signal quality with measurement uncertainty."""
        signal_array = np.array([100.0, 200.0, 300.0])
        measurement_uncertainty = 0.05  # 5% relative

        metrics = analyzer.assess_signal_quality("test_signal", signal_array, measurement_uncertainty=measurement_uncertainty)

        assert isinstance(metrics, SignalQualityMetrics)
        assert metrics.uncertainty_mean > 0

    @pytest.mark.unit
    def test_assess_signal_quality_no_store_maps(self, analyzer):
        """Test assessing signal quality without storing maps."""
        signal_array = np.array([100.0, 200.0, 300.0])

        metrics = analyzer.assess_signal_quality("test_signal", signal_array, store_maps=False)

        assert metrics.snr_map is None
        assert metrics.uncertainty_map is None
        assert metrics.confidence_map is None

    @pytest.mark.unit
    def test_assess_signal_quality_2d(self, analyzer):
        """Test assessing signal quality for 2D array."""
        signal_array = np.array([[100.0, 200.0], [300.0, 400.0]])

        metrics = analyzer.assess_signal_quality("test_signal", signal_array)

        assert isinstance(metrics, SignalQualityMetrics)
        if metrics.snr_map is not None:
            assert metrics.snr_map.shape == signal_array.shape
