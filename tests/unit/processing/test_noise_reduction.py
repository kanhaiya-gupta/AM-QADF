"""
Unit tests for noise reduction.

Tests for OutlierDetector, SignalSmoother, SignalQualityMetrics, and NoiseReductionPipeline.
"""

import pytest
import numpy as np
from am_qadf.processing.noise_reduction import (
    OutlierDetector,
    SignalSmoother,
    SignalQualityMetrics,
    NoiseReductionPipeline,
)


class TestOutlierDetector:
    """Test suite for OutlierDetector class."""

    @pytest.fixture
    def detector(self):
        """Create an OutlierDetector instance."""
        return OutlierDetector()

    @pytest.mark.unit
    def test_outlier_detector_creation_default(self):
        """Test creating OutlierDetector with default parameters."""
        detector = OutlierDetector()

        assert detector.method == "zscore"
        assert detector.threshold == 3.0
        assert detector.use_spatial is True

    @pytest.mark.unit
    def test_outlier_detector_creation_custom(self):
        """Test creating OutlierDetector with custom parameters."""
        detector = OutlierDetector(method="iqr", threshold=2.0, use_spatial=False)

        assert detector.method == "iqr"
        assert detector.threshold == 2.0
        assert detector.use_spatial is False

    @pytest.mark.unit
    def test_detect_zscore(self, detector):
        """Test Z-score outlier detection."""
        # Create signal with outliers
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0, 2.0, 3.0])

        outlier_mask, z_scores = detector.detect_zscore(signal)

        assert len(outlier_mask) == len(signal)
        assert len(z_scores) == len(signal)
        assert outlier_mask.dtype == bool
        # Outlier (100.0) should be detected
        assert outlier_mask[5] == True

    @pytest.mark.unit
    def test_detect_zscore_no_outliers(self, detector):
        """Test Z-score detection with no outliers."""
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        outlier_mask, z_scores = detector.detect_zscore(signal)

        # Should not detect outliers for normal data
        assert np.sum(outlier_mask) == 0

    @pytest.mark.unit
    def test_detect_zscore_empty(self, detector):
        """Test Z-score detection with empty signal."""
        signal = np.array([])

        outlier_mask, z_scores = detector.detect_zscore(signal)

        assert len(outlier_mask) == 0

    @pytest.mark.unit
    def test_detect_zscore_all_zeros(self, detector):
        """Test Z-score detection with all zeros."""
        signal = np.zeros(10)

        outlier_mask, z_scores = detector.detect_zscore(signal)

        # Should handle zeros gracefully
        assert len(outlier_mask) == 10

    @pytest.mark.unit
    def test_detect_iqr(self, detector):
        """Test IQR outlier detection."""
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0, 2.0, 3.0])

        outlier_mask, iqr_scores = detector.detect_iqr(signal)

        assert len(outlier_mask) == len(signal)
        assert len(iqr_scores) == len(signal)
        # Outlier should be detected
        assert outlier_mask[5] == True

    @pytest.mark.unit
    def test_detect_spatial(self, detector):
        """Test spatial outlier detection."""
        # Create 2D signal with spatial outlier
        signal = np.array([[1.0, 2.0, 1.0], [2.0, 100.0, 2.0], [1.0, 2.0, 1.0]])  # Outlier in center

        outlier_mask, spatial_scores = detector.detect_spatial(signal, kernel_size=3)

        assert outlier_mask.shape == signal.shape
        assert spatial_scores.shape == signal.shape
        # Center should be detected as outlier
        assert outlier_mask[1, 1] == True

    @pytest.mark.unit
    def test_detect(self, detector):
        """Test main detect method."""
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])

        outlier_mask, scores = detector.detect(signal)

        assert len(outlier_mask) == len(signal)
        assert len(scores) == len(signal)
        assert outlier_mask[5] == True

    @pytest.mark.unit
    def test_detect_without_spatial(self):
        """Test detection without spatial context."""
        detector = OutlierDetector(use_spatial=False)
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])

        outlier_mask, scores = detector.detect(signal)

        assert len(outlier_mask) == len(signal)

    @pytest.mark.unit
    def test_remove_outliers_median(self, detector):
        """Test removing outliers with median fill."""
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0, 2.0, 3.0])

        cleaned = detector.remove_outliers(signal, fill_method="median")

        assert len(cleaned) == len(signal)
        # Outlier should be replaced
        assert cleaned[5] != 100.0
        assert cleaned[5] == np.median(signal[signal != 100.0])

    @pytest.mark.unit
    def test_remove_outliers_mean(self, detector):
        """Test removing outliers with mean fill."""
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])

        cleaned = detector.remove_outliers(signal, fill_method="mean")

        assert len(cleaned) == len(signal)
        assert cleaned[5] == np.mean(signal[signal != 100.0])

    @pytest.mark.unit
    def test_remove_outliers_zero(self, detector):
        """Test removing outliers with zero fill."""
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])

        cleaned = detector.remove_outliers(signal, fill_method="zero")

        assert len(cleaned) == len(signal)
        assert cleaned[5] == 0.0

    @pytest.mark.unit
    def test_remove_outliers_no_outliers(self, detector):
        """Test removing outliers when none exist."""
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        cleaned = detector.remove_outliers(signal)

        assert np.array_equal(cleaned, signal)


class TestSignalSmoother:
    """Test suite for SignalSmoother class."""

    @pytest.fixture
    def smoother(self):
        """Create a SignalSmoother instance."""
        return SignalSmoother()

    @pytest.mark.unit
    def test_signal_smoother_creation_default(self):
        """Test creating SignalSmoother with default parameters."""
        smoother = SignalSmoother()

        assert smoother.method == "gaussian"
        assert smoother.kernel_size == 1.0

    @pytest.mark.unit
    def test_signal_smoother_creation_custom(self):
        """Test creating SignalSmoother with custom parameters."""
        smoother = SignalSmoother(method="median", kernel_size=3.0)

        assert smoother.method == "median"
        assert smoother.kernel_size == 3.0

    @pytest.mark.unit
    def test_gaussian_smooth(self, smoother):
        """Test Gaussian smoothing."""
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0])

        smoothed = smoother.gaussian_smooth(signal, sigma=1.0)

        assert len(smoothed) == len(signal)
        assert smoothed.dtype == float
        # Smoothed signal should be smoother (less variation)
        assert np.std(smoothed) <= np.std(signal)

    @pytest.mark.unit
    def test_gaussian_smooth_2d(self, smoother):
        """Test Gaussian smoothing on 2D signal."""
        signal = np.array([[1.0, 2.0, 1.0], [2.0, 3.0, 2.0], [1.0, 2.0, 1.0]])

        smoothed = smoother.gaussian_smooth(signal, sigma=1.0)

        assert smoothed.shape == signal.shape

    @pytest.mark.unit
    def test_median_smooth(self, smoother):
        """Test median smoothing."""
        signal = np.array([1.0, 2.0, 100.0, 4.0, 5.0])  # Outlier in middle

        smoothed = smoother.median_smooth(signal, size=3)

        assert len(smoothed) == len(signal)
        # Median filter should reduce outlier
        assert smoothed[2] < 100.0

    @pytest.mark.unit
    def test_median_smooth_auto_size(self, smoother):
        """Test median smoothing with auto size."""
        smoother.kernel_size = 3.0
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        smoothed = smoother.median_smooth(signal)

        assert len(smoothed) == len(signal)

    @pytest.mark.unit
    def test_savgol_smooth(self, smoother):
        """Test Savitzky-Golay smoothing."""
        smoother.method = "savgol"
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])

        smoothed = smoother.savgol_smooth(signal, window_length=5, poly_order=3)

        assert len(smoothed) == len(signal)

    @pytest.mark.unit
    def test_smooth_gaussian(self, smoother):
        """Test smooth method with Gaussian."""
        smoother.method = "gaussian"
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        smoothed = smoother.smooth(signal)

        assert len(smoothed) == len(signal)

    @pytest.mark.unit
    def test_smooth_median(self, smoother):
        """Test smooth method with median."""
        smoother.method = "median"
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        smoothed = smoother.smooth(signal)

        assert len(smoothed) == len(signal)

    @pytest.mark.unit
    def test_smooth_savgol(self, smoother):
        """Test smooth method with Savitzky-Golay."""
        smoother.method = "savgol"
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

        smoothed = smoother.smooth(signal)

        assert len(smoothed) == len(signal)

    @pytest.mark.unit
    def test_smooth_unknown_method(self, smoother):
        """Test smooth method with unknown method (should default to Gaussian)."""
        smoother.method = "unknown"
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        smoothed = smoother.smooth(signal)

        assert len(smoothed) == len(signal)


class TestSignalQualityMetrics:
    """Test suite for SignalQualityMetrics class."""

    @pytest.mark.unit
    def test_compute_snr(self):
        """Test computing Signal-to-Noise Ratio."""
        # Create signal with known SNR
        signal = np.array([100.0, 101.0, 99.0, 100.0, 102.0])  # Low noise

        snr = SignalQualityMetrics.compute_snr(signal)

        assert isinstance(snr, float)
        assert snr >= 0 or np.isinf(snr)

    @pytest.mark.unit
    def test_compute_snr_with_noise_estimate(self):
        """Test computing SNR with noise estimate."""
        signal = np.array([100.0, 101.0, 99.0, 100.0, 102.0])
        noise = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        snr = SignalQualityMetrics.compute_snr(signal, noise_estimate=noise)

        assert isinstance(snr, float)

    @pytest.mark.unit
    def test_compute_snr_empty(self):
        """Test computing SNR with empty signal."""
        signal = np.array([])

        snr = SignalQualityMetrics.compute_snr(signal)

        assert snr == 0.0

    @pytest.mark.unit
    def test_compute_coverage(self):
        """Test computing signal coverage."""
        signal = np.array([1.0, 2.0, 0.0, np.nan, 5.0])

        coverage = SignalQualityMetrics.compute_coverage(signal)

        assert 0.0 <= coverage <= 1.0
        # Should be 3/5 = 0.6 (excluding 0 and NaN)
        assert coverage == 0.6

    @pytest.mark.unit
    def test_compute_coverage_with_threshold(self):
        """Test computing coverage with threshold."""
        signal = np.array([1.0, 2.0, 0.5, 0.1, 5.0])

        coverage = SignalQualityMetrics.compute_coverage(signal, threshold=0.5)

        assert 0.0 <= coverage <= 1.0
        # Should exclude values <= 0.5

    @pytest.mark.unit
    def test_compute_uniformity(self):
        """Test computing signal uniformity."""
        signal = np.array([100.0, 100.0, 100.0, 100.0, 100.0])  # Uniform

        uniformity = SignalQualityMetrics.compute_uniformity(signal)

        assert uniformity >= 0
        # Uniform signal should have low coefficient of variation
        assert uniformity < 0.1

    @pytest.mark.unit
    def test_compute_uniformity_variable(self):
        """Test computing uniformity for variable signal."""
        signal = np.array([1.0, 10.0, 100.0, 50.0, 5.0])  # Variable

        uniformity = SignalQualityMetrics.compute_uniformity(signal)

        assert uniformity > 0
        # Variable signal should have higher coefficient of variation

    @pytest.mark.unit
    def test_compute_uniformity_empty(self):
        """Test computing uniformity with empty signal."""
        signal = np.array([])

        uniformity = SignalQualityMetrics.compute_uniformity(signal)

        assert np.isinf(uniformity)

    @pytest.mark.unit
    def test_compute_statistics(self):
        """Test computing comprehensive statistics."""
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        stats = SignalQualityMetrics.compute_statistics(signal)

        assert isinstance(stats, dict)
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "median" in stats
        assert "snr" in stats
        assert "coverage" in stats
        assert "uniformity" in stats
        assert stats["mean"] == 3.0
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0

    @pytest.mark.unit
    def test_compute_statistics_empty(self):
        """Test computing statistics with empty signal."""
        signal = np.array([])

        stats = SignalQualityMetrics.compute_statistics(signal)

        assert stats["mean"] == 0.0
        assert stats["coverage"] == 0.0
        assert np.isinf(stats["uniformity"])


class TestNoiseReductionPipeline:
    """Test suite for NoiseReductionPipeline class."""

    @pytest.fixture
    def pipeline(self):
        """Create a NoiseReductionPipeline instance."""
        return NoiseReductionPipeline()

    @pytest.mark.unit
    def test_pipeline_creation_default(self):
        """Test creating NoiseReductionPipeline with default parameters."""
        pipeline = NoiseReductionPipeline()

        assert pipeline.outlier_detector is not None
        assert pipeline.smoother is not None
        assert pipeline.quality_metrics is not None

    @pytest.mark.unit
    def test_pipeline_creation_custom(self):
        """Test creating NoiseReductionPipeline with custom parameters."""
        pipeline = NoiseReductionPipeline(
            outlier_method="iqr",
            outlier_threshold=2.0,
            smoothing_method="median",
            smoothing_kernel=3.0,
            use_spatial=False,
        )

        assert pipeline.outlier_detector.method == "iqr"
        assert pipeline.smoother.method == "median"

    @pytest.mark.unit
    def test_process_full(self, pipeline):
        """Test processing signal through full pipeline."""
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0, 2.0, 3.0])

        result = pipeline.process(signal)

        assert "original" in result
        assert "cleaned" in result
        assert "outlier_mask" in result
        assert "outlier_scores" in result
        assert "metrics" in result
        assert np.array_equal(result["original"], signal)
        assert len(result["cleaned"]) == len(signal)

    @pytest.mark.unit
    def test_process_no_outlier_removal(self, pipeline):
        """Test processing without outlier removal."""
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = pipeline.process(signal, remove_outliers=False)

        assert "outlier_mask" in result
        assert np.sum(result["outlier_mask"]) == 0

    @pytest.mark.unit
    def test_process_no_smoothing(self, pipeline):
        """Test processing without smoothing."""
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = pipeline.process(signal, apply_smoothing=False)

        # Cleaned should be same as original (if no outliers)
        assert len(result["cleaned"]) == len(signal)

    @pytest.mark.unit
    def test_process_no_metrics(self, pipeline):
        """Test processing without computing metrics."""
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = pipeline.process(signal, compute_metrics=False)

        assert "metrics" not in result

    @pytest.mark.unit
    def test_process_2d_signal(self, pipeline):
        """Test processing 2D signal."""
        signal = np.array([[1.0, 2.0, 3.0], [4.0, 100.0, 6.0], [7.0, 8.0, 9.0]])  # Outlier

        result = pipeline.process(signal)

        assert result["cleaned"].shape == signal.shape
        assert result["outlier_mask"].shape == signal.shape
