"""
Unit tests for noise reduction (C++ wrapper).

Tests OutlierDetector, SignalSmoother, SignalQualityMetrics, and NoiseReductionPipeline.
OutlierDetector and SignalSmoother delegate to am_qadf_native.correction.SignalNoiseReduction; skip if not built.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from am_qadf.processing.noise_reduction import (
    OutlierDetector,
    SignalSmoother,
    SignalQualityMetrics,
    NoiseReductionPipeline,
)


class TestOutlierDetector:
    """Test suite for OutlierDetector (C++ wrapper)."""

    @pytest.mark.unit
    def test_outlier_detector_requires_cpp(self):
        """Test that OutlierDetector raises when C++ is not available."""
        import am_qadf.processing.noise_reduction as mod
        with patch.object(mod, "CPP_AVAILABLE", False):
            with pytest.raises(ImportError, match=r"C\+\+ bindings not available"):
                OutlierDetector()

    @pytest.mark.unit
    def test_outlier_detector_creation(self):
        """Test creating OutlierDetector when C++ is available."""
        pytest.importorskip("am_qadf_native.correction", reason="SignalNoiseReduction C++ bindings required")
        det = OutlierDetector()
        assert det._reducer is not None

    @pytest.mark.unit
    def test_outlier_detector_detect_not_implemented(self):
        """Test that detect() raises NotImplementedError."""
        pytest.importorskip("am_qadf_native.correction", reason="SignalNoiseReduction C++ bindings required")
        det = OutlierDetector()
        values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        with pytest.raises(NotImplementedError, match="OutlierDetector.detect|not yet fully implemented"):
            det.detect(values, method="iqr")


class TestSignalSmoother:
    """Test suite for SignalSmoother (C++ wrapper)."""

    @pytest.mark.unit
    def test_signal_smoother_requires_cpp(self):
        """Test that SignalSmoother raises when C++ is not available."""
        import am_qadf.processing.noise_reduction as mod
        with patch.object(mod, "CPP_AVAILABLE", False):
            with pytest.raises(ImportError, match=r"C\+\+ bindings not available"):
                SignalSmoother()

    @pytest.mark.unit
    def test_signal_smoother_creation(self):
        """Test creating SignalSmoother when C++ is available."""
        pytest.importorskip("am_qadf_native.correction", reason="SignalNoiseReduction C++ bindings required")
        smoother = SignalSmoother()
        assert smoother._reducer is not None

    @pytest.mark.unit
    def test_signal_smoother_smooth_gaussian(self):
        """Test smooth with gaussian method returns array; uses fixture data when available."""
        pytest.importorskip("am_qadf_native.correction", reason="SignalNoiseReduction C++ bindings required")
        try:
            from tests.fixtures.signals import generate_sample_signals
            signals = generate_sample_signals()
            values = np.asarray(signals["temperature"], dtype=np.float32)
        except ImportError:
            # Fallback: small array (fixtures package not on path)
            values = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        smoother = SignalSmoother()
        result = smoother.smooth(values, method="gaussian", sigma=1.0)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert len(result) == len(values)

    @pytest.mark.unit
    def test_signal_smoother_smooth_unknown_method_raises(self):
        """Test smooth with unknown method raises ValueError."""
        pytest.importorskip("am_qadf_native.correction", reason="SignalNoiseReduction C++ bindings required")
        smoother = SignalSmoother()
        values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        with pytest.raises(ValueError, match="Unknown smoothing method"):
            smoother.smooth(values, method="unknown")


class TestSignalQualityMetrics:
    """Test suite for SignalQualityMetrics (C++ wrapper)."""

    @pytest.mark.unit
    def test_signal_quality_metrics_requires_cpp(self):
        """Test that SignalQualityMetrics raises when C++ is not available."""
        import am_qadf.processing.noise_reduction as mod
        with patch.object(mod, "CPP_AVAILABLE", False):
            with pytest.raises(ImportError, match=r"C\+\+ bindings not available"):
                SignalQualityMetrics()

    @pytest.mark.unit
    def test_signal_quality_metrics_compute_not_implemented(self):
        """Test that compute() raises NotImplementedError."""
        import am_qadf.processing.noise_reduction as mod
        if not getattr(mod, "CPP_AVAILABLE", False):
            pytest.skip("am_qadf_native.correction not available")
        qm = SignalQualityMetrics()
        values = np.array([1.0, 2.0, 3.0])
        with pytest.raises(NotImplementedError, match="SignalQualityMetrics.compute|not yet fully implemented"):
            qm.compute(values)


class TestNoiseReductionPipeline:
    """Test suite for NoiseReductionPipeline (C++ wrapper)."""

    @pytest.mark.unit
    def test_noise_reduction_pipeline_requires_cpp(self):
        """Test that NoiseReductionPipeline raises when C++ is not available."""
        import am_qadf.processing.noise_reduction as mod
        with patch.object(mod, "CPP_AVAILABLE", False):
            with pytest.raises(ImportError, match=r"C\+\+ bindings not available"):
                NoiseReductionPipeline()

    @pytest.mark.unit
    def test_noise_reduction_pipeline_creation(self):
        """Test creating NoiseReductionPipeline when C++ is available."""
        pytest.importorskip("am_qadf_native.correction", reason="SignalNoiseReduction C++ bindings required")
        pipeline = NoiseReductionPipeline()
        assert pipeline._reducer is not None
        assert pipeline._detector is not None
        assert pipeline._smoother is not None

    @pytest.mark.unit
    def test_noise_reduction_pipeline_process_not_implemented(self):
        """Test that process() raises NotImplementedError."""
        pytest.importorskip("am_qadf_native.correction", reason="SignalNoiseReduction C++ bindings required")
        pipeline = NoiseReductionPipeline()
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        with pytest.raises(NotImplementedError, match="NoiseReductionPipeline.process|not yet fully implemented"):
            pipeline.process(values)
