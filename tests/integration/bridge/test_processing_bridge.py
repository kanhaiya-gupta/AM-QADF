"""
Bridge tests: processing (SignalProcessing, SignalGeneration) via am_qadf_native.

Aligned with docs/Tests/Test_New_Plans.md (tests/integration/bridge/).
"""

import pytest


@pytest.mark.integration
@pytest.mark.bridge
class TestProcessingBridge:
    """Python → C++ processing API."""

    def test_signal_processing_normalize(self, native_module):
        """SignalProcessing.normalize maps input to output range [min_value, max_value]."""
        SignalProcessing = native_module.processing.SignalProcessing
        proc = SignalProcessing()
        values = [0.0, 50.0, 100.0]
        # C++ API: normalize(values, out_min, out_max); map to [0, 1]
        out = proc.normalize(values, 0.0, 1.0)
        assert len(out) == 3
        assert out[0] == pytest.approx(0.0)
        assert out[1] == pytest.approx(0.5)
        assert out[2] == pytest.approx(1.0)

    def test_signal_processing_moving_average(self, native_module):
        """SignalProcessing.moving_average returns same-length list."""
        SignalProcessing = native_module.processing.SignalProcessing
        proc = SignalProcessing()
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        out = proc.moving_average(values, 3)
        assert len(out) == len(values)

    def test_signal_processing_derivative(self, native_module):
        """SignalProcessing.derivative returns same-length list."""
        SignalProcessing = native_module.processing.SignalProcessing
        proc = SignalProcessing()
        values = [0.0, 1.0, 2.0, 3.0]
        out = proc.derivative(values)
        assert len(out) == len(values)

    def test_signal_processing_integral(self, native_module):
        """SignalProcessing.integral returns same-length list."""
        SignalProcessing = native_module.processing.SignalProcessing
        proc = SignalProcessing()
        values = [1.0, 1.0, 1.0]
        out = proc.integral(values)
        assert len(out) == len(values)

    def test_signal_generation_generate_random(self, native_module):
        """SignalGeneration.generate_random returns list of floats."""
        SignalGeneration = native_module.processing.SignalGeneration
        gen = SignalGeneration()
        out = gen.generate_random(100, 0.0, 1.0, 42)
        assert len(out) == 100
        assert all(isinstance(x, (int, float)) for x in out)
