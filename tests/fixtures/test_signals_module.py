"""
Tests for signals fixture module.

Tests the loading and generation functions in tests/fixtures/signals/__init__.py
"""

import pytest
import numpy as np
from pathlib import Path

try:
    from tests.fixtures.signals import load_sample_signals, generate_sample_signals

    SIGNALS_MODULE_AVAILABLE = True
except ImportError:
    SIGNALS_MODULE_AVAILABLE = False


@pytest.mark.skipif(not SIGNALS_MODULE_AVAILABLE, reason="Signals module not available")
class TestSignalsLoading:
    """Tests for signal loading functions."""

    def test_load_sample_signals(self):
        """Test loading sample signals."""
        signals = load_sample_signals()
        assert signals is not None
        assert isinstance(signals, dict)
        assert len(signals) > 0

    def test_load_from_npz_file(self):
        """Test that loading works from existing npz file."""
        signals = load_sample_signals()
        # If file exists, it should load from file
        # If not, it generates on-the-fly
        assert signals is not None
        assert isinstance(signals, dict)


@pytest.mark.skipif(not SIGNALS_MODULE_AVAILABLE, reason="Signals module not available")
class TestSignalsGeneration:
    """Tests for signal generation functions."""

    def test_generate_sample_signals(self):
        """Test generating sample signals."""
        signals = generate_sample_signals()
        assert signals is not None
        assert isinstance(signals, dict)
        assert len(signals) > 0

    def test_generated_signals_are_arrays(self):
        """Test that generated signals are numpy arrays."""
        signals = generate_sample_signals()
        for signal_name, signal_data in signals.items():
            assert isinstance(signal_data, np.ndarray), f"Signal {signal_name} should be numpy array"
            assert signal_data.ndim == 1, f"Signal {signal_name} should be 1D array"

    def test_generated_signals_have_expected_keys(self):
        """Test that generated signals have expected signal names."""
        signals = generate_sample_signals()
        expected_signals = [
            "laser_power",
            "scan_speed",
            "temperature",
            "density",
            "porosity",
            "velocity",
            "energy_density",
            "exposure_time",
        ]
        for expected in expected_signals:
            assert expected in signals, f"Expected signal {expected} not found"

    def test_generated_signals_have_correct_length(self):
        """Test that generated signals have correct length."""
        signals = generate_sample_signals()
        expected_length = 1000
        for signal_name, signal_data in signals.items():
            assert len(signal_data) == expected_length, f"Signal {signal_name} should have length {expected_length}"

    def test_generated_signals_have_reasonable_ranges(self):
        """Test that generated signals have reasonable value ranges."""
        signals = generate_sample_signals()

        # Check laser_power range (0-300 W)
        if "laser_power" in signals:
            assert np.all(signals["laser_power"] >= 0)
            assert np.all(signals["laser_power"] <= 300)

        # Check temperature range (0-1000 °C)
        if "temperature" in signals:
            assert np.all(signals["temperature"] >= 0)
            assert np.all(signals["temperature"] <= 1000)

        # Check density range (4.0-5.0 g/cm³)
        if "density" in signals:
            assert np.all(signals["density"] >= 4.0)
            assert np.all(signals["density"] <= 5.0)


@pytest.mark.skipif(not SIGNALS_MODULE_AVAILABLE, reason="Signals module not available")
class TestSignalsReproducibility:
    """Tests for signal generation reproducibility."""

    def test_same_seed_produces_same_signals(self):
        """Test that using the same seed produces reproducible results."""
        signals1 = generate_sample_signals()
        signals2 = generate_sample_signals()

        # Both should have same keys
        assert set(signals1.keys()) == set(signals2.keys())

        # Both should have same values (due to fixed seed)
        for key in signals1.keys():
            np.testing.assert_array_equal(signals1[key], signals2[key])
