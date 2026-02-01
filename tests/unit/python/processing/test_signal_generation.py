"""
Unit tests for signal generation (C++ wrapper).

Tests ThermalFieldGenerator, DensityFieldEstimator, and StressFieldGenerator.
All delegate to am_qadf_native.processing.SignalGeneration; skip if not built.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from am_qadf.processing.signal_generation import (
    ThermalFieldGenerator,
    DensityFieldEstimator,
    StressFieldGenerator,
)


class TestThermalFieldGenerator:
    """Test suite for ThermalFieldGenerator (C++ wrapper)."""

    @pytest.mark.unit
    def test_thermal_field_generator_requires_cpp(self):
        """Test that ThermalFieldGenerator raises when C++ is not available."""
        import am_qadf.processing.signal_generation as mod
        with patch.object(mod, "CPP_AVAILABLE", False):
            with pytest.raises(ImportError, match=r"C\+\+ bindings not available"):
                ThermalFieldGenerator()

    @pytest.mark.unit
    def test_thermal_field_generator_creation(self):
        """Test creating ThermalFieldGenerator when C++ is available."""
        pytest.importorskip("am_qadf_native.processing", reason="SignalGeneration C++ bindings required")
        gen = ThermalFieldGenerator()
        assert gen._generator is not None

    @pytest.mark.unit
    def test_thermal_field_generator_generate(self):
        """Test generate returns array of thermal values."""
        pytest.importorskip("am_qadf_native.processing", reason="SignalGeneration C++ bindings required")
        gen = ThermalFieldGenerator()
        points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
        values = gen.generate(points, amplitude=1.0, frequency=1.0)
        assert isinstance(values, np.ndarray)
        assert values.dtype == np.float32
        assert len(values) == len(points)


class TestDensityFieldEstimator:
    """Test suite for DensityFieldEstimator (C++ wrapper)."""

    @pytest.mark.unit
    def test_density_field_estimator_requires_cpp(self):
        """Test that DensityFieldEstimator raises when C++ is not available."""
        import am_qadf.processing.signal_generation as mod
        with patch.object(mod, "CPP_AVAILABLE", False):
            with pytest.raises(ImportError, match=r"C\+\+ bindings not available"):
                DensityFieldEstimator()

    @pytest.mark.unit
    def test_density_field_estimator_creation(self):
        """Test creating DensityFieldEstimator when C++ is available."""
        pytest.importorskip("am_qadf_native.processing", reason="SignalGeneration C++ bindings required")
        est = DensityFieldEstimator()
        assert est._generator is not None

    @pytest.mark.unit
    def test_density_field_estimator_estimate(self):
        """Test estimate returns array of density values."""
        pytest.importorskip("am_qadf_native.processing", reason="SignalGeneration C++ bindings required")
        est = DensityFieldEstimator()
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
        values = est.estimate(points, center=(0.5, 0.0, 0.0), amplitude=1.0, sigma=1.0)
        assert isinstance(values, np.ndarray)
        assert values.dtype == np.float32
        assert len(values) == len(points)


class TestStressFieldGenerator:
    """Test suite for StressFieldGenerator (C++ wrapper)."""

    @pytest.mark.unit
    def test_stress_field_generator_requires_cpp(self):
        """Test that StressFieldGenerator raises when C++ is not available."""
        import am_qadf.processing.signal_generation as mod
        with patch.object(mod, "CPP_AVAILABLE", False):
            with pytest.raises(ImportError, match=r"C\+\+ bindings not available"):
                StressFieldGenerator()

    @pytest.mark.unit
    def test_stress_field_generator_creation(self):
        """Test creating StressFieldGenerator when C++ is available."""
        pytest.importorskip("am_qadf_native.processing", reason="SignalGeneration C++ bindings required")
        gen = StressFieldGenerator()
        assert gen._generator is not None

    @pytest.mark.unit
    def test_stress_field_generator_generate_synthetic(self):
        """Test generate without expression returns array."""
        pytest.importorskip("am_qadf_native.processing", reason="SignalGeneration C++ bindings required")
        gen = StressFieldGenerator()
        points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
        values = gen.generate(points, amplitude=1.0, frequency=1.0)
        assert isinstance(values, np.ndarray)
        assert values.dtype == np.float32
        assert len(values) == len(points)

    @pytest.mark.unit
    def test_stress_field_generator_generate_from_expression(self):
        """Test generate with expression returns array."""
        pytest.importorskip("am_qadf_native.processing", reason="SignalGeneration C++ bindings required")
        gen = StressFieldGenerator()
        points = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        values = gen.generate(points, expression="x + y + z")
        assert isinstance(values, np.ndarray)
        assert values.dtype == np.float32
        assert len(values) == len(points)
