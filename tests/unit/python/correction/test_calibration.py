"""
Unit tests for calibration (C++ wrapper).

Tests ReferenceMeasurement, CalibrationData, and CalibrationManager.
CalibrationManager delegates to am_qadf_native.correction.Calibration; skip if not built.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

from am_qadf.correction.calibration import (
    ReferenceMeasurement,
    CalibrationData,
    CalibrationManager,
)


class TestReferenceMeasurement:
    """Test suite for ReferenceMeasurement dataclass."""

    @pytest.mark.unit
    def test_reference_measurement_creation(self):
        """Test creating ReferenceMeasurement."""
        m = ReferenceMeasurement(point=(1.0, 2.0, 3.0), value=42.0, timestamp=1000.0)
        assert m.point == (1.0, 2.0, 3.0)
        assert m.value == 42.0
        assert m.timestamp == 1000.0

    @pytest.mark.unit
    def test_reference_measurement_timestamp_optional(self):
        """Test ReferenceMeasurement with optional timestamp."""
        m = ReferenceMeasurement(point=(0.0, 0.0, 0.0), value=0.0)
        assert m.timestamp is None


class TestCalibrationData:
    """Test suite for CalibrationData dataclass."""

    @pytest.mark.unit
    def test_calibration_data_creation(self):
        """Test creating CalibrationData."""
        data = CalibrationData(
            sensor_id="s1",
            calibration_type="intrinsic",
            parameters={"k1": 0.1},
            reference_points=[(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)],
            measured_points=[(0.01, 0.01, 0.01), (1.01, 1.01, 1.01)],
        )
        assert data.sensor_id == "s1"
        assert data.calibration_type == "intrinsic"
        assert data.parameters == {"k1": 0.1}
        assert len(data.reference_points) == 2
        assert len(data.measured_points) == 2


class TestCalibrationManager:
    """Test suite for CalibrationManager (C++ wrapper)."""

    @pytest.mark.unit
    def test_calibration_manager_requires_cpp(self):
        """Test that CalibrationManager raises when am_qadf_native is not available."""
        import am_qadf.correction.calibration as cal_mod
        with patch.object(cal_mod, "CPP_AVAILABLE", False):
            with pytest.raises(ImportError, match=r"C\+\+ bindings not available"):
                CalibrationManager()

    @pytest.mark.unit
    def test_calibration_manager_creation(self):
        """Test creating CalibrationManager when C++ is available."""
        pytest.importorskip("am_qadf_native.correction", reason="Calibration C++ bindings required")
        mgr = CalibrationManager()
        assert mgr._calibration is not None

    @pytest.mark.unit
    def test_compute_calibration(self):
        """Test compute_calibration returns CalibrationData."""
        pytest.importorskip("am_qadf_native", reason="Calibration C++ bindings required")
        mgr = CalibrationManager()
        ref_pts = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
        meas_pts = [(0.01, 0.0, 0.0), (1.01, 0.0, 0.0)]
        data = mgr.compute_calibration(ref_pts, meas_pts, calibration_type="intrinsic")
        assert isinstance(data, CalibrationData)
        # C++ computeCalibration sets calibration_type, reference_points, measured_points, parameters; sensor_id may be left empty
        assert data.calibration_type == "intrinsic"
        assert isinstance(data.reference_points, list)
        assert isinstance(data.measured_points, list)
        assert isinstance(data.parameters, dict)

    @pytest.mark.unit
    def test_validate_calibration(self):
        """Test validate_calibration returns bool."""
        pytest.importorskip("am_qadf_native.correction", reason="Calibration C++ bindings required")
        mgr = CalibrationManager()
        data = CalibrationData(
            sensor_id="s1",
            calibration_type="intrinsic",
            parameters={},
            reference_points=[(0.0, 0.0, 0.0)],
            measured_points=[(0.0, 0.0, 0.0)],
        )
        result = mgr.validate_calibration(data)
        assert isinstance(result, bool)
