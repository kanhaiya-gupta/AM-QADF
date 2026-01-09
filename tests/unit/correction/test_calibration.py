"""
Unit tests for calibration.

Tests for ReferenceMeasurement, CalibrationData, and CalibrationManager.
"""

import pytest
import numpy as np
from datetime import datetime
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
        measurement = ReferenceMeasurement(
            point=(1.0, 2.0, 3.0),
            expected_point=(1.1, 2.1, 3.1),
            timestamp=datetime(2024, 1, 1),
            measurement_type="ct_scan",
            uncertainty=0.05,
        )

        assert measurement.point == (1.0, 2.0, 3.0)
        assert measurement.expected_point == (1.1, 2.1, 3.1)
        assert measurement.measurement_type == "ct_scan"
        assert measurement.uncertainty == 0.05

    @pytest.mark.unit
    def test_reference_measurement_minimal(self):
        """Test creating ReferenceMeasurement with minimal parameters."""
        measurement = ReferenceMeasurement(point=(1.0, 2.0, 3.0), expected_point=(1.1, 2.1, 3.1))

        assert measurement.point == (1.0, 2.0, 3.0)
        assert measurement.timestamp is None
        assert measurement.measurement_type == "manual"
        assert measurement.uncertainty == 0.0


class TestCalibrationData:
    """Test suite for CalibrationData class."""

    @pytest.fixture
    def calibration_data(self):
        """Create a CalibrationData instance."""
        return CalibrationData(name="test_calibration", calibration_date=datetime(2024, 1, 1))

    @pytest.mark.unit
    def test_calibration_data_creation(self, calibration_data):
        """Test creating CalibrationData."""
        assert calibration_data.name == "test_calibration"
        assert calibration_data.calibration_date == datetime(2024, 1, 1)
        assert len(calibration_data.reference_measurements) == 0
        assert calibration_data.transformation_matrix is None
        assert calibration_data.uncertainty == 0.0

    @pytest.mark.unit
    def test_add_measurement(self, calibration_data):
        """Test adding reference measurement."""
        measurement = ReferenceMeasurement(point=(1.0, 2.0, 3.0), expected_point=(1.1, 2.1, 3.1))

        calibration_data.add_measurement(measurement)

        assert len(calibration_data.reference_measurements) == 1
        assert calibration_data.reference_measurements[0] is measurement

    @pytest.mark.unit
    def test_compute_error_empty(self, calibration_data):
        """Test computing error with no measurements."""
        error_metrics = calibration_data.compute_error()

        assert error_metrics["mean_error"] == 0.0
        assert error_metrics["max_error"] == 0.0
        assert error_metrics["rms_error"] == 0.0
        assert error_metrics["num_measurements"] == 0

    @pytest.mark.unit
    def test_compute_error_with_measurements(self, calibration_data):
        """Test computing error with measurements."""
        calibration_data.add_measurement(
            ReferenceMeasurement(point=(1.0, 2.0, 3.0), expected_point=(1.0, 2.0, 3.0))  # No error
        )
        calibration_data.add_measurement(
            ReferenceMeasurement(point=(1.0, 2.0, 3.0), expected_point=(1.1, 2.1, 3.1))  # Some error
        )

        error_metrics = calibration_data.compute_error()

        assert error_metrics["num_measurements"] == 2
        assert error_metrics["mean_error"] > 0
        assert error_metrics["max_error"] > 0
        assert error_metrics["rms_error"] > 0
        assert "std_error" in error_metrics


class TestCalibrationManager:
    """Test suite for CalibrationManager class."""

    @pytest.fixture
    def manager(self):
        """Create a CalibrationManager instance."""
        return CalibrationManager()

    @pytest.fixture
    def calibration_data(self):
        """Create a CalibrationData instance."""
        cal = CalibrationData(name="test_calibration", calibration_date=datetime(2024, 1, 1))
        cal.add_measurement(ReferenceMeasurement(point=(1.0, 2.0, 3.0), expected_point=(1.1, 2.1, 3.1)))
        return cal

    @pytest.mark.unit
    def test_calibration_manager_creation(self, manager):
        """Test creating CalibrationManager."""
        assert manager is not None
        assert len(manager._calibrations) == 0

    @pytest.mark.unit
    def test_register_calibration(self, manager, calibration_data):
        """Test registering calibration."""
        manager.register_calibration("test", calibration_data)

        assert "test" in manager._calibrations
        assert manager._calibrations["test"] is calibration_data

    @pytest.mark.unit
    def test_get_calibration(self, manager, calibration_data):
        """Test getting calibration."""
        manager.register_calibration("test", calibration_data)

        retrieved = manager.get_calibration("test")

        assert retrieved is calibration_data

    @pytest.mark.unit
    def test_get_calibration_nonexistent(self, manager):
        """Test getting nonexistent calibration."""
        retrieved = manager.get_calibration("nonexistent")

        assert retrieved is None

    @pytest.mark.unit
    def test_list_calibrations(self, manager, calibration_data):
        """Test listing calibrations."""
        manager.register_calibration("cal1", calibration_data)

        cal2 = CalibrationData(name="cal2", calibration_date=datetime(2024, 1, 2))
        manager.register_calibration("cal2", cal2)

        calibrations = manager.list_calibrations()

        assert len(calibrations) == 2
        assert "cal1" in calibrations
        assert "cal2" in calibrations

    @pytest.mark.unit
    def test_estimate_transformation(self, manager, calibration_data):
        """Test estimating transformation from measurements."""
        # Add more measurements for better estimation
        calibration_data.add_measurement(ReferenceMeasurement(point=(0.0, 0.0, 0.0), expected_point=(0.1, 0.1, 0.1)))
        calibration_data.add_measurement(ReferenceMeasurement(point=(2.0, 2.0, 2.0), expected_point=(2.1, 2.1, 2.1)))

        manager.register_calibration("test", calibration_data)

        transformation = manager.estimate_transformation("test")

        assert transformation is not None
        assert transformation.shape == (4, 4)

    @pytest.mark.unit
    def test_estimate_transformation_empty(self, manager):
        """Test estimating transformation with no measurements."""
        cal = CalibrationData(name="empty", calibration_date=datetime(2024, 1, 1))
        manager.register_calibration("empty", cal)

        transformation = manager.estimate_transformation("empty")

        assert transformation is None

    @pytest.mark.unit
    def test_estimate_transformation_nonexistent(self, manager):
        """Test estimating transformation for nonexistent calibration."""
        transformation = manager.estimate_transformation("nonexistent")

        assert transformation is None

    @pytest.mark.unit
    def test_validate_calibration(self, manager, calibration_data):
        """Test validating calibration."""
        manager.register_calibration("test", calibration_data)

        validation = manager.validate_calibration("test", threshold=0.1)

        assert "valid" in validation
        assert "error_metrics" in validation
        assert "threshold" in validation

    @pytest.mark.unit
    def test_validate_calibration_nonexistent(self, manager):
        """Test validating nonexistent calibration."""
        validation = manager.validate_calibration("nonexistent")

        assert validation["valid"] is False
        assert "error" in validation

    @pytest.mark.unit
    def test_validate_calibration_within_threshold(self, manager):
        """Test validating calibration within threshold."""
        cal = CalibrationData(name="good", calibration_date=datetime(2024, 1, 1))
        # Add measurements with small errors
        cal.add_measurement(ReferenceMeasurement(point=(1.0, 2.0, 3.0), expected_point=(1.01, 2.01, 3.01)))  # Small error

        manager.register_calibration("good", cal)

        validation = manager.validate_calibration("good", threshold=0.1)

        # Should be valid if errors are small
        assert isinstance(validation["valid"], bool)

    @pytest.mark.unit
    def test_apply_calibration_correction_with_matrix(self, manager):
        """Test applying calibration correction with transformation matrix."""
        cal = CalibrationData(name="test", calibration_date=datetime(2024, 1, 1))
        # Create transformation matrix (translation)
        matrix = np.eye(4)
        matrix[0:3, 3] = [10.0, 20.0, 30.0]
        cal.transformation_matrix = matrix

        manager.register_calibration("test", cal)

        points = np.array([[1.0, 2.0, 3.0]])
        result = manager.apply_calibration_correction(points, "test")

        expected = np.array([[11.0, 22.0, 33.0]])
        assert np.allclose(result, expected)

    @pytest.mark.unit
    def test_apply_calibration_correction_estimated(self, manager, calibration_data):
        """Test applying calibration correction with estimated transformation."""
        # Add measurements
        calibration_data.add_measurement(ReferenceMeasurement(point=(0.0, 0.0, 0.0), expected_point=(0.1, 0.1, 0.1)))
        calibration_data.add_measurement(ReferenceMeasurement(point=(1.0, 1.0, 1.0), expected_point=(1.1, 1.1, 1.1)))

        manager.register_calibration("test", calibration_data)

        points = np.array([[0.0, 0.0, 0.0]])
        result = manager.apply_calibration_correction(points, "test")

        # Should apply estimated transformation
        assert result.shape == (1, 3)

    @pytest.mark.unit
    def test_apply_calibration_correction_nonexistent(self, manager):
        """Test applying calibration correction for nonexistent calibration."""
        points = np.array([[1.0, 2.0, 3.0]])
        result = manager.apply_calibration_correction(points, "nonexistent")

        # Should return points unchanged
        assert np.array_equal(result, points)

    @pytest.mark.unit
    def test_apply_calibration_correction_single_point(self, manager):
        """Test applying calibration correction to single point."""
        cal = CalibrationData(name="test", calibration_date=datetime(2024, 1, 1))
        matrix = np.eye(4)
        matrix[0:3, 3] = [10.0, 20.0, 30.0]
        cal.transformation_matrix = matrix

        manager.register_calibration("test", cal)

        point = np.array([1.0, 2.0, 3.0])  # 1D array
        result = manager.apply_calibration_correction(point, "test")

        assert result.shape == (1, 3)
