"""
Calibration

Calibration data management and reference measurements.
"""

from typing import Optional, Dict, List, Tuple, Any
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ReferenceMeasurement:
    """Represents a reference measurement for calibration."""

    point: Tuple[float, float, float]  # Measured point
    expected_point: Tuple[float, float, float]  # Expected (true) point
    timestamp: Optional[datetime] = None
    measurement_type: str = "manual"  # "manual", "ct_scan", "cmm", etc.
    uncertainty: float = 0.0  # Measurement uncertainty (mm)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CalibrationData:
    """Calibration data for a coordinate system or sensor."""

    name: str
    calibration_date: datetime
    reference_measurements: List[ReferenceMeasurement] = field(default_factory=list)
    transformation_matrix: Optional[np.ndarray] = None  # 4x4 transformation
    distortion_parameters: Dict[str, Any] = field(default_factory=dict)
    uncertainty: float = 0.0  # Overall calibration uncertainty
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_measurement(self, measurement: ReferenceMeasurement):
        """Add a reference measurement."""
        self.reference_measurements.append(measurement)

    def compute_error(self) -> Dict[str, float]:
        """
        Compute calibration error from reference measurements.

        Returns:
            Dictionary of error metrics
        """
        if len(self.reference_measurements) == 0:
            return {
                "mean_error": 0.0,
                "max_error": 0.0,
                "rms_error": 0.0,
                "num_measurements": 0,
            }

        errors = []
        for meas in self.reference_measurements:
            measured = np.array(meas.point)
            expected = np.array(meas.expected_point)
            error = np.linalg.norm(measured - expected)
            errors.append(error)

        errors = np.array(errors)

        return {
            "mean_error": float(np.mean(errors)),
            "max_error": float(np.max(errors)),
            "rms_error": float(np.sqrt(np.mean(errors**2))),
            "std_error": float(np.std(errors)),
            "num_measurements": len(self.reference_measurements),
        }


class CalibrationManager:
    """
    Manage calibration data for multiple coordinate systems and sensors.
    """

    def __init__(self):
        """Initialize calibration manager."""
        self._calibrations: Dict[str, CalibrationData] = {}

    def register_calibration(self, name: str, calibration: CalibrationData):
        """
        Register calibration data.

        Args:
            name: Calibration name/identifier
            calibration: CalibrationData object
        """
        self._calibrations[name] = calibration

    def get_calibration(self, name: str) -> Optional[CalibrationData]:
        """
        Get calibration data.

        Args:
            name: Calibration name

        Returns:
            CalibrationData or None
        """
        return self._calibrations.get(name)

    def list_calibrations(self) -> List[str]:
        """
        List all registered calibrations.

        Returns:
            List of calibration names
        """
        return list(self._calibrations.keys())

    def estimate_transformation(self, calibration_name: str) -> Optional[np.ndarray]:
        """
        Estimate transformation matrix from calibration measurements.

        Uses point cloud registration to find best-fit transformation.

        Args:
            calibration_name: Name of calibration

        Returns:
            4x4 transformation matrix or None
        """
        calibration = self.get_calibration(calibration_name)
        if calibration is None or len(calibration.reference_measurements) == 0:
            return None

        # Extract measured and expected points
        measured_points = np.array([m.point for m in calibration.reference_measurements])
        expected_points = np.array([m.expected_point for m in calibration.reference_measurements])

        # Compute centroids
        measured_centroid = np.mean(measured_points, axis=0)
        expected_centroid = np.mean(expected_points, axis=0)

        # Center points
        measured_centered = measured_points - measured_centroid
        expected_centered = expected_points - expected_centroid

        # Compute rotation using SVD
        H = measured_centered.T @ expected_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Ensure proper rotation (det(R) = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Compute translation
        t = expected_centroid - R @ measured_centroid

        # Build 4x4 transformation matrix
        matrix = np.eye(4)
        matrix[0:3, 0:3] = R
        matrix[0:3, 3] = t

        return matrix

    def validate_calibration(self, calibration_name: str, threshold: float = 0.1) -> Dict[str, Any]:  # mm
        """
        Validate calibration quality.

        Args:
            calibration_name: Name of calibration
            threshold: Maximum acceptable error (mm)

        Returns:
            Dictionary of validation results
        """
        calibration = self.get_calibration(calibration_name)
        if calibration is None:
            return {"valid": False, "error": "Calibration not found"}

        error_metrics = calibration.compute_error()

        # Check if errors are within threshold
        valid = error_metrics["mean_error"] <= threshold and error_metrics["max_error"] <= threshold * 2

        return {
            "valid": valid,
            "error_metrics": error_metrics,
            "threshold": threshold,
            "within_threshold": error_metrics["max_error"] <= threshold * 2,
        }

    def apply_calibration_correction(self, points: np.ndarray, calibration_name: str) -> np.ndarray:
        """
        Apply calibration correction to points.

        Args:
            points: Points to correct (N, 3)
            calibration_name: Name of calibration to use

        Returns:
            Corrected points
        """
        calibration = self.get_calibration(calibration_name)
        if calibration is None:
            return points

        points = np.asarray(points)
        if points.ndim == 1:
            points = points.reshape(1, -1)

        # Use transformation matrix if available
        if calibration.transformation_matrix is not None:
            # Import TransformationMatrix with fallback
            try:
                from ..synchronization.spatial_transformation import (
                    TransformationMatrix,
                )
            except ImportError:
                # Fallback: create a simple transformation wrapper
                import sys
                from pathlib import Path

                current_file = Path(__file__).resolve()
                spatial_path = current_file.parent.parent / "synchronization" / "spatial_transformation.py"
                if spatial_path.exists():
                    import importlib.util

                    spec = importlib.util.spec_from_file_location("spatial_transformation", spatial_path)
                    spatial_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(spatial_module)
                    TransformationMatrix = spatial_module.TransformationMatrix
                else:
                    raise ImportError("Could not import TransformationMatrix")

            trans = TransformationMatrix(matrix=calibration.transformation_matrix)
            return trans.apply(points)

        # Otherwise, estimate from measurements
        transformation = self.estimate_transformation(calibration_name)
        if transformation is not None:
            # Import TransformationMatrix with fallback
            try:
                from ..synchronization.spatial_transformation import (
                    TransformationMatrix,
                )
            except ImportError:
                # Fallback: create a simple transformation wrapper
                import sys
                from pathlib import Path

                current_file = Path(__file__).resolve()
                spatial_path = current_file.parent.parent / "synchronization" / "spatial_transformation.py"
                if spatial_path.exists():
                    import importlib.util

                    spec = importlib.util.spec_from_file_location("spatial_transformation", spatial_path)
                    spatial_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(spatial_module)
                    TransformationMatrix = spatial_module.TransformationMatrix
                else:
                    raise ImportError("Could not import TransformationMatrix")

            trans = TransformationMatrix(matrix=transformation)
            return trans.apply(points)

        return points
