"""
Calibration - C++ Wrapper

Thin Python wrapper for C++ calibration implementation.
All core computation is done in C++.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np

try:
    from am_qadf_native.correction import Calibration, CalibrationData as CppCalibrationData
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False
    Calibration = None
    CppCalibrationData = None


@dataclass
class ReferenceMeasurement:
    """Reference measurement data."""
    point: Tuple[float, float, float]
    value: float
    timestamp: Optional[float] = None


@dataclass
class CalibrationData:
    """Calibration data structure - Python wrapper."""
    sensor_id: str
    calibration_type: str  # "intrinsic", "extrinsic", "distortion"
    parameters: Dict[str, float]
    reference_points: List[Tuple[float, float, float]]
    measured_points: List[Tuple[float, float, float]]


class CalibrationManager:
    """
    Calibration manager - C++ wrapper.
    
    This is a thin wrapper around the C++ Calibration implementation.
    All core computation is done in C++.
    """

    def __init__(self):
        """Initialize calibration manager."""
        if not CPP_AVAILABLE:
            raise ImportError(
                "C++ bindings not available. "
                "Please build am_qadf_native with pybind11 bindings."
            )
        self._calibration = Calibration()

    def load_from_file(self, filename: str) -> CalibrationData:
        """
        Load calibration data from file.

        Args:
            filename: Path to calibration file

        Returns:
            CalibrationData object
        """
        cpp_data = self._calibration.load_from_file(filename)
        
        # Convert C++ CalibrationData to Python CalibrationData
        return CalibrationData(
            sensor_id=cpp_data.sensor_id,
            calibration_type=cpp_data.calibration_type,
            parameters=dict(cpp_data.parameters),
            reference_points=[tuple(p) for p in cpp_data.reference_points],
            measured_points=[tuple(p) for p in cpp_data.measured_points]
        )

    def save_to_file(self, data: CalibrationData, filename: str):
        """
        Save calibration data to file.

        Args:
            data: CalibrationData object
            filename: Path to output file
        """
        # Convert Python CalibrationData to C++ CalibrationData
        cpp_data = CppCalibrationData()
        cpp_data.sensor_id = data.sensor_id
        cpp_data.calibration_type = data.calibration_type
        cpp_data.parameters = data.parameters
        cpp_data.reference_points = [list(p) for p in data.reference_points]
        cpp_data.measured_points = [list(p) for p in data.measured_points]
        
        self._calibration.save_to_file(cpp_data, filename)

    def compute_calibration(
        self,
        reference_points: List[Tuple[float, float, float]],
        measured_points: List[Tuple[float, float, float]],
        calibration_type: str = "intrinsic"
    ) -> CalibrationData:
        """
        Compute calibration parameters from reference and measured points.

        Args:
            reference_points: List of reference point coordinates
            measured_points: List of measured point coordinates
            calibration_type: Type of calibration ('intrinsic', 'extrinsic', 'distortion')

        Returns:
            CalibrationData object
        """
        ref_points_cpp = [list(p) for p in reference_points]
        meas_points_cpp = [list(p) for p in measured_points]
        
        cpp_data = self._calibration.compute_calibration(
            ref_points_cpp, meas_points_cpp, calibration_type
        )
        
        # Convert to Python CalibrationData
        return CalibrationData(
            sensor_id=cpp_data.sensor_id,
            calibration_type=cpp_data.calibration_type,
            parameters=dict(cpp_data.parameters),
            reference_points=[tuple(p) for p in cpp_data.reference_points],
            measured_points=[tuple(p) for p in cpp_data.measured_points]
        )

    def validate_calibration(self, data: CalibrationData) -> bool:
        """
        Validate calibration data.

        Args:
            data: CalibrationData object

        Returns:
            True if calibration is valid
        """
        # Convert to C++ CalibrationData
        cpp_data = CppCalibrationData()
        cpp_data.sensor_id = data.sensor_id
        cpp_data.calibration_type = data.calibration_type
        cpp_data.parameters = data.parameters
        cpp_data.reference_points = [list(p) for p in data.reference_points]
        cpp_data.measured_points = [list(p) for p in data.measured_points]
        
        return self._calibration.validate_calibration(cpp_data)
