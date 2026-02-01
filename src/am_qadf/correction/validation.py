"""
Validation - C++ Wrapper

Thin Python wrapper for C++ validation implementation.
All core computation is done in C++.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np

try:
    from am_qadf_native.correction import Validation, ValidationResult as CppValidationResult
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False
    Validation = None
    CppValidationResult = None

from ..voxelization.uniform_resolution import VoxelGrid
from ..signal_mapping.utils._conversion import voxelgrid_to_floatgrid


@dataclass
class AlignmentQuality:
    """Alignment quality metrics."""
    spatial_error: float
    temporal_error: float
    overall_quality: float
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ValidationMetrics:
    """Validation metrics."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


class CorrectionValidator:
    """
    Correction validator - C++ wrapper.
    
    This is a thin wrapper around the C++ Validation implementation.
    All core computation is done in C++.
    """

    def __init__(self):
        """Initialize correction validator."""
        if not CPP_AVAILABLE:
            raise ImportError(
                "C++ bindings not available. "
                "Please build am_qadf_native with pybind11 bindings."
            )
        self._validator = Validation()

    def validate_grid(self, grid: VoxelGrid, signal_name: Optional[str] = None) -> ValidationMetrics:
        """
        Validate voxel grid.

        Args:
            grid: VoxelGrid to validate
            signal_name: Optional signal name to validate (if None, validates all signals)

        Returns:
            ValidationMetrics object
        """
        # Convert VoxelGrid to FloatGrid
        if signal_name:
            openvdb_grid = voxelgrid_to_floatgrid(grid, signal_name, default=0.0)
        else:
            # Use first available signal
            if len(grid.available_signals) == 0:
                raise ValueError("No signals available in grid")
            signal_name = list(grid.available_signals)[0]
            openvdb_grid = voxelgrid_to_floatgrid(grid, signal_name, default=0.0)

        # Call C++ validator
        cpp_result = self._validator.validate_grid(openvdb_grid)

        # Convert to Python ValidationMetrics
        return ValidationMetrics(
            is_valid=cpp_result.is_valid,
            errors=list(cpp_result.errors),
            warnings=list(cpp_result.warnings),
            metrics=dict(cpp_result.metrics)
        )

    def validate_signal_data(
        self,
        values: np.ndarray,
        min_value: float = 0.0,
        max_value: float = 1.0
    ) -> ValidationMetrics:
        """
        Validate signal data.

        Args:
            values: Array of signal values
            min_value: Minimum expected value
            max_value: Maximum expected value

        Returns:
            ValidationMetrics object
        """
        values_cpp = values.astype(np.float32).tolist()
        cpp_result = self._validator.validate_signal_data(values_cpp, min_value, max_value)

        return ValidationMetrics(
            is_valid=cpp_result.is_valid,
            errors=list(cpp_result.errors),
            warnings=list(cpp_result.warnings),
            metrics=dict(cpp_result.metrics)
        )

    def validate_coordinates(
        self,
        points: np.ndarray,
        bbox_min: Tuple[float, float, float],
        bbox_max: Tuple[float, float, float]
    ) -> ValidationMetrics:
        """
        Validate coordinate data.

        Args:
            points: Array of points (N, 3)
            bbox_min: Minimum bounding box
            bbox_max: Maximum bounding box

        Returns:
            ValidationMetrics object
        """
        points_cpp = [list(p) for p in points]
        bbox_min_cpp = list(bbox_min)
        bbox_max_cpp = list(bbox_max)

        cpp_result = self._validator.validate_coordinates(points_cpp, bbox_min_cpp, bbox_max_cpp)

        return ValidationMetrics(
            is_valid=cpp_result.is_valid,
            errors=list(cpp_result.errors),
            warnings=list(cpp_result.warnings),
            metrics=dict(cpp_result.metrics)
        )

    def check_consistency(
        self,
        grid1: VoxelGrid,
        grid2: VoxelGrid,
        tolerance: float = 1e-6,
        signal_name: Optional[str] = None
    ) -> bool:
        """
        Check consistency between two grids.

        Args:
            grid1: First VoxelGrid
            grid2: Second VoxelGrid
            tolerance: Tolerance for comparison
            signal_name: Optional signal name to compare (if None, uses first available)

        Returns:
            True if grids are consistent
        """
        # Determine signal to compare
        if signal_name is None:
            common_signals = grid1.available_signals & grid2.available_signals
            if len(common_signals) == 0:
                return False
            signal_name = list(common_signals)[0]

        # Convert grids to FloatGrid
        openvdb_grid1 = voxelgrid_to_floatgrid(grid1, signal_name, default=0.0)
        openvdb_grid2 = voxelgrid_to_floatgrid(grid2, signal_name, default=0.0)

        # Call C++ validator
        return self._validator.check_consistency(openvdb_grid1, openvdb_grid2, tolerance)
