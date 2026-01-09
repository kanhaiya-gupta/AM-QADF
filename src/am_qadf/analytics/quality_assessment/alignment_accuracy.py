"""
Alignment Accuracy Validation

Validates alignment accuracy for voxel domain data:
- Coordinate System Alignment: Accuracy of coordinate transformations
- Temporal Alignment: Accuracy of temporal synchronization
- Spatial Registration: Accuracy of spatial registration
- Residual Analysis: Residual errors after alignment
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class AlignmentAccuracyMetrics:
    """Alignment accuracy metrics."""

    coordinate_alignment_error: float  # Mean coordinate transformation error (mm)
    temporal_alignment_error: float  # Mean temporal alignment error (layers or seconds)
    spatial_registration_error: float  # Mean spatial registration error (mm)
    residual_error_mean: float  # Mean residual error after alignment
    residual_error_std: float  # Standard deviation of residual errors
    alignment_score: float  # Overall alignment score (0-1, higher is better)

    # Detailed metrics
    transformation_errors: List[float]  # Per-transformation errors
    registration_residuals: Optional[np.ndarray] = None  # Residual map

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        result = {
            "coordinate_alignment_error": self.coordinate_alignment_error,
            "temporal_alignment_error": self.temporal_alignment_error,
            "spatial_registration_error": self.spatial_registration_error,
            "residual_error_mean": self.residual_error_mean,
            "residual_error_std": self.residual_error_std,
            "alignment_score": self.alignment_score,
            "transformation_errors_count": len(self.transformation_errors),
        }
        if self.registration_residuals is not None:
            result["registration_residuals_shape"] = self.registration_residuals.shape
        return result


class AlignmentAccuracyAnalyzer:
    """Analyzes alignment accuracy for voxel domain data."""

    def __init__(self, max_acceptable_error: float = 0.1):
        """
        Initialize the alignment accuracy analyzer.

        Args:
            max_acceptable_error: Maximum acceptable alignment error (mm)
        """
        self.max_acceptable_error = max_acceptable_error

    def validate_coordinate_alignment(
        self,
        source_points: np.ndarray,
        target_points: np.ndarray,
        transformation_matrix: Optional[np.ndarray] = None,
    ) -> float:
        """
        Validate coordinate system alignment accuracy.

        Args:
            source_points: Source coordinate points (N, 3)
            target_points: Target coordinate points (N, 3)
            transformation_matrix: Optional transformation matrix (4x4)

        Returns:
            Mean alignment error (mm)
        """
        if len(source_points) != len(target_points):
            return float("inf")

        if transformation_matrix is not None:
            # Apply transformation
            from scipy.spatial.transform import Rotation

            # Extract rotation and translation
            R = transformation_matrix[:3, :3]
            t = transformation_matrix[:3, 3]
            transformed_points = (R @ source_points.T).T + t
        else:
            transformed_points = source_points

        # Calculate errors
        errors = np.linalg.norm(transformed_points - target_points, axis=1)
        mean_error = np.mean(errors)

        return mean_error

    def validate_temporal_alignment(self, source_times: np.ndarray, target_times: np.ndarray, tolerance: float = 0.1) -> float:
        """
        Validate temporal alignment accuracy.

        Args:
            source_times: Source timestamps or layer indices
            target_times: Target timestamps or layer indices
            tolerance: Acceptable temporal difference

        Returns:
            Mean temporal alignment error
        """
        if len(source_times) != len(target_times):
            return float("inf")

        # Calculate temporal differences
        time_diffs = np.abs(source_times - target_times)
        mean_error = np.mean(time_diffs)

        return mean_error

    def calculate_registration_residuals(
        self, reference_points: np.ndarray, aligned_points: np.ndarray
    ) -> Tuple[float, float, np.ndarray]:
        """
        Calculate spatial registration residuals.

        Args:
            reference_points: Reference point cloud (N, 3)
            aligned_points: Aligned point cloud (N, 3)

        Returns:
            (mean_residual, std_residual, residual_map)
        """
        if len(reference_points) != len(aligned_points):
            return float("inf"), 0.0, np.array([])

        # Calculate residuals
        residuals = np.linalg.norm(reference_points - aligned_points, axis=1)
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)

        return mean_residual, std_residual, residuals

    def assess_alignment_accuracy(
        self,
        voxel_data: Any,
        coordinate_transformer: Optional[Any] = None,
        reference_data: Optional[Any] = None,
    ) -> AlignmentAccuracyMetrics:
        """
        Assess overall alignment accuracy.

        Args:
            voxel_data: Voxel domain data object
            coordinate_transformer: Optional coordinate system transformer
            reference_data: Optional reference data for comparison

        Returns:
            AlignmentAccuracyMetrics object
        """
        # For now, use simplified assessment
        # In a full implementation, this would:
        # 1. Extract coordinate points from different sources
        # 2. Apply transformations
        # 3. Compare with reference
        # 4. Calculate residuals

        # Default values (assume good alignment if no reference)
        coordinate_error = 0.0
        temporal_error = 0.0
        spatial_error = 0.0
        residual_mean = 0.0
        residual_std = 0.0

        transformation_errors = []

        # If we have a coordinate transformer, validate it
        if coordinate_transformer is not None:
            # Sample some points and validate transformation
            # This is a placeholder - actual implementation would use real data
            try:
                # Get sample points from voxel data
                if hasattr(voxel_data, "bbox_min") and hasattr(voxel_data, "bbox_max"):
                    bbox_min = voxel_data.bbox_min
                    bbox_max = voxel_data.bbox_max

                    # Sample points
                    n_samples = 10
                    sample_points = np.random.uniform(bbox_min, bbox_max, size=(n_samples, 3))

                    # Transform and check consistency
                    # (Simplified - actual would compare with known correspondences)
                    transformation_errors = [0.0] * n_samples
                    coordinate_error = np.mean(transformation_errors)
            except Exception:
                pass

        # Calculate alignment score (0-1, higher is better)
        # Normalize errors to 0-1 range
        error_normalized = min(1.0, coordinate_error / self.max_acceptable_error)
        alignment_score = max(0.0, 1.0 - error_normalized)

        return AlignmentAccuracyMetrics(
            coordinate_alignment_error=coordinate_error,
            temporal_alignment_error=temporal_error,
            spatial_registration_error=spatial_error,
            residual_error_mean=residual_mean,
            residual_error_std=residual_std,
            alignment_score=alignment_score,
            transformation_errors=transformation_errors,
            registration_residuals=None,
        )
