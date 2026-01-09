"""
Geometric Distortion

Models and correction for geometric distortions:
- Warping
- Scaling
- Rotation
- Combined distortions
"""

from typing import Optional, Tuple, Dict, Any, List
import numpy as np
from scipy.interpolate import griddata
from abc import ABC, abstractmethod


class DistortionModel(ABC):
    """
    Abstract base class for geometric distortion models.
    """

    @abstractmethod
    def apply(self, points: np.ndarray) -> np.ndarray:
        """
        Apply distortion to points.

        Args:
            points: Array of points (N, 3)

        Returns:
            Distorted points
        """
        pass

    @abstractmethod
    def correct(self, points: np.ndarray) -> np.ndarray:
        """
        Correct distortion from points.

        Args:
            points: Array of distorted points (N, 3)

        Returns:
            Corrected points
        """
        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get distortion parameters.

        Returns:
            Dictionary of parameters
        """
        pass


class ScalingModel(DistortionModel):
    """
    Scaling distortion model.

    Applies different scaling factors along each axis.
    """

    def __init__(
        self,
        scale_x: float = 1.0,
        scale_y: float = 1.0,
        scale_z: float = 1.0,
        center: Optional[Tuple[float, float, float]] = None,
    ):
        """
        Initialize scaling model.

        Args:
            scale_x: Scaling factor along X axis
            scale_y: Scaling factor along Y axis
            scale_z: Scaling factor along Z axis
            center: Scaling center point (if None, uses origin)
        """
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.scale_z = scale_z
        self.center = center or (0.0, 0.0, 0.0)

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Apply scaling distortion."""
        points = np.asarray(points)
        if points.ndim == 1:
            points = points.reshape(1, -1)

        # Translate to center
        centered = points - np.array(self.center)

        # Apply scaling
        scaled = centered * np.array([self.scale_x, self.scale_y, self.scale_z])

        # Translate back
        return scaled + np.array(self.center)

    def correct(self, points: np.ndarray) -> np.ndarray:
        """Correct scaling distortion."""
        points = np.asarray(points)
        if points.ndim == 1:
            points = points.reshape(1, -1)

        # Translate to center
        centered = points - np.array(self.center)

        # Apply inverse scaling
        corrected = centered / np.array([self.scale_x, self.scale_y, self.scale_z])

        # Translate back
        return corrected + np.array(self.center)

    def get_parameters(self) -> Dict[str, Any]:
        """Get scaling parameters."""
        return {
            "type": "scaling",
            "scale_x": self.scale_x,
            "scale_y": self.scale_y,
            "scale_z": self.scale_z,
            "center": self.center,
        }


class RotationModel(DistortionModel):
    """
    Rotation distortion model.

    Applies rotation around a specified axis and center.
    """

    def __init__(
        self,
        axis: str = "z",
        angle: float = 0.0,  # radians
        center: Optional[Tuple[float, float, float]] = None,
    ):
        """
        Initialize rotation model.

        Args:
            axis: Rotation axis ('x', 'y', 'z')
            angle: Rotation angle (radians)
            center: Rotation center point (if None, uses origin)
        """
        self.axis = axis
        self.angle = angle
        self.center = center or (0.0, 0.0, 0.0)
        self._rotation_matrix = self._compute_rotation_matrix()

    def _compute_rotation_matrix(self) -> np.ndarray:
        """Compute 3x3 rotation matrix."""
        cos_a = np.cos(self.angle)
        sin_a = np.sin(self.angle)

        if self.axis == "x":
            return np.array([[1.0, 0.0, 0.0], [0.0, cos_a, -sin_a], [0.0, sin_a, cos_a]])
        elif self.axis == "y":
            return np.array([[cos_a, 0.0, sin_a], [0.0, 1.0, 0.0], [-sin_a, 0.0, cos_a]])
        else:  # z
            return np.array([[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]])

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Apply rotation distortion."""
        points = np.asarray(points)
        if points.ndim == 1:
            points = points.reshape(1, -1)

        # Translate to center
        centered = points - np.array(self.center)

        # Apply rotation
        rotated = (self._rotation_matrix @ centered.T).T

        # Translate back
        return rotated + np.array(self.center)

    def correct(self, points: np.ndarray) -> np.ndarray:
        """Correct rotation distortion."""
        points = np.asarray(points)
        if points.ndim == 1:
            points = points.reshape(1, -1)

        # Translate to center
        centered = points - np.array(self.center)

        # Apply inverse rotation (transpose of rotation matrix)
        corrected = (self._rotation_matrix.T @ centered.T).T

        # Translate back
        return corrected + np.array(self.center)

    def get_parameters(self) -> Dict[str, Any]:
        """Get rotation parameters."""
        return {
            "type": "rotation",
            "axis": self.axis,
            "angle": self.angle,
            "angle_degrees": np.degrees(self.angle),
            "center": self.center,
        }


class WarpingModel(DistortionModel):
    """
    Warping distortion model.

    Uses a displacement field to model non-linear distortions.
    """

    def __init__(
        self,
        displacement_field: Optional[np.ndarray] = None,
        reference_points: Optional[np.ndarray] = None,
        displacement_vectors: Optional[np.ndarray] = None,
    ):
        """
        Initialize warping model.

        Args:
            displacement_field: Pre-computed displacement field (grid)
            reference_points: Reference points for displacement field
            displacement_vectors: Displacement vectors at reference points
        """
        self.displacement_field = displacement_field
        self.reference_points = reference_points
        self.displacement_vectors = displacement_vectors

    def _interpolate_displacement(self, points: np.ndarray) -> np.ndarray:
        """
        Interpolate displacement field at given points.

        Args:
            points: Points to interpolate at (N, 3)

        Returns:
            Displacement vectors (N, 3)
        """
        if self.displacement_field is not None:
            # Use pre-computed field (simplified - would need proper 3D interpolation)
            # For now, return zero displacement
            return np.zeros_like(points)

        if self.reference_points is not None and self.displacement_vectors is not None:
            # Interpolate from reference points
            if len(self.reference_points) == 0:
                return np.zeros_like(points)

            num_ref_points = len(self.reference_points)

            # For 3D linear interpolation, need at least 5 points for Delaunay triangulation
            # For fewer points, use nearest neighbor interpolation
            # For exact matches, check and return exact displacement
            if num_ref_points == 1:
                # Single reference point - check if points match exactly
                ref_point = self.reference_points[0]
                ref_displacement = self.displacement_vectors[0]

                # Check for exact matches
                distances = np.linalg.norm(points - ref_point, axis=1)
                exact_matches = distances < 1e-10

                displacements = np.zeros_like(points)
                displacements[exact_matches] = ref_displacement
                # For non-matching points, use the single displacement (nearest neighbor)
                displacements[~exact_matches] = ref_displacement

                return displacements
            elif num_ref_points < 5:
                # Use nearest neighbor for small number of points
                method = "nearest"
            else:
                # Use linear interpolation for sufficient points
                method = "linear"

            displacements = griddata(
                self.reference_points,
                self.displacement_vectors,
                points,
                method=method,
                fill_value=0.0,
            )
            return displacements

        return np.zeros_like(points)

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Apply warping distortion."""
        points = np.asarray(points)
        if points.ndim == 1:
            points = points.reshape(1, -1)

        # Get displacement at each point
        displacement = self._interpolate_displacement(points)

        # Apply displacement
        return points + displacement

    def correct(self, points: np.ndarray) -> np.ndarray:
        """Correct warping distortion."""
        points = np.asarray(points)
        if points.ndim == 1:
            points = points.reshape(1, -1)

        # Get displacement at each point
        displacement = self._interpolate_displacement(points)

        # Apply inverse displacement
        return points - displacement

    def get_parameters(self) -> Dict[str, Any]:
        """Get warping parameters."""
        return {
            "type": "warping",
            "has_field": self.displacement_field is not None,
            "num_reference_points": (len(self.reference_points) if self.reference_points is not None else 0),
        }

    def estimate_from_correspondences(self, source_points: np.ndarray, target_points: np.ndarray):
        """
        Estimate warping model from point correspondences.

        Args:
            source_points: Source (distorted) points (N, 3)
            target_points: Target (correct) points (N, 3)
        """
        source_points = np.asarray(source_points)
        target_points = np.asarray(target_points)

        if source_points.shape != target_points.shape:
            raise ValueError("Source and target points must have same shape")

        # Compute displacement vectors
        displacement = target_points - source_points

        # Store as reference points
        self.reference_points = source_points
        self.displacement_vectors = displacement


class CombinedDistortionModel(DistortionModel):
    """
    Combined distortion model.

    Applies multiple distortion models in sequence.
    """

    def __init__(self, models: List[DistortionModel]):
        """
        Initialize combined model.

        Args:
            models: List of distortion models to apply in order
        """
        self.models = models

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Apply all distortions in sequence."""
        result = points.copy()
        for model in self.models:
            result = model.apply(result)
        return result

    def correct(self, points: np.ndarray) -> np.ndarray:
        """Correct all distortions in reverse order."""
        result = points.copy()
        for model in reversed(self.models):
            result = model.correct(result)
        return result

    def get_parameters(self) -> Dict[str, Any]:
        """Get parameters from all models."""
        return {
            "type": "combined",
            "num_models": len(self.models),
            "models": [model.get_parameters() for model in self.models],
        }
