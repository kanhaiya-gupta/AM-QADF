"""
Spatial Transformation

Coordinate system transformations and alignment.
"""

from typing import Optional, Tuple, Dict, List, Any
import numpy as np
from dataclasses import dataclass


@dataclass
class TransformationMatrix:
    """Represents a 4x4 homogeneous transformation matrix."""

    matrix: np.ndarray  # 4x4 matrix

    def __post_init__(self):
        """Validate matrix shape."""
        if self.matrix.shape != (4, 4):
            raise ValueError("Transformation matrix must be 4x4")

    def apply(self, points: np.ndarray) -> np.ndarray:
        """
        Apply transformation to points.

        Args:
            points: Array of points (N, 3) or (3,)

        Returns:
            Transformed points
        """
        points = np.asarray(points)
        if points.ndim == 1:
            points = points.reshape(1, -1)

        # Convert to homogeneous coordinates
        n_points = points.shape[0]
        homogeneous = np.ones((n_points, 4))
        homogeneous[:, :3] = points

        # Apply transformation
        transformed = (self.matrix @ homogeneous.T).T

        # Convert back to 3D
        return transformed[:, :3]

    def inverse(self) -> "TransformationMatrix":
        """Compute inverse transformation."""
        return TransformationMatrix(matrix=np.linalg.inv(self.matrix))

    @classmethod
    def identity(cls) -> "TransformationMatrix":
        """Create identity transformation."""
        return cls(matrix=np.eye(4))

    @classmethod
    def translation(cls, tx: float, ty: float, tz: float) -> "TransformationMatrix":
        """Create translation transformation."""
        matrix = np.eye(4)
        matrix[0:3, 3] = [tx, ty, tz]
        return cls(matrix=matrix)

    @classmethod
    def rotation(cls, axis: str, angle: float) -> "TransformationMatrix":
        """
        Create rotation transformation.

        Args:
            axis: Rotation axis ('x', 'y', 'z')
            angle: Rotation angle (radians)
        """
        matrix = np.eye(4)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        if axis == "x":
            matrix[1:3, 1:3] = [[cos_a, -sin_a], [sin_a, cos_a]]
        elif axis == "z":
            matrix[0:2, 0:2] = [[cos_a, -sin_a], [sin_a, cos_a]]
        else:  # y
            matrix[0:3:2, 0:3:2] = [[cos_a, sin_a], [-sin_a, cos_a]]

        return cls(matrix=matrix)

    @classmethod
    def scale(cls, sx: float, sy: float, sz: float) -> "TransformationMatrix":
        """Create scaling transformation."""
        matrix = np.eye(4)
        matrix[0, 0] = sx
        matrix[1, 1] = sy
        matrix[2, 2] = sz
        return cls(matrix=matrix)


class SpatialTransformer:
    """
    Apply spatial transformations to coordinate data.

    Handles transformations between different coordinate systems:
    - STL model coordinates
    - Build platform coordinates
    - Sensor coordinate systems
    """

    def __init__(self):
        """Initialize spatial transformer."""
        self._transformations: Dict[str, TransformationMatrix] = {}

    def register_transformation(self, name: str, transformation: TransformationMatrix):
        """
        Register a named transformation.

        Args:
            name: Transformation name
            transformation: TransformationMatrix object
        """
        self._transformations[name] = transformation

    def get_transformation(self, name: str) -> Optional[TransformationMatrix]:
        """
        Get a named transformation.

        Args:
            name: Transformation name

        Returns:
            TransformationMatrix or None
        """
        return self._transformations.get(name)

    def transform_points(
        self,
        points: np.ndarray,
        transformation_name: Optional[str] = None,
        transformation: Optional[TransformationMatrix] = None,
    ) -> np.ndarray:
        """
        Transform points using a transformation.

        Args:
            points: Array of points (N, 3)
            transformation_name: Name of registered transformation
            transformation: Direct TransformationMatrix object

        Returns:
            Transformed points
        """
        if transformation is None:
            if transformation_name is None:
                raise ValueError("Must provide transformation_name or transformation")
            transformation = self.get_transformation(transformation_name)
            if transformation is None:
                raise ValueError(f"Transformation '{transformation_name}' not found")

        return transformation.apply(points)

    def align_coordinate_systems(
        self,
        source_points: np.ndarray,
        target_points: np.ndarray,
        method: str = "umeyama",
    ) -> TransformationMatrix:
        """
        Compute transformation to align source points to target points.

        Uses point cloud registration (simplified implementation).

        Args:
            source_points: Source coordinate points (N, 3)
            target_points: Target coordinate points (N, 3)
            method: Registration method ('umeyama', 'svd')

        Returns:
            TransformationMatrix that aligns source to target
        """
        source_points = np.asarray(source_points)
        target_points = np.asarray(target_points)

        if source_points.shape != target_points.shape:
            raise ValueError("Source and target points must have same shape")

        # Compute centroids
        source_centroid = np.mean(source_points, axis=0)
        target_centroid = np.mean(target_points, axis=0)

        # Center points
        source_centered = source_points - source_centroid
        target_centered = target_points - target_centroid

        # Compute rotation using SVD
        H = source_centered.T @ target_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Ensure proper rotation (det(R) = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Compute translation
        t = target_centroid - R @ source_centroid

        # Build transformation matrix
        matrix = np.eye(4)
        matrix[0:3, 0:3] = R
        matrix[0:3, 3] = t

        return TransformationMatrix(matrix=matrix)


class TransformationManager:
    """
    Manage multiple coordinate system transformations.

    Provides a registry for transformations between different
    coordinate systems (STL, build platform, sensors, etc.).
    """

    def __init__(self):
        """Initialize transformation manager."""
        self.transformer = SpatialTransformer()
        self._coordinate_systems: Dict[str, Dict[str, Any]] = {}

    def register_coordinate_system(
        self,
        name: str,
        origin: Optional[Tuple[float, float, float]] = None,
        axes: Optional[Dict[str, Tuple[float, float, float]]] = None,
    ):
        """
        Register a coordinate system.

        Args:
            name: Coordinate system name
            origin: Origin point (x, y, z)
            axes: Dictionary of axis vectors {'x': (1,0,0), 'y': (0,1,0), 'z': (0,0,1)}
        """
        self._coordinate_systems[name] = {
            "origin": origin or (0.0, 0.0, 0.0),
            "axes": axes or {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)},
        }

    def get_transformation(
        self, from_system: str, to_system: str, visited: Optional[set] = None
    ) -> Optional[TransformationMatrix]:
        """
        Get transformation between coordinate systems.

        Supports chaining transformations through intermediate coordinate systems.

        Args:
            from_system: Source coordinate system name
            to_system: Target coordinate system name
            visited: Set of visited coordinate systems (for recursion prevention)

        Returns:
            TransformationMatrix or None
        """
        if from_system == to_system:
            return TransformationMatrix.identity()

        if visited is None:
            visited = set()

        # Prevent infinite recursion - check if we've already visited this path
        path_key = (from_system, to_system)
        if path_key in visited:
            return None

        visited.add(path_key)

        # Try direct transformation
        trans_name = f"{from_system}_to_{to_system}"
        transformation = self.transformer.get_transformation(trans_name)
        if transformation is not None:
            return transformation

        # Try inverse
        trans_name = f"{to_system}_to_{from_system}"
        transformation = self.transformer.get_transformation(trans_name)
        if transformation is not None:
            return transformation.inverse()

        # Try chaining through intermediate coordinate systems
        # Find a path: from_system -> intermediate -> to_system
        # Get all known coordinate systems from both registered systems and transformation names
        known_systems = set(self._coordinate_systems.keys())
        # Extract system names from transformation names (format: "from_system_to_to_system")
        for trans_name in self.transformer._transformations.keys():
            if "_to_" in trans_name:
                parts = trans_name.split("_to_")
                if len(parts) == 2:
                    known_systems.add(parts[0])
                    known_systems.add(parts[1])

        for intermediate in known_systems:
            if intermediate == from_system or intermediate == to_system:
                continue

            # Check if we can go from_system -> intermediate -> to_system
            # Use a new visited set for each intermediate to allow exploring different paths
            trans1 = self.get_transformation(from_system, intermediate, set(visited))
            if trans1 is None:
                continue

            trans2 = self.get_transformation(intermediate, to_system, set(visited))
            if trans2 is None:
                continue

            # Chain transformations: T_to = T_intermediate_to_to @ T_from_to_intermediate
            chained_matrix = trans2.matrix @ trans1.matrix
            chained_trans = TransformationMatrix(matrix=chained_matrix)

            # Cache this transformation for future use
            self.set_transformation(from_system, to_system, chained_trans)

            return chained_trans

        return None

    def set_transformation(self, from_system: str, to_system: str, transformation: TransformationMatrix):
        """
        Set transformation between coordinate systems.

        Args:
            from_system: Source coordinate system name
            to_system: Target coordinate system name
            transformation: TransformationMatrix
        """
        trans_name = f"{from_system}_to_{to_system}"
        self.transformer.register_transformation(trans_name, transformation)

        # Also register inverse
        inv_name = f"{to_system}_to_{from_system}"
        self.transformer.register_transformation(inv_name, transformation.inverse())

    def transform_points(self, points: np.ndarray, from_system: str, to_system: str) -> np.ndarray:
        """
        Transform points between coordinate systems.

        Args:
            points: Array of points (N, 3)
            from_system: Source coordinate system name
            to_system: Target coordinate system name

        Returns:
            Transformed points
        """
        transformation = self.get_transformation(from_system, to_system)
        if transformation is None:
            raise ValueError(f"No transformation found from '{from_system}' to '{to_system}'")

        return self.transformer.transform_points(points, transformation=transformation)
