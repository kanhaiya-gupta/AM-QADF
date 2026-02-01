"""
Coordinate System Transformer - C++ Wrapper

Thin Python wrapper for C++ coordinate transformation implementation.
All core computation is done in C++ using Eigen.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np

try:
    from am_qadf_native import CoordinateTransformer, CoordinateSystem
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False
    CoordinateTransformer = None
    CoordinateSystem = None


class CoordinateSystemTransformer:
    """
    Transform points and data between different coordinate systems.
    
    This is a thin wrapper around the C++ CoordinateTransformer.
    All core computation is done in C++ using Eigen.
    
    Supports transformations between:
    - Build platform coordinates (STL, hatching)
    - CT scan coordinates
    - ISPM sensor coordinates
    """
    
    def __init__(self):
        """Initialize coordinate system transformer."""
        if not CPP_AVAILABLE:
            raise ImportError(
                "C++ bindings not available. "
                "Please build am_qadf_native with pybind11 bindings."
            )
        self._transformer = CoordinateTransformer()
    
    def transform_point(
        self,
        point: Tuple[float, float, float],
        from_system: Dict[str, Any],
        to_system: Dict[str, Any],
    ) -> Tuple[float, float, float]:
        """
        Transform a single point from one coordinate system to another.

        Args:
            point: Point coordinates (x, y, z) in mm
            from_system: Source coordinate system dictionary
            to_system: Target coordinate system dictionary

        Returns:
            Transformed point coordinates (x, y, z) in mm
        """
        self._require_cpp()
        from_coord = self._dict_to_coordinate_system(from_system)
        to_coord = self._dict_to_coordinate_system(to_system)
        
        point_vec = np.array(point, dtype=np.float64)
        result = self._transformer.transform_point(point_vec, from_coord, to_coord)
        return tuple(result)
    
    def transform_points(
        self,
        points: np.ndarray,
        from_system: Dict[str, Any],
        to_system: Dict[str, Any],
    ) -> np.ndarray:
        """
        Transform multiple points from one coordinate system to another.

        Args:
            points: Array of points with shape (n, 3)
            from_system: Source coordinate system dictionary
            to_system: Target coordinate system dictionary

        Returns:
            Transformed points array with shape (n, 3)
        """
        self._require_cpp()
        if len(points) == 0:
            return points.copy()
        
        from_coord = self._dict_to_coordinate_system(from_system)
        to_coord = self._dict_to_coordinate_system(to_system)
        
        points_array = np.asarray(points, dtype=np.float64)
        if points_array.ndim == 1:
            points_array = points_array.reshape(1, -1)
        
        if points_array.shape[1] != 3:
            raise ValueError(f"Points must have shape (n, 3), got {points_array.shape}")
        
        result = self._transformer.transform_points(points_array, from_coord, to_coord)
        return result
    
    def _dict_to_coordinate_system(self, system_dict: Dict[str, Any]) -> CoordinateSystem:
        """Convert Python dict to C++ CoordinateSystem."""
        coord = CoordinateSystem()
        
        # Origin
        origin = system_dict.get("origin", [0.0, 0.0, 0.0])
        if isinstance(origin, dict):
            origin = [origin.get("x", 0.0), origin.get("y", 0.0), origin.get("z", 0.0)]
        coord.origin = np.array(origin, dtype=np.float64)
        
        # Rotation
        rotation = system_dict.get("rotation", {})
        if "axis" in rotation and "angle" in rotation:
            # Axis-angle format
            axis_map = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}
            axis_name = rotation["axis"].lower()
            coord.rotation_axis = np.array(axis_map.get(axis_name, [0, 0, 1]), dtype=np.float64)
            coord.rotation_angle = np.deg2rad(rotation["angle"])
            coord.use_axis_angle = True
        else:
            # Euler angles
            if "x_deg" in rotation:
                coord.rotation_euler = np.deg2rad([
                    rotation["x_deg"],
                    rotation["y_deg"],
                    rotation["z_deg"]
                ])
            elif "x" in rotation:
                coord.rotation_euler = np.array([
                    rotation["x"],
                    rotation["y"],
                    rotation["z"]
                ], dtype=np.float64)
            else:
                coord.rotation_euler = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        
        # Scale
        scale = system_dict.get("scale_factor", {"x": 1.0, "y": 1.0, "z": 1.0})
        if isinstance(scale, dict):
            coord.scale = np.array([
                scale.get("x", 1.0),
                scale.get("y", 1.0),
                scale.get("z", 1.0)
            ], dtype=np.float64)
        elif isinstance(scale, (list, tuple)):
            coord.scale = np.array(scale, dtype=np.float64)
        elif isinstance(scale, (int, float)):
            coord.scale = np.array([scale, scale, scale], dtype=np.float64)
        else:
            coord.scale = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        
        return coord
    
    def align_data_sources(
        self,
        model_id: str,
        stl_coord_system: Dict[str, Any],
        hatching_coord_system: Optional[Dict[str, Any]] = None,
        ct_coord_system: Optional[Dict[str, Any]] = None,
        ispm_coord_system: Optional[Dict[str, Any]] = None,
        target_system: str = "build_platform",
    ) -> Dict[str, Any]:
        """
        Align coordinate systems from multiple data sources.

        Args:
            model_id: Model UUID
            stl_coord_system: STL model coordinate system
            hatching_coord_system: Hatching layer coordinate system
            ct_coord_system: CT scan coordinate system
            ispm_coord_system: ISPM monitoring coordinate system
            target_system: Target coordinate system name ('build_platform', 'ct_scan', 'ispm')

        Returns:
            Dictionary with aligned coordinate systems and transformation matrices
        """
        # Use STL/build platform as reference by default
        reference_system = stl_coord_system

        # Determine target system
        if target_system == "ct_scan" and ct_coord_system:
            target_coord_system = ct_coord_system
        elif target_system == "ispm" and ispm_coord_system:
            target_coord_system = ispm_coord_system
        else:
            target_coord_system = stl_coord_system

        # Build transformation mappings
        transformations = {
            "stl": {
                "system": stl_coord_system,
                "to_target": lambda p: (
                    self.transform_point(p, stl_coord_system, target_coord_system)
                    if stl_coord_system != target_coord_system
                    else p
                ),
            }
        }

        if hatching_coord_system:
            transformations["hatching"] = {
                "system": hatching_coord_system,
                "to_target": lambda p: (
                    self.transform_point(p, hatching_coord_system, target_coord_system)
                    if hatching_coord_system != target_coord_system
                    else p
                ),
            }

        if ct_coord_system:
            transformations["ct_scan"] = {
                "system": ct_coord_system,
                "to_target": lambda p: (
                    self.transform_point(p, ct_coord_system, target_coord_system)
                    if ct_coord_system != target_coord_system
                    else p
                ),
            }

        if ispm_coord_system:
            transformations["ispm"] = {
                "system": ispm_coord_system,
                "to_target": lambda p: (
                    self.transform_point(p, ispm_coord_system, target_coord_system)
                    if ispm_coord_system != target_coord_system
                    else p
                ),
            }

        return {
            "model_id": model_id,
            "target_system": target_system,
            "target_coord_system": target_coord_system,
            "transformations": transformations,
            "reference_system": reference_system,
        }
    
    def validate_coordinate_system(self, coord_system: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate a coordinate system dictionary.

        Args:
            coord_system: Coordinate system dictionary

        Returns:
            Tuple of (is_valid, error_message)
        """
        required_fields = ["origin"]

        for field in required_fields:
            if field not in coord_system:
                return False, f"Missing required field: {field}"

        # Validate origin
        origin = coord_system.get("origin", [0.0, 0.0, 0.0])
        if isinstance(origin, dict):
            origin = [origin.get("x", 0.0), origin.get("y", 0.0), origin.get("z", 0.0)]

        if len(origin) != 3:
            return False, "Origin must have 3 coordinates (x, y, z)"
        
        # Convert to C++ CoordinateSystem and validate
        try:
            coord = self._dict_to_coordinate_system(coord_system)
            is_valid = self._transformer.validate_coordinate_system(coord)
            if not is_valid:
                return False, "Coordinate system validation failed"
            return True, None
        except Exception as e:
            return False, f"Validation error: {str(e)}"
