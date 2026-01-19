"""
CT Scan Client

Query client for CT scan data.
Supports querying CT scan volumes and aligning with build geometry.
"""

from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import gzip
import io

# Import base classes
try:
    from .base_query_client import (
        BaseQueryClient,
        QueryResult,
        SpatialQuery,
        TemporalQuery,
        SignalType,
    )
except ImportError:
    # Fallback for direct import
    import sys
    from pathlib import Path

    current_file = Path(__file__).resolve()
    base_path = current_file.parent / "base_query_client.py"

    if base_path.exists():
        import importlib.util

        spec = importlib.util.spec_from_file_location("base_query_client", base_path)
        base_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(base_module)
        sys.modules["base_query_client"] = base_module

        BaseQueryClient = base_module.BaseQueryClient
        QueryResult = base_module.QueryResult
        SpatialQuery = base_module.SpatialQuery
        TemporalQuery = base_module.TemporalQuery
        SignalType = base_module.SignalType
    else:
        raise ImportError("Could not import base_query_client")


class CTScanClient(BaseQueryClient):
    """
    Query client for CT scan data.

    Supports:
    - Querying CT scan volumes from MongoDB (with GridFS)
    - In-memory CT scan data (for simulation/testing)
    - Voxelizing CT scan data
    - Aligning with build geometry
    - Density/porosity analysis
    """

    def __init__(
        self,
        data_source: Optional[str] = None,
        ct_volume: Optional[np.ndarray] = None,
        ct_spacing: Optional[Tuple[float, float, float]] = None,
        ct_origin: Optional[Tuple[float, float, float]] = None,
        mongo_client=None,
        use_mongodb: bool = False,
    ):
        """
        Initialize CT scan client.

        Args:
            data_source: Path to CT scan file or database connection
            ct_volume: CT scan volume data (3D numpy array) - for simulation/testing
            ct_spacing: Voxel spacing (dx, dy, dz) in mm
            ct_origin: Origin coordinates (x, y, z) in mm
            mongo_client: MongoDBClient instance (for MongoDB data warehouse)
            use_mongodb: If True, use MongoDB backend; if False, use in-memory data
        """
        super().__init__(data_source=data_source or ("mongodb_warehouse" if use_mongodb else "in_memory"))
        self._available_signals = [SignalType.DENSITY]

        # CT scan data (for simulation/testing)
        self.ct_volume = ct_volume
        self.ct_spacing = ct_spacing or (0.1, 0.1, 0.1)  # Default 0.1mm spacing
        self.ct_origin = ct_origin or (0.0, 0.0, 0.0)

        # MongoDB support
        self.mongo_client = mongo_client
        self.use_mongodb = use_mongodb
        self.collection_name = "ct_scan_data"

    def set_mongo_client(self, mongo_client):
        """Set MongoDB client instance (for MongoDB mode)."""
        self.mongo_client = mongo_client
        self.use_mongodb = True

    def _get_collection(self):
        """Get MongoDB collection."""
        if self.mongo_client is None:
            raise RuntimeError("MongoDB client not initialized. Call set_mongo_client() first.")
        if not self.mongo_client.is_connected():
            raise ConnectionError("MongoDB client not connected")
        return self.mongo_client.get_collection(self.collection_name)

    def set_ct_data(
        self,
        volume: np.ndarray,
        spacing: Tuple[float, float, float],
        origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        """
        Set CT scan volume data.

        Args:
            volume: 3D numpy array of CT scan data (Hounsfield units or density)
            spacing: Voxel spacing (dx, dy, dz) in mm
            origin: Origin coordinates (x, y, z) in mm
        """
        self.ct_volume = volume
        self.ct_spacing = spacing
        self.ct_origin = origin

    def query(
        self,
        spatial: Optional[SpatialQuery] = None,
        temporal: Optional[TemporalQuery] = None,
        signal_types: Optional[List[SignalType]] = None,
    ) -> QueryResult:
        """
        Query CT scan data.

        Supports both MongoDB warehouse and in-memory data modes.

        Args:
            spatial: Spatial query parameters (bounding box, component_id = model_id for MongoDB)
            temporal: Temporal query parameters (not typically used for CT scans)
            signal_types: Signal types to retrieve (DENSITY)

        Returns:
            QueryResult with CT scan points and density values
        """
        # Use MongoDB backend if enabled
        if self.use_mongodb and self.mongo_client:
            return self._query_mongodb(spatial, temporal, signal_types)

        # Fall back to in-memory data mode
        return self._query_in_memory(spatial, temporal, signal_types)

    def _query_mongodb(
        self,
        spatial: Optional[SpatialQuery] = None,
        temporal: Optional[TemporalQuery] = None,
        signal_types: Optional[List[SignalType]] = None,
    ) -> QueryResult:
        """Query CT scan data from MongoDB."""
        if spatial is None or spatial.component_id is None:
            raise ValueError("Spatial query must include component_id (model_id) for MongoDB queries")

        model_id = spatial.component_id
        collection = self._get_collection()

        # Get CT scan document
        doc = collection.find_one({"model_id": model_id})
        if doc is None:
            return QueryResult(
                points=[],
                signals={},
                metadata={"error": f"No CT scan data found for model_id: {model_id}"},
            )

        # Retrieve density values and porosity map from GridFS
        density_values = None
        porosity_map = None

        # Check both top-level and nested data_storage location for GridFS IDs
        density_gridfs_id = doc.get("density_values_gridfs_id") or doc.get("data_storage", {}).get("density_values_gridfs_id")
        porosity_gridfs_id = doc.get("porosity_map_gridfs_id") or doc.get("data_storage", {}).get("porosity_map_gridfs_id")

        if density_gridfs_id:
            density_data = self.mongo_client.get_file(density_gridfs_id)
            if density_data:
                # Decompress
                density_data = gzip.decompress(density_data)
                density_values = np.load(io.BytesIO(density_data))

        if porosity_gridfs_id:
            porosity_data = self.mongo_client.get_file(porosity_gridfs_id)
            if porosity_data:
                # Decompress
                porosity_data = gzip.decompress(porosity_data)
                porosity_map = np.load(io.BytesIO(porosity_data))

        # Get metadata and coordinate system
        metadata = doc.get("metadata", {})
        coord_system = doc.get("coordinate_system", metadata.get("coordinate_system", {}))

        # Get spacing - check multiple locations
        if "voxel_spacing" in coord_system:
            spacing_dict = coord_system["voxel_spacing"]
            spacing = (
                spacing_dict.get("x", 0.67),
                spacing_dict.get("y", 0.67),
                spacing_dict.get("z", 0.67),
            )
        elif isinstance(coord_system.get("voxel_spacing"), (list, tuple)):
            spacing = tuple(coord_system.get("voxel_spacing", [0.67, 0.67, 0.67]))
        else:
            spacing = tuple(metadata.get("statistics", {}).get("spacing", [0.67, 0.67, 0.67]))

        # Get origin - check multiple locations
        if "origin" in coord_system:
            origin_dict = coord_system["origin"]
            origin = (
                origin_dict.get("x", 0.0),
                origin_dict.get("y", 0.0),
                origin_dict.get("z", 0.0),
            )
        elif isinstance(coord_system.get("origin"), (list, tuple)):
            origin = tuple(coord_system.get("origin", [0.0, 0.0, 0.0]))
        else:
            origin = tuple(metadata.get("statistics", {}).get("origin", [0.0, 0.0, 0.0]))

        grid_dimensions = tuple(
            metadata.get("statistics", {}).get(
                "dimensions",
                density_values.shape if density_values is not None else (30, 30, 30),
            )
        )

        # Generate points and signals
        points = []
        signals = {}

        # First, try to use defect_locations if available (more efficient and accurate)
        defect_locations = doc.get("defect_locations", [])
        # Ensure defect_count is an integer, not a MagicMock
        defect_count_raw = doc.get("defect_count", 0)
        if isinstance(defect_count_raw, (int, float, np.integer, np.floating)):
            defect_count = int(defect_count_raw)
        else:
            defect_count = 0

        if defect_locations and len(defect_locations) > 0:
            # Convert defect locations from voxel indices to world coordinates
            # defect_locations are stored as voxel indices [i, j, k], need to convert to world coords
            spacing_arr = np.array(spacing)
            origin_arr = np.array(origin)

            defect_points = []
            defect_voxel_indices = []  # Store voxel indices for density lookup

            for loc in defect_locations:
                if isinstance(loc, (list, tuple)) and len(loc) == 3:
                    # Check if already in world coordinates (large values) or voxel indices (small integers)
                    voxel_idx = np.array(loc)

                    # If values are small integers (< 1000), assume voxel indices
                    # If values are large floats, assume already world coordinates
                    if np.all(voxel_idx < 1000) and np.all(voxel_idx == np.round(voxel_idx)):
                        # Convert from voxel indices to world coordinates
                        world_pos = origin_arr + voxel_idx * spacing_arr
                        defect_points.append(tuple(world_pos))
                        defect_voxel_indices.append(tuple(voxel_idx.astype(int)))
                    else:
                        # Already in world coordinates - try to convert back to voxel indices for density lookup
                        world_pos = np.array(loc)
                        voxel_idx = ((world_pos - origin_arr) / spacing_arr).astype(int)
                        defect_points.append(tuple(loc))
                        defect_voxel_indices.append(tuple(voxel_idx))

            # Apply spatial filtering if provided
            if spatial and spatial.bbox_min and spatial.bbox_max:
                bbox_min = np.array(spatial.bbox_min)
                bbox_max = np.array(spatial.bbox_max)

                # Filter defect locations within bounding box
                filtered_defects = []
                filtered_indices = []
                for defect_point, voxel_idx in zip(defect_points, defect_voxel_indices):
                    defect_arr = np.array(defect_point)
                    if np.all((defect_arr >= bbox_min) & (defect_arr <= bbox_max)):
                        filtered_defects.append(defect_point)
                        filtered_indices.append(voxel_idx)

                points = filtered_defects
                defect_voxel_indices = filtered_indices
            else:
                points = defect_points

            # Try to get density values for defect locations if density_values exists
            if density_values is not None and SignalType.DENSITY in (signal_types or [SignalType.DENSITY]):
                density_list = []
                for voxel_idx in defect_voxel_indices:
                    try:
                        i, j, k = voxel_idx
                        # Clamp to valid range
                        i = max(0, min(i, density_values.shape[0] - 1))
                        j = max(0, min(j, density_values.shape[1] - 1))
                        k = max(0, min(k, density_values.shape[2] - 1))
                        density_list.append(float(density_values[i, j, k]))
                    except (IndexError, ValueError):
                        # If index is out of bounds, use a default value or skip
                        density_list.append(0.0)

                # Only add signals if we have matching density values
                if len(density_list) == len(points):
                    signals["density"] = density_list

            # Add defect count to metadata
            metadata = {
                "model_id": model_id,
                "source": "mongodb",
                "spacing": spacing,
                "origin": origin,
                "grid_dimensions": grid_dimensions,
                "has_porosity_map": porosity_map is not None,
                "defect_count": defect_count,
                "n_defect_locations": len(defect_locations),
                "n_points_returned": len(points),
                "has_density_signals": "density" in signals,
            }

            return QueryResult(points=points, signals=signals, metadata=metadata, component_id=model_id)

        # Fallback: Generate points from density_values voxel grid
        if density_values is not None:
            density_list = []

            # Apply spatial filtering if provided
            if spatial and spatial.bbox_min and spatial.bbox_max:
                bbox_min = np.array(spatial.bbox_min)
                bbox_max = np.array(spatial.bbox_max)

                # Convert to voxel indices
                idx_min = ((bbox_min - np.array(origin)) / np.array(spacing)).astype(int)
                idx_max = ((bbox_max - np.array(origin)) / np.array(spacing)).astype(int)

                # Clamp to volume bounds
                idx_min = np.maximum(idx_min, 0)
                idx_max = np.minimum(idx_max, np.array(density_values.shape) - 1)

                # Extract subvolume
                slices = tuple(slice(idx_min[i], idx_max[i] + 1) for i in range(3))
                subvolume = density_values[slices]

                # Generate points
                for i in range(subvolume.shape[0]):
                    for j in range(subvolume.shape[1]):
                        for k in range(subvolume.shape[2]):
                            voxel_idx = np.array([i, j, k]) + idx_min
                            world_pos = np.array(origin) + voxel_idx * np.array(spacing)
                            points.append(tuple(world_pos))
                            density_list.append(float(subvolume[i, j, k]))
            else:
                # Use full volume
                for i in range(density_values.shape[0]):
                    for j in range(density_values.shape[1]):
                        for k in range(density_values.shape[2]):
                            voxel_idx = np.array([i, j, k])
                            world_pos = np.array(origin) + voxel_idx * np.array(spacing)
                            points.append(tuple(world_pos))
                            density_list.append(float(density_values[i, j, k]))

            if SignalType.DENSITY in (signal_types or [SignalType.DENSITY]):
                signals["density"] = density_list

        return QueryResult(
            points=points,
            signals=signals,
            metadata={
                "model_id": model_id,
                "source": "mongodb",
                "spacing": spacing,
                "origin": origin,
                "grid_dimensions": grid_dimensions,
                "has_porosity_map": porosity_map is not None,
                "defect_count": (
                    defect_count
                    if isinstance(defect_count, (int, float, np.integer, np.floating)) and defect_count > 0
                    else None
                ),
                "n_defect_locations": len(defect_locations) if defect_locations else 0,
            },
            component_id=model_id,
        )

    def _query_in_memory(
        self,
        spatial: Optional[SpatialQuery] = None,
        temporal: Optional[TemporalQuery] = None,
        signal_types: Optional[List[SignalType]] = None,
    ) -> QueryResult:
        """Query CT scan data from in-memory volume (original implementation)."""
        if self.ct_volume is None:
            return QueryResult(
                points=[],
                signals={},
                metadata={"error": "No CT scan data available. Use set_ct_data() or connect to database."},
            )

        # Determine which signals to retrieve
        if signal_types is None:
            signal_types = self._available_signals

        # Apply spatial filtering if provided
        if spatial and spatial.bbox_min and spatial.bbox_max:
            bbox_min = np.array(spatial.bbox_min)
            bbox_max = np.array(spatial.bbox_max)
        else:
            # Use full volume
            bbox_min = np.array(self.ct_origin)
            bbox_max = np.array(self.ct_origin) + np.array(self.ct_spacing) * np.array(self.ct_volume.shape)

        # Convert bounding box to voxel indices
        origin = np.array(self.ct_origin)
        spacing = np.array(self.ct_spacing)

        idx_min = ((bbox_min - origin) / spacing).astype(int)
        idx_max = ((bbox_max - origin) / spacing).astype(int)

        # Clamp to volume bounds
        idx_min = np.maximum(idx_min, 0)
        idx_max = np.minimum(idx_max, np.array(self.ct_volume.shape) - 1)

        # Extract subvolume
        slices = tuple(slice(idx_min[i], idx_max[i] + 1) for i in range(3))
        subvolume = self.ct_volume[slices]

        # Generate points and signals
        points = []
        signals = {}

        if SignalType.DENSITY in signal_types:
            density_values = []

            # Iterate through voxels
            for i in range(subvolume.shape[0]):
                for j in range(subvolume.shape[1]):
                    for k in range(subvolume.shape[2]):
                        # Calculate world coordinates
                        voxel_idx = np.array([i, j, k]) + idx_min
                        world_pos = origin + voxel_idx * spacing

                        points.append(world_pos.tolist())
                        density_values.append(float(subvolume[i, j, k]))

            signals["density"] = density_values

        return QueryResult(
            points=points,
            signals=signals,
            metadata={
                "source": "ct_scan",
                "spacing": self.ct_spacing,
                "origin": self.ct_origin,
                "volume_shape": self.ct_volume.shape,
                "subvolume_shape": subvolume.shape,
            },
        )

    def get_bounding_box(
        self, component_id: Optional[str] = None
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """
        Get bounding box of CT scan data.

        Args:
            component_id: Model ID (for MongoDB mode) or None (for in-memory mode)

        Returns:
            Tuple of (bbox_min, bbox_max)
        """
        if self.use_mongodb and self.mongo_client and component_id:
            # Get from MongoDB
            collection = self._get_collection()
            doc = collection.find_one({"model_id": component_id})
            if doc:
                metadata = doc.get("metadata", {})
                coord_system = metadata.get("coordinate_system", {})
                origin = tuple(coord_system.get("origin", [0.0, 0.0, 0.0]))
                spacing = tuple(coord_system.get("voxel_spacing", [0.67, 0.67, 0.67]))
                grid_dimensions = tuple(metadata.get("statistics", {}).get("dimensions", [30, 30, 30]))

                bbox_min = origin
                bbox_max = tuple(np.array(origin) + np.array(spacing) * np.array(grid_dimensions))
                return (bbox_min, bbox_max)

        # In-memory mode
        if self.ct_volume is None:
            raise ValueError("No CT scan data available")

        origin = np.array(self.ct_origin)
        spacing = np.array(self.ct_spacing)
        shape = np.array(self.ct_volume.shape)

        bbox_min = tuple(origin)
        bbox_max = tuple(origin + spacing * shape)

        return (bbox_min, bbox_max)

    def get_available_signals(self) -> List[SignalType]:
        """Get available signal types."""
        return self._available_signals

    def voxelize_to_grid(self, target_grid, alignment_transform=None):
        """
        Voxelize CT scan data to target voxel grid.

        Args:
            target_grid: Target VoxelGrid object
            alignment_transform: Optional transformation matrix for alignment

        Returns:
            VoxelGrid with CT scan data interpolated
        """
        if self.ct_volume is None:
            raise ValueError("No CT scan data available")

        # Query all CT scan data
        ct_result = self.query()

        if not ct_result.points:
            return target_grid

        # Apply alignment transform if provided
        points = np.array(ct_result.points)
        if alignment_transform is not None:
            # Apply 4x4 transformation matrix
            ones = np.ones((points.shape[0], 1))
            points_homogeneous = np.hstack([points, ones])
            points_transformed = (alignment_transform @ points_homogeneous.T).T[:, :3]
            points = points_transformed

        # Add points to target grid
        density_values = ct_result.signals.get("density", [])
        for i, point in enumerate(points):
            x, y, z = point
            signals = {}
            if i < len(density_values):
                signals["density"] = density_values[i]

            target_grid.add_point(x, y, z, signals)

        target_grid.finalize()
        return target_grid

    # MongoDB-specific helper methods
    def get_scan(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get CT scan document for a model (MongoDB mode only).

        Args:
            model_id: Model UUID

        Returns:
            CT scan document or None if not found
        """
        if not self.use_mongodb:
            raise RuntimeError("get_scan() requires MongoDB mode. Call set_mongo_client() first.")

        collection = self._get_collection()
        return collection.find_one({"model_id": model_id})

    def get_density_values(self, model_id: str) -> Optional[np.ndarray]:
        """
        Get density values array from GridFS (MongoDB mode only).

        Args:
            model_id: Model UUID

        Returns:
            Density values array or None if not found
        """
        if not self.use_mongodb:
            raise RuntimeError("get_density_values() requires MongoDB mode. Call set_mongo_client() first.")

        doc = self.get_scan(model_id)
        if doc is None:
            return None

        # Check both top-level and nested data_storage location for GridFS ID
        density_gridfs_id = doc.get("density_values_gridfs_id") or doc.get("data_storage", {}).get("density_values_gridfs_id")
        if not density_gridfs_id:
            return None

        density_data = self.mongo_client.get_file(density_gridfs_id)
        if density_data:
            density_data = gzip.decompress(density_data)
            return np.load(io.BytesIO(density_data))
        return None

    def get_porosity_map(self, model_id: str) -> Optional[np.ndarray]:
        """
        Get porosity map array from GridFS (MongoDB mode only).

        Args:
            model_id: Model UUID

        Returns:
            Porosity map array or None if not found
        """
        if not self.use_mongodb:
            raise RuntimeError("get_porosity_map() requires MongoDB mode. Call set_mongo_client() first.")

        doc = self.get_scan(model_id)
        if doc is None or "porosity_map_gridfs_id" not in doc:
            return None

        porosity_data = self.mongo_client.get_file(doc["porosity_map_gridfs_id"])
        if porosity_data:
            porosity_data = gzip.decompress(porosity_data)
            return np.load(io.BytesIO(porosity_data))
        return None

    def get_defect_locations(self, model_id: str) -> List[Tuple[int, int, int]]:
        """
        Get defect locations for a model (MongoDB mode only).

        Args:
            model_id: Model UUID

        Returns:
            List of defect locations as (x, y, z) tuples
        """
        if not self.use_mongodb:
            raise RuntimeError("get_defect_locations() requires MongoDB mode. Call set_mongo_client() first.")

        doc = self.get_scan(model_id)
        if doc is None:
            return []

        defect_locations = doc.get("defect_locations", [])
        return [tuple(loc) for loc in defect_locations]

    def get_coordinate_system(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get coordinate system information for a model (MongoDB mode only).

        Args:
            model_id: Model UUID

        Returns:
            Coordinate system dictionary or None
        """
        if not self.use_mongodb:
            raise RuntimeError("get_coordinate_system() requires MongoDB mode. Call set_mongo_client() first.")

        doc = self.get_scan(model_id)
        if doc is None:
            return None

        metadata = doc.get("metadata", {})
        return metadata.get("coordinate_system", {})