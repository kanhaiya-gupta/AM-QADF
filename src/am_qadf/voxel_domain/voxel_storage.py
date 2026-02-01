"""
Voxel Grid Storage

Storage and retrieval of voxel grids in MongoDB.
Uses OpenVDB format (.vdb files) stored in GridFS for efficient storage.
Each signal is stored as a named FloatGrid in a single .vdb file.

REQUIRES: OpenVDB C++ bindings (am_qadf_native.io, am_qadf_native.voxelization).
Raises ImportError if OpenVDB bindings are not available.
No fallback to legacy format - OpenVDB is required.
"""

import numpy as np
import pickle
import gzip
import io
import tempfile
import os
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def _to_bson_safe(obj: Any) -> Any:
    """Convert numpy types and nested structures to BSON-serializable Python types."""
    if isinstance(obj, dict):
        return {k: _to_bson_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_bson_safe(x) for x in obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# Import OpenVDB IO bindings (required, no fallback)
from am_qadf_native.io import VDBWriter, OpenVDBReader
from am_qadf_native import numpy_to_openvdb, openvdb_to_numpy


class VoxelGridStorage:
    """
    Storage and retrieval of voxel grids in MongoDB.

    Stores:
    - Grid metadata (resolution, bounding box, dimensions) in a collection
    - Signal arrays in GridFS (for large arrays)
    - Voxel data structure (sparse representation) in GridFS
    """

    def __init__(self, mongo_client):
        """
        Initialize voxel grid storage.

        Args:
            mongo_client: MongoDBClient instance
        """
        self.mongo_client = mongo_client
        self.collection_name = "voxel_grids"
        self.gridfs_bucket = "voxel_grid_data"

    def save_voxel_grid(
        self,
        model_id: str,
        grid_name: str,
        voxel_grid: Any,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        model_name: Optional[str] = None,
        configuration_metadata: Optional[Dict[str, Any]] = None,
        local_vdb_hatching_path: Optional[str] = None,
        local_vdb_stl_geometry_path: Optional[str] = None,
    ) -> str:
        """
        Save a voxel grid to MongoDB.

        Args:
            model_id: Model ID this grid belongs to
            grid_name: Name for the grid (must be unique per model)
            voxel_grid: VoxelGrid or AdaptiveResolutionGrid instance
            description: Optional description
            tags: Optional tags for categorization
            model_name: Optional model name for easier identification
            configuration_metadata: Optional dictionary with user-selected configuration
            local_vdb_hatching_path: Optional path to local VDB file (hatching/signals)
            local_vdb_stl_geometry_path: Optional path to local VDB file (STL geometry/occupancy)

            configuration_metadata may contain:
                - grid_type: Type of grid (uniform/adaptive/multi)
                - resolution_mode: Resolution mode (uniform/per_axis/adaptive)
                - uniform_resolution: Resolution value for uniform mode
                - x_resolution, y_resolution, z_resolution: Per-axis resolutions
                - bbox_mode: Bounding box mode (model/custom/interactive)
                - coordinate_system: Dict with type, origin, rotation
                - aggregation_method: Aggregation method (mean/max/min/etc)
                - sparse_storage: Whether sparse storage is enabled
                - compression: Whether compression is enabled
                - adaptive_strategy: Strategy for adaptive grids

        Returns:
            Grid ID (document _id as string)

        Note:
            Multiple grids can be saved for the same model with different configurations.
            The configuration_metadata allows recreating the exact grid settings later.
        """
        if not self.mongo_client or not self.mongo_client.is_connected():
            raise ConnectionError("MongoDB client not connected")

        # Check if grid with same name already exists for this model
        collection = self.mongo_client.get_collection(self.collection_name)
        existing = collection.find_one({"model_id": model_id, "grid_name": grid_name})

        result = None
        if existing:
            # Update existing grid
            grid_id = str(existing["_id"])
            logger.info(f"Updating existing grid: {grid_id}")
        else:
            # Create grid document first to get the grid_id
            # This ensures signals are stored with the correct grid_id in metadata
            metadata = self._extract_metadata(voxel_grid)

            # Store configuration metadata in nested structure
            if configuration_metadata:
                metadata["configuration_metadata"] = configuration_metadata

            # Create minimal document first to get the ID (BSON-safe: no numpy.int64/float64)
            initial_doc = _to_bson_safe({
                "model_id": model_id,
                "model_name": model_name or "",
                "grid_name": grid_name,
                "description": description or "",
                "tags": tags or [],
                "metadata": metadata,
                "signal_references": {},  # Will be updated
                "voxel_data_reference": None,  # Will be updated
                "available_signals": list(voxel_grid.available_signals),
                "local_vdb_hatching_path": local_vdb_hatching_path,
                "local_vdb_stl_geometry_path": local_vdb_stl_geometry_path,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
            })
            result = collection.insert_one(initial_doc)
            grid_id = str(result.inserted_id)
            logger.info(f"Created new grid document: {grid_id}")

        # Extract grid metadata (if not already done for new grids)
        if existing:
            metadata = self._extract_metadata(voxel_grid)

            # Store configuration metadata in nested structure
            if configuration_metadata:
                metadata["configuration_metadata"] = configuration_metadata
                logger.debug(f"Stored configuration metadata: {list(configuration_metadata.keys())}")
            else:
                logger.warning(f"No configuration_metadata provided for grid {grid_name}. Grid will have limited metadata.")

        # Store grids in OpenVDB format (required, no fallback)
        signal_references = {}
        
        # Store as OpenVDB .vdb file (one file with multiple named grids)
        vdb_file_id = self._store_voxel_grid_openvdb(grid_id, voxel_grid)
        if not vdb_file_id:
            raise RuntimeError(f"Failed to store voxel grid as OpenVDB format for grid {grid_id}")
        
        # Store reference to .vdb file
        signal_references["_openvdb_file"] = vdb_file_id
        logger.info(f"Stored voxel grid as OpenVDB format: {vdb_file_id}")

        # Update document with signal references and voxel data reference (BSON-safe).
        # OpenVDB path: no separate voxel_data_reference; all data is in the .vdb file.
        voxel_data_ref = None
        grid_doc = _to_bson_safe({
            "model_id": model_id,
            "model_name": model_name or "",  # Store model_name for easier identification
            "grid_name": grid_name,
            "description": description or "",
            "tags": tags or [],
            "metadata": metadata,
            "signal_references": signal_references,
            "voxel_data_reference": voxel_data_ref,
            "available_signals": list(voxel_grid.available_signals),
            "local_vdb_hatching_path": local_vdb_hatching_path,
            "local_vdb_stl_geometry_path": local_vdb_stl_geometry_path,
            "updated_at": datetime.utcnow(),
        })

        if existing:
            # Update existing
            collection.update_one({"_id": existing["_id"]}, {"$set": grid_doc})
        else:
            # Update the document we just created
            collection.update_one({"_id": result.inserted_id}, {"$set": grid_doc})

        logger.info(f"Saved voxel grid: {grid_id} ({grid_name}) for model {model_id}")
        return grid_id

    def _store_voxel_grid_openvdb(self, grid_id: str, voxel_grid: Any) -> Optional[str]:
        """
        Store voxel grid as OpenVDB .vdb file.
        
        Each signal is stored as a named FloatGrid in a single .vdb file.
        
        Args:
            grid_id: Grid ID
            voxel_grid: VoxelGrid or AdaptiveResolutionGrid instance
            
        Returns:
            GridFS file ID for the .vdb file, or None if failed
        """
        
        try:
            writer = VDBWriter()
            grids = []
            
            # Convert each signal to FloatGrid
            for signal_name in voxel_grid.available_signals:
                try:
                    # Get signal as numpy array
                    signal_array = voxel_grid.get_signal_array(signal_name, default=0.0)
                    
                    # Convert to OpenVDB FloatGrid
                    float_grid = numpy_to_openvdb(signal_array, voxel_grid.resolution)
                    
                    # Set grid name to signal name (OpenVDB supports named grids)
                    # Note: This requires accessing the grid's name property
                    # OpenVDB grids can be named for identification in multi-grid files
                    grids.append(float_grid)
                except Exception as e:
                    logger.warning(f"Failed to convert signal {signal_name} to FloatGrid: {e}")
                    continue
            
            if len(grids) == 0:
                logger.warning("No signals to store in OpenVDB format")
                return None
            
            # Write all grids to temporary .vdb file
            with tempfile.NamedTemporaryFile(suffix='.vdb', delete=False) as tmp_file:
                vdb_filename = tmp_file.name
            
            try:
                # Write multiple grids with signal names to single .vdb file
                signal_names = list(voxel_grid.available_signals)
                writer.write_multiple_with_names(grids, signal_names, vdb_filename)
                
                # Read .vdb file and store in GridFS
                with open(vdb_filename, 'rb') as f:
                    vdb_data = f.read()
                
                # Store in GridFS (MongoDBClient uses single GridFS bucket, no bucket_name)
                file_id = self.mongo_client.store_file(
                    vdb_data,
                    filename=f"{grid_id}.vdb",
                    metadata={
                        "grid_id": grid_id,
                        "data_type": "voxel_grid",
                        "format": "openvdb",
                        "num_signals": len(grids),
                        "signals": list(voxel_grid.available_signals),
                    }
                )
                
                return str(file_id)
            finally:
                # Clean up temporary file
                if os.path.exists(vdb_filename):
                    os.unlink(vdb_filename)
                    
        except Exception as e:
            logger.error(f"Error storing voxel grid as OpenVDB format: {e}", exc_info=True)
            return None

    def _store_voxel_grid_legacy(self, grid_id: str, voxel_grid: Any) -> Tuple[Dict[str, str], Optional[str]]:
        """
        Store voxel grid in legacy format (dictionary/numpy arrays).
        
        This is kept for backward compatibility and as fallback.
        
        Returns:
            Tuple of (signal_references dict, voxel_data_ref)
        """
        signal_references = {}
        for signal_name in voxel_grid.available_signals:
            try:
                # Use sparse storage (much faster than dense array conversion)
                signal_ref = self._store_signal_sparse(grid_id, signal_name, voxel_grid)
                signal_references[signal_name] = signal_ref
            except Exception as e:
                logger.warning(f"Failed to store signal {signal_name}: {e}")
                # Fallback to dense array if sparse storage fails
                try:
                    signal_array = voxel_grid.get_signal_array(signal_name, default=0.0)
                    if signal_array is not None and signal_array.size > 0:
                        signal_ref = self._store_signal_array(grid_id, signal_name, signal_array)
                        signal_references[signal_name] = signal_ref
                except Exception as e2:
                    logger.warning(f"Failed to store signal {signal_name} (dense fallback): {e2}")

        # Store voxel data structure (sparse representation)
        voxel_data_ref = None
        # Check if grid has voxels attribute (VoxelGrid) or region_grids (AdaptiveResolutionGrid)
        if hasattr(voxel_grid, "voxels") and voxel_grid.voxels:
            try:
                # OPTIMIZED: Use dict comprehension for faster conversion
                voxel_data = {
                    str(idx): (dict(voxel_obj.signals) if hasattr(voxel_obj, "signals") else {})
                    for idx, voxel_obj in voxel_grid.voxels.items()
                }

                voxel_data_ref = self._store_voxel_data(grid_id, voxel_data)
            except Exception as e:
                logger.warning(f"Failed to store voxel data: {e}")
        elif hasattr(voxel_grid, "region_grids") and voxel_grid.region_grids:
            # Handle AdaptiveResolutionGrid - store data from region grids
            try:
                voxel_data = {}
                for region_key, region_grid in voxel_grid.region_grids.items():
                    if hasattr(region_grid, "voxels") and region_grid.voxels:
                        for idx, voxel_obj in region_grid.voxels.items():
                            key = f"{region_key}_{idx}"
                            voxel_data[key] = dict(voxel_obj.signals) if hasattr(voxel_obj, "signals") else {}
                if voxel_data:
                    voxel_data_ref = self._store_voxel_data(grid_id, voxel_data)
            except Exception as e:
                logger.warning(f"Failed to store adaptive grid voxel data: {e}")
        
        return signal_references, voxel_data_ref

    def load_voxel_grid(self, grid_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a voxel grid from MongoDB.

        Args:
            grid_id: Grid ID (document _id)

        Returns:
            Dictionary with grid data and metadata, or None if not found
        """
        if not self.mongo_client or not self.mongo_client.is_connected():
            raise ConnectionError("MongoDB client not connected")

        from bson import ObjectId
        from bson.errors import InvalidId

        try:
            collection = self.mongo_client.get_collection(self.collection_name)
            # Try to convert to ObjectId, but handle mock IDs gracefully
            try:
                object_id = ObjectId(grid_id)
            except (ValueError, TypeError, InvalidId):
                # If it's not a valid ObjectId (e.g., mock ID), use as string
                object_id = grid_id
            grid_doc = collection.find_one({"_id": object_id})
            if not grid_doc:
                logger.warning(f"Grid not found: {grid_id}")
                return None

            # Load from OpenVDB format (required, no fallback)
            signal_references = grid_doc.get("signal_references", {})
            if "_openvdb_file" not in signal_references:
                raise ValueError(
                    f"Grid {grid_id} is not stored in OpenVDB format. "
                    "Legacy format is no longer supported. Please re-save the grid."
                )
            
            # Load from OpenVDB format
            signal_arrays, voxel_data = self._load_voxel_grid_openvdb(signal_references["_openvdb_file"])

            return {
                "grid_id": grid_id,
                "model_id": grid_doc["model_id"],
                "grid_name": grid_doc["grid_name"],
                "description": grid_doc.get("description", ""),
                "tags": grid_doc.get("tags", []),
                "metadata": grid_doc.get("metadata", {}),
                "available_signals": grid_doc.get("available_signals", []),
                "signal_arrays": signal_arrays,
                "voxel_data": voxel_data,
                "created_at": grid_doc.get("created_at"),
                "updated_at": grid_doc.get("updated_at"),
            }
        except Exception as e:
            logger.error(f"Error loading grid {grid_id}: {e}")
            return None

    def list_grids(self, model_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        List available voxel grids.

        Args:
            model_id: Optional model ID to filter by
            limit: Maximum number of grids to return

        Returns:
            List of grid metadata dictionaries
        """
        if not self.mongo_client or not self.mongo_client.is_connected():
            raise ConnectionError("MongoDB client not connected")

        query = {}
        if model_id:
            query["model_id"] = model_id

        collection = self.mongo_client.get_collection(self.collection_name)
        grids = list(
            collection.find(
                query,
                {
                    "_id": 1,
                    "model_id": 1,
                    "grid_name": 1,
                    "description": 1,
                    "tags": 1,
                    "metadata": 1,
                    "available_signals": 1,
                    "created_at": 1,
                    "updated_at": 1,
                },
            )
            .sort("created_at", -1)
            .limit(limit)
        )

        # Convert ObjectId to string
        for grid in grids:
            grid["grid_id"] = str(grid["_id"])
            del grid["_id"]

        return grids

    def delete_grid(self, grid_id: str) -> bool:
        """
        Delete a voxel grid from MongoDB.

        Args:
            grid_id: Grid ID to delete

        Returns:
            True if deleted, False otherwise
        """
        if not self.mongo_client or not self.mongo_client.is_connected():
            raise ConnectionError("MongoDB client not connected")

        from bson import ObjectId
        from bson.errors import InvalidId

        try:
            # Get grid document to find GridFS references
            collection = self.mongo_client.get_collection(self.collection_name)
            # Try to convert to ObjectId, but handle mock IDs gracefully
            try:
                object_id = ObjectId(grid_id)
            except (ValueError, TypeError, InvalidId):
                # If it's not a valid ObjectId (e.g., mock ID), use as string
                object_id = grid_id
            grid_doc = collection.find_one({"_id": object_id})
            if not grid_doc:
                return False

            # Delete signal arrays from GridFS
            for signal_ref in grid_doc.get("signal_references", {}).values():
                try:
                    self._delete_gridfs_file(signal_ref)
                except Exception as e:
                    logger.warning(f"Failed to delete signal array: {e}")

            # Delete voxel data from GridFS
            if grid_doc.get("voxel_data_reference"):
                try:
                    self._delete_gridfs_file(grid_doc["voxel_data_reference"])
                except Exception as e:
                    logger.warning(f"Failed to delete voxel data: {e}")

            # Delete metadata document
            collection = self.mongo_client.get_collection(self.collection_name)
            # Use the same object_id we used for find_one
            result = collection.delete_one({"_id": object_id})

            logger.info(f"Deleted grid: {grid_id}")
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Error deleting grid {grid_id}: {e}")
            return False

    def _extract_metadata(self, voxel_grid: Any) -> Dict[str, Any]:
        """Extract metadata from voxel grid (BSON-safe: native int/float for MongoDB)."""
        metadata = {}

        if hasattr(voxel_grid, "bbox_min") and hasattr(voxel_grid, "bbox_max"):
            bmin = voxel_grid.bbox_min.tolist() if isinstance(voxel_grid.bbox_min, np.ndarray) else list(voxel_grid.bbox_min)
            bmax = voxel_grid.bbox_max.tolist() if isinstance(voxel_grid.bbox_max, np.ndarray) else list(voxel_grid.bbox_max)
            metadata["bbox_min"] = [float(x) for x in bmin]
            metadata["bbox_max"] = [float(x) for x in bmax]

        if hasattr(voxel_grid, "resolution"):
            metadata["resolution"] = float(voxel_grid.resolution)

        if hasattr(voxel_grid, "dims"):
            dims = voxel_grid.dims.tolist() if isinstance(voxel_grid.dims, np.ndarray) else list(voxel_grid.dims)
            metadata["dims"] = [int(x) for x in dims]

        if hasattr(voxel_grid, "aggregation"):
            metadata["aggregation"] = voxel_grid.aggregation

        # Determine grid type
        if hasattr(voxel_grid, "__class__"):
            class_name = voxel_grid.__class__.__name__
            metadata["grid_type"] = class_name

        return metadata

    def _store_signal_sparse(self, grid_id: str, signal_name: str, voxel_grid: Any) -> str:
        """
        Store signal in sparse format (OPTIMIZED - avoids dense array creation).

        This is much faster than get_signal_array() + dense storage because:
        1. No dense array creation (saves memory)
        2. Direct sparse extraction (faster iteration)
        3. Smaller storage size (only non-zero values)
        """
        # Extract sparse representation directly from voxel grid
        indices = []
        values = []

        # Handle VoxelGrid
        if hasattr(voxel_grid, "voxels") and voxel_grid.voxels:
            # Optimized: single pass through voxels
            for (i, j, k), voxel in voxel_grid.voxels.items():
                if hasattr(voxel, "signals") and signal_name in voxel.signals:
                    indices.append([i, j, k])
                    values.append(float(voxel.signals[signal_name]))
        # Handle AdaptiveResolutionGrid
        elif hasattr(voxel_grid, "region_grids") and voxel_grid.region_grids:
            # Extract from all region grids
            for region_key, region_grid in voxel_grid.region_grids.items():
                if hasattr(region_grid, "voxels") and region_grid.voxels:
                    for (i, j, k), voxel in region_grid.voxels.items():
                        if hasattr(voxel, "signals") and signal_name in voxel.signals:
                            indices.append([i, j, k])
                            values.append(float(voxel.signals[signal_name]))

        if len(indices) == 0:
            # No data for this signal
            raise ValueError(f"No data found for signal {signal_name}")

        # Convert to numpy arrays
        indices_array = np.array(indices, dtype=np.int32)
        values_array = np.array(values, dtype=np.float32)

        # Get grid dimensions
        if hasattr(voxel_grid, "dims"):
            dims = voxel_grid.dims.tolist() if isinstance(voxel_grid.dims, np.ndarray) else list(voxel_grid.dims)
        elif hasattr(voxel_grid, "region_grids") and voxel_grid.region_grids:
            # For AdaptiveResolutionGrid, use dimensions from first region grid
            first_grid = next(iter(voxel_grid.region_grids.values()))
            if hasattr(first_grid, "dims"):
                dims = first_grid.dims.tolist() if isinstance(first_grid.dims, np.ndarray) else list(first_grid.dims)
            else:
                # Fallback: calculate from bbox and base resolution
                size = voxel_grid.bbox_max - voxel_grid.bbox_min
                dims = np.ceil(size / voxel_grid.base_resolution).astype(int).tolist()
        else:
            # Fallback: calculate from bbox and base resolution
            size = voxel_grid.bbox_max - voxel_grid.bbox_min
            dims = np.ceil(size / voxel_grid.base_resolution).astype(int).tolist()

        # Store sparse representation
        sparse_data = {
            "indices": indices_array,
            "values": values_array,
            "dims": dims,
            "default": 0.0,
            "format": "sparse",
        }

        # Compress and store
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **sparse_data)
        compressed_data = gzip.compress(buffer.getvalue())

        filename = f"{grid_id}_{signal_name}_sparse.npz.gz"
        file_id = self.mongo_client.store_file(
            compressed_data,
            filename,
            metadata={
                "grid_id": grid_id,
                "signal_name": signal_name,
                "data_type": "signal_array",
                "format": "sparse_numpy_gzip",
                "dims": dims,
                "dtype": str(values_array.dtype),
                "num_values": len(values_array),
            },
        )

        return str(file_id)

    def _store_signal_array(self, grid_id: str, signal_name: str, signal_array: np.ndarray) -> str:
        """
        Store signal array in GridFS (dense format).

        Note: Prefer _store_signal_sparse() for better performance.
        This method is kept for backward compatibility and fallback.
        """
        # Compress numpy array
        buffer = io.BytesIO()
        np.save(buffer, signal_array)
        compressed_data = gzip.compress(buffer.getvalue())

        filename = f"{grid_id}_{signal_name}.npy.gz"
        file_id = self.mongo_client.store_file(
            compressed_data,
            filename,
            metadata={
                "grid_id": grid_id,
                "signal_name": signal_name,
                "data_type": "signal_array",
                "format": "numpy_gzip",
                "shape": list(signal_array.shape),
                "dtype": str(signal_array.dtype),
            },
        )

        return str(file_id)

    def _load_signal_array(self, file_id: str) -> np.ndarray:
        """
        Load signal array from GridFS.

        Supports both sparse and dense formats automatically.
        """
        file_data = self.mongo_client.get_file(file_id)
        if not file_data:
            raise ValueError(f"File not found: {file_id}")

        decompressed = gzip.decompress(file_data)
        buffer = io.BytesIO(decompressed)

        # Try to load as sparse format first
        try:
            data = np.load(buffer, allow_pickle=True)

            # Check if it's sparse format (npz file with 'format' key)
            is_sparse = False
            if isinstance(data, np.lib.npyio.NpzFile):
                # For npz files, check if 'format' is in the files list
                if "format" in data.files:
                    format_value = data["format"]
                    if isinstance(format_value, np.ndarray):
                        format_value = format_value.item() if format_value.size == 1 else str(format_value)
                    is_sparse = (format_value == "sparse" or format_value == b"sparse")
            elif isinstance(data, dict):
                # For dict-like objects
                is_sparse = data.get("format") == "sparse"
            
            if is_sparse:
                # Reconstruct dense array from sparse representation
                indices = data["indices"]
                values = data["values"]
                dims = tuple(data["dims"])
                default = float(data["default"].item() if hasattr(data["default"], "item") else data["default"]) if "default" in (data.files if isinstance(data, np.lib.npyio.NpzFile) else data) else 0.0

                # Create dense array
                signal_array = np.full(dims, default, dtype=values.dtype)
                if len(indices) > 0:
                    signal_array[indices[:, 0], indices[:, 1], indices[:, 2]] = values

                return signal_array
            else:
                # Dense format - handle both npz and npy
                if isinstance(data, np.lib.npyio.NpzFile):
                    # Get first array from npz file
                    if len(data.files) > 0:
                        first_key = data.files[0]
                        signal_array = data[first_key]
                        return signal_array
                    else:
                        raise ValueError("Empty npz file")
                elif isinstance(data, np.ndarray):
                    # Direct numpy array
                    return data
                else:
                    # Fallback: try to reload
                    buffer.seek(0)
                    signal_array = np.load(buffer)
                    return signal_array
        except Exception as e:
            # Fallback: try as regular numpy file
            buffer.seek(0)
            try:
                signal_array = np.load(buffer)
                return signal_array
            except Exception as e2:
                raise ValueError(f"Failed to load signal array: {e2}")

    def _load_voxel_grid_openvdb(self, file_id: str) -> Tuple[Dict[str, np.ndarray], Optional[Dict]]:
        """
        Load voxel grid from OpenVDB .vdb file.
        
        Args:
            file_id: GridFS file ID for the .vdb file
            
        Returns:
            Tuple of (signal_arrays dict, voxel_data dict)
        """
        
        try:
            # Get .vdb file from GridFS
            vdb_data = self.mongo_client.get_file(file_id)
            
            if not vdb_data:
                raise ValueError(f"OpenVDB file not found: {file_id}")
            
            # Write to temporary file
            with tempfile.NamedTemporaryFile(suffix='.vdb', delete=False) as tmp_file:
                vdb_filename = tmp_file.name
                tmp_file.write(vdb_data)
            
            try:
                # Load all grids from .vdb file (by name)
                reader = OpenVDBReader()
                
                # Load all grids as a map (grid_name -> FloatGridPtr)
                grids_map = reader.load_all_grids(vdb_filename)
                
                # Convert each grid to numpy array
                signal_arrays = {}
                for signal_name, float_grid in grids_map.items():
                    try:
                        # Convert FloatGrid to numpy array
                        signal_array = openvdb_to_numpy(float_grid)
                        signal_arrays[signal_name] = signal_array
                    except Exception as e:
                        logger.warning(f"Failed to convert grid {signal_name} to numpy: {e}")
                
                # Return signal arrays and None for voxel_data (not needed for OpenVDB format)
                return signal_arrays, None
                
            finally:
                # Clean up temporary file
                if os.path.exists(vdb_filename):
                    os.unlink(vdb_filename)
                    
        except Exception as e:
            logger.error(f"Error loading voxel grid from OpenVDB format: {e}", exc_info=True)
            raise

    def _load_voxel_grid_legacy(self, grid_doc: Dict[str, Any]) -> Tuple[Dict[str, np.ndarray], Optional[Dict]]:
        """
        Load voxel grid from legacy format (dictionary/numpy arrays).
        
        DEPRECATED: This method is kept for backward compatibility only.
        It is NOT used as a fallback - OpenVDB is required.
        May be used for reading old data only if explicitly needed.
        
        Returns:
            Tuple of (signal_arrays dict, voxel_data dict)
        """
        signal_arrays = {}
        for signal_name, signal_ref in grid_doc.get("signal_references", {}).items():
            if signal_name == "_openvdb_file":
                continue  # Skip OpenVDB file reference
            try:
                signal_array = self._load_signal_array(signal_ref)
                signal_arrays[signal_name] = signal_array
            except Exception as e:
                logger.warning(f"Failed to load signal {signal_name}: {e}")

        # Load voxel data structure
        voxel_data = None
        if grid_doc.get("voxel_data_reference"):
            try:
                voxel_data = self._load_voxel_data(grid_doc["voxel_data_reference"])
            except Exception as e:
                logger.warning(f"Failed to load voxel data: {e}")
        
        return signal_arrays, voxel_data

    def _store_voxel_data(self, grid_id: str, voxel_data: Dict) -> str:
        """Store voxel data structure in GridFS."""
        # Serialize and compress
        serialized = pickle.dumps(voxel_data)
        compressed_data = gzip.compress(serialized)

        filename = f"{grid_id}_voxel_data.pkl.gz"
        file_id = self.mongo_client.store_file(
            compressed_data,
            filename,
            metadata={
                "grid_id": grid_id,
                "data_type": "voxel_data",
                "format": "pickle_gzip",
            },
        )

        return str(file_id)

    def _load_voxel_data(self, file_id: str) -> Dict:
        """Load voxel data structure from GridFS."""
        file_data = self.mongo_client.get_file(file_id)
        if file_data:
            decompressed = gzip.decompress(file_data)
            voxel_data = pickle.loads(decompressed)
            return voxel_data
        else:
            raise ValueError(f"File not found: {file_id}")

    def _delete_gridfs_file(self, file_id: str):
        """Delete a file from GridFS."""
        self.mongo_client.delete_file(file_id)
