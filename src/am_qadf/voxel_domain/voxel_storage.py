"""
Voxel Grid Storage

Storage and retrieval of voxel grids in MongoDB.
Uses GridFS for large signal arrays and metadata collection for grid information.
"""

import numpy as np
import pickle
import gzip
import io
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


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
            configuration_metadata: Optional dictionary with user-selected configuration:
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

            # Create minimal document first to get the ID
            initial_doc = {
                "model_id": model_id,
                "model_name": model_name or "",
                "grid_name": grid_name,
                "description": description or "",
                "tags": tags or [],
                "metadata": metadata,
                "signal_references": {},  # Will be updated
                "voxel_data_reference": None,  # Will be updated
                "available_signals": list(voxel_grid.available_signals),
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
            }
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

        # Store signal arrays in GridFS (now with actual grid_id)
        # OPTIMIZATION: Store sparse representation instead of dense arrays
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

        # Update document with signal references and voxel data reference
        grid_doc = {
            "model_id": model_id,
            "model_name": model_name or "",  # Store model_name for easier identification
            "grid_name": grid_name,
            "description": description or "",
            "tags": tags or [],
            "metadata": metadata,
            "signal_references": signal_references,
            "voxel_data_reference": voxel_data_ref,
            "available_signals": list(voxel_grid.available_signals),
            "updated_at": datetime.utcnow(),
        }

        if existing:
            # Update existing
            collection.update_one({"_id": existing["_id"]}, {"$set": grid_doc})
        else:
            # Update the document we just created
            collection.update_one({"_id": result.inserted_id}, {"$set": grid_doc})

        logger.info(f"Saved voxel grid: {grid_id} ({grid_name}) for model {model_id}")
        return grid_id

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

            # Load signal arrays from GridFS
            signal_arrays = {}
            for signal_name, signal_ref in grid_doc.get("signal_references", {}).items():
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
        """Extract metadata from voxel grid."""
        metadata = {}

        if hasattr(voxel_grid, "bbox_min") and hasattr(voxel_grid, "bbox_max"):
            metadata["bbox_min"] = (
                voxel_grid.bbox_min.tolist() if isinstance(voxel_grid.bbox_min, np.ndarray) else list(voxel_grid.bbox_min)
            )
            metadata["bbox_max"] = (
                voxel_grid.bbox_max.tolist() if isinstance(voxel_grid.bbox_max, np.ndarray) else list(voxel_grid.bbox_max)
            )

        if hasattr(voxel_grid, "resolution"):
            metadata["resolution"] = float(voxel_grid.resolution)

        if hasattr(voxel_grid, "dims"):
            metadata["dims"] = voxel_grid.dims.tolist() if isinstance(voxel_grid.dims, np.ndarray) else list(voxel_grid.dims)

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

            # Check if it's sparse format
            if "format" in data and data["format"] == "sparse":
                # Reconstruct dense array from sparse representation
                indices = data["indices"]
                values = data["values"]
                dims = tuple(data["dims"])
                default = float(data["default"]) if "default" in data else 0.0

                # Create dense array
                signal_array = np.full(dims, default, dtype=values.dtype)
                signal_array[indices[:, 0], indices[:, 1], indices[:, 2]] = values

                return signal_array
            else:
                # Dense format (legacy)
                buffer.seek(0)  # Reset buffer
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
