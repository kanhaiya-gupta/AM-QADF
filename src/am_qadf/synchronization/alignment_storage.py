"""
Alignment Results Storage

Store temporal and spatial alignment results in MongoDB.
"""

import logging
from typing import Optional, Dict, List, Any
from datetime import datetime
import numpy as np
import uuid
import io
import gzip

logger = logging.getLogger(__name__)


class AlignmentStorage:
    """
    Storage client for alignment results.

    Stores temporal and spatial alignment results with metadata including:
    - alignment_id: Unique identifier for the alignment
    - model_id: Associated model ID
    - model_name: Model name for easier identification
    - transformation_matrix: Spatial transformation matrix (if applicable)
    - temporal_mapping: Temporal alignment mapping (if applicable)
    - alignment_metrics: Accuracy and quality metrics
    - configuration: Alignment configuration parameters
    """

    def __init__(self, mongo_client):
        """
        Initialize alignment storage.

        Args:
            mongo_client: MongoDB client instance
        """
        if not mongo_client or not mongo_client.is_connected():
            raise ConnectionError("MongoDB client not connected")

        self.mongo_client = mongo_client
        self.collection_name = "alignment_results"
        logger.info("AlignmentStorage initialized")

    def save_alignment(
        self,
        model_id: str,
        alignment_mode: str,
        transformation_matrix: Optional[np.ndarray] = None,
        temporal_mapping: Optional[Dict[str, Any]] = None,
        alignment_metrics: Optional[Dict[str, Any]] = None,
        aligned_data_sources: Optional[List[str]] = None,
        configuration: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        aligned_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save alignment results to MongoDB.

        Args:
            model_id: Model ID
            alignment_mode: 'temporal', 'spatial', or 'both'
            transformation_matrix: 4x4 transformation matrix (for spatial alignment)
            temporal_mapping: Temporal alignment mapping data
            alignment_metrics: Alignment accuracy and quality metrics
            aligned_data_sources: List of data sources that were aligned
            configuration: Alignment configuration parameters
            model_name: Model name for easier identification
            description: Optional description
            tags: Optional tags
            aligned_data: Dictionary of aligned data points (keyed by source name, e.g., 'hatching', 'laser', 'ct', 'ispm')

        Returns:
            alignment_id: Unique identifier for the saved alignment
        """
        if not self.mongo_client.is_connected():
            raise ConnectionError("MongoDB client not connected")

        collection = self.mongo_client.get_collection(self.collection_name)

        # Generate unique alignment ID
        alignment_id = str(uuid.uuid4())

        # Prepare transformation matrix for storage
        transformation_data = None
        if transformation_matrix is not None:
            transformation_data = {
                "matrix": (
                    transformation_matrix.tolist() if isinstance(transformation_matrix, np.ndarray) else transformation_matrix
                ),
                "shape": (list(transformation_matrix.shape) if isinstance(transformation_matrix, np.ndarray) else None),
            }

        # Prepare temporal mapping for storage
        temporal_data = None
        if temporal_mapping is not None:
            # Convert numpy arrays to lists
            temporal_data = {}
            for key, value in temporal_mapping.items():
                if isinstance(value, np.ndarray):
                    temporal_data[key] = value.tolist()
                elif isinstance(value, (dict, list)):
                    temporal_data[key] = self._convert_numpy_recursive(value)
                else:
                    temporal_data[key] = value

        # Prepare metrics for storage
        metrics_data = None
        if alignment_metrics is not None:
            metrics_data = {}
            for key, value in alignment_metrics.items():
                if isinstance(value, (np.ndarray, np.generic)):
                    metrics_data[key] = value.tolist() if hasattr(value, "tolist") else float(value)
                elif isinstance(value, (dict, list)):
                    metrics_data[key] = self._convert_numpy_recursive(value)
                else:
                    metrics_data[key] = value

        # Store aligned data points in GridFS (if provided)
        aligned_data_references = {}
        if aligned_data is not None:
            for source_name, source_data in aligned_data.items():
                if source_data is None:
                    continue

                # Extract points array
                points = None
                signals = None
                times = None
                layers = None

                if isinstance(source_data, dict):
                    points = source_data.get("points")
                    signals = source_data.get("signals")
                    times = source_data.get("times")
                    layers = source_data.get("layers")
                elif isinstance(source_data, np.ndarray):
                    points = source_data

                # Store points in GridFS if available
                if points is not None and isinstance(points, np.ndarray) and len(points) > 0:
                    try:
                        # Compress numpy array
                        buffer = io.BytesIO()
                        np.save(buffer, points)
                        compressed_data = gzip.compress(buffer.getvalue())

                        filename = f"{alignment_id}_{source_name}_points.npy.gz"
                        file_id = self.mongo_client.store_file(
                            compressed_data,
                            filename,
                            metadata={
                                "alignment_id": alignment_id,
                                "model_id": model_id,
                                "source_name": source_name,
                                "data_type": "aligned_points",
                                "format": "numpy_gzip",
                                "shape": list(points.shape),
                                "dtype": str(points.dtype),
                                "num_points": int(len(points)),
                            },
                        )
                        aligned_data_references[source_name] = {
                            "points_gridfs_id": str(file_id),
                            "num_points": int(len(points)),
                            "shape": list(points.shape),
                            "dtype": str(points.dtype),
                        }

                        # Store signals if available
                        if signals is not None:
                            # Check if signals dict is not empty
                            if isinstance(signals, dict) and len(signals) > 0:
                                # Store each signal separately
                                signal_refs = {}
                                for signal_name, signal_values in signals.items():
                                    if (
                                        signal_values is not None
                                        and isinstance(signal_values, np.ndarray)
                                        and len(signal_values) > 0
                                    ):
                                        try:
                                            sig_buffer = io.BytesIO()
                                            np.save(sig_buffer, signal_values)
                                            sig_compressed = gzip.compress(sig_buffer.getvalue())
                                            sig_filename = f"{alignment_id}_{source_name}_{signal_name}_signals.npy.gz"
                                            sig_file_id = self.mongo_client.store_file(
                                                sig_compressed,
                                                sig_filename,
                                                metadata={
                                                    "alignment_id": alignment_id,
                                                    "source_name": source_name,
                                                    "signal_name": signal_name,
                                                    "data_type": "aligned_signals",
                                                    "format": "numpy_gzip",
                                                },
                                            )
                                            signal_refs[signal_name] = str(sig_file_id)
                                        except Exception as e:
                                            logger.warning(f"Failed to store signal {signal_name} for {source_name}: {e}")
                                if signal_refs:
                                    aligned_data_references[source_name]["signals_gridfs_ids"] = signal_refs
                                else:
                                    logger.warning(
                                        f"No signals stored for {source_name} - all signal arrays were empty or invalid"
                                    )
                            elif isinstance(signals, np.ndarray) and len(signals) > 0:
                                # Single signal array
                                try:
                                    sig_buffer = io.BytesIO()
                                    np.save(sig_buffer, signals)
                                    sig_compressed = gzip.compress(sig_buffer.getvalue())
                                    sig_filename = f"{alignment_id}_{source_name}_signals.npy.gz"
                                    sig_file_id = self.mongo_client.store_file(
                                        sig_compressed,
                                        sig_filename,
                                        metadata={
                                            "alignment_id": alignment_id,
                                            "source_name": source_name,
                                            "data_type": "aligned_signals",
                                            "format": "numpy_gzip",
                                        },
                                    )
                                    aligned_data_references[source_name]["signals_gridfs_id"] = str(sig_file_id)
                                except Exception as e:
                                    logger.warning(f"Failed to store signals for {source_name}: {e}")
                            else:
                                logger.warning(f"Signals for {source_name} are None, empty, or invalid type: {type(signals)}")

                        # Store times and layers if available (smaller arrays, can be in document)
                        if times is not None and isinstance(times, np.ndarray):
                            aligned_data_references[source_name]["times"] = times.tolist() if len(times) < 10000 else None
                        if layers is not None and isinstance(layers, np.ndarray):
                            aligned_data_references[source_name]["layers"] = layers.tolist() if len(layers) < 10000 else None

                    except Exception as e:
                        logger.warning(f"Failed to store aligned data for {source_name}: {e}")

        # Create document
        alignment_doc = {
            "alignment_id": alignment_id,
            "model_id": model_id,
            "model_name": model_name or "",
            "alignment_mode": alignment_mode,
            "transformation_matrix": transformation_data,
            "temporal_mapping": temporal_data,
            "alignment_metrics": metrics_data,
            "aligned_data_sources": aligned_data_sources or [],
            "aligned_data_references": aligned_data_references,  # GridFS references for aligned data
            "configuration": configuration or {},
            "description": description or "",
            "tags": tags or [],
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }

        # Insert document
        result = collection.insert_one(alignment_doc)
        # Return the alignment_id from the document, not the MongoDB _id
        saved_alignment_id = alignment_doc["alignment_id"]

        logger.info(f"Saved alignment: {saved_alignment_id} for model {model_id} ({model_name or 'N/A'})")
        return saved_alignment_id

    def load_alignment(self, alignment_id: str, load_aligned_data: bool = False) -> Optional[Dict[str, Any]]:
        """
        Load alignment results from MongoDB.

        Args:
            alignment_id: Alignment ID
            load_aligned_data: If True, also load aligned data points from GridFS

        Returns:
            Alignment document (with aligned_data if load_aligned_data=True) or None if not found
        """
        if not self.mongo_client.is_connected():
            raise ConnectionError("MongoDB client not connected")

        collection = self.mongo_client.get_collection(self.collection_name)

        # Try to find by alignment_id field first, then by _id
        doc = collection.find_one({"alignment_id": alignment_id})
        if doc is None:
            doc = collection.find_one({"_id": alignment_id})

        if doc is None:
            logger.warning(f"Alignment not found: {alignment_id}")
            return None

        # Convert _id to string
        if "_id" in doc:
            doc["_id"] = str(doc["_id"])

        # Load aligned data from GridFS if requested
        if load_aligned_data and "aligned_data_references" in doc:
            aligned_data = {}
            for source_name, refs in doc["aligned_data_references"].items():
                source_data = {}

                # Load points
                if "points_gridfs_id" in refs:
                    try:
                        points_data = self.mongo_client.get_file(refs["points_gridfs_id"])
                        if points_data:
                            points_data = gzip.decompress(points_data)
                            points = np.load(io.BytesIO(points_data))
                            source_data["points"] = points
                    except Exception as e:
                        logger.warning(f"Failed to load points for {source_name}: {e}")

                # Load signals
                if "signals_gridfs_id" in refs:
                    try:
                        signals_data = self.mongo_client.get_file(refs["signals_gridfs_id"])
                        if signals_data:
                            signals_data = gzip.decompress(signals_data)
                            signals = np.load(io.BytesIO(signals_data))
                            source_data["signals"] = signals
                    except Exception as e:
                        logger.warning(f"Failed to load signals for {source_name}: {e}")
                elif "signals_gridfs_ids" in refs:
                    # Multiple signals
                    source_data["signals"] = {}
                    for signal_name, signal_file_id in refs["signals_gridfs_ids"].items():
                        try:
                            signal_data = self.mongo_client.get_file(signal_file_id)
                            if signal_data:
                                signal_data = gzip.decompress(signal_data)
                                signal_values = np.load(io.BytesIO(signal_data))
                                source_data["signals"][signal_name] = signal_values
                        except Exception as e:
                            logger.warning(f"Failed to load signal {signal_name} for {source_name}: {e}")

                # Load times and layers (if stored in document)
                if "times" in refs:
                    source_data["times"] = np.array(refs["times"]) if refs["times"] else None
                if "layers" in refs:
                    source_data["layers"] = np.array(refs["layers"]) if refs["layers"] else None

                if source_data:
                    aligned_data[source_name] = source_data

            doc["aligned_data"] = aligned_data

        logger.info(f"Loaded alignment: {alignment_id}")
        return doc

    def list_alignments(
        self,
        model_id: Optional[str] = None,
        alignment_mode: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        List alignment results.

        Args:
            model_id: Filter by model ID (optional)
            alignment_mode: Filter by alignment mode (optional)
            limit: Maximum number of results

        Returns:
            List of alignment documents
        """
        if not self.mongo_client.is_connected():
            raise ConnectionError("MongoDB client not connected")

        collection = self.mongo_client.get_collection(self.collection_name)

        # Build query
        query = {}
        if model_id:
            query["model_id"] = model_id
        if alignment_mode:
            query["alignment_mode"] = alignment_mode

        # Find alignments
        alignments = list(collection.find(query).sort("created_at", -1).limit(limit))

        # Convert _id to string
        for doc in alignments:
            if "_id" in doc:
                doc["_id"] = str(doc["_id"])

        logger.info(f"Found {len(alignments)} alignment(s)")
        return alignments

    def delete_alignment(self, alignment_id: str) -> bool:
        """
        Delete an alignment result.

        Args:
            alignment_id: Alignment ID

        Returns:
            True if deleted, False if not found
        """
        if not self.mongo_client.is_connected():
            raise ConnectionError("MongoDB client not connected")

        collection = self.mongo_client.get_collection(self.collection_name)

        # Try to delete by alignment_id field first, then by _id
        result = collection.delete_one({"alignment_id": alignment_id})
        if result.deleted_count == 0:
            result = collection.delete_one({"_id": alignment_id})

        if result.deleted_count > 0:
            logger.info(f"Deleted alignment: {alignment_id}")
            return True
        else:
            logger.warning(f"Alignment not found for deletion: {alignment_id}")
            return False

    def _convert_numpy_recursive(self, obj: Any) -> Any:
        """Recursively convert numpy arrays to lists."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: self._convert_numpy_recursive(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_recursive(item) for item in obj]
        else:
            return obj
