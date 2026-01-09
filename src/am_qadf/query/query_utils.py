"""
Query Utilities for MongoDB Data Warehouse

Common query patterns and helpers for warehouse-specific queries.
Provides builders for spatial, temporal, and aggregation queries.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from datetime import datetime


def build_spatial_query(
    bbox_min: Optional[Tuple[float, float, float]] = None,
    bbox_max: Optional[Tuple[float, float, float]] = None,
    field_name: str = "spatial_coordinates",
) -> Dict[str, Any]:
    """
    Build a MongoDB spatial query for bounding box.

    Args:
        bbox_min: Minimum bounding box coordinates (x_min, y_min, z_min)
        bbox_max: Maximum bounding box coordinates (x_max, y_max, z_max)
        field_name: Name of the spatial coordinates field

    Returns:
        MongoDB query dictionary
    """
    query = {}

    if bbox_min is not None and bbox_max is not None:
        # Query for points within bounding box
        # MongoDB query: field[0] >= bbox_min[0] AND field[0] <= bbox_max[0] AND ...
        query[field_name] = {"$elemMatch": {"$gte": bbox_min[0], "$lte": bbox_max[0]}}
        # Note: MongoDB doesn't natively support 3D bounding box queries on arrays
        # This is a simplified version. For full 3D queries, we'll filter in Python
        # or use MongoDB's 2dsphere index if we convert to GeoJSON format

    return query


def build_temporal_query(
    time_start: Optional[datetime] = None,
    time_end: Optional[datetime] = None,
    layer_start: Optional[int] = None,
    layer_end: Optional[int] = None,
    time_field: str = "timestamp",
    layer_field: str = "layer_index",
) -> Dict[str, Any]:
    """
    Build a MongoDB temporal query.

    Args:
        time_start: Start time (datetime)
        time_end: End time (datetime)
        layer_start: Start layer index
        layer_end: End layer index
        time_field: Name of the timestamp field
        layer_field: Name of the layer index field

    Returns:
        MongoDB query dictionary
    """
    query: Dict[str, Any] = {}

    # Time range query
    if time_start is not None or time_end is not None:
        time_query = {}
        if time_start is not None:
            time_query["$gte"] = time_start
        if time_end is not None:
            time_query["$lte"] = time_end
        if time_query:
            query[time_field] = time_query

    # Layer range query
    if layer_start is not None or layer_end is not None:
        layer_query = {}
        if layer_start is not None:
            layer_query["$gte"] = layer_start
        if layer_end is not None:
            layer_query["$lte"] = layer_end
        if layer_query:
            query[layer_field] = layer_query

    return query


def build_model_query(
    model_id: Optional[str] = None,
    model_name: Optional[str] = None,
    filename: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a MongoDB query for model filtering.

    Args:
        model_id: Model UUID
        model_name: Model name
        filename: STL filename

    Returns:
        MongoDB query dictionary
    """
    query = {}

    if model_id:
        query["model_id"] = model_id
    if model_name:
        query["model_name"] = model_name
    if filename:
        query["filename"] = filename

    return query


def build_parameter_range_query(
    field_name: Optional[str] = None,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Build a MongoDB query for parameter range filtering.

    Args:
        field_name: Name of the parameter field
        min_value: Minimum value
        max_value: Maximum value

    Returns:
        MongoDB query dictionary fragment
    """
    if field_name is None or (min_value is None and max_value is None):
        return {}

    query = {}
    if min_value is not None and max_value is not None:
        query[field_name] = {"$gte": min_value, "$lte": max_value}
    elif min_value is not None:
        query[field_name] = {"$gte": min_value}
    elif max_value is not None:
        query[field_name] = {"$lte": max_value}

    return query


def combine_queries(*queries: Dict[str, Any]) -> Dict[str, Any]:
    """
    Combine multiple MongoDB queries with AND logic.

    Args:
        *queries: Variable number of query dictionaries

    Returns:
        Combined MongoDB query dictionary
    """
    combined = {}
    for query in queries:
        if query:
            combined.update(query)
    return combined


def filter_points_in_bbox(
    points: np.ndarray,
    bbox_min: Tuple[float, float, float],
    bbox_max: Tuple[float, float, float],
) -> np.ndarray:
    """
    Filter points that are within a 3D bounding box.

    Args:
        points: Array of points with shape (n, 3) or (n,)
        bbox_min: Minimum bounding box (x_min, y_min, z_min)
        bbox_max: Maximum bounding box (x_max, y_max, z_max)

    Returns:
        Boolean mask array indicating which points are in bbox
    """
    if len(points) == 0:
        return np.array([], dtype=bool)

    points = np.asarray(points)
    if points.ndim == 1:
        # Single point
        return np.all((points >= bbox_min) & (points <= bbox_max))

    # Multiple points
    mask = np.all((points >= bbox_min) & (points <= bbox_max), axis=1)
    return mask


def extract_coordinate_system(doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Extract coordinate system information from a document.

    Args:
        doc: MongoDB document

    Returns:
        Coordinate system dictionary or None
    """
    # Try different possible field names
    coord_system = doc.get("coordinate_system") or doc.get("metadata", {}).get("coordinate_system")
    return coord_system


def get_bounding_box_from_doc(
    doc: Dict[str, Any],
) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
    """
    Extract bounding box from a document.

    Args:
        doc: MongoDB document

    Returns:
        Tuple of (bbox_min, bbox_max) or None
    """
    # Try different possible locations and formats
    # Direct bbox_min/bbox_max format
    if "bbox_min" in doc and "bbox_max" in doc:
        bbox_min = doc["bbox_min"]
        bbox_max = doc["bbox_max"]
        if isinstance(bbox_min, (list, tuple)) and isinstance(bbox_max, (list, tuple)):
            return (tuple(bbox_min), tuple(bbox_max))

    # Nested bounding_box format
    bbox = doc.get("bounding_box") or doc.get("metadata", {}).get("bounding_box")

    if bbox is None:
        return None

    # Handle different formats
    if isinstance(bbox, dict):
        if "min" in bbox and "max" in bbox:
            return (tuple(bbox["min"]), tuple(bbox["max"]))
        elif "x" in bbox and "y" in bbox and "z" in bbox:
            # Format: {'x': (min, max), 'y': (min, max), 'z': (min, max)}
            return (
                (bbox["x"][0], bbox["y"][0], bbox["z"][0]),
                (bbox["x"][1], bbox["y"][1], bbox["z"][1]),
            )

    return None


def build_aggregation_pipeline(
    match_stage: Optional[Dict[str, Any]] = None,
    group_stage: Optional[Dict[str, Any]] = None,
    project_stage: Optional[Dict[str, Any]] = None,
    sort_stage: Optional[List[Tuple[str, int]]] = None,
    limit_stage: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Build a MongoDB aggregation pipeline.

    Args:
        match_stage: $match stage query
        group_stage: $group stage specification
        project_stage: $project stage specification
        sort_stage: $sort stage (list of (field, direction) tuples)
        limit_stage: $limit stage value

    Returns:
        Aggregation pipeline list
    """
    pipeline = []

    if match_stage:
        pipeline.append({"$match": match_stage})

    if group_stage:
        pipeline.append({"$group": group_stage})

    if project_stage:
        pipeline.append({"$project": project_stage})

    if sort_stage:
        pipeline.append({"$sort": {field: direction for field, direction in sort_stage}})

    if limit_stage:
        pipeline.append({"$limit": limit_stage})

    return pipeline


def aggregate_by_layer(
    collection,
    model_id: str,
    field_to_aggregate: str,
    aggregation_type: str = "avg",  # "avg", "min", "max", "sum"
) -> Dict[int, float]:
    """
    Aggregate a field by layer for a given model.

    Args:
        collection: MongoDB collection
        model_id: Model ID
        field_to_aggregate: Field name to aggregate
        aggregation_type: Type of aggregation (avg, min, max, sum)

    Returns:
        Dictionary mapping layer_index to aggregated value
    """
    pipeline = [
        {"$match": {"model_id": model_id}},
        {
            "$group": {
                "_id": "$layer_index",
                "value": {
                    (
                        "$avg"
                        if aggregation_type == "avg"
                        else ("$min" if aggregation_type == "min" else "$max" if aggregation_type == "max" else "$sum")
                    ): f"${field_to_aggregate}"
                },
            }
        },
        {"$sort": {"_id": 1}},
    ]

    results = {}
    for doc in collection.aggregate(pipeline):
        results[doc["_id"]] = doc["value"]

    return results
