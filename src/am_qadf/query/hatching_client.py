"""
Hatching Layer Query Client

Query client for hatching layers and laser scan paths from MongoDB data warehouse.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Handle both relative import (when used as package) and direct import (when loaded directly)
try:
    from .base_query_client import (
        BaseQueryClient,
        QueryResult,
        SpatialQuery,
        TemporalQuery,
        SignalType,
    )
    from .query_utils import (
        build_model_query,
        build_temporal_query,
        combine_queries,
        filter_points_in_bbox,
        extract_coordinate_system,
    )
except ImportError:
    # Fallback for direct module loading
    try:
        from base_query_client import (
            BaseQueryClient,
            QueryResult,
            SpatialQuery,
            TemporalQuery,
            SignalType,
        )
        from query_utils import (
            build_model_query,
            build_temporal_query,
            combine_queries,
            filter_points_in_bbox,
            extract_coordinate_system,
        )
    except ImportError:
        import sys
        from pathlib import Path

        current_file = Path(__file__).resolve()
        base_path = current_file.parent / "base_query_client.py"
        utils_path = current_file.parent / "query_utils.py"

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

        if utils_path.exists():
            import importlib.util

            spec = importlib.util.spec_from_file_location("query_utils", utils_path)
            utils_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(utils_module)
            sys.modules["query_utils"] = utils_module
            build_model_query = utils_module.build_model_query
            build_temporal_query = utils_module.build_temporal_query
            combine_queries = utils_module.combine_queries
            filter_points_in_bbox = utils_module.filter_points_in_bbox
            extract_coordinate_system = utils_module.extract_coordinate_system
        else:
            raise ImportError("Could not import required modules")


class HatchingClient(BaseQueryClient):
    """
    Query client for hatching layers from MongoDB data warehouse.

    Provides methods to query hatching layers, retrieve laser scan paths,
    and access coordinate system information.
    """

    def __init__(self, mongo_client=None, data_source: Optional[str] = None):
        """
        Initialize hatching layer query client.

        Args:
            mongo_client: MongoDBClient instance (if None, will create one)
            data_source: Optional identifier for the data source
        """
        super().__init__(data_source)
        self.mongo_client = mongo_client
        self.collection_name = "hatching_layers"
        self._available_signals = [
            SignalType.POWER,
            SignalType.VELOCITY,
            SignalType.ENERGY,
        ]

    def _get_collection(self):
        """Get MongoDB collection."""
        if self.mongo_client is None:
            raise RuntimeError("MongoDB client not initialized. Call set_mongo_client() first.")
        if not self.mongo_client.is_connected():
            raise ConnectionError("MongoDB client not connected")
        return self.mongo_client.get_collection(self.collection_name)

    def set_mongo_client(self, mongo_client):
        """Set MongoDB client instance."""
        self.mongo_client = mongo_client

    def get_layers(self, model_id: str, layer_range: Optional[Tuple[int, int]] = None) -> List[Dict[str, Any]]:
        """
        Get hatching layers for a model.

        Args:
            model_id: Model UUID
            layer_range: Optional (start_layer, end_layer) tuple

        Returns:
            List of layer documents
        """
        collection = self._get_collection()
        query = {"model_id": model_id}

        if layer_range:
            query["layer_index"] = {"$gte": layer_range[0], "$lte": layer_range[1]}

        cursor = collection.find(query).sort("layer_index", 1)
        return list(cursor)

    def get_layer(self, model_id: str, layer_index: int) -> Optional[Dict[str, Any]]:
        """
        Get a specific hatching layer.

        Args:
            model_id: Model UUID
            layer_index: Layer index

        Returns:
            Layer document or None if not found
        """
        collection = self._get_collection()
        doc = collection.find_one({"model_id": model_id, "layer_index": layer_index})
        return doc

    def get_hatch_paths(self, model_id: str, layer_index: int) -> List[Dict[str, Any]]:
        """
        Get hatch paths for a specific layer.

        Args:
            model_id: Model UUID
            layer_index: Layer index

        Returns:
            List of hatch path dictionaries
        """
        layer = self.get_layer(model_id, layer_index)
        if layer is None:
            return []

        return layer.get("hatches", [])

    def get_all_points(self, model_id: str, layer_range: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Get all laser path points for a model.

        Args:
            model_id: Model UUID
            layer_range: Optional (start_layer, end_layer) tuple

        Returns:
            Numpy array of points with shape (n, 3)
        """
        layers = self.get_layers(model_id, layer_range)
        all_points = []

        for layer in layers:
            hatches = layer.get("hatches", [])
            for hatch in hatches:
                points = hatch.get("points", [])
                if points:
                    all_points.extend(points)

        if not all_points:
            return np.array([], dtype=np.float64).reshape(0, 3)

        return np.array(all_points, dtype=np.float64)

    def query_spatial(
        self,
        model_id: str,
        bbox: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
    ) -> List[Dict[str, Any]]:
        """
        Query hatch paths within a spatial bounding box.

        Args:
            model_id: Model UUID
            bbox: Bounding box ((x_min, y_min, z_min), (x_max, y_max, z_max))

        Returns:
            List of hatch paths within the bounding box
        """
        layers = self.get_layers(model_id)
        bbox_min, bbox_max = bbox
        matching_hatches = []

        for layer in layers:
            hatches = layer.get("hatches", [])
            for hatch in hatches:
                points = hatch.get("points", [])
                if points:
                    points_array = np.array(points)
                    mask = filter_points_in_bbox(points_array, bbox_min, bbox_max)
                    if np.any(mask):
                        # Include hatch if any point is in bbox
                        matching_hatches.append(hatch)

        return matching_hatches

    def get_coordinate_system(self, model_id: str, layer_index: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Get coordinate system information for a layer.

        Args:
            model_id: Model UUID
            layer_index: Optional layer index (if None, gets from first layer)

        Returns:
            Coordinate system dictionary or None
        """
        if layer_index is None:
            # Get from first layer
            layers = self.get_layers(model_id, layer_range=(0, 0))
            if not layers:
                return None
            layer = layers[0]
        else:
            layer = self.get_layer(model_id, layer_index)
            if layer is None:
                return None

        return extract_coordinate_system(layer)

    def query(
        self,
        spatial: Optional[SpatialQuery] = None,
        temporal: Optional[TemporalQuery] = None,
        signal_types: Optional[List[SignalType]] = None,
    ) -> QueryResult:
        """
        Query hatching data (implements BaseQueryClient interface).

        Args:
            spatial: Spatial query parameters (requires component_id = model_id)
            temporal: Temporal query parameters (layer range)
            signal_types: Signal types to retrieve (POWER, VELOCITY, ENERGY)

        Returns:
            QueryResult with points and signals
        """
        if spatial is None or spatial.component_id is None:
            raise ValueError("Spatial query must include component_id (model_id)")

        model_id = spatial.component_id
        layer_range = None

        if temporal and temporal.layer_start is not None and temporal.layer_end is not None:
            layer_range = (temporal.layer_start, temporal.layer_end)
        elif spatial and spatial.layer_range:
            layer_range = spatial.layer_range

        # Get layers
        layers = self.get_layers(model_id, layer_range)

        # Extract points and signals
        all_points = []
        power_values = []
        velocity_values = []
        energy_values = []
        metadata_list = []

        for layer in layers:
            hatches = layer.get("hatches", [])
            for hatch in hatches:
                points = hatch.get("points", [])
                if points:
                    all_points.extend(points)

                    # Extract signals
                    laser_power = hatch.get("laser_power", 0.0)
                    scan_speed = hatch.get("scan_speed", 0.0)
                    energy_density = hatch.get("energy_density", 0.0)

                    # Repeat values for each point in the hatch
                    n_points = len(points)
                    power_values.extend([laser_power] * n_points)
                    velocity_values.extend([scan_speed] * n_points)
                    energy_values.extend([energy_density] * n_points)

                    # Metadata
                    metadata_list.append(
                        {
                            "layer_index": layer.get("layer_index"),
                            "hatch_type": hatch.get("hatch_type"),
                            "laser_beam_width": hatch.get("laser_beam_width"),
                            "hatch_spacing": hatch.get("hatch_spacing"),
                            "overlap_percentage": hatch.get("overlap_percentage"),
                        }
                    )

        # Apply spatial filtering if bbox provided
        if spatial and spatial.bbox_min and spatial.bbox_max:
            points_array = np.array(all_points)
            mask = filter_points_in_bbox(points_array, spatial.bbox_min, spatial.bbox_max)
            all_points = [tuple(p) for p, m in zip(points_array, mask) if m]
            power_values = [v for v, m in zip(power_values, mask) if m]
            velocity_values = [v for v, m in zip(velocity_values, mask) if m]
            energy_values = [v for v, m in zip(energy_values, mask) if m]

        # Build signals dictionary
        signals = {}
        if signal_types is None or SignalType.POWER in signal_types:
            signals["power"] = power_values
        if signal_types is None or SignalType.VELOCITY in signal_types:
            signals["velocity"] = velocity_values
        if signal_types is None or SignalType.ENERGY in signal_types:
            signals["energy"] = energy_values

        return QueryResult(
            points=[tuple(p) for p in all_points],
            signals=signals,
            metadata={
                "model_id": model_id,
                "n_layers": len(layers),
                "n_hatches": len(metadata_list),
                "hatch_metadata": metadata_list,
            },
            component_id=model_id,
        )

    def get_available_signals(self) -> List[SignalType]:
        """Get available signal types."""
        return self._available_signals

    def get_bounding_box(
        self, component_id: Optional[str] = None
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """
        Get bounding box (implements BaseQueryClient interface).

        Args:
            component_id: Model ID

        Returns:
            Tuple of (bbox_min, bbox_max)
        """
        if component_id is None:
            return ((0.0, 0.0, 0.0), (100.0, 100.0, 100.0))

        coord_system = self.get_coordinate_system(component_id)
        if coord_system and "bounding_box" in coord_system:
            bbox = coord_system["bounding_box"]
            if "min" in bbox and "max" in bbox:
                return (tuple(bbox["min"]), tuple(bbox["max"]))

        return ((0.0, 0.0, 0.0), (100.0, 100.0, 100.0))
