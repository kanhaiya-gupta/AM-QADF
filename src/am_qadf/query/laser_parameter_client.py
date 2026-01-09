"""
Laser Parameter Query Client

Query client for laser parameters (power, velocity, energy) from hatching data.
Wraps existing hatching generation code to provide standardized query interface.
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
    except ImportError:
        # If still fails, try to get it from sys.modules (injected by loader)
        import sys

        if "base_query_client" in sys.modules:
            base_module = sys.modules["base_query_client"]
            BaseQueryClient = base_module.BaseQueryClient
            QueryResult = base_module.QueryResult
            SpatialQuery = base_module.SpatialQuery
            TemporalQuery = base_module.TemporalQuery
            SignalType = base_module.SignalType
        else:
            raise ImportError("Could not import BaseQueryClient. Make sure base_query_client module is loaded first.")


class LaserParameterClient(BaseQueryClient):
    """
    Query client for laser parameters from hatching/path data.

    Supports both:
    1. Generated data (pyslm layers) - original functionality
    2. MongoDB data warehouse - new functionality

    This client wraps hatching generation and provides a query interface
    for accessing laser power, velocity, and energy data.
    """

    def __init__(
        self,
        stl_part=None,
        generated_layers: Optional[List] = None,
        generated_build_styles: Optional[Dict] = None,
        mongo_client=None,
        use_mongodb: bool = False,
    ):
        """
        Initialize laser parameter client.

        Args:
            stl_part: pyslm.Part object (optional, for generated data)
            generated_layers: List of generated pyslm.geometry.Layer objects (for generated data)
            generated_build_styles: Dictionary of build style ID to BuildStyle objects (for generated data)
            mongo_client: MongoDBClient instance (for MongoDB data warehouse)
            use_mongodb: If True, use MongoDB backend; if False, use generated data
        """
        super().__init__(data_source="hatching_data" if not use_mongodb else "mongodb_warehouse")
        self.stl_part = stl_part
        self.generated_layers = generated_layers or []
        self.generated_build_styles = generated_build_styles or {}
        self.mongo_client = mongo_client
        self.use_mongodb = use_mongodb
        self.collection_name = "laser_parameters"
        self._available_signals = [
            SignalType.POWER,
            SignalType.VELOCITY,
            SignalType.ENERGY,
        ]

    def set_data(
        self,
        stl_part=None,
        generated_layers: Optional[List] = None,
        generated_build_styles: Optional[Dict] = None,
    ):
        """
        Set or update the data source (for generated data mode).

        Args:
            stl_part: pyslm.Part object
            generated_layers: List of generated layers
            generated_build_styles: Dictionary of build styles
        """
        if stl_part is not None:
            self.stl_part = stl_part
        if generated_layers is not None:
            self.generated_layers = generated_layers
        if generated_build_styles is not None:
            self.generated_build_styles = generated_build_styles

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

    def query(
        self,
        spatial: Optional[SpatialQuery] = None,
        temporal: Optional[TemporalQuery] = None,
        signal_types: Optional[List[SignalType]] = None,
    ) -> QueryResult:
        """
        Query laser parameters from hatching data.

        Supports both MongoDB warehouse and generated data modes.

        Args:
            spatial: Spatial query parameters (requires component_id = model_id for MongoDB mode)
            temporal: Temporal query parameters (layer range)
            signal_types: Signal types to retrieve (None = all available)

        Returns:
            QueryResult with points and signal values
        """
        # Use MongoDB backend if enabled
        if self.use_mongodb and self.mongo_client:
            return self._query_mongodb(spatial, temporal, signal_types)

        # Fall back to generated data mode
        return self._query_generated(spatial, temporal, signal_types)

    def _query_mongodb(
        self,
        spatial: Optional[SpatialQuery] = None,
        temporal: Optional[TemporalQuery] = None,
        signal_types: Optional[List[SignalType]] = None,
    ) -> QueryResult:
        """Query laser parameters from MongoDB."""
        if spatial is None or spatial.component_id is None:
            raise ValueError("Spatial query must include component_id (model_id) for MongoDB queries")

        model_id = spatial.component_id
        collection = self._get_collection()

        # Build query
        query = {"model_id": model_id}

        # Add layer filter
        if temporal and temporal.layer_start is not None:
            query["layer_index"] = {"$gte": temporal.layer_start}
        if temporal and temporal.layer_end is not None:
            if "layer_index" in query:
                query["layer_index"]["$lte"] = temporal.layer_end
            else:
                query["layer_index"] = {"$lte": temporal.layer_end}
        elif spatial and spatial.layer_range:
            query["layer_index"] = {
                "$gte": spatial.layer_range[0],
                "$lte": spatial.layer_range[1],
            }

        # Execute query
        cursor = collection.find(query)

        # Collect data
        all_points = []
        power_values = []
        velocity_values = []
        energy_values = []

        for doc in cursor:
            coords = doc.get("spatial_coordinates", [])
            if len(coords) == 3:
                all_points.append(tuple(coords))
                power_values.append(doc.get("laser_power", 0.0))
                velocity_values.append(doc.get("scan_speed", 0.0))
                energy_values.append(doc.get("energy_density", 0.0))

        # Apply spatial filtering if bbox provided
        if spatial and spatial.bbox_min and spatial.bbox_max:
            # Try to import filter_points_in_bbox
            filter_points_in_bbox = None
            try:
                from .query_utils import filter_points_in_bbox
            except ImportError:
                try:
                    import sys

                    query_utils_module = sys.modules.get("query_utils")
                    if query_utils_module:
                        filter_points_in_bbox = getattr(query_utils_module, "filter_points_in_bbox", None)
                except Exception:
                    pass

            if filter_points_in_bbox:
                points_array = np.array(all_points)
                mask = filter_points_in_bbox(points_array, spatial.bbox_min, spatial.bbox_max)
                all_points = [tuple(p) for p, m in zip(points_array, mask) if m]
                power_values = [v for v, m in zip(power_values, mask) if m]
                velocity_values = [v for v, m in zip(velocity_values, mask) if m]
                energy_values = [v for v, m in zip(energy_values, mask) if m]
            else:
                # Fallback: manual filtering
                bbox_min = np.array(spatial.bbox_min)
                bbox_max = np.array(spatial.bbox_max)
                filtered_points = []
                filtered_power = []
                filtered_velocity = []
                filtered_energy = []
                for i, point in enumerate(all_points):
                    point_arr = np.array(point)
                    if np.all((point_arr >= bbox_min) & (point_arr <= bbox_max)):
                        filtered_points.append(point)
                        filtered_power.append(power_values[i])
                        filtered_velocity.append(velocity_values[i])
                        filtered_energy.append(energy_values[i])
                all_points = filtered_points
                power_values = filtered_power
                velocity_values = filtered_velocity
                energy_values = filtered_energy

        # Build signals dictionary
        signals = {}
        if signal_types is None or SignalType.POWER in signal_types:
            signals["power"] = power_values
        if signal_types is None or SignalType.VELOCITY in signal_types:
            signals["velocity"] = velocity_values
        if signal_types is None or SignalType.ENERGY in signal_types:
            signals["energy"] = energy_values

        return QueryResult(
            points=all_points,
            signals=signals,
            metadata={
                "model_id": model_id,
                "source": "mongodb",
                "n_points": len(all_points),
            },
            component_id=model_id,
        )

    def _query_generated(
        self,
        spatial: Optional[SpatialQuery] = None,
        temporal: Optional[TemporalQuery] = None,
        signal_types: Optional[List[SignalType]] = None,
    ) -> QueryResult:
        """Query laser parameters from generated data (original implementation)."""
        self.validate_query(spatial, temporal)

        if not self.generated_layers:
            return QueryResult(
                points=[],
                signals={},
                metadata={"message": "No hatching data available"},
            )

        # Determine which layers to query
        layer_start = 0
        layer_end = len(self.generated_layers) - 1

        if temporal and temporal.layer_start is not None:
            layer_start = max(0, temporal.layer_start)
        if temporal and temporal.layer_end is not None:
            layer_end = min(len(self.generated_layers) - 1, temporal.layer_end)
        if spatial and spatial.layer_range:
            layer_start = max(layer_start, spatial.layer_range[0])
            layer_end = min(layer_end, spatial.layer_range[1])

        # Determine which signals to retrieve
        if signal_types is None:
            signal_types = self._available_signals

        # Collect points and signals
        all_points = []
        all_signals = {signal.value: [] for signal in signal_types}

        for layer_idx in range(layer_start, layer_end + 1):
            layer = self.generated_layers[layer_idx]
            z_height = float(layer.z) / 1000.0  # Convert to mm

            # Get build style for this layer
            build_style = None
            if layer.geometry:
                first_geom = layer.geometry[0]
                build_style = self.generated_build_styles.get(first_geom.bid)

            # Extract power, velocity, energy from build style
            power = build_style.laserPower if build_style else 200.0
            velocity = build_style.laserSpeed if build_style else 500.0
            energy = power / velocity if velocity > 0 else 0.0

            # Process geometry in this layer
            for geom in layer.geometry:
                if hasattr(geom, "coords"):
                    coords = geom.coords
                    if len(coords) >= 2:
                        # Extract points along the path
                        for i in range(len(coords) - 1):
                            x1, y1 = coords[i]
                            x2, y2 = coords[i + 1]

                            # Check spatial filter
                            if spatial and spatial.bbox_min and spatial.bbox_max:
                                # Simple check: if both endpoints are outside bbox, skip
                                if (
                                    (x1 < spatial.bbox_min[0] and x2 < spatial.bbox_min[0])
                                    or (x1 > spatial.bbox_max[0] and x2 > spatial.bbox_max[0])
                                    or (y1 < spatial.bbox_min[1] and y2 < spatial.bbox_min[1])
                                    or (y1 > spatial.bbox_max[1] and y2 > spatial.bbox_max[1])
                                    or (z_height < spatial.bbox_min[2] or z_height > spatial.bbox_max[2])
                                ):
                                    continue

                            # Add midpoint of segment (or both points)
                            x = (x1 + x2) / 2.0
                            y = (y1 + y2) / 2.0
                            z = z_height

                            all_points.append((x, y, z))

                            # Add signal values
                            if SignalType.POWER in signal_types:
                                all_signals["power"].append(power)
                            if SignalType.VELOCITY in signal_types:
                                all_signals["velocity"].append(velocity)
                            if SignalType.ENERGY in signal_types:
                                all_signals["energy"].append(energy)

        # Build metadata
        metadata = {
            "num_layers": layer_end - layer_start + 1,
            "layer_range": (layer_start, layer_end),
            "total_points": len(all_points),
            "available_signals": [s.value for s in signal_types],
        }

        return QueryResult(points=all_points, signals=all_signals, metadata=metadata)

    def get_available_signals(self) -> List[SignalType]:
        """Get available signal types."""
        return self._available_signals.copy()

    def get_bounding_box(
        self, component_id: Optional[str] = None
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """
        Get bounding box of the hatching data.

        Args:
            component_id: Not used for this client (single component assumed)

        Returns:
            Tuple of (bbox_min, bbox_max)
        """
        if not self.generated_layers:
            return ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

        # Collect all points to find bounding box
        all_x = []
        all_y = []
        all_z = []

        for layer in self.generated_layers:
            z_height = float(layer.z) / 1000.0
            for geom in layer.geometry:
                if hasattr(geom, "coords"):
                    for coord in geom.coords:
                        if len(coord) >= 2:
                            all_x.append(coord[0])
                            all_y.append(coord[1])
                            all_z.append(z_height)

        if not all_x:
            return ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

        bbox_min = (min(all_x), min(all_y), min(all_z))
        bbox_max = (max(all_x), max(all_y), max(all_z))

        return (bbox_min, bbox_max)

    def get_layer_count(self) -> int:
        """Get total number of layers."""
        if self.use_mongodb and self.mongo_client:
            # Query MongoDB for layer count
            collection = self._get_collection()
            # This would need model_id, so return 0 for now
            return 0
        return len(self.generated_layers)

    def get_build_styles(self) -> Dict:
        """Get dictionary of build styles."""
        return self.generated_build_styles.copy()

    # MongoDB-specific helper methods
    def get_points(self, model_id: str, filters: Optional[Dict] = None) -> np.ndarray:
        """
        Get all points for a model (MongoDB mode only).

        Args:
            model_id: Model UUID
            filters: Optional MongoDB query filters

        Returns:
            Numpy array of points with shape (n, 3)
        """
        if not self.use_mongodb:
            raise RuntimeError("get_points() requires MongoDB mode. Call set_mongo_client() first.")

        collection = self._get_collection()
        query = {"model_id": model_id}
        if filters:
            query.update(filters)

        cursor = collection.find(query)
        points = []
        for doc in cursor:
            coords = doc.get("spatial_coordinates", [])
            if len(coords) == 3:
                points.append(coords)

        if not points:
            return np.array([], dtype=np.float64).reshape(0, 3)

        return np.array(points, dtype=np.float64)

    def get_power(self, model_id: str, filters: Optional[Dict] = None) -> np.ndarray:
        """Get laser power values for a model (MongoDB mode only)."""
        if not self.use_mongodb:
            raise RuntimeError("get_power() requires MongoDB mode. Call set_mongo_client() first.")

        collection = self._get_collection()
        query = {"model_id": model_id}
        if filters:
            query.update(filters)

        cursor = collection.find(query)
        values = [doc.get("laser_power", 0.0) for doc in cursor]
        return np.array(values, dtype=np.float64)

    def get_velocity(self, model_id: str, filters: Optional[Dict] = None) -> np.ndarray:
        """Get scan speed values for a model (MongoDB mode only)."""
        if not self.use_mongodb:
            raise RuntimeError("get_velocity() requires MongoDB mode. Call set_mongo_client() first.")

        collection = self._get_collection()
        query = {"model_id": model_id}
        if filters:
            query.update(filters)

        cursor = collection.find(query)
        values = [doc.get("scan_speed", 0.0) for doc in cursor]
        return np.array(values, dtype=np.float64)

    def get_energy_density(self, model_id: str, filters: Optional[Dict] = None) -> np.ndarray:
        """Get energy density values for a model (MongoDB mode only)."""
        if not self.use_mongodb:
            raise RuntimeError("get_energy_density() requires MongoDB mode. Call set_mongo_client() first.")

        collection = self._get_collection()
        query = {"model_id": model_id}
        if filters:
            query.update(filters)

        cursor = collection.find(query)
        values = [doc.get("energy_density", 0.0) for doc in cursor]
        return np.array(values, dtype=np.float64)

    def aggregate_by_layer(self, model_id: str) -> Dict[int, Dict[str, float]]:
        """
        Aggregate laser parameters by layer (MongoDB mode only).

        Args:
            model_id: Model UUID

        Returns:
            Dictionary mapping layer_index to aggregated values
        """
        if not self.use_mongodb:
            raise RuntimeError("aggregate_by_layer() requires MongoDB mode. Call set_mongo_client() first.")

        # Try to import aggregate_by_layer
        agg_by_layer = None
        try:
            from .query_utils import aggregate_by_layer as agg_by_layer
        except ImportError:
            try:
                import sys

                query_utils_module = sys.modules.get("query_utils")
                if query_utils_module:
                    agg_by_layer = getattr(query_utils_module, "aggregate_by_layer", None)
            except Exception:
                pass

        if not agg_by_layer:
            raise ImportError("Could not import aggregate_by_layer from query_utils")

        collection = self._get_collection()

        return {
            "power_avg": agg_by_layer(collection, model_id, "laser_power", "avg"),
            "power_min": agg_by_layer(collection, model_id, "laser_power", "min"),
            "power_max": agg_by_layer(collection, model_id, "laser_power", "max"),
            "velocity_avg": agg_by_layer(collection, model_id, "scan_speed", "avg"),
            "velocity_min": agg_by_layer(collection, model_id, "scan_speed", "min"),
            "velocity_max": agg_by_layer(collection, model_id, "scan_speed", "max"),
            "energy_avg": agg_by_layer(collection, model_id, "energy_density", "avg"),
            "energy_min": agg_by_layer(collection, model_id, "energy_density", "min"),
            "energy_max": agg_by_layer(collection, model_id, "energy_density", "max"),
        }
