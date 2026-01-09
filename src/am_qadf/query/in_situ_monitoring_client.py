"""
In-Situ Monitoring Client

Query client for real-time in-situ monitoring data.
Supports streaming data integration and live visualization updates.
"""

from typing import Optional, List, Tuple, Dict, Any, Callable
import numpy as np
from datetime import datetime, timedelta

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


class InSituMonitoringClient(BaseQueryClient):
    """
    Query client for in-situ monitoring data.

    Supports:
    - Real-time monitoring data queries
    - Streaming data integration
    - Live visualization updates
    - Temporal alignment with build process
    """

    def __init__(
        self,
        data_source: Optional[str] = None,
        streaming_enabled: bool = False,
        update_callback: Optional[Callable] = None,
        mongo_client=None,
        use_mongodb: bool = False,
    ):
        """
        Initialize in-situ monitoring client.

        Args:
            data_source: Database connection string or file path
            streaming_enabled: Whether to enable streaming updates
            update_callback: Callback function for streaming updates
            mongo_client: MongoDBClient instance (for MongoDB data warehouse)
            use_mongodb: If True, use MongoDB backend; if False, use simulated data
        """
        super().__init__(data_source=data_source or ("mongodb_warehouse" if use_mongodb else "simulated"))
        self._available_signals = [
            SignalType.THERMAL,
            SignalType.TEMPERATURE,
            SignalType.POWER,
            SignalType.VELOCITY,
        ]

        self.streaming_enabled = streaming_enabled
        self.update_callback = update_callback

        # MongoDB support
        self.mongo_client = mongo_client
        self.use_mongodb = use_mongodb
        self.collection_name = "ispm_monitoring_data"

        # Simulated data storage (for testing)
        self._monitoring_data: List[Dict[str, Any]] = []
        self._last_update_time: Optional[datetime] = None

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

    def add_monitoring_point(
        self,
        x: float,
        y: float,
        z: float,
        timestamp: datetime,
        temperature: Optional[float] = None,
        power: Optional[float] = None,
        velocity: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Add a monitoring data point (for simulation/testing).

        Args:
            x, y, z: Spatial coordinates (mm)
            timestamp: Timestamp of measurement
            temperature: Temperature reading (°C)
            power: Laser power (W)
            velocity: Scan velocity (mm/s)
            metadata: Additional metadata
        """
        point_data = {
            "point": (x, y, z),
            "timestamp": timestamp,
            "signals": {},
            "metadata": metadata or {},
        }

        if temperature is not None:
            point_data["signals"]["temperature"] = temperature
        if power is not None:
            point_data["signals"]["power"] = power
        if velocity is not None:
            point_data["signals"]["velocity"] = velocity

        self._monitoring_data.append(point_data)
        self._last_update_time = timestamp

        # Trigger callback if streaming is enabled
        if self.streaming_enabled and self.update_callback:
            self.update_callback(point_data)

    def query(
        self,
        spatial: Optional[SpatialQuery] = None,
        temporal: Optional[TemporalQuery] = None,
        signal_types: Optional[List[SignalType]] = None,
    ) -> QueryResult:
        """
        Query in-situ monitoring data.

        Supports both MongoDB warehouse and simulated data modes.

        Args:
            spatial: Spatial query parameters (requires component_id = model_id for MongoDB)
            temporal: Temporal query parameters (time range, layer range)
            signal_types: Signal types to retrieve

        Returns:
            QueryResult with monitoring points and signals
        """
        # Use MongoDB backend if enabled
        if self.use_mongodb and self.mongo_client:
            return self._query_mongodb(spatial, temporal, signal_types)

        # Fall back to simulated data mode
        return self._query_simulated(spatial, temporal, signal_types)

    def _query_mongodb(
        self,
        spatial: Optional[SpatialQuery] = None,
        temporal: Optional[TemporalQuery] = None,
        signal_types: Optional[List[SignalType]] = None,
    ) -> QueryResult:
        """Query ISPM data from MongoDB."""
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

        # Add time filter
        if temporal and temporal.time_start:
            if "timestamp" not in query:
                query["timestamp"] = {}
            query["timestamp"]["$gte"] = (
                temporal.time_start.isoformat() if isinstance(temporal.time_start, datetime) else temporal.time_start
            )

        if temporal and temporal.time_end:
            if "timestamp" not in query:
                query["timestamp"] = {}
            query["timestamp"]["$lte"] = (
                temporal.time_end.isoformat() if isinstance(temporal.time_end, datetime) else temporal.time_end
            )

        # Execute query
        cursor = collection.find(query)

        # Collect data
        all_points = []
        temperature_values = []
        melt_pool_size_values = []
        peak_temp_values = []
        cooling_rate_values = []
        gradient_values = []

        for doc in cursor:
            coords = doc.get("spatial_coordinates", [])
            if len(coords) == 3:
                all_points.append(tuple(coords))
                temperature_values.append(doc.get("melt_pool_temperature", 0.0))
                melt_pool_size_values.append(doc.get("melt_pool_size", {}))
                peak_temp_values.append(doc.get("peak_temperature", 0.0))
                cooling_rate_values.append(doc.get("cooling_rate", 0.0))
                gradient_values.append(doc.get("temperature_gradient", 0.0))

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
                temperature_values = [v for v, m in zip(temperature_values, mask) if m]
                melt_pool_size_values = [v for v, m in zip(melt_pool_size_values, mask) if m]
                peak_temp_values = [v for v, m in zip(peak_temp_values, mask) if m]
                cooling_rate_values = [v for v, m in zip(cooling_rate_values, mask) if m]
                gradient_values = [v for v, m in zip(gradient_values, mask) if m]
            else:
                # Fallback: manual filtering
                bbox_min = np.array(spatial.bbox_min)
                bbox_max = np.array(spatial.bbox_max)
                filtered_points = []
                filtered_temp = []
                filtered_melt = []
                filtered_peak = []
                filtered_cooling = []
                filtered_gradient = []
                for i, point in enumerate(all_points):
                    point_arr = np.array(point)
                    if np.all((point_arr >= bbox_min) & (point_arr <= bbox_max)):
                        filtered_points.append(point)
                        filtered_temp.append(temperature_values[i])
                        filtered_melt.append(melt_pool_size_values[i])
                        filtered_peak.append(peak_temp_values[i])
                        filtered_cooling.append(cooling_rate_values[i])
                        filtered_gradient.append(gradient_values[i])
                all_points = filtered_points
                temperature_values = filtered_temp
                melt_pool_size_values = filtered_melt
                peak_temp_values = filtered_peak
                cooling_rate_values = filtered_cooling
                gradient_values = filtered_gradient

        # Build signals dictionary
        signals = {}
        if signal_types is None or SignalType.TEMPERATURE in signal_types or SignalType.THERMAL in signal_types:
            signals["temperature"] = temperature_values
            signals["peak_temperature"] = peak_temp_values
        if signal_types is None or SignalType.THERMAL in signal_types:
            signals["cooling_rate"] = cooling_rate_values
            signals["temperature_gradient"] = gradient_values

        return QueryResult(
            points=all_points,
            signals=signals,
            metadata={
                "model_id": model_id,
                "source": "mongodb",
                "n_points": len(all_points),
                "has_melt_pool_data": len(melt_pool_size_values) > 0,
            },
            component_id=model_id,
        )

    def _query_simulated(
        self,
        spatial: Optional[SpatialQuery] = None,
        temporal: Optional[TemporalQuery] = None,
        signal_types: Optional[List[SignalType]] = None,
    ) -> QueryResult:
        """Query ISPM data from simulated data (original implementation)."""
        # Determine which signals to retrieve
        if signal_types is None:
            signal_types = self._available_signals

        # Filter data based on spatial and temporal constraints
        filtered_data = self._monitoring_data.copy()

        # Apply temporal filtering
        if temporal and temporal.time_start:
            filtered_data = [d for d in filtered_data if d["timestamp"] >= temporal.time_start]

        if temporal and temporal.time_end:
            filtered_data = [d for d in filtered_data if d["timestamp"] <= temporal.time_end]

        # Apply spatial filtering
        if spatial and spatial.bbox_min and spatial.bbox_max:
            bbox_min = np.array(spatial.bbox_min)
            bbox_max = np.array(spatial.bbox_max)

            filtered_data = [
                d
                for d in filtered_data
                if np.all(np.array(d["point"]) >= bbox_min) and np.all(np.array(d["point"]) <= bbox_max)
            ]

        # Collect points and signals
        points = []
        signals = {signal.value: [] for signal in signal_types}

        for data_point in filtered_data:
            points.append(list(data_point["point"]))

            for signal_type in signal_types:
                signal_name = signal_type.value
                if signal_name in data_point["signals"]:
                    signals[signal_name].append(data_point["signals"][signal_name])
                else:
                    signals[signal_name].append(None)

        return QueryResult(
            points=points,
            signals=signals,
            metadata={
                "source": "in_situ_monitoring",
                "streaming_enabled": self.streaming_enabled,
                "num_points": len(points),
                "last_update": (self._last_update_time.isoformat() if self._last_update_time else None),
            },
        )

    def get_bounding_box(
        self, component_id: Optional[str] = None
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """
        Get bounding box of monitoring data.

        Args:
            component_id: Model ID (for MongoDB mode) or None (for simulated mode)

        Returns:
            Tuple of (bbox_min, bbox_max)
        """
        if self.use_mongodb and self.mongo_client and component_id:
            # Get from MongoDB
            collection = self._get_collection()
            cursor = collection.find({"model_id": component_id})
            points = []
            for doc in cursor:
                coords = doc.get("spatial_coordinates", [])
                if len(coords) == 3:
                    points.append(coords)

            if points:
                points_array = np.array(points)
                bbox_min = tuple(np.min(points_array, axis=0))
                bbox_max = tuple(np.max(points_array, axis=0))
                return (bbox_min, bbox_max)

        # Simulated mode
        if not self._monitoring_data:
            raise ValueError("No monitoring data available")

        points = np.array([d["point"] for d in self._monitoring_data])
        bbox_min = tuple(np.min(points, axis=0))
        bbox_max = tuple(np.max(points, axis=0))

        return (bbox_min, bbox_max)

    def get_available_signals(self) -> List[SignalType]:
        """Get available signal types."""
        return self._available_signals

    def enable_streaming(self, callback: Optional[Callable] = None):
        """
        Enable streaming data updates.

        Args:
            callback: Callback function called when new data arrives
        """
        self.streaming_enabled = True
        if callback:
            self.update_callback = callback

    def disable_streaming(self):
        """Disable streaming data updates."""
        self.streaming_enabled = False
        self.update_callback = None

    def get_latest_data(self, num_points: int = 100, signal_types: Optional[List[SignalType]] = None) -> QueryResult:
        """
        Get the most recent monitoring data points.

        Args:
            num_points: Number of recent points to retrieve
            signal_types: Signal types to retrieve

        Returns:
            QueryResult with latest data points
        """
        if not self._monitoring_data:
            return QueryResult(
                points=[],
                signals={},
                metadata={"message": "No monitoring data available"},
            )

        # Sort by timestamp (most recent first)
        sorted_data = sorted(self._monitoring_data, key=lambda x: x["timestamp"], reverse=True)

        # Get most recent points
        recent_data = sorted_data[:num_points]

        # Determine which signals to retrieve
        if signal_types is None:
            signal_types = self._available_signals

        # Collect points and signals
        points = []
        signals = {signal.value: [] for signal in signal_types}

        for data_point in recent_data:
            points.append(list(data_point["point"]))

            for signal_type in signal_types:
                signal_name = signal_type.value
                if signal_name in data_point["signals"]:
                    signals[signal_name].append(data_point["signals"][signal_name])
                else:
                    signals[signal_name].append(None)

        return QueryResult(
            points=points,
            signals=signals,
            metadata={
                "source": "in_situ_monitoring",
                "num_points": len(points),
                "latest_timestamp": (recent_data[0]["timestamp"].isoformat() if recent_data else None),
            },
        )

    # MongoDB-specific helper methods
    def get_temperature_data(self, model_id: str, layer_range: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Get temperature data for a model (MongoDB mode only).

        Args:
            model_id: Model UUID
            layer_range: Optional (start_layer, end_layer) tuple

        Returns:
            Numpy array of temperature values
        """
        if not self.use_mongodb:
            raise RuntimeError("get_temperature_data() requires MongoDB mode. Call set_mongo_client() first.")

        collection = self._get_collection()
        query = {"model_id": model_id}
        if layer_range:
            query["layer_index"] = {"$gte": layer_range[0], "$lte": layer_range[1]}

        cursor = collection.find(query)
        values = [doc.get("melt_pool_temperature", 0.0) for doc in cursor]
        return np.array(values, dtype=np.float64)

    def get_melt_pool_data(self, model_id: str, layer_range: Optional[Tuple[int, int]] = None) -> List[Dict[str, Any]]:
        """
        Get melt pool data for a model (MongoDB mode only).

        Args:
            model_id: Model UUID
            layer_range: Optional (start_layer, end_layer) tuple

        Returns:
            List of melt pool data dictionaries
        """
        if not self.use_mongodb:
            raise RuntimeError("get_melt_pool_data() requires MongoDB mode. Call set_mongo_client() first.")

        collection = self._get_collection()
        query = {"model_id": model_id}
        if layer_range:
            query["layer_index"] = {"$gte": layer_range[0], "$lte": layer_range[1]}

        cursor = collection.find(query)
        return [
            {
                "spatial_coordinates": doc.get("spatial_coordinates", []),
                "melt_pool_size": doc.get("melt_pool_size", {}),
                "melt_pool_temperature": doc.get("melt_pool_temperature", 0.0),
                "layer_index": doc.get("layer_index", 0),
                "timestamp": doc.get("timestamp"),
            }
            for doc in cursor
        ]

    def get_thermal_gradients(self, model_id: str, layer_range: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Get thermal gradient data for a model (MongoDB mode only).

        Args:
            model_id: Model UUID
            layer_range: Optional (start_layer, end_layer) tuple

        Returns:
            Numpy array of thermal gradient values
        """
        if not self.use_mongodb:
            raise RuntimeError("get_thermal_gradients() requires MongoDB mode. Call set_mongo_client() first.")

        collection = self._get_collection()
        query = {"model_id": model_id}
        if layer_range:
            query["layer_index"] = {"$gte": layer_range[0], "$lte": layer_range[1]}

        cursor = collection.find(query)
        values = [doc.get("temperature_gradient", 0.0) for doc in cursor]
        return np.array(values, dtype=np.float64)

    def get_process_events(self, model_id: str) -> List[Dict[str, Any]]:
        """
        Get process events for a model (MongoDB mode only).

        Args:
            model_id: Model UUID

        Returns:
            List of process event dictionaries
        """
        if not self.use_mongodb:
            raise RuntimeError("get_process_events() requires MongoDB mode. Call set_mongo_client() first.")

        collection = self._get_collection()
        cursor = collection.find({"model_id": model_id, "process_event": {"$exists": True, "$ne": None}})

        return [
            {
                "spatial_coordinates": doc.get("spatial_coordinates", []),
                "process_event": doc.get("process_event"),
                "layer_index": doc.get("layer_index", 0),
                "timestamp": doc.get("timestamp"),
            }
            for doc in cursor
        ]

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

        collection = self._get_collection()
        doc = collection.find_one({"model_id": model_id})
        if doc is None:
            return None

        return doc.get("coordinate_system", {})
