"""
In-Situ Monitoring Client (ISPM_Thermal)

Query client for real-time in-situ thermal monitoring data.
Supports streaming data integration and live visualization updates.

Note: ISPM (In-Situ Process Monitoring) is a broad category. This client handles
ISPM_Thermal (thermal monitoring - melt pool temperature, thermal gradients, etc.).
Other ISPM types include:
- ISPM_Optical: Photodiodes, cameras, melt pool imaging
- ISPM_Acoustic: Acoustic emissions, sound sensors
- ISPM_Strain: Strain gauges, deformation sensors
- ISPM_Plume: Vapor plume monitoring

IMPORTANT: All MongoDB queries use the C++ backend (MongoDBQueryClient) to ensure
all ISPM field calculations (melt_pool_area, eccentricity, perimeter, TOT metrics)
happen in C++ for maximum performance. No Python calculations are performed.
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


class ISPMThermalClient(BaseQueryClient):
    """
    Query client for ISPM_Thermal (In-Situ Process Monitoring - Thermal) data.

    All queries use MongoDB backend with C++ client for maximum performance.
    All ISPM_Thermal field calculations (area, eccentricity, perimeter, TOT metrics) happen in C++.

    Note: ISPM (In-Situ Process Monitoring) is a broad category. This client handles
    ISPM_Thermal (thermal monitoring). Other ISPM types (Optical, Acoustic, Strain, Plume)
    will have separate clients.

    Supports:
    - Real-time thermal monitoring data queries (MongoDB + C++ backend)
    - Streaming data integration
    - Live visualization updates
    - Temporal alignment with build process
    
    Note: MongoDB backend is required. Simulated mode removed for performance.
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
            data_source: Database connection string (MongoDB required)
            streaming_enabled: Whether to enable streaming updates
            update_callback: Callback function for streaming updates
            mongo_client: MongoDBClient instance (required for queries)
            use_mongodb: Must be True - MongoDB backend required (all queries use C++ backend)
        """
        if not use_mongodb:
            raise ValueError("MongoDB backend required (use_mongodb=True). Simulated mode removed.")
        
        super().__init__(data_source=data_source or "mongodb_warehouse")
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

        # Note: Simulated mode removed - all queries must use MongoDB (C++ backend)

    def set_mongo_client(self, mongo_client):
        """Set MongoDB client instance (for MongoDB mode)."""
        self.mongo_client = mongo_client
        self.use_mongodb = True

    def query(
        self,
        spatial: Optional[SpatialQuery] = None,
        temporal: Optional[TemporalQuery] = None,
        signal_types: Optional[List[SignalType]] = None,
    ) -> QueryResult:
        """
        Query in-situ monitoring data from MongoDB (C++ backend).

        All queries use C++ backend for maximum performance. All ISPM field calculations
        (melt_pool_area, eccentricity, perimeter, TOT metrics) happen in C++.

        Args:
            spatial: Spatial query parameters (requires component_id = model_id)
            temporal: Temporal query parameters (time range, layer range)
            signal_types: Signal types to retrieve

        Returns:
            QueryResult with monitoring points and signals (all calculated in C++)
        """
        # All queries must use MongoDB (C++ backend)
        if not self.use_mongodb or not self.mongo_client:
            raise RuntimeError(
                "MongoDB backend required. Call set_mongo_client() or initialize with use_mongodb=True. "
                "Simulated mode removed - all queries use C++ backend for performance."
            )
        
        if spatial is None or spatial.component_id is None:
            raise ValueError("Spatial query must include component_id (model_id) for MongoDB queries")

        model_id = spatial.component_id
        
        # Use C++ client for all queries (calculations happen in C++)
        try:
            import am_qadf_native
            from am_qadf_native import MongoDBQueryClient
            import logging
            logger = logging.getLogger(__name__)
            
            # Get MongoDB connection info from mongo_client config (no hardcoded defaults)
            if not self.mongo_client:
                raise RuntimeError("mongo_client is required but not set")
            
            # Get database name from config (primary source)
            if hasattr(self.mongo_client, 'config') and hasattr(self.mongo_client.config, 'database'):
                db_name = self.mongo_client.config.database
            elif hasattr(self.mongo_client, '_database') and self.mongo_client._database:
                db_name = self.mongo_client._database.name
            else:
                raise RuntimeError("Cannot determine database name from mongo_client. Ensure mongo_client has config.database set.")
            
            # Get URI from config
            if hasattr(self.mongo_client, 'config') and hasattr(self.mongo_client.config, 'url'):
                uri = self.mongo_client.config.url
            elif hasattr(self.mongo_client, 'client') and hasattr(self.mongo_client.client, 'address'):
                host, port = self.mongo_client.client.address
                uri = f"mongodb://{host}:{port}"
            else:
                import os
                env_uri = os.getenv('MONGODB_URI') or os.getenv('MONGODB_URL')
                if env_uri:
                    uri = env_uri
                else:
                    uri = "mongodb://localhost:27017"  # Last resort default
            
            # Create C++ query client
            logger.info(f"C++ Query: Creating MongoDBQueryClient for ISPM data (database: {db_name})")
            cpp_client = MongoDBQueryClient(uri, db_name)
            
            # Convert time range to float timestamps (seconds since epoch)
            time_start = 0.0
            time_end = 0.0
            if temporal:
                if temporal.time_start:
                    if isinstance(temporal.time_start, datetime):
                        time_start = temporal.time_start.timestamp()
                    else:
                        time_start = float(temporal.time_start)
                if temporal.time_end:
                    if isinstance(temporal.time_end, datetime):
                        time_end = temporal.time_end.timestamp()
                    else:
                        time_end = float(temporal.time_end)
            
            # Prepare layer range (all filtering in C++)
            layer_start = -1
            layer_end = -1
            if temporal and (temporal.layer_start is not None or temporal.layer_end is not None):
                layer_start = temporal.layer_start if temporal.layer_start is not None else -1
                layer_end = temporal.layer_end if temporal.layer_end is not None else -1
            elif spatial and spatial.layer_range:
                layer_start, layer_end = spatial.layer_range
            
            # Prepare bbox (all filtering in C++)
            bbox_min = [float('-inf'), float('-inf'), float('-inf')]
            bbox_max = [float('inf'), float('inf'), float('inf')]
            if spatial and spatial.bbox_min and spatial.bbox_max:
                bbox_min = list(spatial.bbox_min)
                bbox_max = list(spatial.bbox_max)
            
            # Query using C++ (all calculations and filtering happen in C++)
            logger.info(f"C++ Query: Calling query_ispm_thermal for model_id={model_id[:20]}..., "
                       f"time_range=({time_start}, {time_end}), "
                       f"layer_range=({layer_start}, {layer_end}), "
                       f"bbox_min={bbox_min}, bbox_max={bbox_max}")
            cpp_result = cpp_client.query_ispm_thermal(
                model_id, time_start, time_end, layer_start, layer_end, bbox_min, bbox_max
            )
            logger.info(f"C++ Query: query_ispm_thermal completed, processing result...")
            
            # Convert C++ QueryResult to Python QueryResult
            # Extract points and values (already filtered in C++)
            all_points = [tuple(p) for p in cpp_result.points] if cpp_result.points else []
            all_values = list(cpp_result.values) if cpp_result.values else []
            all_timestamps = list(cpp_result.timestamps) if cpp_result.timestamps else []
            all_layers = list(cpp_result.layers) if cpp_result.layers else []
            
            # Extract ISPM_Thermal-specific data (all calculated in C++)
            ispm_thermal_data_list = []
            if hasattr(cpp_result, 'ispm_thermal_data') and cpp_result.ispm_thermal_data:
                for ispm in cpp_result.ispm_thermal_data:
                    ispm_thermal_data_list.append({
                        "melt_pool_temperature": ispm.melt_pool_temperature,
                        "peak_temperature": ispm.peak_temperature,
                        "melt_pool_width": ispm.melt_pool_width,
                        "melt_pool_length": ispm.melt_pool_length,
                        "melt_pool_depth": ispm.melt_pool_depth,
                        "melt_pool_area": ispm.melt_pool_area,  # Calculated in C++
                        "melt_pool_eccentricity": ispm.melt_pool_eccentricity,  # Calculated in C++
                        "melt_pool_perimeter": ispm.melt_pool_perimeter,  # Calculated in C++
                        "cooling_rate": ispm.cooling_rate,
                        "temperature_gradient": ispm.temperature_gradient,
                        "time_over_threshold_1200K": ispm.time_over_threshold_1200K,  # Calculated in C++
                        "time_over_threshold_1680K": ispm.time_over_threshold_1680K,  # Calculated in C++
                        "time_over_threshold_2400K": ispm.time_over_threshold_2400K,  # Calculated in C++
                        "process_event": ispm.process_event,
                    })
            
            # Build signals dictionary from ISPM_Thermal data
            signals = {}
            if signal_types is None or SignalType.TEMPERATURE in signal_types or SignalType.THERMAL in signal_types:
                signals["temperature"] = [ispm.get("melt_pool_temperature", 0.0) for ispm in ispm_thermal_data_list] if ispm_thermal_data_list else all_values
                signals["peak_temperature"] = [ispm.get("peak_temperature", 0.0) for ispm in ispm_thermal_data_list] if ispm_thermal_data_list else []
            if signal_types is None or SignalType.THERMAL in signal_types:
                signals["cooling_rate"] = [ispm.get("cooling_rate", 0.0) for ispm in ispm_thermal_data_list] if ispm_thermal_data_list else []
                signals["temperature_gradient"] = [ispm.get("temperature_gradient", 0.0) for ispm in ispm_thermal_data_list] if ispm_thermal_data_list else []
            
            # Add ISPM_Thermal-specific signals
            if ispm_thermal_data_list:
                signals["melt_pool_area"] = [ispm.get("melt_pool_area", 0.0) for ispm in ispm_thermal_data_list]
                signals["melt_pool_eccentricity"] = [ispm.get("melt_pool_eccentricity", 0.0) for ispm in ispm_thermal_data_list]
                signals["melt_pool_perimeter"] = [ispm.get("melt_pool_perimeter", 0.0) for ispm in ispm_thermal_data_list]
                signals["time_over_threshold_1200K"] = [ispm.get("time_over_threshold_1200K", 0.0) for ispm in ispm_thermal_data_list]
                signals["time_over_threshold_1680K"] = [ispm.get("time_over_threshold_1680K", 0.0) for ispm in ispm_thermal_data_list]
                signals["time_over_threshold_2400K"] = [ispm.get("time_over_threshold_2400K", 0.0) for ispm in ispm_thermal_data_list]
            
            # Time and layer for 2D Analysis tab (Panel 1 time series, Panel 4 signal vs layer).
            n = len(all_points)
            if n > 0:
                if all_timestamps and len(all_timestamps) >= n:
                    signals["time"] = [float(t) for t in all_timestamps[:n]]
                else:
                    signals["time"] = [float(i) for i in range(n)]
                # Signal comes with points, timestamp, and layer_index from backend (no fallback).
                if all_layers and len(all_layers) >= n:
                    layer_arr = [int(all_layers[i]) for i in range(n)]
                    signals["layer_index"] = layer_arr
                    signals["layer"] = layer_arr
            
            return QueryResult(
                points=all_points,
                signals=signals,
                metadata={
                    "model_id": model_id,
                    "source": "mongodb_cpp",  # Indicates C++ backend
                    "n_points": len(all_points),
                    "has_ispm_thermal_data": len(ispm_thermal_data_list) > 0,
                    "ispm_fields_calculated_in_cpp": True,  # All calculations in C++
                },
                component_id=model_id,
            )
            
        except ImportError:
            raise ImportError("am_qadf_native module not available. Install C++ extensions for ISPM calculations.")

    def get_bounding_box(
        self, component_id: Optional[str] = None
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """
        Get bounding box of monitoring data.

        Args:
            component_id: Model ID (required for MongoDB queries)

        Returns:
            Tuple of (bbox_min, bbox_max)
        """
        if not self.use_mongodb or not self.mongo_client:
            raise RuntimeError("MongoDB backend required. Call set_mongo_client() first.")
        
        if not component_id:
            raise ValueError("component_id (model_id) required for MongoDB queries")
        
        # Get from MongoDB using C++ backend
        try:
            cpp_client = self._get_cpp_client()
            
            # Query using C++ backend (all layers, all space)
            cpp_result = cpp_client.query_ispm_thermal(
                component_id,
                0.0, 0.0,  # All time
                -1, -1,  # All layers
                [float('-inf'), float('-inf'), float('-inf')],
                [float('inf'), float('inf'), float('inf')]
            )
            
            if not cpp_result.points or len(cpp_result.points) == 0:
                raise ValueError(f"No monitoring data found for model_id: {component_id}")
            
            points_array = np.array(cpp_result.points)
            bbox_min = tuple(np.min(points_array, axis=0))
            bbox_max = tuple(np.max(points_array, axis=0))
            return (bbox_min, bbox_max)
        except ImportError:
            raise ImportError("am_qadf_native module not available. Install C++ extensions for ISPM_Thermal queries.")

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
        Get the most recent monitoring data points (MongoDB mode only).

        Args:
            num_points: Number of recent points to retrieve
            signal_types: Signal types to retrieve

        Returns:
            QueryResult with latest data points
        """
        if not self.use_mongodb or not self.mongo_client:
            raise RuntimeError("MongoDB backend required. Call set_mongo_client() first.")
        
        # Use C++ backend to get latest data
        # Note: C++ query doesn't support sorting by timestamp yet, so we'll get all and sort in Python
        # For now, get a reasonable amount of data and sort
        try:
            cpp_client = self._get_cpp_client()
            
            # Query using C++ backend (get more than needed, then sort and limit)
            cpp_result = cpp_client.query_ispm_thermal(
                "",  # Empty model_id to get all (or we need to pass a specific one)
                0.0, 0.0,  # All time
                -1, -1,  # All layers
                [float('-inf'), float('-inf'), float('-inf')],
                [float('inf'), float('inf'), float('inf')]
            )
            
            # Sort by timestamp descending and take latest num_points
            if cpp_result.points and cpp_result.timestamps:
                # Create list of (timestamp, point, value) tuples
                data_list = list(zip(
                    cpp_result.timestamps,
                    cpp_result.points,
                    cpp_result.values if cpp_result.values else [0.0] * len(cpp_result.points)
                ))
                # Sort by timestamp descending
                data_list.sort(key=lambda x: x[0], reverse=True)
                # Take latest num_points
                data_list = data_list[:num_points]
                
                points = [tuple(p) for _, p, _ in data_list]
                signals = {
                    "temperature": [v for _, _, v in data_list]
                }
            else:
                points = []
                signals = {}
            
            return QueryResult(
                points=points,
                signals=signals,
                metadata={
                    "source": "mongodb_cpp",
                    "num_points": len(points),
                },
            )
        except ImportError:
            raise ImportError("am_qadf_native module not available. Install C++ extensions for ISPM_Thermal queries.")

    # MongoDB-specific helper methods
    def get_temperature_data(self, model_id: str, layer_range: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Get temperature data for a model (uses C++ backend).

        Args:
            model_id: Model UUID
            layer_range: Optional (start_layer, end_layer) tuple

        Returns:
            Numpy array of temperature values
        """
        if not self.use_mongodb:
            raise RuntimeError("get_temperature_data() requires MongoDB mode. Call set_mongo_client() first.")

        try:
            cpp_client = self._get_cpp_client()
            
            # Prepare layer range
            layer_start = -1
            layer_end = -1
            if layer_range:
                layer_start, layer_end = layer_range
            
            # Query using C++ backend
            cpp_result = cpp_client.query_ispm_thermal(
                model_id,
                0.0, 0.0,  # All time
                layer_start, layer_end,
                [float('-inf'), float('-inf'), float('-inf')],
                [float('inf'), float('inf'), float('inf')]
            )
            
            # Extract temperature values from ispm_thermal_data
            values = []
            if hasattr(cpp_result, 'ispm_thermal_data') and cpp_result.ispm_thermal_data:
                values = [ispm.melt_pool_temperature for ispm in cpp_result.ispm_thermal_data]
            
            return np.array(values, dtype=np.float64) if values else np.array([], dtype=np.float64)
        except ImportError:
            raise ImportError("am_qadf_native module not available. Install C++ extensions for ISPM_Thermal queries.")

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

        try:
            cpp_client = self._get_cpp_client()
            
            # Prepare layer range
            layer_start = -1
            layer_end = -1
            if layer_range:
                layer_start, layer_end = layer_range
            
            # Query using C++ backend
            cpp_result = cpp_client.query_ispm_thermal(
                model_id,
                0.0, 0.0,  # All time
                layer_start, layer_end,
                [float('-inf'), float('-inf'), float('-inf')],
                [float('inf'), float('inf'), float('inf')]
            )
            
            # Reconstruct melt pool data from C++ result
            melt_pool_data = []
            if hasattr(cpp_result, 'ispm_thermal_data') and cpp_result.ispm_thermal_data:
                for i, ispm in enumerate(cpp_result.ispm_thermal_data):
                    point = cpp_result.points[i] if i < len(cpp_result.points) else [0.0, 0.0, 0.0]
                    timestamp = cpp_result.timestamps[i] if i < len(cpp_result.timestamps) else 0.0
                    layer_idx = cpp_result.layers[i] if i < len(cpp_result.layers) else 0
                    
                    melt_pool_data.append({
                        "spatial_coordinates": list(point),
                        "melt_pool_size": {
                            "width": ispm.melt_pool_width,
                            "length": ispm.melt_pool_length,
                            "depth": ispm.melt_pool_depth
                        },
                        "melt_pool_temperature": ispm.melt_pool_temperature,
                        "layer_index": layer_idx,
                        "timestamp": timestamp,
                    })
            
            return melt_pool_data
        except ImportError:
            raise ImportError("am_qadf_native module not available. Install C++ extensions for ISPM_Thermal queries.")

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

        try:
            cpp_client = self._get_cpp_client()
            
            # Prepare layer range
            layer_start = -1
            layer_end = -1
            if layer_range:
                layer_start, layer_end = layer_range
            
            # Query using C++ backend
            cpp_result = cpp_client.query_ispm_thermal(
                model_id,
                0.0, 0.0,  # All time
                layer_start, layer_end,
                [float('-inf'), float('-inf'), float('-inf')],
                [float('inf'), float('inf'), float('inf')]
            )
            
            # Extract thermal gradient values from ispm_thermal_data
            values = []
            if hasattr(cpp_result, 'ispm_thermal_data') and cpp_result.ispm_thermal_data:
                values = [ispm.temperature_gradient for ispm in cpp_result.ispm_thermal_data]
            
            return np.array(values, dtype=np.float64) if values else np.array([], dtype=np.float64)
        except ImportError:
            raise ImportError("am_qadf_native module not available. Install C++ extensions for ISPM_Thermal queries.")

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

        try:
            cpp_client = self._get_cpp_client()
            
            # Query using C++ backend
            cpp_result = cpp_client.query_ispm_thermal(
                model_id,
                0.0, 0.0,  # All time
                -1, -1,  # All layers
                [float('-inf'), float('-inf'), float('-inf')],
                [float('inf'), float('inf'), float('inf')]
            )
            
            # Filter for process events from C++ result
            process_events = []
            if hasattr(cpp_result, 'ispm_thermal_data') and cpp_result.ispm_thermal_data:
                for i, ispm in enumerate(cpp_result.ispm_thermal_data):
                    if ispm.process_event and ispm.process_event != "":
                        point = cpp_result.points[i] if i < len(cpp_result.points) else [0.0, 0.0, 0.0]
                        timestamp = cpp_result.timestamps[i] if i < len(cpp_result.timestamps) else 0.0
                        layer_idx = cpp_result.layers[i] if i < len(cpp_result.layers) else 0
                        
                        process_events.append({
                            "spatial_coordinates": list(point),
                            "process_event": ispm.process_event,
                            "layer_index": layer_idx,
                            "timestamp": timestamp,
                        })
            
            return process_events
        except ImportError:
            raise ImportError("am_qadf_native module not available. Install C++ extensions for ISPM_Thermal queries.")

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

        try:
            cpp_client = self._get_cpp_client()
            
            # Query using C++ backend to get one document
            cpp_result = cpp_client.query_ispm_thermal(
                model_id,
                0.0, 0.0,  # All time
                -1, -1,  # All layers
                [float('-inf'), float('-inf'), float('-inf')],
                [float('inf'), float('inf'), float('inf')]
            )
            
            # Coordinate system is not directly in C++ result
            # We need to extract it from metadata or reconstruct
            # For now, return None (coordinate system should be stored separately or in metadata)
            # TODO: Add coordinate system extraction from C++ result if available
            return None
        except ImportError:
            raise ImportError("am_qadf_native module not available. Install C++ extensions for ISPM_Thermal queries.")
