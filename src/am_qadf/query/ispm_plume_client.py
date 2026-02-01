"""
In-Situ Monitoring Client (ISPM_Plume)

Query client for real-time in-situ vapor plume monitoring data for PBF-LB/M processes.
Supports vapor plume characteristics, geometry, composition, dynamics, and process quality indicators.

Note: ISPM (In-Situ Process Monitoring) is a broad category. This client handles
ISPM_Plume (vapor plume monitoring - monitoring the vapor plume above the melt pool).
Other ISPM types include:
- ISPM_Thermal: Thermal monitoring
- ISPM_Optical: Photodiodes, cameras, melt pool imaging
- ISPM_Acoustic: Acoustic emissions, sound sensors
- ISPM_Strain: Strain gauges, deformation sensors

IMPORTANT: All MongoDB queries use the C++ backend (MongoDBQueryClient) to ensure
all ISPM_Plume field extraction happens in C++ for maximum performance. No Python calculations are performed.
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


class ISPMPlumeClient(BaseQueryClient):
    """
    Query client for ISPM_Plume (In-Situ Process Monitoring - Plume) data.

    All queries use MongoDB backend with C++ client for maximum performance.
    All ISPM_Plume field extraction happens in C++.

    Note: ISPM (In-Situ Process Monitoring) is a broad category. This client handles
    ISPM_Plume (vapor plume monitoring - monitoring the vapor plume above the melt pool).
    Other ISPM types (Thermal, Optical, Acoustic, Strain) have separate clients.

    Supports:
    - Real-time vapor plume monitoring data queries (MongoDB + C++ backend)
    - Plume characteristics (intensity, density, temperature, velocity)
    - Plume geometry (height, width, angle, spread, area)
    - Plume composition (particle concentration, metal vapor, gas composition)
    - Plume dynamics (fluctuation rate, instability, turbulence)
    - Process quality indicators (process stability, plume stability)
    - Event detection (excessive plume, unstable plume, contamination, anomalies)
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
        Initialize in-situ vapor plume monitoring client.

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
            SignalType.POWER,  # Plume intensity can be treated as power signal
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

    def _get_cpp_client(self):
        """Get C++ MongoDB query client (internal helper)."""
        try:
            import am_qadf_native
            from am_qadf_native import MongoDBQueryClient
            
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
            
            return MongoDBQueryClient(uri, db_name)
        except ImportError:
            raise ImportError("am_qadf_native module not available. Install C++ extensions for ISPM_Plume queries.")

    def query(
        self,
        spatial: Optional[SpatialQuery] = None,
        temporal: Optional[TemporalQuery] = None,
        signal_types: Optional[List[SignalType]] = None,
    ) -> QueryResult:
        """
        Query in-situ vapor plume monitoring data from MongoDB (C++ backend).

        All queries use C++ backend for maximum performance. All ISPM_Plume field extraction happens in C++.

        Args:
            spatial: Spatial query parameters (requires component_id = model_id)
            temporal: Temporal query parameters (time range, layer range)
            signal_types: Signal types to retrieve

        Returns:
            QueryResult with monitoring points and signals (all extracted in C++)
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
        
        # Use C++ client for all queries (extraction happens in C++)
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
            logger.info(f"C++ Query: Creating MongoDBQueryClient for ISPM_Plume data (database: {db_name})")
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
            
            # Query using C++ (all extraction and filtering happen in C++)
            logger.info(f"C++ Query: Calling query_ispm_plume for model_id={model_id[:20]}..., "
                       f"time_range=({time_start}, {time_end}), "
                       f"layer_range=({layer_start}, {layer_end}), "
                       f"bbox_min={bbox_min}, bbox_max={bbox_max}")
            cpp_result = cpp_client.query_ispm_plume(
                model_id, time_start, time_end, layer_start, layer_end, bbox_min, bbox_max
            )
            logger.info(f"C++ Query: query_ispm_plume completed, processing result...")
            
            # Convert C++ QueryResult to Python QueryResult
            # Extract points and values (already filtered in C++)
            all_points = [tuple(p) for p in cpp_result.points] if cpp_result.points else []
            all_values = list(cpp_result.values) if cpp_result.values else []
            all_timestamps = list(cpp_result.timestamps) if cpp_result.timestamps else []
            all_layers = list(cpp_result.layers) if cpp_result.layers else []
            
            # Extract ISPM_Plume-specific data (all extracted in C++)
            ispm_plume_data_list = []
            if hasattr(cpp_result, 'ispm_plume_data') and cpp_result.ispm_plume_data:
                for ispm in cpp_result.ispm_plume_data:
                    ispm_plume_data_list.append({
                        "plume_intensity": ispm.plume_intensity,
                        "plume_density": ispm.plume_density,
                        "plume_temperature": ispm.plume_temperature,
                        "plume_velocity": ispm.plume_velocity,
                        "plume_velocity_x": ispm.plume_velocity_x,
                        "plume_velocity_y": ispm.plume_velocity_y,
                        "plume_height": ispm.plume_height,
                        "plume_width": ispm.plume_width,
                        "plume_angle": ispm.plume_angle,
                        "plume_spread": ispm.plume_spread,
                        "plume_area": ispm.plume_area,
                        "particle_concentration": ispm.particle_concentration,
                        "metal_vapor_concentration": ispm.metal_vapor_concentration,
                        "gas_composition_ratio": ispm.gas_composition_ratio,
                        "plume_fluctuation_rate": ispm.plume_fluctuation_rate,
                        "plume_instability_index": ispm.plume_instability_index,
                        "plume_turbulence": ispm.plume_turbulence,
                        "process_stability": ispm.process_stability,
                        "plume_stability": ispm.plume_stability,
                        "intensity_variation": ispm.intensity_variation,
                        "excessive_plume_event": ispm.excessive_plume_event,
                        "unstable_plume_event": ispm.unstable_plume_event,
                        "contamination_event": ispm.contamination_event,
                        "anomaly_detected": ispm.anomaly_detected,
                        "anomaly_type": ispm.anomaly_type if ispm.anomaly_type else None,
                        "plume_energy": ispm.plume_energy,
                        "energy_density": ispm.energy_density,
                        "process_event": ispm.process_event if ispm.process_event else None,
                        "signal_to_noise_ratio": ispm.signal_to_noise_ratio,
                        "plume_momentum": ispm.plume_momentum if ispm.plume_momentum >= 0 else None,
                        "plume_pressure": ispm.plume_pressure if ispm.plume_pressure >= 0 else None,
                    })
            
            # Build signals dictionary from ISPM_Plume data
            signals = {}
            if signal_types is None or SignalType.POWER in signal_types:
                signals["plume_intensity"] = [ispm.get("plume_intensity", 0.0) for ispm in ispm_plume_data_list] if ispm_plume_data_list else all_values
            
            # Add ISPM_Plume-specific signals
            if ispm_plume_data_list:
                signals["plume_density"] = [ispm.get("plume_density", 0.0) for ispm in ispm_plume_data_list]
                signals["plume_temperature"] = [ispm.get("plume_temperature", 0.0) for ispm in ispm_plume_data_list]
                signals["plume_velocity"] = [ispm.get("plume_velocity", 0.0) for ispm in ispm_plume_data_list]
                signals["plume_height"] = [ispm.get("plume_height", 0.0) for ispm in ispm_plume_data_list]
                signals["plume_width"] = [ispm.get("plume_width", 0.0) for ispm in ispm_plume_data_list]
                signals["plume_area"] = [ispm.get("plume_area", 0.0) for ispm in ispm_plume_data_list]
                signals["particle_concentration"] = [ispm.get("particle_concentration", 0.0) for ispm in ispm_plume_data_list]
                signals["process_stability"] = [ispm.get("process_stability", 0.0) for ispm in ispm_plume_data_list]
                signals["plume_stability"] = [ispm.get("plume_stability", 0.0) for ispm in ispm_plume_data_list]
                signals["intensity_variation"] = [ispm.get("intensity_variation", 0.0) for ispm in ispm_plume_data_list]
                signals["excessive_plume_event"] = [1.0 if ispm.get("excessive_plume_event", False) else 0.0 for ispm in ispm_plume_data_list]
                signals["unstable_plume_event"] = [1.0 if ispm.get("unstable_plume_event", False) else 0.0 for ispm in ispm_plume_data_list]
                signals["contamination_event"] = [1.0 if ispm.get("contamination_event", False) else 0.0 for ispm in ispm_plume_data_list]
                signals["anomaly_detected"] = [1.0 if ispm.get("anomaly_detected", False) else 0.0 for ispm in ispm_plume_data_list]
            
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
                    "has_ispm_plume_data": len(ispm_plume_data_list) > 0,
                    "ispm_fields_extracted_in_cpp": True,  # All extraction in C++
                },
                component_id=model_id,
            )
            
        except ImportError:
            raise ImportError("am_qadf_native module not available. Install C++ extensions for ISPM_Plume queries.")

    def get_bounding_box(
        self, component_id: Optional[str] = None
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """
        Get bounding box of monitoring data (uses C++ backend).

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
            cpp_result = cpp_client.query_ispm_plume(
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
            raise ImportError("am_qadf_native module not available. Install C++ extensions for ISPM_Plume queries.")

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

    def get_plume_intensity(self, model_id: str, layer_range: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Get plume intensity data for a model (uses C++ backend).

        Args:
            model_id: Model UUID
            layer_range: Optional (start_layer, end_layer) tuple

        Returns:
            Numpy array of plume intensity values
        """
        if not self.use_mongodb:
            raise RuntimeError("get_plume_intensity() requires MongoDB mode. Call set_mongo_client() first.")

        try:
            cpp_client = self._get_cpp_client()
            
            # Prepare layer range
            layer_start = -1
            layer_end = -1
            if layer_range:
                layer_start, layer_end = layer_range
            
            # Query using C++ backend
            cpp_result = cpp_client.query_ispm_plume(
                model_id,
                0.0, 0.0,  # All time
                layer_start, layer_end,
                [float('-inf'), float('-inf'), float('-inf')],
                [float('inf'), float('inf'), float('inf')]
            )
            
            # Extract plume intensity values from ispm_plume_data
            values = []
            if hasattr(cpp_result, 'ispm_plume_data') and cpp_result.ispm_plume_data:
                values = [ispm.plume_intensity for ispm in cpp_result.ispm_plume_data]
            
            return np.array(values, dtype=np.float64) if values else np.array([], dtype=np.float64)
        except ImportError:
            raise ImportError("am_qadf_native module not available. Install C++ extensions for ISPM_Plume queries.")

    def get_excessive_plume_events(self, model_id: str, layer_range: Optional[Tuple[int, int]] = None) -> List[Dict[str, Any]]:
        """
        Get excessive plume events for a model (uses C++ backend).

        Args:
            model_id: Model UUID
            layer_range: Optional (start_layer, end_layer) tuple

        Returns:
            List of excessive plume event dictionaries
        """
        if not self.use_mongodb:
            raise RuntimeError("get_excessive_plume_events() requires MongoDB mode. Call set_mongo_client() first.")

        try:
            cpp_client = self._get_cpp_client()
            
            # Prepare layer range
            layer_start = -1
            layer_end = -1
            if layer_range:
                layer_start, layer_end = layer_range
            
            # Query using C++ backend
            cpp_result = cpp_client.query_ispm_plume(
                model_id,
                0.0, 0.0,  # All time
                layer_start, layer_end,
                [float('-inf'), float('-inf'), float('-inf')],
                [float('inf'), float('inf'), float('inf')]
            )
            
            # Filter for excessive plume events from C++ result
            excessive_plume_events = []
            if hasattr(cpp_result, 'ispm_plume_data') and cpp_result.ispm_plume_data:
                for i, ispm in enumerate(cpp_result.ispm_plume_data):
                    if ispm.excessive_plume_event:
                        point = cpp_result.points[i] if i < len(cpp_result.points) else [0.0, 0.0, 0.0]
                        timestamp = cpp_result.timestamps[i] if i < len(cpp_result.timestamps) else 0.0
                        layer_idx = cpp_result.layers[i] if i < len(cpp_result.layers) else 0
                        
                        excessive_plume_events.append({
                            "spatial_coordinates": list(point),
                            "plume_intensity": ispm.plume_intensity,
                            "plume_density": ispm.plume_density,
                            "plume_temperature": ispm.plume_temperature,
                            "plume_velocity": ispm.plume_velocity,
                            "layer_index": layer_idx,
                            "timestamp": timestamp,
                        })
            
            return excessive_plume_events
        except ImportError:
            raise ImportError("am_qadf_native module not available. Install C++ extensions for ISPM_Plume queries.")

    def get_unstable_plume_events(self, model_id: str, layer_range: Optional[Tuple[int, int]] = None) -> List[Dict[str, Any]]:
        """
        Get unstable plume events for a model (uses C++ backend).

        Args:
            model_id: Model UUID
            layer_range: Optional (start_layer, end_layer) tuple

        Returns:
            List of unstable plume event dictionaries
        """
        if not self.use_mongodb:
            raise RuntimeError("get_unstable_plume_events() requires MongoDB mode. Call set_mongo_client() first.")

        try:
            cpp_client = self._get_cpp_client()
            
            # Prepare layer range
            layer_start = -1
            layer_end = -1
            if layer_range:
                layer_start, layer_end = layer_range
            
            # Query using C++ backend
            cpp_result = cpp_client.query_ispm_plume(
                model_id,
                0.0, 0.0,  # All time
                layer_start, layer_end,
                [float('-inf'), float('-inf'), float('-inf')],
                [float('inf'), float('inf'), float('inf')]
            )
            
            # Filter for unstable plume events from C++ result
            unstable_plume_events = []
            if hasattr(cpp_result, 'ispm_plume_data') and cpp_result.ispm_plume_data:
                for i, ispm in enumerate(cpp_result.ispm_plume_data):
                    if ispm.unstable_plume_event:
                        point = cpp_result.points[i] if i < len(cpp_result.points) else [0.0, 0.0, 0.0]
                        timestamp = cpp_result.timestamps[i] if i < len(cpp_result.timestamps) else 0.0
                        layer_idx = cpp_result.layers[i] if i < len(cpp_result.layers) else 0
                        
                        unstable_plume_events.append({
                            "spatial_coordinates": list(point),
                            "plume_instability_index": ispm.plume_instability_index,
                            "plume_stability": ispm.plume_stability,
                            "plume_fluctuation_rate": ispm.plume_fluctuation_rate,
                            "plume_turbulence": ispm.plume_turbulence,
                            "plume_intensity": ispm.plume_intensity,
                            "layer_index": layer_idx,
                            "timestamp": timestamp,
                        })
            
            return unstable_plume_events
        except ImportError:
            raise ImportError("am_qadf_native module not available. Install C++ extensions for ISPM_Plume queries.")

    def get_contamination_events(self, model_id: str, layer_range: Optional[Tuple[int, int]] = None) -> List[Dict[str, Any]]:
        """
        Get contamination events for a model (uses C++ backend).

        Args:
            model_id: Model UUID
            layer_range: Optional (start_layer, end_layer) tuple

        Returns:
            List of contamination event dictionaries
        """
        if not self.use_mongodb:
            raise RuntimeError("get_contamination_events() requires MongoDB mode. Call set_mongo_client() first.")

        try:
            cpp_client = self._get_cpp_client()
            
            # Prepare layer range
            layer_start = -1
            layer_end = -1
            if layer_range:
                layer_start, layer_end = layer_range
            
            # Query using C++ backend
            cpp_result = cpp_client.query_ispm_plume(
                model_id,
                0.0, 0.0,  # All time
                layer_start, layer_end,
                [float('-inf'), float('-inf'), float('-inf')],
                [float('inf'), float('inf'), float('inf')]
            )
            
            # Filter for contamination events from C++ result
            contamination_events = []
            if hasattr(cpp_result, 'ispm_plume_data') and cpp_result.ispm_plume_data:
                for i, ispm in enumerate(cpp_result.ispm_plume_data):
                    if ispm.contamination_event:
                        point = cpp_result.points[i] if i < len(cpp_result.points) else [0.0, 0.0, 0.0]
                        timestamp = cpp_result.timestamps[i] if i < len(cpp_result.timestamps) else 0.0
                        layer_idx = cpp_result.layers[i] if i < len(cpp_result.layers) else 0
                        
                        contamination_events.append({
                            "spatial_coordinates": list(point),
                            "particle_concentration": ispm.particle_concentration,
                            "metal_vapor_concentration": ispm.metal_vapor_concentration,
                            "gas_composition_ratio": ispm.gas_composition_ratio,
                            "plume_intensity": ispm.plume_intensity,
                            "layer_index": layer_idx,
                            "timestamp": timestamp,
                        })
            
            return contamination_events
        except ImportError:
            raise ImportError("am_qadf_native module not available. Install C++ extensions for ISPM_Plume queries.")

    def get_process_stability(self, model_id: str, layer_range: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Get process stability metrics for a model (uses C++ backend).

        Args:
            model_id: Model UUID
            layer_range: Optional (start_layer, end_layer) tuple

        Returns:
            Numpy array of process stability values (0-1 scale)
        """
        if not self.use_mongodb:
            raise RuntimeError("get_process_stability() requires MongoDB mode. Call set_mongo_client() first.")

        try:
            cpp_client = self._get_cpp_client()
            
            # Prepare layer range
            layer_start = -1
            layer_end = -1
            if layer_range:
                layer_start, layer_end = layer_range
            
            # Query using C++ backend
            cpp_result = cpp_client.query_ispm_plume(
                model_id,
                0.0, 0.0,  # All time
                layer_start, layer_end,
                [float('-inf'), float('-inf'), float('-inf')],
                [float('inf'), float('inf'), float('inf')]
            )
            
            # Extract process stability values from ispm_plume_data
            values = []
            if hasattr(cpp_result, 'ispm_plume_data') and cpp_result.ispm_plume_data:
                values = [ispm.process_stability for ispm in cpp_result.ispm_plume_data]
            
            return np.array(values, dtype=np.float64) if values else np.array([], dtype=np.float64)
        except ImportError:
            raise ImportError("am_qadf_native module not available. Install C++ extensions for ISPM_Plume queries.")
