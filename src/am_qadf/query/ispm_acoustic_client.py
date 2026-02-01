"""
In-Situ Monitoring Client (ISPM_Acoustic)

Query client for real-time in-situ acoustic monitoring data for PBF-LB/M processes.
Supports acoustic emission signals, event detection (spatter, defects, anomalies), and process stability.

Note: ISPM (In-Situ Process Monitoring) is a broad category. This client handles
ISPM_Acoustic (acoustic monitoring - acoustic emissions, sound sensors).
Other ISPM types include:
- ISPM_Thermal: Thermal monitoring
- ISPM_Optical: Photodiodes, cameras, melt pool imaging
- ISPM_Strain: Strain gauges, deformation sensors
- ISPM_Plume: Vapor plume monitoring

IMPORTANT: All MongoDB queries use the C++ backend (MongoDBQueryClient) to ensure
all ISPM_Acoustic field extraction happens in C++ for maximum performance. No Python calculations are performed.
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


class ISPMAcousticClient(BaseQueryClient):
    """
    Query client for ISPM_Acoustic (In-Situ Process Monitoring - Acoustic) data.

    All queries use MongoDB backend with C++ client for maximum performance.
    All ISPM_Acoustic field extraction happens in C++.

    Note: ISPM (In-Situ Process Monitoring) is a broad category. This client handles
    ISPM_Acoustic (acoustic monitoring - acoustic emissions, sound sensors).
    Other ISPM types (Thermal, Optical, Strain, Plume) have separate clients.

    Supports:
    - Real-time acoustic monitoring data queries (MongoDB + C++ backend)
    - Acoustic emission signal analysis
    - Event detection (spatter, defects, anomalies)
    - Process stability metrics
    - Frequency domain analysis
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
        Initialize in-situ acoustic monitoring client.

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
            SignalType.POWER,  # Acoustic amplitude can be treated as power signal
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
            raise ImportError("am_qadf_native module not available. Install C++ extensions for ISPM_Acoustic queries.")

    def query(
        self,
        spatial: Optional[SpatialQuery] = None,
        temporal: Optional[TemporalQuery] = None,
        signal_types: Optional[List[SignalType]] = None,
    ) -> QueryResult:
        """
        Query in-situ acoustic monitoring data from MongoDB (C++ backend).

        All queries use C++ backend for maximum performance. All ISPM_Acoustic field extraction happens in C++.

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
            logger.info(f"C++ Query: Creating MongoDBQueryClient for ISPM_Acoustic data (database: {db_name})")
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
            logger.info(f"C++ Query: Calling query_ispm_acoustic for model_id={model_id[:20]}..., "
                       f"time_range=({time_start}, {time_end}), "
                       f"layer_range=({layer_start}, {layer_end}), "
                       f"bbox_min={bbox_min}, bbox_max={bbox_max}")
            cpp_result = cpp_client.query_ispm_acoustic(
                model_id, time_start, time_end, layer_start, layer_end, bbox_min, bbox_max
            )
            logger.info(f"C++ Query: query_ispm_acoustic completed, processing result...")
            
            # Convert C++ QueryResult to Python QueryResult
            # Extract points and values (already filtered in C++)
            all_points = [tuple(p) for p in cpp_result.points] if cpp_result.points else []
            all_values = list(cpp_result.values) if cpp_result.values else []
            all_timestamps = list(cpp_result.timestamps) if cpp_result.timestamps else []
            all_layers = list(cpp_result.layers) if cpp_result.layers else []
            
            # Extract ISPM_Acoustic-specific data (all extracted in C++)
            ispm_acoustic_data_list = []
            if hasattr(cpp_result, 'ispm_acoustic_data') and cpp_result.ispm_acoustic_data:
                for ispm in cpp_result.ispm_acoustic_data:
                    ispm_acoustic_data_list.append({
                        "acoustic_amplitude": ispm.acoustic_amplitude,
                        "acoustic_frequency": ispm.acoustic_frequency,
                        "acoustic_rms": ispm.acoustic_rms,
                        "acoustic_peak": ispm.acoustic_peak,
                        "dominant_frequency": ispm.dominant_frequency,
                        "frequency_bandwidth": ispm.frequency_bandwidth,
                        "spectral_centroid": ispm.spectral_centroid,
                        "spectral_energy": ispm.spectral_energy,
                        "spectral_rolloff": ispm.spectral_rolloff if ispm.spectral_rolloff >= 0 else None,
                        "spatter_event_detected": ispm.spatter_event_detected,
                        "spatter_event_amplitude": ispm.spatter_event_amplitude if ispm.spatter_event_amplitude >= 0 else None,
                        "defect_event_detected": ispm.defect_event_detected,
                        "defect_event_amplitude": ispm.defect_event_amplitude if ispm.defect_event_amplitude >= 0 else None,
                        "anomaly_detected": ispm.anomaly_detected,
                        "anomaly_type": ispm.anomaly_type if ispm.anomaly_type else None,
                        "process_stability": ispm.process_stability,
                        "acoustic_variation": ispm.acoustic_variation,
                        "signal_to_noise_ratio": ispm.signal_to_noise_ratio,
                        "zero_crossing_rate": ispm.zero_crossing_rate if ispm.zero_crossing_rate >= 0 else None,
                        "autocorrelation_peak": ispm.autocorrelation_peak if ispm.autocorrelation_peak >= 0 else None,
                        "harmonic_ratio": ispm.harmonic_ratio if ispm.harmonic_ratio >= 0 else None,
                        "spectral_flatness": ispm.spectral_flatness if ispm.spectral_flatness >= 0 else None,
                        "spectral_crest": ispm.spectral_crest if ispm.spectral_crest >= 0 else None,
                        "process_event": ispm.process_event if ispm.process_event else None,
                        "acoustic_energy": ispm.acoustic_energy,
                    })
            
            # Build signals dictionary from ISPM_Acoustic data
            signals = {}
            if signal_types is None or SignalType.POWER in signal_types:
                signals["acoustic_amplitude"] = [ispm.get("acoustic_amplitude", 0.0) for ispm in ispm_acoustic_data_list] if ispm_acoustic_data_list else all_values
            
            # Add ISPM_Acoustic-specific signals
            if ispm_acoustic_data_list:
                signals["acoustic_frequency"] = [ispm.get("acoustic_frequency", 0.0) for ispm in ispm_acoustic_data_list]
                signals["acoustic_rms"] = [ispm.get("acoustic_rms", 0.0) for ispm in ispm_acoustic_data_list]
                signals["acoustic_peak"] = [ispm.get("acoustic_peak", 0.0) for ispm in ispm_acoustic_data_list]
                signals["dominant_frequency"] = [ispm.get("dominant_frequency", 0.0) for ispm in ispm_acoustic_data_list]
                signals["process_stability"] = [ispm.get("process_stability", 0.0) for ispm in ispm_acoustic_data_list]
                signals["acoustic_variation"] = [ispm.get("acoustic_variation", 0.0) for ispm in ispm_acoustic_data_list]
                signals["signal_to_noise_ratio"] = [ispm.get("signal_to_noise_ratio", 0.0) for ispm in ispm_acoustic_data_list]
                signals["spatter_event_detected"] = [1.0 if ispm.get("spatter_event_detected", False) else 0.0 for ispm in ispm_acoustic_data_list]
                signals["defect_event_detected"] = [1.0 if ispm.get("defect_event_detected", False) else 0.0 for ispm in ispm_acoustic_data_list]
                signals["anomaly_detected"] = [1.0 if ispm.get("anomaly_detected", False) else 0.0 for ispm in ispm_acoustic_data_list]
                signals["acoustic_energy"] = [ispm.get("acoustic_energy", 0.0) for ispm in ispm_acoustic_data_list]
            
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
                    "has_ispm_acoustic_data": len(ispm_acoustic_data_list) > 0,
                    "ispm_fields_extracted_in_cpp": True,  # All extraction in C++
                },
                component_id=model_id,
            )
            
        except ImportError:
            raise ImportError("am_qadf_native module not available. Install C++ extensions for ISPM_Acoustic queries.")

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
            cpp_result = cpp_client.query_ispm_acoustic(
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
            raise ImportError("am_qadf_native module not available. Install C++ extensions for ISPM_Acoustic queries.")

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

    def get_amplitude_data(self, model_id: str, layer_range: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Get acoustic amplitude data for a model (uses C++ backend).

        Args:
            model_id: Model UUID
            layer_range: Optional (start_layer, end_layer) tuple

        Returns:
            Numpy array of acoustic amplitude values
        """
        if not self.use_mongodb:
            raise RuntimeError("get_amplitude_data() requires MongoDB mode. Call set_mongo_client() first.")

        try:
            cpp_client = self._get_cpp_client()
            
            # Prepare layer range
            layer_start = -1
            layer_end = -1
            if layer_range:
                layer_start, layer_end = layer_range
            
            # Query using C++ backend
            cpp_result = cpp_client.query_ispm_acoustic(
                model_id,
                0.0, 0.0,  # All time
                layer_start, layer_end,
                [float('-inf'), float('-inf'), float('-inf')],
                [float('inf'), float('inf'), float('inf')]
            )
            
            # Extract acoustic amplitude values from ispm_acoustic_data
            values = []
            if hasattr(cpp_result, 'ispm_acoustic_data') and cpp_result.ispm_acoustic_data:
                values = [ispm.acoustic_amplitude for ispm in cpp_result.ispm_acoustic_data]
            
            return np.array(values, dtype=np.float64) if values else np.array([], dtype=np.float64)
        except ImportError:
            raise ImportError("am_qadf_native module not available. Install C++ extensions for ISPM_Acoustic queries.")

    def get_spatter_events(self, model_id: str, layer_range: Optional[Tuple[int, int]] = None) -> List[Dict[str, Any]]:
        """
        Get spatter events for a model (uses C++ backend).

        Args:
            model_id: Model UUID
            layer_range: Optional (start_layer, end_layer) tuple

        Returns:
            List of spatter event dictionaries
        """
        if not self.use_mongodb:
            raise RuntimeError("get_spatter_events() requires MongoDB mode. Call set_mongo_client() first.")

        try:
            cpp_client = self._get_cpp_client()
            
            # Prepare layer range
            layer_start = -1
            layer_end = -1
            if layer_range:
                layer_start, layer_end = layer_range
            
            # Query using C++ backend
            cpp_result = cpp_client.query_ispm_acoustic(
                model_id,
                0.0, 0.0,  # All time
                layer_start, layer_end,
                [float('-inf'), float('-inf'), float('-inf')],
                [float('inf'), float('inf'), float('inf')]
            )
            
            # Filter for spatter events from C++ result
            spatter_events = []
            if hasattr(cpp_result, 'ispm_acoustic_data') and cpp_result.ispm_acoustic_data:
                for i, ispm in enumerate(cpp_result.ispm_acoustic_data):
                    if ispm.spatter_event_detected:
                        point = cpp_result.points[i] if i < len(cpp_result.points) else [0.0, 0.0, 0.0]
                        timestamp = cpp_result.timestamps[i] if i < len(cpp_result.timestamps) else 0.0
                        layer_idx = cpp_result.layers[i] if i < len(cpp_result.layers) else 0
                        
                        spatter_events.append({
                            "spatial_coordinates": list(point),
                            "spatter_event_amplitude": ispm.spatter_event_amplitude if ispm.spatter_event_amplitude >= 0 else None,
                            "acoustic_amplitude": ispm.acoustic_amplitude,
                            "dominant_frequency": ispm.dominant_frequency,
                            "layer_index": layer_idx,
                            "timestamp": timestamp,
                        })
            
            return spatter_events
        except ImportError:
            raise ImportError("am_qadf_native module not available. Install C++ extensions for ISPM_Acoustic queries.")

    def get_defect_events(self, model_id: str, layer_range: Optional[Tuple[int, int]] = None) -> List[Dict[str, Any]]:
        """
        Get defect events for a model (uses C++ backend).

        Args:
            model_id: Model UUID
            layer_range: Optional (start_layer, end_layer) tuple

        Returns:
            List of defect event dictionaries
        """
        if not self.use_mongodb:
            raise RuntimeError("get_defect_events() requires MongoDB mode. Call set_mongo_client() first.")

        try:
            cpp_client = self._get_cpp_client()
            
            # Prepare layer range
            layer_start = -1
            layer_end = -1
            if layer_range:
                layer_start, layer_end = layer_range
            
            # Query using C++ backend
            cpp_result = cpp_client.query_ispm_acoustic(
                model_id,
                0.0, 0.0,  # All time
                layer_start, layer_end,
                [float('-inf'), float('-inf'), float('-inf')],
                [float('inf'), float('inf'), float('inf')]
            )
            
            # Filter for defect events from C++ result
            defect_events = []
            if hasattr(cpp_result, 'ispm_acoustic_data') and cpp_result.ispm_acoustic_data:
                for i, ispm in enumerate(cpp_result.ispm_acoustic_data):
                    if ispm.defect_event_detected:
                        point = cpp_result.points[i] if i < len(cpp_result.points) else [0.0, 0.0, 0.0]
                        timestamp = cpp_result.timestamps[i] if i < len(cpp_result.timestamps) else 0.0
                        layer_idx = cpp_result.layers[i] if i < len(cpp_result.layers) else 0
                        
                        defect_events.append({
                            "spatial_coordinates": list(point),
                            "defect_event_amplitude": ispm.defect_event_amplitude if ispm.defect_event_amplitude >= 0 else None,
                            "acoustic_amplitude": ispm.acoustic_amplitude,
                            "dominant_frequency": ispm.dominant_frequency,
                            "layer_index": layer_idx,
                            "timestamp": timestamp,
                        })
            
            return defect_events
        except ImportError:
            raise ImportError("am_qadf_native module not available. Install C++ extensions for ISPM_Acoustic queries.")

    def get_anomaly_events(self, model_id: str, layer_range: Optional[Tuple[int, int]] = None) -> List[Dict[str, Any]]:
        """
        Get anomaly events for a model (uses C++ backend).

        Args:
            model_id: Model UUID
            layer_range: Optional (start_layer, end_layer) tuple

        Returns:
            List of anomaly event dictionaries
        """
        if not self.use_mongodb:
            raise RuntimeError("get_anomaly_events() requires MongoDB mode. Call set_mongo_client() first.")

        try:
            cpp_client = self._get_cpp_client()
            
            # Prepare layer range
            layer_start = -1
            layer_end = -1
            if layer_range:
                layer_start, layer_end = layer_range
            
            # Query using C++ backend
            cpp_result = cpp_client.query_ispm_acoustic(
                model_id,
                0.0, 0.0,  # All time
                layer_start, layer_end,
                [float('-inf'), float('-inf'), float('-inf')],
                [float('inf'), float('inf'), float('inf')]
            )
            
            # Filter for anomaly events from C++ result
            anomaly_events = []
            if hasattr(cpp_result, 'ispm_acoustic_data') and cpp_result.ispm_acoustic_data:
                for i, ispm in enumerate(cpp_result.ispm_acoustic_data):
                    if ispm.anomaly_detected:
                        point = cpp_result.points[i] if i < len(cpp_result.points) else [0.0, 0.0, 0.0]
                        timestamp = cpp_result.timestamps[i] if i < len(cpp_result.timestamps) else 0.0
                        layer_idx = cpp_result.layers[i] if i < len(cpp_result.layers) else 0
                        
                        anomaly_events.append({
                            "spatial_coordinates": list(point),
                            "anomaly_type": ispm.anomaly_type if ispm.anomaly_type else "unknown",
                            "acoustic_amplitude": ispm.acoustic_amplitude,
                            "dominant_frequency": ispm.dominant_frequency,
                            "process_stability": ispm.process_stability,
                            "layer_index": layer_idx,
                            "timestamp": timestamp,
                        })
            
            return anomaly_events
        except ImportError:
            raise ImportError("am_qadf_native module not available. Install C++ extensions for ISPM_Acoustic queries.")

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
            cpp_result = cpp_client.query_ispm_acoustic(
                model_id,
                0.0, 0.0,  # All time
                layer_start, layer_end,
                [float('-inf'), float('-inf'), float('-inf')],
                [float('inf'), float('inf'), float('inf')]
            )
            
            # Extract process stability values from ispm_acoustic_data
            values = []
            if hasattr(cpp_result, 'ispm_acoustic_data') and cpp_result.ispm_acoustic_data:
                values = [ispm.process_stability for ispm in cpp_result.ispm_acoustic_data]
            
            return np.array(values, dtype=np.float64) if values else np.array([], dtype=np.float64)
        except ImportError:
            raise ImportError("am_qadf_native module not available. Install C++ extensions for ISPM_Acoustic queries.")
