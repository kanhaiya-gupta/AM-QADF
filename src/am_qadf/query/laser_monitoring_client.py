"""
Laser Beam Diagnostics (LBD) / Laser Monitoring Query Client

Query client for laser parameters (commanded_power, commanded_scan_speed, commanded_energy)
and laser monitoring data from hatching/path data. LBD = Laser Beam Diagnostics.
Wraps existing hatching generation and C++ query to provide a standardized query interface.
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


class LaserMonitoringClient(BaseQueryClient):
    """
    Laser Beam Diagnostics (LBD) query client for laser parameters from hatching/path data.

    Supports both:
    1. Generated data (pyslm layers) - original functionality
    2. MongoDB data warehouse - new functionality (C++ LaserMonitoringQuery / LBD)

    This client wraps hatching generation and provides a query interface
    for accessing laser commanded_power, commanded_scan_speed, commanded_energy,
    and laser monitoring / LBD temporal data (power, beam, system health).
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
            raise ImportError("am_qadf_native module not available. Install C++ extensions for laser parameter queries.")

    def query(
        self,
        spatial: Optional[SpatialQuery] = None,
        temporal: Optional[TemporalQuery] = None,
        signal_types: Optional[List[SignalType]] = None,
    ) -> QueryResult:
        """
        Query laser parameters from MongoDB (C++ backend).
        
        All queries use C++ backend for performance - no Python-side filtering or calculations.
        Supports both MongoDB warehouse and generated data modes.

        Args:
            spatial: Spatial query parameters (requires component_id = model_id for MongoDB mode)
            temporal: Temporal query parameters (layer range)
            signal_types: Signal types to retrieve (None = all available)

        Returns:
            QueryResult with points and signal values
        """
        # Fall back to generated data mode if MongoDB not enabled
        if not self.use_mongodb or not self.mongo_client:
            return self._query_generated(spatial, temporal, signal_types)
        
        # All MongoDB queries use C++ backend
        if spatial is None or spatial.component_id is None:
            raise ValueError("Spatial query must include component_id (model_id) for MongoDB queries")

        model_id = spatial.component_id
        
        # Use C++ client for all queries (all filtering happens in C++)
        try:
            import am_qadf_native
            from am_qadf_native import MongoDBQueryClient
            from datetime import datetime
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
            logger.info(f"C++ Query: Creating MongoDBQueryClient for laser parameters (database: {db_name})")
            cpp_client = MongoDBQueryClient(uri, db_name)
            
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
            
            # Query using C++ (all filtering happens in C++)
            logger.info(f"C++ Query: Calling query_laser_monitoring_data for model_id={model_id[:20]}..., "
                       f"layer_range=({layer_start}, {layer_end}), "
                       f"bbox_min={bbox_min}, bbox_max={bbox_max}")
            cpp_result = cpp_client.query_laser_monitoring_data(
                model_id, layer_start, layer_end, bbox_min, bbox_max
            )
            logger.info(f"C++ Query: query_laser_monitoring_data completed, processing result...")
            
            # Convert C++ QueryResult to Python QueryResult
            # Extract points and values (already filtered in C++)
            all_points = [tuple(p) for p in cpp_result.points] if cpp_result.points else []
            all_timestamps = list(cpp_result.timestamps) if cpp_result.timestamps else []
            all_layers = list(cpp_result.layers) if cpp_result.layers else []
            
            # Extract laser temporal sensor data (all extracted in C++)
            laser_temporal_data_list = []
            if hasattr(cpp_result, 'laser_temporal_data') and cpp_result.laser_temporal_data:
                for temporal in cpp_result.laser_temporal_data:
                    laser_temporal_data_list.append({
                        # Process Parameters
                        "commanded_power": temporal.commanded_power,
                        "commanded_scan_speed": temporal.commanded_scan_speed,
                        # Category 3.1 - Laser Power Sensors
                        "actual_power": temporal.actual_power,
                        "power_setpoint": temporal.power_setpoint,
                        "power_error": temporal.power_error,
                        "power_stability": temporal.power_stability,
                        "power_fluctuation_amplitude": temporal.power_fluctuation_amplitude,
                        "power_fluctuation_frequency": temporal.power_fluctuation_frequency,
                        # Category 3.2 - Beam Temporal Characteristics
                        "pulse_frequency": temporal.pulse_frequency,
                        "pulse_duration": temporal.pulse_duration,
                        "pulse_energy": temporal.pulse_energy,
                        "duty_cycle": temporal.duty_cycle,
                        "beam_modulation_frequency": temporal.beam_modulation_frequency,
                        # Category 3.3 - Laser System Health
                        "laser_temperature": temporal.laser_temperature,
                        "laser_cooling_water_temp": temporal.laser_cooling_water_temp,
                        "laser_cooling_flow_rate": temporal.laser_cooling_flow_rate,
                        "laser_power_supply_voltage": temporal.laser_power_supply_voltage,
                        "laser_power_supply_current": temporal.laser_power_supply_current,
                        "laser_diode_current": temporal.laser_diode_current,
                        "laser_diode_temperature": temporal.laser_diode_temperature,
                        "laser_operating_hours": temporal.laser_operating_hours,
                        "laser_pulse_count": temporal.laser_pulse_count,
                    })
            
            # Build signals dictionary from structured data (extract ALL available signals)
            signals = {}
            if laser_temporal_data_list:
                n_temporal = len(laser_temporal_data_list)
                # Time/sample index for 2D time series: use timestamps if same length, else 0..n-1
                if all_timestamps and len(all_timestamps) >= n_temporal:
                    signals["time"] = [float(t) for t in all_timestamps[:n_temporal]]
                else:
                    signals["time"] = [float(i) for i in range(n_temporal)]

                # Per-point layer_index: signal comes with points, timestamp, and layer_index from backend.
                # Only set when C++ returns layers (no fallback).
                if all_layers and len(all_layers) >= n_temporal:
                    layer_arr = [int(all_layers[i]) for i in range(n_temporal)]
                    signals["layer_index"] = layer_arr
                    signals["layer"] = layer_arr

                # Extract power, velocity, energy from structured data
                power_values = [t.get("commanded_power", 0.0) for t in laser_temporal_data_list]
                velocity_values = [t.get("commanded_scan_speed", 0.0) for t in laser_temporal_data_list]
                # Calculate energy density: power / (speed * hatch_spacing) or power / speed
                energy_values = []
                for t in laser_temporal_data_list:
                    power = t.get("commanded_power", 0.0)
                    speed = t.get("commanded_scan_speed", 0.0)
                    if speed > 0:
                        energy_values.append(power / speed)  # J/mm² approximation
                    else:
                        energy_values.append(0.0)
                
                # Store signals with descriptive names (no duplicates)
                if signal_types is None or SignalType.POWER in signal_types:
                    signals["commanded_power"] = power_values  # Setpoint/commanded laser power (Watts)
                if signal_types is None or SignalType.VELOCITY in signal_types:
                    signals["commanded_scan_speed"] = velocity_values  # Setpoint/commanded scan speed (mm/s)
                if signal_types is None or SignalType.ENERGY in signal_types:
                    signals["commanded_energy"] = energy_values  # Calculated from commanded_power/commanded_scan_speed (J/mm² approximation)
                
                # Extract ALL temporal sensor signals (Category 3.1 - Laser Power Sensors)
                signals["actual_power"] = [t.get("actual_power", 0.0) for t in laser_temporal_data_list]
                signals["power_setpoint"] = [t.get("power_setpoint", 0.0) for t in laser_temporal_data_list]
                signals["power_error"] = [t.get("power_error", 0.0) for t in laser_temporal_data_list]
                signals["power_stability"] = [t.get("power_stability", 0.0) for t in laser_temporal_data_list]
                signals["power_fluctuation_amplitude"] = [t.get("power_fluctuation_amplitude", 0.0) for t in laser_temporal_data_list]
                signals["power_fluctuation_frequency"] = [t.get("power_fluctuation_frequency", 0.0) for t in laser_temporal_data_list]
                
                # Extract ALL beam temporal characteristics (Category 3.2)
                signals["pulse_frequency"] = [t.get("pulse_frequency", 0.0) for t in laser_temporal_data_list]
                signals["pulse_duration"] = [t.get("pulse_duration", 0.0) for t in laser_temporal_data_list]
                signals["pulse_energy"] = [t.get("pulse_energy", 0.0) for t in laser_temporal_data_list]
                signals["duty_cycle"] = [t.get("duty_cycle", 0.0) for t in laser_temporal_data_list]
                signals["beam_modulation_frequency"] = [t.get("beam_modulation_frequency", 0.0) for t in laser_temporal_data_list]
                
                # Extract ALL laser system health signals (Category 3.3)
                signals["laser_temperature"] = [t.get("laser_temperature", 0.0) for t in laser_temporal_data_list]
                signals["laser_cooling_water_temp"] = [t.get("laser_cooling_water_temp", 0.0) for t in laser_temporal_data_list]
                signals["laser_cooling_flow_rate"] = [t.get("laser_cooling_flow_rate", 0.0) for t in laser_temporal_data_list]
                signals["laser_power_supply_voltage"] = [t.get("laser_power_supply_voltage", 0.0) for t in laser_temporal_data_list]
                signals["laser_power_supply_current"] = [t.get("laser_power_supply_current", 0.0) for t in laser_temporal_data_list]
                signals["laser_diode_current"] = [t.get("laser_diode_current", 0.0) for t in laser_temporal_data_list]
                signals["laser_diode_temperature"] = [t.get("laser_diode_temperature", 0.0) for t in laser_temporal_data_list]
                signals["laser_operating_hours"] = [t.get("laser_operating_hours", 0.0) for t in laser_temporal_data_list]
                signals["laser_pulse_count"] = [float(t.get("laser_pulse_count", 0)) for t in laser_temporal_data_list]
            
            # Align points with signal length for 2D (spatial heatmap requires points.length === signal.length)
            n_sig = len(laser_temporal_data_list)
            if n_sig > 0 and len(all_points) != n_sig:
                if len(all_points) > n_sig:
                    all_points = all_points[:n_sig]
                elif all_points:
                    all_points = list(all_points) + [all_points[-1]] * (n_sig - len(all_points))

            return QueryResult(
                points=all_points,
                signals=signals,
                metadata={
                    "model_id": model_id,
                    "source": "mongodb_cpp",  # Indicates C++ backend
                    "n_points": len(all_points),
                    "has_laser_temporal_data": len(laser_temporal_data_list) > 0,
                    "laser_temporal_fields_extracted_in_cpp": True,  # All extraction in C++
                },
                component_id=model_id,
            )
            
        except ImportError:
            raise ImportError("am_qadf_native module not available. Install C++ extensions for laser parameter queries.")

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

        # Collect points and signals (include layer for 2D Analysis Panel 4: signal vs layer)
        all_points = []
        all_signals = {}
        all_signals["layer"] = []
        if SignalType.POWER in signal_types:
            all_signals["commanded_power"] = []
        if SignalType.VELOCITY in signal_types:
            all_signals["commanded_scan_speed"] = []
        if SignalType.ENERGY in signal_types:
            all_signals["commanded_energy"] = []

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
                            all_signals["layer"].append(layer_idx)

                            # Add signal values
                            if SignalType.POWER in signal_types:
                                all_signals["commanded_power"].append(power)
                            if SignalType.VELOCITY in signal_types:
                                all_signals["commanded_scan_speed"].append(velocity)
                            if SignalType.ENERGY in signal_types:
                                all_signals["commanded_energy"].append(energy)

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
        """Get total number of layers (uses C++ backend)."""
        if self.use_mongodb and self.mongo_client:
            # Use C++ backend to get layer count
            # Note: This requires a model_id, so we can't get a general count
            # Return 0 if no model_id provided (or use a default query)
            return 0
        return len(self.generated_layers)

    def get_build_styles(self) -> Dict:
        """Get dictionary of build styles."""
        return self.generated_build_styles.copy()

    # MongoDB-specific helper methods (all use C++ backend)
    def get_points(self, model_id: str, filters: Optional[Dict] = None) -> np.ndarray:
        """
        Get all points for a model (uses C++ backend).

        Args:
            model_id: Model UUID
            filters: Optional filters (layer_range, bbox) - converted to C++ query parameters

        Returns:
            Numpy array of points with shape (n, 3)
        """
        if not self.use_mongodb:
            raise RuntimeError("get_points() requires MongoDB mode. Call set_mongo_client() first.")

        try:
            cpp_client = self._get_cpp_client()
            
            # Extract layer range and bbox from filters
            layer_start = -1
            layer_end = -1
            bbox_min = [float('-inf'), float('-inf'), float('-inf')]
            bbox_max = [float('inf'), float('inf'), float('inf')]
            
            if filters:
                if "layer_range" in filters:
                    layer_start, layer_end = filters["layer_range"]
                if "bbox_min" in filters and "bbox_max" in filters:
                    bbox_min = list(filters["bbox_min"])
                    bbox_max = list(filters["bbox_max"])
            
            # Query using C++ backend
            cpp_result = cpp_client.query_laser_monitoring_data(
                model_id,
                layer_start,
                layer_end,
                bbox_min,
                bbox_max
            )
            
            # Extract points from C++ result
            if cpp_result.points:
                return np.array(cpp_result.points, dtype=np.float64)
            else:
                return np.array([], dtype=np.float64).reshape(0, 3)
        except ImportError:
            raise ImportError("am_qadf_native module not available. Install C++ extensions for laser parameter queries.")

    def get_power(self, model_id: str, filters: Optional[Dict] = None) -> np.ndarray:
        """Get laser power values for a model (uses C++ backend)."""
        if not self.use_mongodb:
            raise RuntimeError("get_power() requires MongoDB mode. Call set_mongo_client() first.")

        try:
            cpp_client = self._get_cpp_client()
            
            # Extract layer range and bbox from filters
            layer_start = -1
            layer_end = -1
            bbox_min = [float('-inf'), float('-inf'), float('-inf')]
            bbox_max = [float('inf'), float('inf'), float('inf')]
            
            if filters:
                if "layer_range" in filters:
                    layer_start, layer_end = filters["layer_range"]
                if "bbox_min" in filters and "bbox_max" in filters:
                    bbox_min = list(filters["bbox_min"])
                    bbox_max = list(filters["bbox_max"])
            
            # Query using C++ backend
            cpp_result = cpp_client.query_laser_monitoring_data(
                model_id,
                layer_start,
                layer_end,
                bbox_min,
                bbox_max
            )
            
            # Extract power values from laser_temporal_data
            power_values = []
            if hasattr(cpp_result, 'laser_temporal_data') and cpp_result.laser_temporal_data:
                power_values = [t.commanded_power for t in cpp_result.laser_temporal_data]
            
            return np.array(power_values, dtype=np.float64) if power_values else np.array([], dtype=np.float64)
        except ImportError:
            raise ImportError("am_qadf_native module not available. Install C++ extensions for laser parameter queries.")

    def get_velocity(self, model_id: str, filters: Optional[Dict] = None) -> np.ndarray:
        """Get scan speed values for a model (uses C++ backend)."""
        if not self.use_mongodb:
            raise RuntimeError("get_velocity() requires MongoDB mode. Call set_mongo_client() first.")

        try:
            cpp_client = self._get_cpp_client()
            
            # Extract layer range and bbox from filters
            layer_start = -1
            layer_end = -1
            bbox_min = [float('-inf'), float('-inf'), float('-inf')]
            bbox_max = [float('inf'), float('inf'), float('inf')]
            
            if filters:
                if "layer_range" in filters:
                    layer_start, layer_end = filters["layer_range"]
                if "bbox_min" in filters and "bbox_max" in filters:
                    bbox_min = list(filters["bbox_min"])
                    bbox_max = list(filters["bbox_max"])
            
            # Query using C++ backend
            cpp_result = cpp_client.query_laser_monitoring_data(
                model_id,
                layer_start,
                layer_end,
                bbox_min,
                bbox_max
            )
            
            # Extract velocity values from laser_temporal_data
            velocity_values = []
            if hasattr(cpp_result, 'laser_temporal_data') and cpp_result.laser_temporal_data:
                velocity_values = [t.commanded_scan_speed for t in cpp_result.laser_temporal_data]
            
            return np.array(velocity_values, dtype=np.float64) if velocity_values else np.array([], dtype=np.float64)
        except ImportError:
            raise ImportError("am_qadf_native module not available. Install C++ extensions for laser parameter queries.")

    def get_energy_density(self, model_id: str, filters: Optional[Dict] = None) -> np.ndarray:
        """Get energy density values for a model (uses C++ backend)."""
        if not self.use_mongodb:
            raise RuntimeError("get_energy_density() requires MongoDB mode. Call set_mongo_client() first.")

        try:
            cpp_client = self._get_cpp_client()
            
            # Extract layer range and bbox from filters
            layer_start = -1
            layer_end = -1
            bbox_min = [float('-inf'), float('-inf'), float('-inf')]
            bbox_max = [float('inf'), float('inf'), float('inf')]
            
            if filters:
                if "layer_range" in filters:
                    layer_start, layer_end = filters["layer_range"]
                if "bbox_min" in filters and "bbox_max" in filters:
                    bbox_min = list(filters["bbox_min"])
                    bbox_max = list(filters["bbox_max"])
            
            # Query using C++ backend
            cpp_result = cpp_client.query_laser_monitoring_data(
                model_id,
                layer_start,
                layer_end,
                bbox_min,
                bbox_max
            )
            
            # Calculate energy density from laser_temporal_data: power / speed
            energy_values = []
            if hasattr(cpp_result, 'laser_temporal_data') and cpp_result.laser_temporal_data:
                for t in cpp_result.laser_temporal_data:
                    if t.commanded_scan_speed > 0:
                        energy_values.append(t.commanded_power / t.commanded_scan_speed)
                    else:
                        energy_values.append(0.0)
            
            return np.array(energy_values, dtype=np.float64) if energy_values else np.array([], dtype=np.float64)
        except ImportError:
            raise ImportError("am_qadf_native module not available. Install C++ extensions for laser parameter queries.")

    def aggregate_by_layer(self, model_id: str) -> Dict[int, Dict[str, float]]:
        """
        Aggregate laser parameters by layer (uses C++ backend).

        Args:
            model_id: Model UUID

        Returns:
            Dictionary mapping layer_index to aggregated values
        """
        if not self.use_mongodb:
            raise RuntimeError("aggregate_by_layer() requires MongoDB mode. Call set_mongo_client() first.")

        try:
            cpp_client = self._get_cpp_client()
            
            # Query all data using C++ backend
            cpp_result = cpp_client.query_laser_monitoring_data(
                model_id,
                -1, -1,  # All layers
                [float('-inf'), float('-inf'), float('-inf')],
                [float('inf'), float('inf'), float('inf')]
            )
            
            # Aggregate by layer from C++ result
            layer_data = {}
            if hasattr(cpp_result, 'laser_temporal_data') and cpp_result.laser_temporal_data:
                for i, temporal in enumerate(cpp_result.laser_temporal_data):
                    layer_idx = cpp_result.layers[i] if i < len(cpp_result.layers) else 0
                    
                    if layer_idx not in layer_data:
                        layer_data[layer_idx] = {
                            "power": [],
                            "velocity": [],
                            "energy": []
                        }
                    
                    layer_data[layer_idx]["power"].append(temporal.commanded_power)
                    layer_data[layer_idx]["velocity"].append(temporal.commanded_scan_speed)
                    if temporal.commanded_scan_speed > 0:
                        layer_data[layer_idx]["energy"].append(temporal.commanded_power / temporal.commanded_scan_speed)
                    else:
                        layer_data[layer_idx]["energy"].append(0.0)
            
            # Calculate aggregates
            result = {}
            for layer_idx, data in layer_data.items():
                result[layer_idx] = {
                    "power_avg": float(np.mean(data["power"])) if data["power"] else 0.0,
                    "power_min": float(np.min(data["power"])) if data["power"] else 0.0,
                    "power_max": float(np.max(data["power"])) if data["power"] else 0.0,
                    "velocity_avg": float(np.mean(data["velocity"])) if data["velocity"] else 0.0,
                    "velocity_min": float(np.min(data["velocity"])) if data["velocity"] else 0.0,
                    "velocity_max": float(np.max(data["velocity"])) if data["velocity"] else 0.0,
                    "energy_avg": float(np.mean(data["energy"])) if data["energy"] else 0.0,
                    "energy_min": float(np.min(data["energy"])) if data["energy"] else 0.0,
                    "energy_max": float(np.max(data["energy"])) if data["energy"] else 0.0,
                }
            
            return result
        except ImportError:
            raise ImportError("am_qadf_native module not available. Install C++ extensions for laser parameter queries.")


