"""
Hatching Layer Query Client

Query client for hatching layers and laser scan paths from MongoDB data warehouse.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)


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
        self._available_signals = [
            SignalType.POWER,
            SignalType.VELOCITY,
            SignalType.ENERGY,
        ]

    def set_mongo_client(self, mongo_client):
        """Set MongoDB client instance."""
        self.mongo_client = mongo_client

    def get_connection_info(self) -> Tuple[str, str]:
        """
        Get (uri, db_name) for the current MongoDB connection.
        Used by C++ visualization API and other callers that need connection params.
        """
        uri = "mongodb://localhost:27017"
        db_name = "am_qadf"
        if self.mongo_client:
            if hasattr(self.mongo_client, "config") and hasattr(self.mongo_client.config, "url"):
                uri = self.mongo_client.config.url
                if hasattr(self.mongo_client.config, "database"):
                    db_name = self.mongo_client.config.database
                elif hasattr(self.mongo_client, "_database") and self.mongo_client._database:
                    db_name = self.mongo_client._database.name
            else:
                import os
                env_uri = os.getenv("MONGODB_URI") or os.getenv("MONGODB_URL")
                env_db = os.getenv("MONGODB_DB") or os.getenv("MONGO_DATABASE", "am_qadf_data")
                if env_uri:
                    uri = env_uri
                    db_name = env_db
                elif hasattr(self.mongo_client, "client") and hasattr(self.mongo_client.client, "address"):
                    host, port = self.mongo_client.client.address
                    uri = f"mongodb://{host}:{port}"
        return (uri, db_name)

    def _get_cpp_client(self):
        """Get C++ MongoDB query client (internal helper)."""
        try:
            import am_qadf_native
            from am_qadf_native import MongoDBQueryClient

            uri, db_name = self.get_connection_info()
            return MongoDBQueryClient(uri, db_name)
        except ImportError:
            raise ImportError("am_qadf_native module not available. Install C++ extensions for hatching queries.")

    def get_layers(self, model_id: str, layer_range: Optional[Tuple[int, int]] = None) -> List[Dict[str, Any]]:
        """
        Get hatching layers for a model (uses C++ backend).

        Args:
            model_id: Model UUID
            layer_range: Optional (start_layer, end_layer) tuple

        Returns:
            List of layer documents (reconstructed from C++ query result)
        """
        try:
            cpp_client = self._get_cpp_client()
            
            # Prepare layer range
            layer_start = -1
            layer_end = -1
            if layer_range:
                layer_start, layer_end = layer_range
            
            # Query using C++ backend
            cpp_result = cpp_client.query_hatching_data(
                model_id,
                layer_start,
                layer_end,
                [float('-inf'), float('-inf'), float('-inf')],
                [float('inf'), float('inf'), float('inf')]
            )
            
            # Reconstruct layer documents from C++ result
            # Group by layer_index from vectordata
            layers_dict = {}
            
            # First, process vectordata to create layer structure
            if hasattr(cpp_result, 'vectordata') and cpp_result.vectordata:
                for vd in cpp_result.vectordata:
                    layer_idx = vd.layer_index
                    if layer_idx not in layers_dict:
                        layers_dict[layer_idx] = {
                            "model_id": model_id,
                            "layer_index": layer_idx,
                            "vectors": [],
                            "vectordata": []
                        }
                    layers_dict[layer_idx]["vectordata"].append({
                        "dataindex": vd.dataindex,
                        "partid": vd.partid,
                        "type": vd.type,
                        "scanner": vd.scanner,
                        "laserpower": vd.laserpower,
                        "scannerspeed": vd.scannerspeed,
                        "laser_beam_width": vd.laser_beam_width,
                        "hatch_spacing": vd.hatch_spacing,
                        "layer_index": vd.layer_index
                    })
            
            # Then, process vectors and associate them with layers via dataindex
            if hasattr(cpp_result, 'vectors') and cpp_result.vectors:
                # Create a map from dataindex to layer_index
                dataindex_to_layer = {}
                for layer_idx, layer_data in layers_dict.items():
                    for vd in layer_data["vectordata"]:
                        dataindex_to_layer[vd["dataindex"]] = layer_idx
                
                for vec in cpp_result.vectors:
                    # Find layer_index from vectordata using dataindex
                    layer_idx = dataindex_to_layer.get(vec.dataindex)
                    if layer_idx is None:
                        # If no vectordata match, try to infer from layers array
                        # Use the layer_index from the layers array at the same position
                        if cpp_result.layers and len(cpp_result.layers) > 0:
                            # Find layer by matching z coordinate approximately
                            vec_z = vec.z
                            matching_layer = None
                            for i, layer_idx_candidate in enumerate(cpp_result.layers):
                                if layer_idx_candidate not in layers_dict:
                                    layers_dict[layer_idx_candidate] = {
                                        "model_id": model_id,
                                        "layer_index": layer_idx_candidate,
                                        "vectors": [],
                                        "vectordata": []
                                    }
                                matching_layer = layer_idx_candidate
                                break  # Use first available layer
                            layer_idx = matching_layer if matching_layer is not None else 0
                        else:
                            layer_idx = 0
                    
                    if layer_idx not in layers_dict:
                        layers_dict[layer_idx] = {
                            "model_id": model_id,
                            "layer_index": layer_idx,
                            "vectors": [],
                            "vectordata": []
                        }
                    
                    layers_dict[layer_idx]["vectors"].append({
                        "x1": vec.x1, "y1": vec.y1, "x2": vec.x2, "y2": vec.y2,
                        "z": vec.z, "timestamp": vec.timestamp, "dataindex": vec.dataindex
                    })
            
            # If no vectors/vectordata, create layers from layers array
            if not layers_dict and cpp_result.layers:
                for layer_idx in set(cpp_result.layers):
                    layers_dict[layer_idx] = {
                        "model_id": model_id,
                        "layer_index": layer_idx,
                        "vectors": [],
                        "vectordata": []
                    }

            # Sort each layer's vectors and vectordata by scan order, matching pyslm hatching pattern.
            # pyslm default: scanContourFirst=False → hatch first, then contour (hatching.py lines 881–884).
            # Hatch lines are ordered spatially (line 1, line 2, ...) so the path is sequential.
            def _type_sort_key(t: str) -> int:
                if t is None:
                    return 1
                t = (t or "").lower()
                if t in ("outer", "inner", "contour"):
                    return 1  # contour second (drawn after hatches when scan_contour_first=False)
                return 0  # hatch first

            for layer_data in layers_dict.values():
                vd_list = layer_data.get("vectordata", [])
                vec_list = layer_data.get("vectors", [])
                if not vec_list:
                    continue
                # Build dataindex -> type from vectordata
                di_to_type = {vd.get("dataindex"): vd.get("type") for vd in vd_list}
                # Align vec and vd by index (vd_list may be shorter; match by dataindex)
                vd_by_di = {vd.get("dataindex"): vd for vd in vd_list}
                pairs = []
                for v in vec_list:
                    di = v.get("dataindex")
                    vd = vd_by_di.get(di, {}) if vd_list else {}
                    pairs.append((v, vd))
                # Split into hatch-type and contour-type (pyslm default: hatch first, then contour)
                hatch_pairs = [(v, vd) for v, vd in pairs if _type_sort_key(vd.get("type")) == 0]
                contour_pairs = [(v, vd) for v, vd in pairs if _type_sort_key(vd.get("type")) != 0]
                # Sort contours by dataindex (scan order along contour)
                contour_pairs.sort(key=lambda p: (p[0].get("dataindex") is None, p[0].get("dataindex") or 0))
                # Sort hatch vectors by spatial position so adjacent hatch lines are consecutive
                # (perpendicular to hatch direction = order along the hatch spacing direction).
                if hatch_pairs:
                    dx_sum, dy_sum = 0.0, 0.0
                    for v, _ in hatch_pairs:
                        x1, y1 = float(v.get("x1", 0)), float(v.get("y1", 0))
                        x2, y2 = float(v.get("x2", 0)), float(v.get("y2", 0))
                        dx_sum += x2 - x1
                        dy_sum += y2 - y1
                    n_hatch = len(hatch_pairs)
                    if n_hatch > 0 and (dx_sum != 0 or dy_sum != 0):
                        # Perpendicular to average hatch direction (normalized)
                        ax, ay = dx_sum / n_hatch, dy_sum / n_hatch
                        length = (ax * ax + ay * ay) ** 0.5
                        if length > 1e-10:
                            perp_x, perp_y = -ay / length, ax / length
                            hatch_pairs.sort(
                                key=lambda p: (
                                    (float(p[0].get("x1", 0)) + float(p[0].get("x2", 0))) / 2 * perp_x
                                    + (float(p[0].get("y1", 0)) + float(p[0].get("y2", 0))) / 2 * perp_y,
                                    (float(p[0].get("x1", 0)) + float(p[0].get("x2", 0))) / 2,
                                    (float(p[0].get("y1", 0)) + float(p[0].get("y2", 0))) / 2,
                                )
                            )
                        else:
                            hatch_pairs.sort(
                                key=lambda p: (
                                    (float(p[0].get("y1", 0)) + float(p[0].get("y2", 0))) / 2,
                                    (float(p[0].get("x1", 0)) + float(p[0].get("x2", 0))) / 2,
                                )
                            )
                    else:
                        hatch_pairs.sort(
                            key=lambda p: (
                                (float(p[0].get("y1", 0)) + float(p[0].get("y2", 0))) / 2,
                                (float(p[0].get("x1", 0)) + float(p[0].get("x2", 0))) / 2,
                            )
                        )
                # Rebuild in order: hatches first (spatially ordered), then contours (pyslm default)
                ordered_pairs = hatch_pairs + contour_pairs
                layer_data["vectors"] = [p[0] for p in ordered_pairs]
                layer_data["vectordata"] = [p[1] for p in ordered_pairs]  # keep 1:1 with vectors (p[1] may be {})

            # Convert to sorted list
            return sorted(layers_dict.values(), key=lambda x: x["layer_index"])
        except ImportError:
            raise ImportError("am_qadf_native module not available. Install C++ extensions for hatching queries.")

    def get_layer(self, model_id: str, layer_index: int) -> Optional[Dict[str, Any]]:
        """
        Get a specific hatching layer (uses C++ backend).

        Args:
            model_id: Model UUID
            layer_index: Layer index

        Returns:
            Layer document or None if not found
        """
        layers = self.get_layers(model_id, layer_range=(layer_index, layer_index))
        return layers[0] if layers else None

    def get_hatch_paths(self, model_id: str, layer_index: int) -> List[Dict[str, Any]]:
        """
        Get hatch paths for a specific layer (uses C++ backend).

        Args:
            model_id: Model UUID
            layer_index: Layer index

        Returns:
            List of hatch path dictionaries (reconstructed from vectors)
        """
        layer = self.get_layer(model_id, layer_index)
        if layer is None:
            return []
        
        # Reconstruct hatch paths from vectors/vectordata
        hatches = []
        if "vectors" in layer and "vectordata" in layer:
            vectordata_map = {vd["dataindex"]: vd for vd in layer["vectordata"]}
            current_hatch = None
            
            for vec in layer["vectors"]:
                vd = vectordata_map.get(vec["dataindex"], {})
                hatch_type = vd.get("type", "unknown")
                
                # Group vectors by type into hatches
                if current_hatch is None or current_hatch.get("hatch_type") != hatch_type:
                    if current_hatch:
                        hatches.append(current_hatch)
                    current_hatch = {
                        "hatch_type": hatch_type,
                        "points": [],
                        "laser_power": vd.get("laserpower", 0.0),
                        "scan_speed": vd.get("scannerspeed", 0.0),
                        "laser_beam_width": vd.get("laser_beam_width", 0.0),
                        "hatch_spacing": vd.get("hatch_spacing", 0.0),
                    }
                
                # Add points from vector
                current_hatch["points"].append((vec["x1"], vec["y1"], vec["z"]))
                current_hatch["points"].append((vec["x2"], vec["y2"], vec["z"]))
            
            if current_hatch:
                hatches.append(current_hatch)
        
        return hatches

    def get_all_points(self, model_id: str, layer_range: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Get all laser path points for a model (uses C++ backend).

        Args:
            model_id: Model UUID
            layer_range: Optional (start_layer, end_layer) tuple

        Returns:
            Numpy array of points with shape (n, 3)
        """
        try:
            cpp_client = self._get_cpp_client()
            
            # Prepare layer range
            layer_start = -1
            layer_end = -1
            if layer_range:
                layer_start, layer_end = layer_range
            
            # Query using C++ backend
            cpp_result = cpp_client.query_hatching_data(
                model_id,
                layer_start,
                layer_end,
                [float('-inf'), float('-inf'), float('-inf')],
                [float('inf'), float('inf'), float('inf')]
            )
            
            # Extract points from C++ result
            if cpp_result.points:
                return np.array(cpp_result.points, dtype=np.float64)
            else:
                return np.array([], dtype=np.float64).reshape(0, 3)
        except ImportError:
            raise ImportError("am_qadf_native module not available. Install C++ extensions for hatching queries.")

    def query_spatial(
        self,
        model_id: str,
        bbox: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
    ) -> List[Dict[str, Any]]:
        """
        Query hatch paths within a spatial bounding box (uses C++ backend).

        Args:
            model_id: Model UUID
            bbox: Bounding box ((x_min, y_min, z_min), (x_max, y_max, z_max))

        Returns:
            List of hatch path dictionaries within the bounding box
        """
        bbox_min, bbox_max = bbox
        
        # Use main query method with spatial filtering
        from .base_query_client import SpatialQuery
        spatial_query = SpatialQuery(
            component_id=model_id,
            bbox_min=bbox_min,
            bbox_max=bbox_max
        )
        
        result = self.query(spatial=spatial_query)
        
        # Reconstruct hatch paths from query result
        # Group points by layer and create hatch dictionaries
        hatches = []
        if result.points:
            # For now, return a simplified structure
            # Full hatch reconstruction would require more complex grouping
            hatches.append({
                "points": result.points,
                "signals": result.signals
            })
        
        return hatches

    def get_coordinate_system(self, model_id: str, layer_index: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Get coordinate system information for a layer (uses C++ backend).

        Args:
            model_id: Model UUID
            layer_index: Optional layer index (if None, gets from first layer)

        Returns:
            Coordinate system dictionary or None
        """
        # Get layer using C++ backend
        if layer_index is None:
            layers = self.get_layers(model_id, layer_range=(0, 0))
            if not layers:
                return None
            layer = layers[0]
        else:
            layer = self.get_layer(model_id, layer_index)
            if layer is None:
                return None
        
        # Extract coordinate system from layer document
        return extract_coordinate_system(layer)

    def query(
        self,
        spatial: Optional[SpatialQuery] = None,
        temporal: Optional[TemporalQuery] = None,
        signal_types: Optional[List[SignalType]] = None,
    ) -> QueryResult:
        """
        Query hatching data using C++ backend (required for performance).

        All queries use C++ backend for maximum performance. No Python fallback.

        Args:
            spatial: Spatial query parameters (requires component_id = model_id)
            temporal: Temporal query parameters (layer range)
            signal_types: Signal types to retrieve (POWER, VELOCITY, ENERGY)

        Returns:
            QueryResult with points and signals

        Raises:
            ImportError: If C++ backend (am_qadf_native) is not available
            RuntimeError: If C++ query fails
        """
        if spatial is None or spatial.component_id is None:
            raise ValueError("Spatial query must include component_id (model_id)")

        model_id = spatial.component_id
        layer_range = None

        if temporal and temporal.layer_start is not None and temporal.layer_end is not None:
            layer_range = (temporal.layer_start, temporal.layer_end)
        elif spatial and spatial.layer_range:
            layer_range = spatial.layer_range

        # All queries use C++ backend for performance (required)
        # C++ supports both vector-based and point-based formats
        try:
            import am_qadf_native
            from am_qadf_native import MongoDBQueryClient
            
            # Get MongoDB connection info with authentication
            # Try to get from mongo_client, otherwise use defaults
            uri = "mongodb://localhost:27017"
            db_name = "am_qadf"
            
            if self.mongo_client:
                # Try to extract connection info from mongo_client
                # Priority: config.url (includes auth) > environment variables > client.address (no auth)
                if hasattr(self.mongo_client, 'config') and hasattr(self.mongo_client.config, 'url'):
                    # MongoDBClient wrapper - get full URI with authentication
                    uri = self.mongo_client.config.url
                    if hasattr(self.mongo_client.config, 'database'):
                        db_name = self.mongo_client.config.database
                    elif hasattr(self.mongo_client, '_database') and self.mongo_client._database:
                        db_name = self.mongo_client._database.name
                    logger.debug(f"Using MongoDB URI from config: {uri[:50]}... (database: {db_name})")
                else:
                    # Fallback: try environment variables
                    import os
                    env_uri = os.getenv('MONGODB_URI') or os.getenv('MONGODB_URL')
                    env_db = os.getenv('MONGODB_DB') or os.getenv('MONGO_DATABASE', 'am_qadf_data')
                    
                    if env_uri:
                        uri = env_uri
                        db_name = env_db
                        logger.debug(f"Using MongoDB URI from environment: {uri[:50]}... (database: {db_name})")
                    elif hasattr(self.mongo_client, 'client') and hasattr(self.mongo_client.client, 'address'):
                        host, port = self.mongo_client.client.address
                        uri = f"mongodb://{host}:{port}"
                    elif hasattr(self.mongo_client, 'get_database'):
                        # Custom MongoDBClient wrapper - try to get database name
                        try:
                            db = self.mongo_client.get_database("am_qadf_data")
                            db_name = db.name
                        except:
                            pass
            
            # Create C++ query client
            logger.info(f"C++ Query: Creating MongoDBQueryClient for database: {db_name}")
            cpp_client = MongoDBQueryClient(uri, db_name)
            logger.info(f"C++ Query: Client created successfully")
            
            # Prepare layer range
            layer_start = -1
            layer_end = -1
            if layer_range:
                layer_start, layer_end = layer_range
            
            # Prepare bbox
            bbox_min = [float('-inf'), float('-inf'), float('-inf')]
            bbox_max = [float('inf'), float('inf'), float('inf')]
            if spatial and spatial.bbox_min and spatial.bbox_max:
                bbox_min = list(spatial.bbox_min)
                bbox_max = list(spatial.bbox_max)
            
            logger.info(f"C++ Query: Calling query_hatching_data for model_id={model_id[:20]}..., "
                       f"layer_range=({layer_start}, {layer_end}), "
                       f"bbox_min={bbox_min}, bbox_max={bbox_max}")
            
            # Query using C++
            cpp_result = cpp_client.query_hatching_data(
                model_id,
                layer_start,
                layer_end,
                bbox_min,
                bbox_max
            )
            
            logger.info(f"C++ Query: query_hatching_data completed, processing result...")
            
            # Check if C++ result has vector-based format
            has_vectors_cpp = hasattr(cpp_result, 'vectors') and len(cpp_result.vectors) > 0
            
            if has_vectors_cpp:
                # NEW FORMAT: Vector-based (C++ now supports it!)
                # Convert C++ vectors to Python format
                all_vectors = []
                for vec in cpp_result.vectors:
                    all_vectors.append({
                        "x1": vec.x1, "y1": vec.y1, "x2": vec.x2, "y2": vec.y2,
                        "z": vec.z, "timestamp": vec.timestamp, "dataindex": vec.dataindex
                    })
                
                # Convert C++ vectordata to Python format
                all_vectordata = []
                for vd in cpp_result.vectordata:
                    all_vectordata.append({
                        "dataindex": vd.dataindex,
                        "partid": vd.partid,
                        "type": vd.type,
                        "scanner": vd.scanner,
                        "laserpower": vd.laserpower,
                        "scannerspeed": vd.scannerspeed,
                        "laser_beam_width": vd.laser_beam_width,
                        "hatch_spacing": vd.hatch_spacing,
                        "layer_index": vd.layer_index
                    })
                
                # Extract points from vectors for backward compatibility
                points = []
                power_values = []
                velocity_values = []
                energy_values = []
                layer_values = []
                
                # Create vectordata lookup map
                vectordata_map = {vd["dataindex"]: vd for vd in all_vectordata}
                
                for vec in all_vectors:
                    points.append((vec["x1"], vec["y1"], vec["z"]))
                    points.append((vec["x2"], vec["y2"], vec["z"]))
                    
                    vd = vectordata_map.get(vec["dataindex"], {})
                    power = vd.get("laserpower", 0.0)
                    velocity = vd.get("scannerspeed", 0.0)
                    energy = power / (velocity + 1e-6) if velocity > 0 else 0.0
                    layer_index = vd.get("layer_index", 0)
                    
                    power_values.extend([power, power])
                    velocity_values.extend([velocity, velocity])
                    energy_values.extend([energy, energy])
                    layer_values.extend([layer_index, layer_index])
                
                signals = {}
                if signal_types is None or SignalType.POWER in signal_types:
                    signals["power"] = power_values
                if signal_types is None or SignalType.VELOCITY in signal_types:
                    signals["velocity"] = velocity_values
                    signals["speed"] = velocity_values  # alias for 2D/frontend "Speed" selector
                if signal_types is None or SignalType.ENERGY in signal_types:
                    signals["energy"] = energy_values
                # Per-point layer index for 2D layer view and power/layer heatmap (layer_index is canonical)
                signals["layer_index"] = layer_values
                signals["layer"] = layer_values
            else:
                # OLD FORMAT: Points-based (deprecated - use vector-based format)
                # Signals not available in old format without vectordata
                signals = {}
                logger.warning("Old point-based format detected. Signals not available. Use vector-based format.")
                
                # Convert points to tuples
                points = [tuple(p) for p in cpp_result.points]
            
            # Convert hatch metadata to Python format (infill types, etc.)
            hatch_metadata_list = []
            for i, hatch_meta in enumerate(cpp_result.hatch_metadata):
                hatch_dict = {
                    "hatch_type": hatch_meta.hatch_type,  # "raster", "line", "infill", etc.
                    "laser_beam_width": hatch_meta.laser_beam_width,
                    "hatch_spacing": hatch_meta.hatch_spacing,
                    "overlap_percentage": hatch_meta.overlap_percentage,
                    "start_index": cpp_result.hatch_start_indices[i] if i < len(cpp_result.hatch_start_indices) else 0,
                }
                hatch_metadata_list.append(hatch_dict)
            
            # Convert contours to Python format (for visualization)
            contours_data = []
            if cpp_result.has_contours():
                for i, contour in enumerate(cpp_result.contours):
                    contour_dict = {
                        "points": [tuple(p) for p in contour.points],
                        "sub_type": contour.sub_type,
                        "subType": contour.sub_type,  # Also include for compatibility
                        "color": contour.color,
                        "linewidth": contour.linewidth,
                        "layer_index": cpp_result.contour_layers[i] if i < len(cpp_result.contour_layers) else 0,
                    }
                    contours_data.append(contour_dict)
            
            # Build metadata
            metadata = {
                "model_id": model_id,
                "source": "mongodb_cpp",  # Indicates C++ backend
                "n_layers": len(set(cpp_result.layers)),
                "n_points": len(points),
                "n_hatches": len(hatch_metadata_list),
                "n_contours": len(contours_data),
                "hatch_metadata": hatch_metadata_list,  # Include hatch/infill metadata
                "contours": contours_data,  # Include contours for visualization
                "cpp_implementation": True,
            }
            
            # Add vector-based format data if available
            if has_vectors_cpp:
                metadata["vectors"] = all_vectors
                metadata["vectordata"] = all_vectordata
                metadata["format"] = getattr(cpp_result, 'format', 'vector-based')
                metadata["n_vectors"] = len(all_vectors)
                metadata["n_vectordata"] = len(all_vectordata)
            
            return QueryResult(
                points=points,
                signals=signals,
                metadata=metadata,
                component_id=model_id,
            )
        except ImportError:
            logger.error("am_qadf_native module not available. C++ backend required for hatching queries.")
            raise ImportError("am_qadf_native module not available. Install C++ extensions for hatching queries.")
        except Exception as e:
            logger.error(f"C++ hatching query failed: {e}")
            raise RuntimeError(f"C++ hatching query failed: {e}") from e

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
