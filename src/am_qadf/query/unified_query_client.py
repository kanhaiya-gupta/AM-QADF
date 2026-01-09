"""
Unified Query Client

Multi-source query client that merges data from all data sources:
- STL models
- Hatching layers
- Laser parameters
- CT scan data
- ISPM monitoring data

Uses coordinate system transformations to align data from different sources.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Handle both relative import (when used as package) and direct import (when loaded directly)
try:
    from .stl_model_client import STLModelClient
    from .hatching_client import HatchingClient
    from .laser_parameter_client import LaserParameterClient
    from .ct_scan_client import CTScanClient
    from .in_situ_monitoring_client import InSituMonitoringClient
    from .base_query_client import SpatialQuery, TemporalQuery
    from ..voxelization.transformer import CoordinateSystemTransformer
except ImportError:
    # Fallback for direct module loading
    import sys
    from pathlib import Path

    current_file = Path(__file__).resolve()
    query_dir = current_file.parent

    # Try to load modules
    modules_to_load = [
        "stl_model_client",
        "hatching_client",
        "laser_parameter_client",
        "ct_scan_client",
        "in_situ_monitoring_client",
        "base_query_client",
    ]

    for module_name in modules_to_load:
        module_path = query_dir / f"{module_name}.py"
        if module_path.exists():
            import importlib.util

            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            sys.modules[module_name] = module

    # Load coordinate system transformer
    voxel_dir = query_dir.parent / "voxelization"
    transformer_path = voxel_dir / "transformer.py"
    if transformer_path.exists():
        import importlib.util

        spec = importlib.util.spec_from_file_location("transformer", transformer_path)
        transformer_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(transformer_module)
        sys.modules["transformer"] = transformer_module
        CoordinateSystemTransformer = transformer_module.CoordinateSystemTransformer
    else:
        CoordinateSystemTransformer = None

    # Load query_utils
    query_utils_path = query_dir / "query_utils.py"
    query_utils_module = None
    if query_utils_path.exists():
        import importlib.util

        spec = importlib.util.spec_from_file_location("query_utils", query_utils_path)
        query_utils_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(query_utils_module)
        sys.modules["query_utils"] = query_utils_module

    # Get classes from loaded modules
    stl_module = sys.modules.get("stl_model_client")
    hatching_module = sys.modules.get("hatching_client")
    laser_module = sys.modules.get("laser_parameter_client")
    ct_module = sys.modules.get("ct_scan_client")
    ispm_module = sys.modules.get("in_situ_monitoring_client")
    base_module = sys.modules.get("base_query_client")

    STLModelClient = getattr(stl_module, "STLModelClient", None) if stl_module else None
    HatchingClient = getattr(hatching_module, "HatchingClient", None) if hatching_module else None
    LaserParameterClient = getattr(laser_module, "LaserParameterClient", None) if laser_module else None
    CTScanClient = getattr(ct_module, "CTScanClient", None) if ct_module else None
    InSituMonitoringClient = getattr(ispm_module, "InSituMonitoringClient", None) if ispm_module else None
    SpatialQuery = getattr(base_module, "SpatialQuery", None) if base_module else None
    TemporalQuery = getattr(base_module, "TemporalQuery", None) if base_module else None


class UnifiedQueryClient:
    """
    Unified query client for multi-source data retrieval and merging.

    Provides a single interface to query and merge data from all sources:
    - STL models
    - Hatching layers
    - Laser parameters
    - CT scan data
    - ISPM monitoring data
    """

    def __init__(self, mongo_client=None):
        """
        Initialize unified query client.

        Args:
            mongo_client: MongoDBClient instance
        """
        self.mongo_client = mongo_client

        # Initialize individual query clients
        if STLModelClient:
            self.stl_client = STLModelClient(mongo_client=mongo_client)
        else:
            self.stl_client = None

        if HatchingClient:
            self.hatching_client = HatchingClient(mongo_client=mongo_client)
        else:
            self.hatching_client = None

        if LaserParameterClient:
            self.laser_client = LaserParameterClient(mongo_client=mongo_client, use_mongodb=True)
        else:
            self.laser_client = None

        if CTScanClient:
            self.ct_client = CTScanClient(mongo_client=mongo_client, use_mongodb=True)
        else:
            self.ct_client = None

        if InSituMonitoringClient:
            self.ispm_client = InSituMonitoringClient(mongo_client=mongo_client, use_mongodb=True)
        else:
            self.ispm_client = None

        # Coordinate system transformer
        if CoordinateSystemTransformer:
            self.coord_transformer = CoordinateSystemTransformer()
        else:
            self.coord_transformer = None

    def set_mongo_client(self, mongo_client):
        """Set MongoDB client for all query clients."""
        self.mongo_client = mongo_client

        if self.stl_client:
            self.stl_client.set_mongo_client(mongo_client)
        if self.hatching_client:
            self.hatching_client.set_mongo_client(mongo_client)
        if self.laser_client:
            self.laser_client.set_mongo_client(mongo_client)
        if self.ct_client:
            self.ct_client.set_mongo_client(mongo_client)
        if self.ispm_client:
            self.ispm_client.set_mongo_client(mongo_client)

    def get_all_data(self, model_id: str) -> Dict[str, Any]:
        """
        Get all data for a model from all sources.

        Args:
            model_id: Model UUID

        Returns:
            Dictionary containing data from all sources
        """
        result = {
            "model_id": model_id,
            "stl_model": None,
            "hatching_layers": None,
            "laser_parameters": None,
            "ct_scan": None,
            "ispm_monitoring": None,
        }

        # Get STL model
        if self.stl_client:
            result["stl_model"] = self.stl_client.get_model(model_id)

        # Get hatching layers
        if self.hatching_client:
            result["hatching_layers"] = self.hatching_client.get_layers(model_id)

        # Get laser parameters (sample)
        if self.laser_client:
            try:
                spatial_query = SpatialQuery(component_id=model_id)
                result["laser_parameters"] = self.laser_client.query(spatial=spatial_query)
            except Exception as e:
                result["laser_parameters"] = {"error": str(e)}

        # Get CT scan data
        if self.ct_client:
            result["ct_scan"] = self.ct_client.get_scan(model_id)

        # Get ISPM monitoring data (sample)
        if self.ispm_client:
            try:
                spatial_query = SpatialQuery(component_id=model_id)
                result["ispm_monitoring"] = self.ispm_client.query(spatial=spatial_query)
            except Exception as e:
                result["ispm_monitoring"] = {"error": str(e)}

        return result

    def merge_spatial_data(
        self,
        model_id: str,
        bbox: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
        sources: List[str] = ["hatching", "laser", "ct", "ispm"],
        target_coord_system: str = "build_platform",
    ) -> Dict[str, Any]:
        """
        Merge spatial data from multiple sources within a bounding box.

        Args:
            model_id: Model UUID
            bbox: Bounding box ((x_min, y_min, z_min), (x_max, y_max, z_max))
            sources: List of sources to include ('hatching', 'laser', 'ct', 'ispm')
            target_coord_system: Target coordinate system for alignment

        Returns:
            Dictionary with merged spatial data
        """
        bbox_min, bbox_max = bbox

        # Get coordinate systems
        coord_systems = {}
        if self.stl_client:
            stl_model = self.stl_client.get_model(model_id)
            if stl_model:
                # Try to import extract_coordinate_system
                extract_coordinate_system = None
                try:
                    from .query_utils import extract_coordinate_system
                except ImportError:
                    try:
                        import sys

                        query_utils_module = sys.modules.get("query_utils")
                        if query_utils_module:
                            extract_coordinate_system = getattr(query_utils_module, "extract_coordinate_system", None)
                    except Exception:
                        pass

                if extract_coordinate_system:
                    coord_systems["stl"] = extract_coordinate_system(stl_model) or {}
                else:
                    # Fallback: try to get from metadata directly
                    coord_systems["stl"] = stl_model.get("metadata", {}).get("coordinate_system", {})

        if "hatching" in sources and self.hatching_client:
            coord_systems["hatching"] = self.hatching_client.get_coordinate_system(model_id) or {}

        if "ct" in sources and self.ct_client:
            coord_systems["ct"] = self.ct_client.get_coordinate_system(model_id) or {}

        if "ispm" in sources and self.ispm_client:
            coord_systems["ispm"] = self.ispm_client.get_coordinate_system(model_id) or {}

        # Use STL/build platform as reference
        reference_system = coord_systems.get("stl", {})

        # Query data from each source
        merged_data = {
            "model_id": model_id,
            "bbox": bbox,
            "target_coord_system": target_coord_system,
            "sources": {},
        }

        # Query hatching data
        if "hatching" in sources and self.hatching_client:
            try:
                spatial_query = SpatialQuery(component_id=model_id, bbox_min=bbox_min, bbox_max=bbox_max)
                hatching_result = self.hatching_client.query(spatial=spatial_query)
                merged_data["sources"]["hatching"] = {
                    "points": hatching_result.points,
                    "signals": hatching_result.signals,
                    "metadata": hatching_result.metadata,
                }
            except Exception as e:
                merged_data["sources"]["hatching"] = {"error": str(e)}

        # Query laser parameters
        if "laser" in sources and self.laser_client:
            try:
                spatial_query = SpatialQuery(component_id=model_id, bbox_min=bbox_min, bbox_max=bbox_max)
                laser_result = self.laser_client.query(spatial=spatial_query)
                merged_data["sources"]["laser"] = {
                    "points": laser_result.points,
                    "signals": laser_result.signals,
                    "metadata": laser_result.metadata,
                }
            except Exception as e:
                merged_data["sources"]["laser"] = {"error": str(e)}

        # Query CT scan data
        if "ct" in sources and self.ct_client:
            try:
                spatial_query = SpatialQuery(component_id=model_id, bbox_min=bbox_min, bbox_max=bbox_max)
                ct_result = self.ct_client.query(spatial=spatial_query)
                merged_data["sources"]["ct"] = {
                    "points": ct_result.points,
                    "signals": ct_result.signals,
                    "metadata": ct_result.metadata,
                }
            except Exception as e:
                merged_data["sources"]["ct"] = {"error": str(e)}

        # Query ISPM data
        if "ispm" in sources and self.ispm_client:
            try:
                spatial_query = SpatialQuery(component_id=model_id, bbox_min=bbox_min, bbox_max=bbox_max)
                ispm_result = self.ispm_client.query(spatial=spatial_query)
                merged_data["sources"]["ispm"] = {
                    "points": ispm_result.points,
                    "signals": ispm_result.signals,
                    "metadata": ispm_result.metadata,
                }
            except Exception as e:
                merged_data["sources"]["ispm"] = {"error": str(e)}

        return merged_data

    def merge_temporal_data(
        self,
        model_id: str,
        layer_range: Tuple[int, int],
        sources: List[str] = ["hatching", "laser", "ispm"],
    ) -> Dict[str, Any]:
        """
        Merge temporal data from multiple sources for a layer range.

        Args:
            model_id: Model UUID
            layer_range: (start_layer, end_layer) tuple
            sources: List of sources to include ('hatching', 'laser', 'ispm')

        Returns:
            Dictionary with merged temporal data
        """
        merged_data = {"model_id": model_id, "layer_range": layer_range, "sources": {}}

        temporal_query = TemporalQuery(layer_start=layer_range[0], layer_end=layer_range[1])

        # Query hatching data
        if "hatching" in sources and self.hatching_client:
            try:
                layers = self.hatching_client.get_layers(model_id, layer_range)
                merged_data["sources"]["hatching"] = {
                    "layers": layers,
                    "n_layers": len(layers),
                }
            except Exception as e:
                merged_data["sources"]["hatching"] = {"error": str(e)}

        # Query laser parameters
        if "laser" in sources and self.laser_client:
            try:
                spatial_query = SpatialQuery(component_id=model_id, layer_range=layer_range)
                laser_result = self.laser_client.query(spatial=spatial_query, temporal=temporal_query)
                merged_data["sources"]["laser"] = {
                    "points": laser_result.points,
                    "signals": laser_result.signals,
                    "metadata": laser_result.metadata,
                }
            except Exception as e:
                merged_data["sources"]["laser"] = {"error": str(e)}

        # Query ISPM data
        if "ispm" in sources and self.ispm_client:
            try:
                spatial_query = SpatialQuery(component_id=model_id, layer_range=layer_range)
                ispm_result = self.ispm_client.query(spatial=spatial_query, temporal=temporal_query)
                merged_data["sources"]["ispm"] = {
                    "points": ispm_result.points,
                    "signals": ispm_result.signals,
                    "metadata": ispm_result.metadata,
                }
            except Exception as e:
                merged_data["sources"]["ispm"] = {"error": str(e)}

        return merged_data

    def correlate_data(
        self,
        model_id: str,
        spatial_point: Tuple[float, float, float],
        layer: Optional[int] = None,
        radius: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Correlate data from all sources at a specific spatial point.

        Args:
            model_id: Model UUID
            spatial_point: Point coordinates (x, y, z) in mm
            layer: Optional layer index
            radius: Search radius in mm (for finding nearby data points)

        Returns:
            Dictionary with correlated data from all sources
        """
        x, y, z = spatial_point

        # Create bounding box around point
        bbox_min = (x - radius, y - radius, z - radius)
        bbox_max = (x + radius, y + radius, z + radius)

        correlated_data = {
            "model_id": model_id,
            "spatial_point": spatial_point,
            "layer": layer,
            "radius": radius,
            "sources": {},
        }

        # Query hatching data
        if self.hatching_client:
            try:
                spatial_query = SpatialQuery(component_id=model_id, bbox_min=bbox_min, bbox_max=bbox_max)
                if layer is not None:
                    spatial_query.layer_range = (layer, layer)

                hatching_result = self.hatching_client.query(spatial=spatial_query)
                correlated_data["sources"]["hatching"] = {
                    "points": hatching_result.points,
                    "signals": hatching_result.signals,
                    "n_points": len(hatching_result.points),
                }
            except Exception as e:
                correlated_data["sources"]["hatching"] = {"error": str(e)}

        # Query laser parameters
        if self.laser_client:
            try:
                spatial_query = SpatialQuery(component_id=model_id, bbox_min=bbox_min, bbox_max=bbox_max)
                if layer is not None:
                    spatial_query.layer_range = (layer, layer)

                laser_result = self.laser_client.query(spatial=spatial_query)
                correlated_data["sources"]["laser"] = {
                    "points": laser_result.points,
                    "signals": laser_result.signals,
                    "n_points": len(laser_result.points),
                }
            except Exception as e:
                correlated_data["sources"]["laser"] = {"error": str(e)}

        # Query CT scan data
        if self.ct_client:
            try:
                spatial_query = SpatialQuery(component_id=model_id, bbox_min=bbox_min, bbox_max=bbox_max)
                ct_result = self.ct_client.query(spatial=spatial_query)
                correlated_data["sources"]["ct"] = {
                    "points": ct_result.points,
                    "signals": ct_result.signals,
                    "n_points": len(ct_result.points),
                }
            except Exception as e:
                correlated_data["sources"]["ct"] = {"error": str(e)}

        # Query ISPM data
        if self.ispm_client:
            try:
                spatial_query = SpatialQuery(component_id=model_id, bbox_min=bbox_min, bbox_max=bbox_max)
                if layer is not None:
                    spatial_query.layer_range = (layer, layer)

                ispm_result = self.ispm_client.query(spatial=spatial_query)
                correlated_data["sources"]["ispm"] = {
                    "points": ispm_result.points,
                    "signals": ispm_result.signals,
                    "n_points": len(ispm_result.points),
                }
            except Exception as e:
                correlated_data["sources"]["ispm"] = {"error": str(e)}

        return correlated_data

    def query_all_sources(
        self,
        model_id: str,
        spatial: Optional[SpatialQuery] = None,
        temporal: Optional[TemporalQuery] = None,
        signal_types: Optional[List] = None,
    ) -> Dict[str, Any]:
        """
        Query all data sources for a given model.

        Args:
            model_id: Model UUID
            spatial: Optional spatial query constraints
            temporal: Optional temporal query constraints
            signal_types: Optional list of signal types to query

        Returns:
            Dictionary containing QueryResult objects from all sources
        """
        results = {}

        # Create spatial query with model_id if not provided
        if spatial is None:
            spatial = SpatialQuery(component_id=model_id)
        elif spatial.component_id is None:
            spatial = SpatialQuery(
                component_id=model_id,
                bbox_min=spatial.bbox_min,
                bbox_max=spatial.bbox_max,
                layer_range=spatial.layer_range,
            )

        # Query each source
        if self.stl_client:
            try:
                results["stl"] = self.stl_client.query(spatial=spatial, temporal=temporal, signal_types=signal_types)
            except Exception as e:
                results["stl"] = {"error": str(e)}

        if self.hatching_client:
            try:
                results["hatching"] = self.hatching_client.query(spatial=spatial, temporal=temporal, signal_types=signal_types)
            except Exception as e:
                results["hatching"] = {"error": str(e)}

        if self.laser_client:
            try:
                results["laser"] = self.laser_client.query(spatial=spatial, temporal=temporal, signal_types=signal_types)
            except Exception as e:
                results["laser"] = {"error": str(e)}

        if self.ct_client:
            try:
                results["ct"] = self.ct_client.query(spatial=spatial, temporal=temporal, signal_types=signal_types)
            except Exception as e:
                results["ct"] = {"error": str(e)}

        if self.ispm_client:
            try:
                results["ispm"] = self.ispm_client.query(spatial=spatial, temporal=temporal, signal_types=signal_types)
            except Exception as e:
                results["ispm"] = {"error": str(e)}

        return results

    def get_coordinate_transformer(self):
        """
        Get the coordinate system transformer.

        Returns:
            CoordinateSystemTransformer instance or None
        """
        return self.coord_transformer
