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
    from .laser_monitoring_client import LaserMonitoringClient
    from .ct_scan_client import CTScanClient
    from .ispm_thermal_client import ISPMThermalClient
    from .ispm_optical_client import ISPMOpticalClient
    from .ispm_acoustic_client import ISPMAcousticClient
    from .ispm_strain_client import ISPMStrainClient
    from .ispm_plume_client import ISPMPlumeClient
    from .base_query_client import SpatialQuery, TemporalQuery
    from ..coordinate_systems import CoordinateSystemTransformer
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
        "laser_monitoring_client",
        "ct_scan_client",
        "ispm_thermal_client",
        "ispm_optical_client",
        "ispm_acoustic_client",
        "ispm_strain_client",
        "ispm_plume_client",
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
    coord_dir = query_dir.parent / "coordinate_systems"
    transformer_path = coord_dir / "transformer.py"
    if transformer_path.exists():
        import importlib.util

        spec = importlib.util.spec_from_file_location("coordinate_systems.transformer", transformer_path)
        transformer_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(transformer_module)
        sys.modules["coordinate_systems.transformer"] = transformer_module
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
    laser_module = sys.modules.get("laser_monitoring_client")
    ct_module = sys.modules.get("ct_scan_client")
    ispm_thermal_module = sys.modules.get("ispm_thermal_client")
    ispm_optical_module = sys.modules.get("ispm_optical_client")
    ispm_acoustic_module = sys.modules.get("ispm_acoustic_client")
    ispm_strain_module = sys.modules.get("ispm_strain_client")
    ispm_plume_module = sys.modules.get("ispm_plume_client")
    base_module = sys.modules.get("base_query_client")

    STLModelClient = getattr(stl_module, "STLModelClient", None) if stl_module else None
    HatchingClient = getattr(hatching_module, "HatchingClient", None) if hatching_module else None
    LaserMonitoringClient = getattr(laser_module, "LaserMonitoringClient", None) if laser_module else None
    CTScanClient = getattr(ct_module, "CTScanClient", None) if ct_module else None
    ISPMThermalClient = getattr(ispm_thermal_module, "ISPMThermalClient", None) if ispm_thermal_module else None
    ISPMOpticalClient = getattr(ispm_optical_module, "ISPMOpticalClient", None) if ispm_optical_module else None
    ISPMAcousticClient = getattr(ispm_acoustic_module, "ISPMAcousticClient", None) if ispm_acoustic_module else None
    ISPMStrainClient = getattr(ispm_strain_module, "ISPMStrainClient", None) if ispm_strain_module else None
    ISPMPlumeClient = getattr(ispm_plume_module, "ISPMPlumeClient", None) if ispm_plume_module else None
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
    - ISPM monitoring data (Thermal, Optical, Acoustic, Strain)
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

        if LaserMonitoringClient:
            self.laser_client = LaserMonitoringClient(mongo_client=mongo_client, use_mongodb=True)
        else:
            self.laser_client = None

        if CTScanClient:
            self.ct_client = CTScanClient(mongo_client=mongo_client, use_mongodb=True)
        else:
            self.ct_client = None

        if ISPMThermalClient:
            self.ispm_thermal_client = ISPMThermalClient(mongo_client=mongo_client, use_mongodb=True)
        else:
            self.ispm_thermal_client = None

        if ISPMOpticalClient:
            self.ispm_optical_client = ISPMOpticalClient(mongo_client=mongo_client, use_mongodb=True)
        else:
            self.ispm_optical_client = None

        if ISPMAcousticClient:
            self.ispm_acoustic_client = ISPMAcousticClient(mongo_client=mongo_client, use_mongodb=True)
        else:
            self.ispm_acoustic_client = None

        if ISPMStrainClient:
            self.ispm_strain_client = ISPMStrainClient(mongo_client=mongo_client, use_mongodb=True)
        else:
            self.ispm_strain_client = None

        if ISPMPlumeClient:
            self.ispm_plume_client = ISPMPlumeClient(mongo_client=mongo_client, use_mongodb=True)
        else:
            self.ispm_plume_client = None

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
        if self.ispm_thermal_client:
            self.ispm_thermal_client.set_mongo_client(mongo_client)
        if self.ispm_optical_client:
            self.ispm_optical_client.set_mongo_client(mongo_client)
        if self.ispm_acoustic_client:
            self.ispm_acoustic_client.set_mongo_client(mongo_client)

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
            "laser_monitoring_data": None,
            "ct_scan": None,
            "ispm_thermal": None,
            "ispm_optical": None,
            "ispm_acoustic": None,
            "ispm_strain": None,
            "ispm_plume": None,
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
                result["laser_monitoring_data"] = self.laser_client.query(spatial=spatial_query)
            except Exception as e:
                result["laser_monitoring_data"] = {"error": str(e)}

        # Get CT scan data
        if self.ct_client:
            result["ct_scan"] = self.ct_client.get_scan(model_id)

        # Get ISPM thermal monitoring data (sample)
        if self.ispm_thermal_client:
            try:
                spatial_query = SpatialQuery(component_id=model_id)
                result["ispm_thermal"] = self.ispm_thermal_client.query(spatial=spatial_query)
            except Exception as e:
                result["ispm_thermal"] = {"error": str(e)}

        # Get ISPM optical monitoring data (sample)
        if self.ispm_optical_client:
            try:
                spatial_query = SpatialQuery(component_id=model_id)
                result["ispm_optical"] = self.ispm_optical_client.query(spatial=spatial_query)
            except Exception as e:
                result["ispm_optical"] = {"error": str(e)}

        return result

    def merge_spatial_data(
        self,
        model_id: str,
        bbox: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
        sources: List[str] = ["hatching", "laser_monitoring", "ct", "ispm"],
        target_coord_system: str = "build_platform",
    ) -> Dict[str, Any]:
        """
        Merge spatial data from multiple sources within a bounding box.

        Args:
            model_id: Model UUID
            bbox: Bounding box ((x_min, y_min, z_min), (x_max, y_max, z_max))
            sources: List of sources to include ('hatching', 'laser_monitoring', 'ct', 'ispm')
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

        if "ispm" in sources or "ispm_thermal" in sources:
            if self.ispm_thermal_client:
                coord_systems["ispm_thermal"] = self.ispm_thermal_client.get_coordinate_system(model_id) or {}
        if "ispm" in sources or "ispm_optical" in sources:
            if self.ispm_optical_client:
                coord_systems["ispm_optical"] = self.ispm_optical_client.get_coordinate_system(model_id) or {}
        if "ispm" in sources or "ispm_acoustic" in sources:
            if self.ispm_acoustic_client:
                coord_systems["ispm_acoustic"] = self.ispm_acoustic_client.get_coordinate_system(model_id) or {}
        if "ispm" in sources or "ispm_strain" in sources:
            if self.ispm_strain_client:
                coord_systems["ispm_strain"] = self.ispm_strain_client.get_coordinate_system(model_id) or {}
        if "ispm" in sources or "ispm_plume" in sources:
            if self.ispm_plume_client:
                coord_systems["ispm_plume"] = self.ispm_plume_client.get_coordinate_system(model_id) or {}

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

        # Query laser monitoring parameters
        if "laser_monitoring" in sources and self.laser_client:
            try:
                spatial_query = SpatialQuery(component_id=model_id, bbox_min=bbox_min, bbox_max=bbox_max)
                laser_result = self.laser_client.query(spatial=spatial_query)
                merged_data["sources"]["laser_monitoring"] = {
                    "points": laser_result.points,
                    "signals": laser_result.signals,
                    "metadata": laser_result.metadata,
                }
            except Exception as e:
                merged_data["sources"]["laser_monitoring"] = {"error": str(e)}

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

        # Query ISPM thermal data
        if ("ispm" in sources or "ispm_thermal" in sources) and self.ispm_thermal_client:
            try:
                spatial_query = SpatialQuery(component_id=model_id, bbox_min=bbox_min, bbox_max=bbox_max)
                ispm_thermal_result = self.ispm_thermal_client.query(spatial=spatial_query)
                merged_data["sources"]["ispm_thermal"] = {
                    "points": ispm_thermal_result.points,
                    "signals": ispm_thermal_result.signals,
                    "metadata": ispm_thermal_result.metadata,
                }
            except Exception as e:
                merged_data["sources"]["ispm_thermal"] = {"error": str(e)}

        # Query ISPM optical data
        if ("ispm" in sources or "ispm_optical" in sources) and self.ispm_optical_client:
            try:
                spatial_query = SpatialQuery(component_id=model_id, bbox_min=bbox_min, bbox_max=bbox_max)
                ispm_optical_result = self.ispm_optical_client.query(spatial=spatial_query)
                merged_data["sources"]["ispm_optical"] = {
                    "points": ispm_optical_result.points,
                    "signals": ispm_optical_result.signals,
                    "metadata": ispm_optical_result.metadata,
                }
            except Exception as e:
                merged_data["sources"]["ispm_optical"] = {"error": str(e)}

        # Query ISPM acoustic data
        if ("ispm" in sources or "ispm_acoustic" in sources) and self.ispm_acoustic_client:
            try:
                spatial_query = SpatialQuery(component_id=model_id, bbox_min=bbox_min, bbox_max=bbox_max)
                ispm_acoustic_result = self.ispm_acoustic_client.query(spatial=spatial_query)
                merged_data["sources"]["ispm_acoustic"] = {
                    "points": ispm_acoustic_result.points,
                    "signals": ispm_acoustic_result.signals,
                    "metadata": ispm_acoustic_result.metadata,
                }
            except Exception as e:
                merged_data["sources"]["ispm_acoustic"] = {"error": str(e)}

        # Query ISPM strain data
        if ("ispm" in sources or "ispm_strain" in sources) and self.ispm_strain_client:
            try:
                spatial_query = SpatialQuery(component_id=model_id, bbox_min=bbox_min, bbox_max=bbox_max)
                ispm_strain_result = self.ispm_strain_client.query(spatial=spatial_query)
                merged_data["sources"]["ispm_strain"] = {
                    "points": ispm_strain_result.points,
                    "signals": ispm_strain_result.signals,
                    "metadata": ispm_strain_result.metadata,
                }
            except Exception as e:
                merged_data["sources"]["ispm_strain"] = {"error": str(e)}

        return merged_data

    def merge_temporal_data(
        self,
        model_id: str,
        layer_range: Tuple[int, int],
        sources: List[str] = ["hatching", "laser_monitoring", "ispm"],
    ) -> Dict[str, Any]:
        """
        Merge temporal data from multiple sources for a layer range.

        Args:
            model_id: Model UUID
            layer_range: (start_layer, end_layer) tuple
            sources: List of sources to include ('hatching', 'laser_monitoring', 'ispm')

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

        # Query laser monitoring parameters
        if "laser_monitoring" in sources and self.laser_client:
            try:
                spatial_query = SpatialQuery(component_id=model_id, layer_range=layer_range)
                laser_result = self.laser_client.query(spatial=spatial_query, temporal=temporal_query)
                merged_data["sources"]["laser_monitoring"] = {
                    "points": laser_result.points,
                    "signals": laser_result.signals,
                    "metadata": laser_result.metadata,
                }
            except Exception as e:
                merged_data["sources"]["laser_monitoring"] = {"error": str(e)}

        # Query ISPM thermal data
        if ("ispm" in sources or "ispm_thermal" in sources) and self.ispm_thermal_client:
            try:
                spatial_query = SpatialQuery(component_id=model_id, layer_range=layer_range)
                ispm_thermal_result = self.ispm_thermal_client.query(spatial=spatial_query, temporal=temporal_query)
                merged_data["sources"]["ispm_thermal"] = {
                    "points": ispm_thermal_result.points,
                    "signals": ispm_thermal_result.signals,
                    "metadata": ispm_thermal_result.metadata,
                }
            except Exception as e:
                merged_data["sources"]["ispm_thermal"] = {"error": str(e)}

        # Query ISPM optical data
        if ("ispm" in sources or "ispm_optical" in sources) and self.ispm_optical_client:
            try:
                spatial_query = SpatialQuery(component_id=model_id, layer_range=layer_range)
                ispm_optical_result = self.ispm_optical_client.query(spatial=spatial_query, temporal=temporal_query)
                merged_data["sources"]["ispm_optical"] = {
                    "points": ispm_optical_result.points,
                    "signals": ispm_optical_result.signals,
                    "metadata": ispm_optical_result.metadata,
                }
            except Exception as e:
                merged_data["sources"]["ispm_optical"] = {"error": str(e)}

        # Query ISPM acoustic data
        if ("ispm" in sources or "ispm_acoustic" in sources) and self.ispm_acoustic_client:
            try:
                spatial_query = SpatialQuery(component_id=model_id, layer_range=layer_range)
                ispm_acoustic_result = self.ispm_acoustic_client.query(spatial=spatial_query, temporal=temporal_query)
                merged_data["sources"]["ispm_acoustic"] = {
                    "points": ispm_acoustic_result.points,
                    "signals": ispm_acoustic_result.signals,
                    "metadata": ispm_acoustic_result.metadata,
                }
            except Exception as e:
                merged_data["sources"]["ispm_acoustic"] = {"error": str(e)}

        # Query ISPM strain data
        if ("ispm" in sources or "ispm_strain" in sources) and self.ispm_strain_client:
            try:
                spatial_query = SpatialQuery(component_id=model_id, layer_range=layer_range)
                ispm_strain_result = self.ispm_strain_client.query(spatial=spatial_query, temporal=temporal_query)
                merged_data["sources"]["ispm_strain"] = {
                    "points": ispm_strain_result.points,
                    "signals": ispm_strain_result.signals,
                    "metadata": ispm_strain_result.metadata,
                }
            except Exception as e:
                merged_data["sources"]["ispm_strain"] = {"error": str(e)}

        # Query ISPM plume data
        if ("ispm" in sources or "ispm_plume" in sources) and self.ispm_plume_client:
            try:
                spatial_query = SpatialQuery(component_id=model_id, layer_range=layer_range)
                ispm_plume_result = self.ispm_plume_client.query(spatial=spatial_query, temporal=temporal_query)
                merged_data["sources"]["ispm_plume"] = {
                    "points": ispm_plume_result.points,
                    "signals": ispm_plume_result.signals,
                    "metadata": ispm_plume_result.metadata,
                }
            except Exception as e:
                merged_data["sources"]["ispm_plume"] = {"error": str(e)}

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
                correlated_data["sources"]["laser_monitoring"] = {
                    "points": laser_result.points,
                    "signals": laser_result.signals,
                    "n_points": len(laser_result.points),
                }
            except Exception as e:
                correlated_data["sources"]["laser_monitoring"] = {"error": str(e)}

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

        # Query ISPM thermal data
        if self.ispm_thermal_client:
            try:
                spatial_query = SpatialQuery(component_id=model_id, bbox_min=bbox_min, bbox_max=bbox_max)
                if layer is not None:
                    spatial_query.layer_range = (layer, layer)

                ispm_thermal_result = self.ispm_thermal_client.query(spatial=spatial_query)
                correlated_data["sources"]["ispm_thermal"] = {
                    "points": ispm_thermal_result.points,
                    "signals": ispm_thermal_result.signals,
                    "n_points": len(ispm_thermal_result.points),
                }
            except Exception as e:
                correlated_data["sources"]["ispm_thermal"] = {"error": str(e)}

        # Query ISPM optical data
        if self.ispm_optical_client:
            try:
                spatial_query = SpatialQuery(component_id=model_id, bbox_min=bbox_min, bbox_max=bbox_max)
                if layer is not None:
                    spatial_query.layer_range = (layer, layer)

                ispm_optical_result = self.ispm_optical_client.query(spatial=spatial_query)
                correlated_data["sources"]["ispm_optical"] = {
                    "points": ispm_optical_result.points,
                    "signals": ispm_optical_result.signals,
                    "n_points": len(ispm_optical_result.points),
                }
            except Exception as e:
                correlated_data["sources"]["ispm_optical"] = {"error": str(e)}

        return correlated_data

    def query_all_sources(
        self,
        model_id: str,
        spatial: Optional[SpatialQuery] = None,
        temporal: Optional[TemporalQuery] = None,
        signal_types: Optional[List] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Query all data sources for a given model.
        
        Supports caching - if same query was executed before, returns cached result.

        Args:
            model_id: Model UUID
            spatial: Optional spatial query constraints
            temporal: Optional temporal query constraints
            signal_types: Optional list of signal types to query
            use_cache: Whether to use cached results if available (default: True)

        Returns:
            Dictionary containing QueryResult objects from all sources
        """
        # Build cache key from query parameters
        # Serialize spatial and temporal queries to dicts for cache key
        spatial_dict = None
        if spatial:
            try:
                spatial_dict = spatial.__dict__ if hasattr(spatial, '__dict__') else dict(spatial)
            except:
                spatial_dict = {"component_id": getattr(spatial, 'component_id', None)}
        
        temporal_dict = None
        if temporal:
            try:
                temporal_dict = temporal.__dict__ if hasattr(temporal, '__dict__') else dict(temporal)
            except:
                temporal_dict = {}
        
        cache_key = {
            "model_id": model_id,
            "spatial": spatial_dict,
            "temporal": temporal_dict,
            "signal_types": [str(s) for s in signal_types] if signal_types else None,
            "method": "query_all_sources"
        }
        
        # Try to get from cache first
        if use_cache:
            try:
                from src.infrastructure.cache.framework_cache import get_cached_query_result
                cached_result = get_cached_query_result(cache_key)
                if cached_result is not None:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.info(f"Using cached query result for model {model_id}")
                    return cached_result
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.debug(f"Cache check failed (will query fresh): {e}")
        
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
                results["laser_monitoring"] = self.laser_client.query(spatial=spatial, temporal=temporal, signal_types=signal_types)
            except Exception as e:
                results["laser_monitoring"] = {"error": str(e)}

        if self.ct_client:
            try:
                results["ct"] = self.ct_client.query(spatial=spatial, temporal=temporal, signal_types=signal_types)
            except Exception as e:
                results["ct"] = {"error": str(e)}

        if self.ispm_thermal_client:
            try:
                results["ispm_thermal"] = self.ispm_thermal_client.query(spatial=spatial, temporal=temporal, signal_types=signal_types)
            except Exception as e:
                results["ispm_thermal"] = {"error": str(e)}

        if self.ispm_optical_client:
            try:
                results["ispm_optical"] = self.ispm_optical_client.query(spatial=spatial, temporal=temporal, signal_types=signal_types)
            except Exception as e:
                results["ispm_optical"] = {"error": str(e)}

        if self.ispm_acoustic_client:
            try:
                results["ispm_acoustic"] = self.ispm_acoustic_client.query(spatial=spatial, temporal=temporal, signal_types=signal_types)
            except Exception as e:
                results["ispm_acoustic"] = {"error": str(e)}

        if self.ispm_strain_client:
            try:
                results["ispm_strain"] = self.ispm_strain_client.query(spatial=spatial, temporal=temporal, signal_types=signal_types)
            except Exception as e:
                results["ispm_strain"] = {"error": str(e)}

        if self.ispm_plume_client:
            try:
                results["ispm_plume"] = self.ispm_plume_client.query(spatial=spatial, temporal=temporal, signal_types=signal_types)
            except Exception as e:
                results["ispm_plume"] = {"error": str(e)}

        # Cache the results for future use
        if use_cache:
            try:
                from src.infrastructure.cache.framework_cache import cache_query_result
                cache_query_result(cache_key, results)
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.debug(f"Failed to cache query result: {e}")

        return results

    def _query_source(
        self,
        model_id: str,
        source_type: str,
        spatial: Optional[SpatialQuery] = None,
        temporal: Optional[TemporalQuery] = None,
    ) -> Optional[Any]:
        """
        Query a single source by type. Returns QueryResult or None if source unavailable.

        Args:
            model_id: Model UUID
            source_type: One of 'hatching', 'laser_monitoring', 'ct', 'ispm_thermal',
                        'ispm_optical', 'ispm_acoustic', 'ispm_strain', 'ispm_plume'
            spatial: Optional spatial query
            temporal: Optional temporal query

        Returns:
            QueryResult with .points and .signals, or None
        """
        if spatial is None:
            spatial = SpatialQuery(component_id=model_id)
        elif getattr(spatial, "component_id", None) is None:
            spatial = SpatialQuery(
                component_id=model_id,
                bbox_min=getattr(spatial, "bbox_min", None),
                bbox_max=getattr(spatial, "bbox_max", None),
                layer_range=getattr(spatial, "layer_range", None),
            )
        try:
            if source_type == "hatching" and self.hatching_client:
                return self.hatching_client.query(spatial=spatial, temporal=temporal)
            if source_type == "laser_monitoring" and self.laser_client:
                return self.laser_client.query(spatial=spatial, temporal=temporal)
            if source_type == "ct" and self.ct_client:
                return self.ct_client.query(spatial=spatial, temporal=temporal)
            if source_type == "ispm_thermal" and self.ispm_thermal_client:
                return self.ispm_thermal_client.query(spatial=spatial, temporal=temporal)
            if source_type == "ispm_optical" and self.ispm_optical_client:
                return self.ispm_optical_client.query(spatial=spatial, temporal=temporal)
            if source_type == "ispm_acoustic" and self.ispm_acoustic_client:
                return self.ispm_acoustic_client.query(spatial=spatial, temporal=temporal)
            if source_type == "ispm_strain" and self.ispm_strain_client:
                return self.ispm_strain_client.query(spatial=spatial, temporal=temporal)
            if source_type == "ispm_plume" and self.ispm_plume_client:
                return self.ispm_plume_client.query(spatial=spatial, temporal=temporal)
        except Exception:
            pass
        return None

    def _save_transformed_points_to_mongodb(
        self,
        model_id: str,
        source_types: List[str],
        raw_results: Dict[str, Any],
        transformed_points: Dict[str, np.ndarray],
        signals: Dict[str, Dict[str, np.ndarray]],
        transformations: Dict[str, Dict[str, Any]],
        unified_bounds: Any,
        layer_indices_per_source: Dict[str, Any],
        mongo_uri: str,
        db_name: str,
        batch_size: int = 10000,
    ) -> None:
        """
        Save transformed points to MongoDB via C++; all iteration and padding in C++, zero-copy buffers.

        Only non-reference sources are written. The reference source (e.g. hatching)
        is not saved to a Processed_* collection. For scale (billions of points), pass
        layer_indices_per_source as dict of numpy int32 arrays; pass timestamps_per_source
        as None to use default timestamp, or as dict of numpy S28 arrays to avoid list build.
        """
        if not mongo_uri or not db_name:
            raise ValueError("mongo_uri and db_name are required")
        try:
            from am_qadf_native.io import ensure_mongocxx_initialized, save_transformed_points_to_mongodb
            ensure_mongocxx_initialized()
        except ImportError as e:
            raise ImportError(
                "_save_transformed_points_to_mongodb requires am_qadf_native.io (save_transformed_points_to_mongodb). "
                "Install the native extension."
            ) from e
        save_transformed_points_to_mongodb(
            model_id,
            source_types,
            transformed_points,
            signals,
            layer_indices_per_source,
            None,  # timestamps: C++ uses default for all points; no list built here
            transformations,
            unified_bounds,
            mongo_uri,
            db_name,
            batch_size,
        )

    def query_and_transform_points(
        self,
        model_id: str,
        source_types: List[str],
        reference_source: str = "hatching",
        layer_range: Optional[Tuple[int, int]] = None,
        bbox: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None,
        use_full_extent_for_transform: bool = True,
        validation_tolerance: float = 1e-6,
        adaptive_tolerance_pct: float = 0.01,
        save_processed_data: bool = False,
        mongo_uri: Optional[str] = None,
        db_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Query points from all sources, transform to reference coordinate system, and optionally save.

        Uses existing query() for each source; transforms non-reference sources to reference
        using bounding-box corner correspondence (8 corner pairs) and C++ Kabsch+Umeyama.
        Correspondence is established by matching bbox coordinates only (no point-based matching).
        When spatial alignment completes, temporal (layer) alignment is run automatically and
        the result is included as layer_alignment_result and layer_alignment_source_order.

        Bounding box for transformation: By default (use_full_extent_for_transform=True) the
        bbox is computed from the full dataset (no bbox/layer filter); if you pass bbox or
        layer_range, points returned and saved still respect that filter. Set to False to
        compute bbox from the queried subset only.

        Args:
            model_id: Model UUID
            source_types: List of source keys, e.g. ['hatching', 'ispm_thermal']
            reference_source: Source to use as reference (e.g. 'hatching')
            layer_range: Optional (layer_start, layer_end) for spatial/temporal filter
            bbox: Optional ((x_min,y_min,z_min), (x_max,y_max,z_max)) for spatial filter
            use_full_extent_for_transform: If True (default), query each source without
                bbox/layer_range to compute bounding boxes and transformation; points
                returned/saved still use bbox/layer_range when provided.
            validation_tolerance: Tolerance for transformation validation
            adaptive_tolerance_pct: Fraction of ref bbox max extent for adaptive tolerance (default 0.01 = 1%%).
                Effective tolerance = max(validation_tolerance, adaptive_tolerance_pct * max_extent, 1e-3).
            save_processed_data: If True, save transformed points to MongoDB via MongoDBWriter
            mongo_uri: MongoDB URI (required if save_processed_data is True)
            db_name: MongoDB database name (required if save_processed_data is True)

        Saved data (when save_processed_data is True):
            Only non-reference sources are written to Processed_<source_type>_data.
            The reference source (e.g. hatching) is not saved: it is the coordinate
            reference and is not transformed, so no Processed_hatching_data collection
            is created. Use the raw hatching collection for reference geometry.

        Returns:
            Dict with:
                - transformed_points: { source_type: np.ndarray (n, 3) }
                - signals: { source_type: { signal_name: np.ndarray } }
                - unified_bounds: BoundingBox (C++ sync namespace)
                - transformations: { source_type: { "matrix", "quality", ... } }
                - validation_results: { source_type: ValidationResult }
                - raw_results: { source_type: QueryResult } (before transform)
                - layer_indices: { source_type: list of int } (per-point layer)
                - layer_alignment_result: LayerAlignmentResult (when native available; unique_layers and indices_per_layer_per_source for per-layer slicing)
                - layer_alignment_source_order: list of source keys matching indices_per_layer_per_source columns
        """
        try:
            from am_qadf_native import (
                points_to_eigen_matrix,
                TransformationComputer,
                TransformationValidator,
                PointTransformer,
                UnifiedBoundsComputer,
                BoundingBox,
                decompose_similarity_transform,
            )
            from am_qadf_native.io import MongoDBWriter
        except ImportError as e:
            raise ImportError(
                "query_and_transform_points requires am_qadf_native (C++ bindings). "
                "Build the native extension and ensure points_to_eigen_matrix, "
                "TransformationComputer, PointTransformer, UnifiedBoundsComputer are available."
            ) from e

        if reference_source not in source_types:
            source_types = list(source_types) + [reference_source]
        bbox_min = bbox_max = None
        if bbox is not None:
            bbox_min, bbox_max = bbox
        spatial_for_points = SpatialQuery(
            component_id=model_id,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            layer_range=layer_range,
        )
        temporal_for_points = None
        if layer_range is not None:
            temporal_for_points = TemporalQuery(layer_start=layer_range[0], layer_end=layer_range[1])
        has_filter = bbox is not None or layer_range is not None
        use_full_for_bounds = use_full_extent_for_transform and has_filter

        raw_results: Dict[str, Any] = {}
        for st in source_types:
            r = self._query_source(model_id, st, spatial=spatial_for_points, temporal=temporal_for_points)
            if r is not None and getattr(r, "points", None) and len(r.points) > 0:
                raw_results[st] = r

        raw_results_for_bounds: Dict[str, Any] = {}
        if use_full_for_bounds:
            spatial_full = SpatialQuery(component_id=model_id)
            for st in source_types:
                r = self._query_source(model_id, st, spatial=spatial_full, temporal=None)
                if r is not None and getattr(r, "points", None) and len(r.points) > 0:
                    raw_results_for_bounds[st] = r

        if reference_source not in raw_results:
            raise ValueError(
                f"Reference source '{reference_source}' has no points. "
                f"Available: {list(raw_results.keys())}. Ensure data exists and query returns points."
            )
        if use_full_for_bounds and reference_source not in raw_results_for_bounds:
            use_full_for_bounds = False

        def _points_to_list_of_list(points):
            return [[float(x), float(y), float(z)] for (x, y, z) in points]

        ref_result = raw_results[reference_source]
        ref_result_for_bounds = raw_results_for_bounds.get(reference_source, ref_result)
        ref_points_list = _points_to_list_of_list(ref_result_for_bounds.points)
        ref_points_eigen = points_to_eigen_matrix(ref_points_list)
        ref_points_eigen_return = points_to_eigen_matrix(_points_to_list_of_list(ref_result.points))

        transformed_points: Dict[str, np.ndarray] = {reference_source: np.array(ref_result.points, dtype=np.float64)}
        signals: Dict[str, Dict[str, np.ndarray]] = {
            reference_source: {k: np.array(v, dtype=np.float64) for k, v in (ref_result.signals or {}).items()}
        }
        transformations: Dict[str, Dict[str, Any]] = {}
        validation_results: Dict[str, Any] = {}
        # Layer indices per source (for temporal alignment / per-layer fusion)
        n_ref = int(np.asarray(ref_result.points).shape[0])
        ref_layer_indices = list(
            ref_result.metadata.get("layers", []) or []
        ) if hasattr(ref_result, "metadata") else []
        if len(ref_layer_indices) != n_ref:
            ref_layer_indices = (
                [ref_layer_indices[i % len(ref_layer_indices)] for i in range(n_ref)]
                if ref_layer_indices
                else [0] * n_ref
            )
        layer_indices_result: Dict[str, List[int]] = {reference_source: ref_layer_indices}

        computer = TransformationComputer()
        validator = TransformationValidator()
        transformer = PointTransformer()
        bounds_computer = UnifiedBoundsComputer()

        point_sets_for_bounds = [ref_points_eigen_return]

        for st in source_types:
            if st == reference_source:
                continue
            if st not in raw_results:
                continue
            res = raw_results[st]
            res_for_bounds = raw_results_for_bounds.get(st, res) if use_full_for_bounds else res
            src_points_list_bounds = _points_to_list_of_list(res_for_bounds.points)
            src_points_eigen_bounds = points_to_eigen_matrix(src_points_list_bounds)

            # ############
            # (1) Get bounding box and 8 corners (C++) from full data when use_full_for_bounds
            # ############
            src_bbox = bounds_computer.compute_bounds_from_points(src_points_eigen_bounds)
            ref_bbox = bounds_computer.compute_bounds_from_points(ref_points_eigen)
            src_sampled = src_bbox.corners()
            ref_sampled = ref_bbox.corners()

            # ############
            # (2) Compute transform from bbox corners: 24 mappings × 56 triplets, fit on 3 / validate on 8 (C++)
            #    Returns per-fit errors and best_ref_corners (reference reordered by best permutation).
            #    Validation must use best_ref_corners so T*source is compared to the correct target order.
            # ############
            transform, quality, fit_errors, best_ref_corners = computer.compute_transformation_from_bbox_corners_with_fit_errors(
                src_sampled, ref_sampled
            )
            fit_errors_list = [
                {
                    "permutation_index": f.permutation_index,
                    "triplet_index": f.triplet_index,
                    "max_error": f.max_error,
                    "mean_error": f.mean_error,
                    "rms_error": f.rms_error,
                }
                for f in fit_errors
            ]
            max_errors = [f.max_error for f in fit_errors]
            best_idx = int(np.argmin(max_errors))
            best_fit = fit_errors[best_idx]

            # ############
            # (3) Validate transform on 8 corners (pass/fail + tolerance) using best_ref_corners (reordered by best perm)
            # ############
            ref_arr = np.array(best_ref_corners)
            if ref_arr.size >= 3:
                max_extent = float(np.max(np.ptp(ref_arr, axis=0)))
                adaptive_val_tol = max(adaptive_tolerance_pct * max_extent, 1e-3)
                effective_val_tol = max(validation_tolerance, adaptive_val_tol)
            else:
                effective_val_tol = max(validation_tolerance, 1e-3)
            validation = validator.validate_with_matrix(
                src_sampled, transform, best_ref_corners, effective_val_tol
            )
            transformations[st] = {
                "matrix": transform,
                "quality": quality,
                "fit_errors": fit_errors_list,
                "best_fit": {
                    "permutation_index": best_fit.permutation_index,
                    "triplet_index": best_fit.triplet_index,
                    "max_error": best_fit.max_error,
                    "mean_error": best_fit.mean_error,
                    "rms_error": best_fit.rms_error,
                },
                "fit_errors_summary": {
                    "min_max_error": float(np.min(max_errors)),
                    "max_max_error": float(np.max(max_errors)),
                    "num_fits": len(fit_errors),
                },
                "validation_tolerance": effective_val_tol,
            }
            dec = decompose_similarity_transform(np.asarray(transform, dtype=np.float64))
            transformations[st]["scale"] = dec.scale
            transformations[st]["translation"] = (dec.tx, dec.ty, dec.tz)
            transformations[st]["rotation_euler_deg"] = (dec.rot_x_deg, dec.rot_y_deg, dec.rot_z_deg)
            validation_results[st] = validation
            import logging
            _log = logging.getLogger(__name__)
            if not getattr(validation, "is_valid", True):
                _log.warning(
                    "Transformation validation failed for source %r: max_error=%s (tolerance=%s). "
                    "Best fit: perm=%s triplet=%s (max_error=%s). All 24×56 fits: min_max_error=%s max_max_error=%s. "
                    "Continuing with transform; check validation_results and transformations[%r]['fit_errors'] for details.",
                    st,
                    getattr(validation, "max_error", None),
                    effective_val_tol,
                    best_fit.permutation_index,
                    best_fit.triplet_index,
                    best_fit.max_error,
                    float(np.min(max_errors)),
                    float(np.max(max_errors)),
                    st,
                )
            else:
                _log.debug(
                    "Source %r: best fit perm=%s triplet=%s max_error=%s; fits range max_error [%s, %s].",
                    st, best_fit.permutation_index, best_fit.triplet_index, best_fit.max_error,
                    float(np.min(max_errors)), float(np.max(max_errors)),
                )

            # ############
            # (4) Apply transform to points to return (filtered res.points)
            # ############
            src_points_list = _points_to_list_of_list(res.points)
            src_points_eigen = points_to_eigen_matrix(src_points_list)
            transformed_eigen = transformer.transform_with_matrix(src_points_eigen, transform)
            transformed_points[st] = np.array(transformed_eigen)
            signals[st] = {k: np.array(v, dtype=np.float64) for k, v in (res.signals or {}).items()}
            point_sets_for_bounds.append(transformed_eigen)
            n_pts = int(transformed_points[st].shape[0])
            layer_indices = list(
                res.metadata.get("layers", []) or []
            ) if hasattr(res, "metadata") else []
            if len(layer_indices) != n_pts:
                layer_indices = (
                    [layer_indices[i % len(layer_indices)] for i in range(n_pts)]
                    if layer_indices
                    else [0] * n_pts
                )
            layer_indices_result[st] = np.asarray(layer_indices, dtype=np.int32, order="C")

            # ############
            # (5) Validate point correspondence: 8 corners + 1 centre (9 pairs, C++) using best_ref_corners
            # ############
            try:
                bbox_val = validator.validate_bbox_corners_and_centre(src_sampled, best_ref_corners, transform)
                transformations[st]["correspondence_validation"] = {
                    "mean_distance": bbox_val.mean_distance,
                    "max_distance": bbox_val.max_distance,
                    "num_pairs": bbox_val.num_pairs,
                    "type": "corners_and_centre",
                }
            except Exception:  # noqa: BLE001
                pass  # optional; do not fail pipeline

        unified_bounds = bounds_computer.compute_union_bounds(point_sets_for_bounds)

        # Temporal alignment: run automatically after successful spatial alignment (before save)
        import logging
        from .query_utils import format_query_and_transform_summary
        _log = logging.getLogger(__name__)
        layer_alignment_result = None
        layer_alignment_source_order: Optional[List[str]] = None
        try:
            layer_alignment_result, layer_alignment_source_order = self.align_points_by_layer(
                transformed_points, layer_indices_result
            )
        except ImportError:
            pass  # am_qadf_native not built; leave layer_alignment_result None
        except Exception as e:  # noqa: BLE001
            _log.debug("Temporal alignment skipped: %s", e)

        if save_processed_data:
            if not mongo_uri or not db_name:
                raise ValueError("save_processed_data=True requires mongo_uri and db_name")
            # Saves only non-reference sources; reference (e.g. hatching) is not written
            self._save_transformed_points_to_mongodb(
                model_id=model_id,
                source_types=source_types,
                raw_results=raw_results,
                transformed_points=transformed_points,
                signals=signals,
                transformations=transformations,
                unified_bounds=unified_bounds,
                layer_indices_per_source=layer_indices_result,
                mongo_uri=mongo_uri,
                db_name=db_name,
                batch_size=10000,
            )

        # Log summary for user (pass/fail, two tables, unified bounds)
        lines = format_query_and_transform_summary(
            source_types, reference_source,
            transformed_points, validation_results, transformations, unified_bounds,
            adaptive_tolerance_pct=adaptive_tolerance_pct,
        )
        _log.info("\n".join(lines))

        out = {
            "transformed_points": transformed_points,
            "signals": signals,
            "unified_bounds": unified_bounds,
            "transformations": transformations,
            "validation_results": validation_results,
            "raw_results": raw_results,
            "layer_indices": layer_indices_result,
        }
        if layer_alignment_result is not None:
            out["layer_alignment_result"] = layer_alignment_result
            out["layer_alignment_source_order"] = layer_alignment_source_order
        return out

    def align_points_by_layer(
        self,
        transformed_points: Dict[str, np.ndarray],
        layer_indices_per_source: Dict[str, List[int]],
        source_order: Optional[List[str]] = None,
    ) -> Any:
        """
        Align transformed point sets by layer for per-layer fusion.

        Usually not needed when using query_and_transform_points, which runs this
        automatically and returns layer_alignment_result and layer_alignment_source_order.
        Use this when you have transformed points and layer indices from elsewhere.

        All logic (order, validation, conversion, grouping) runs in C++; this is a thin
        wrapper. Returns unique_layers and indices_per_layer_per_source for slicing per
        layer per source.

        Args:
            transformed_points: Dict[source_type, points array (N x 3)]
            layer_indices_per_source: Dict[source_type, array-like of int] length N per source
            source_order: Order of sources for the returned indices (default: keys of transformed_points)

        Returns:
            Tuple of (LayerAlignmentResult, source_order_used). Slice:
            points_for_layer_l_source_s = transformed_points[source_order_used[s]][result.indices_per_layer_per_source[l][s]].
        """
        try:
            from am_qadf_native import PointTemporalAlignment
        except ImportError as e:
            raise ImportError(
                "align_points_by_layer requires am_qadf_native (PointTemporalAlignment). "
                "Install the native extension."
            ) from e
        aligner = PointTemporalAlignment()
        return aligner.align_points_by_layer(
            transformed_points, layer_indices_per_source, source_order
        )

    def get_coordinate_transformer(self):
        """
        Get the coordinate system transformer.

        Returns:
            CoordinateSystemTransformer instance or None
        """
        return self.coord_transformer
