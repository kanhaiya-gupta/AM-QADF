"""
Voxel Domain Client

Main client for creating unified voxel domain representations from multi-source data.
Provides high-level interface for:
- Voxel grid generation
- Data synchronization
- Signal mapping
- Visualization
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging

# Import existing components
import sys
from pathlib import Path

# Try relative imports first (when used as package)
try:
    from ..voxelization.uniform_resolution import VoxelGrid
    from ..voxelization.adaptive_resolution import (
        AdaptiveResolutionGrid,
        SpatialResolutionMap,
        TemporalResolutionMap,
    )
    from ..signal_mapping.execution.sequential import (
        interpolate_to_voxels,
        interpolate_hatching_paths,
    )
    from ..query.unified_query_client import UnifiedQueryClient
    from ..coordinate_systems import CoordinateSystemTransformer
except (ImportError, ValueError):
    # Fallback for direct imports (when loaded from notebook)
    current_file = Path(__file__).resolve()
    parent_dir = current_file.parent.parent

    # Import voxelization components - need to load uniform_resolution first
    voxel_dir = parent_dir / "voxelization"
    if (voxel_dir / "uniform_resolution.py").exists():
        import importlib.util

        # Load uniform_resolution first and register it
        spec = importlib.util.spec_from_file_location("voxelization.uniform_resolution", voxel_dir / "uniform_resolution.py")
        voxel_module = importlib.util.module_from_spec(spec)
        sys.modules["voxelization.uniform_resolution"] = voxel_module  # type: ignore[assignment]
        spec.loader.exec_module(voxel_module)
        VoxelGrid = voxel_module.VoxelGrid

        # Also register as 'voxel_grid' for interpolation module fallback
        sys.modules["voxel_grid"] = voxel_module  # type: ignore[assignment]

        # Load adaptive_resolution
        spec = importlib.util.spec_from_file_location("voxelization.adaptive_resolution", voxel_dir / "adaptive_resolution.py")
        adaptive_module = importlib.util.module_from_spec(spec)
        sys.modules["voxelization.adaptive_resolution"] = adaptive_module  # type: ignore[assignment]
        spec.loader.exec_module(adaptive_module)
        AdaptiveResolutionGrid = adaptive_module.AdaptiveResolutionGrid
        SpatialResolutionMap = adaptive_module.SpatialResolutionMap
        TemporalResolutionMap = adaptive_module.TemporalResolutionMap

        # Load signal mapping execution - VoxelGrid should now be available via sys.modules
        signal_mapping_dir = parent_dir / "signal_mapping" / "execution"
        if (signal_mapping_dir / "sequential.py").exists():
            spec = importlib.util.spec_from_file_location(
                "signal_mapping.execution.sequential",
                signal_mapping_dir / "sequential.py",
            )
            seq_module = importlib.util.module_from_spec(spec)
            sys.modules["signal_mapping.execution.sequential"] = seq_module  # type: ignore[assignment]
            spec.loader.exec_module(seq_module)
            interpolate_to_voxels = seq_module.interpolate_to_voxels
            interpolate_hatching_paths = seq_module.interpolate_hatching_paths
        else:
            raise ImportError("Could not find signal_mapping.execution.sequential module")
    else:
        raise ImportError("Could not import voxelization components")

    # Import query client
    query_dir = parent_dir / "query"
    if (query_dir / "unified_query_client.py").exists():
        spec = importlib.util.spec_from_file_location("query.unified_query_client", query_dir / "unified_query_client.py")
        query_module = importlib.util.module_from_spec(spec)
        sys.modules["query.unified_query_client"] = query_module  # type: ignore[assignment]
        spec.loader.exec_module(query_module)
        UnifiedQueryClient = query_module.UnifiedQueryClient
    else:
        raise ImportError("Could not import UnifiedQueryClient")

    # Import coordinate transformer (required, no fallback)
    coord_dir = parent_dir / "coordinate_systems"
    if not (coord_dir / "transformer.py").exists():
        raise ImportError(
            "Could not find coordinate_systems.transformer module. "
            "C++ bindings are required."
        )
    spec = importlib.util.spec_from_file_location("coordinate_systems.transformer", coord_dir / "transformer.py")
    coord_module = importlib.util.module_from_spec(spec)
    sys.modules["coordinate_systems.transformer"] = coord_module  # type: ignore[assignment]
    spec.loader.exec_module(coord_module)
    CoordinateSystemTransformer = coord_module.CoordinateSystemTransformer

logger = logging.getLogger(__name__)


class VoxelDomainClient:
    """
    Main client for creating unified voxel domain representations.

    Integrates data from multiple sources (STL, hatching, laser, CT, ISPM) into
    a single, queryable voxel grid with proper synchronization and transformation.
    """

    def __init__(
        self,
        unified_query_client: UnifiedQueryClient,
        base_resolution: float = 0.5,
        adaptive: bool = True,
        target_coordinate_system: str = "build_platform",
    ):
        """
        Initialize voxel domain client.

        Args:
            unified_query_client: UnifiedQueryClient instance for querying data
            base_resolution: Base voxel resolution in mm (default: 0.5mm)
            adaptive: Whether to use adaptive resolution (default: True)
            target_coordinate_system: Target coordinate system for all data (default: 'build_platform')
        """
        self.unified_client = unified_query_client
        self.base_resolution = base_resolution
        self.adaptive = adaptive
        self.target_coordinate_system = target_coordinate_system

        # Initialize coordinate transformer (required, no fallback)
        if not CoordinateSystemTransformer:
            raise ImportError(
                "C++ bindings not available. "
                "Please build am_qadf_native with pybind11 bindings."
            )
        self.coord_transformer = CoordinateSystemTransformer()

    def create_voxel_grid(
        self,
        model_id: str,
        resolution: Optional[float] = None,
        bbox_min: Optional[Tuple[float, float, float]] = None,
        bbox_max: Optional[Tuple[float, float, float]] = None,
        adaptive: Optional[bool] = None,
        signals: Optional[List[str]] = None,
    ) -> Union[VoxelGrid, AdaptiveResolutionGrid]:
        """
        Create a voxel grid for a given model.

        Args:
            model_id: Model ID to create grid for
            resolution: Voxel resolution in mm (uses base_resolution if None)
            bbox_min: Minimum bounding box (uses STL bbox if None)
            bbox_max: Maximum bounding box (uses STL bbox if None)
            adaptive: Whether to use adaptive resolution (uses self.adaptive if None)
            signals: List of signal names to include (optional, for metadata)

        Returns:
            VoxelGrid or AdaptiveResolutionGrid instance
        """
        # Get resolution
        if resolution is None:
            resolution = self.base_resolution

        # Get bounding box from STL model if not provided
        if bbox_min is None or bbox_max is None:
            # Use stl_client from unified_client
            if hasattr(self.unified_client, "stl_client") and self.unified_client.stl_client:
                # Try using the dedicated get_model_bounding_box method first (most reliable)
                if hasattr(self.unified_client.stl_client, "get_model_bounding_box"):
                    try:
                        bbox_result = self.unified_client.stl_client.get_model_bounding_box(model_id)
                        if bbox_result:
                            bbox_min_result, bbox_max_result = bbox_result
                            if bbox_min is None:
                                bbox_min = bbox_min_result
                            if bbox_max is None:
                                bbox_max = bbox_max_result
                        else:
                            raise ValueError(f"get_model_bounding_box returned None for model {model_id}")
                    except Exception as e:
                        logger.warning(f"Error using get_model_bounding_box: {e}, falling back to get_model")
                        # Fall through to fallback method
                        bbox_result = None

                # Fallback: get model and extract bounding box
                if bbox_min is None or bbox_max is None:
                    stl_data = self.unified_client.stl_client.get_model(model_id)
                    if not stl_data:
                        raise ValueError(f"Could not retrieve STL data for model {model_id}")

                    # Try different possible bounding box formats
                    bbox = None
                    if "bounding_box" in stl_data:
                        bbox = stl_data["bounding_box"]
                    elif "metadata" in stl_data and "bounding_box" in stl_data["metadata"]:
                        bbox = stl_data["metadata"]["bounding_box"]

                    if bbox:
                        # Handle different bbox formats
                        if isinstance(bbox, dict):
                            if "min" in bbox and "max" in bbox:
                                if bbox_min is None:
                                    bbox_min = tuple(bbox["min"])
                                if bbox_max is None:
                                    bbox_max = tuple(bbox["max"])
                            elif "bbox_min" in bbox and "bbox_max" in bbox:
                                if bbox_min is None:
                                    bbox_min = tuple(bbox["bbox_min"])
                                if bbox_max is None:
                                    bbox_max = tuple(bbox["bbox_max"])
                        elif isinstance(bbox, (list, tuple)) and len(bbox) == 2:
                            # Format: [(min_x, min_y, min_z), (max_x, max_y, max_z)]
                            if bbox_min is None:
                                bbox_min = tuple(bbox[0])
                            if bbox_max is None:
                                bbox_max = tuple(bbox[1])
                    else:
                        logger.warning(
                            f"No bounding box found in STL data for model {model_id}. Available keys: {list(stl_data.keys())}"
                        )
                        raise ValueError(
                            f"Could not retrieve bounding box for model {model_id}. Available keys: {list(stl_data.keys())}"
                        )
            else:
                # Fallback: try to get from get_all_data
                all_data = self.unified_client.get_all_data(model_id)
                stl_data = all_data.get("stl_model")
                if stl_data and "bounding_box" in stl_data:
                    bbox = stl_data["bounding_box"]
                    if isinstance(bbox, dict) and "min" in bbox and "max" in bbox:
                        if bbox_min is None:
                            bbox_min = tuple(bbox["min"])
                        if bbox_max is None:
                            bbox_max = tuple(bbox["max"])
                    else:
                        raise ValueError(f"Unexpected bounding box format for model {model_id}")
                else:
                    raise ValueError(f"Could not retrieve bounding box for model {model_id}")

        # Determine if adaptive
        use_adaptive = adaptive if adaptive is not None else self.adaptive

        if use_adaptive:
            # Create adaptive resolution grid
            grid: Union[VoxelGrid, AdaptiveResolutionGrid] = AdaptiveResolutionGrid(
                bbox_min=bbox_min, bbox_max=bbox_max, base_resolution=resolution
            )
        else:
            # Create regular voxel grid
            grid = VoxelGrid(bbox_min=bbox_min, bbox_max=bbox_max, resolution=resolution)

        logger.info(
            f"Created voxel grid for model {model_id}: "
            f"bbox={bbox_min} to {bbox_max}, resolution={resolution}mm, "
            f"adaptive={use_adaptive}"
        )

        return grid

    def map_signals_to_voxels(
        self,
        model_id: str,
        voxel_grid: Union[VoxelGrid, AdaptiveResolutionGrid],
        sources: Optional[List[str]] = None,
        spatial_query: Optional[Any] = None,
        temporal_query: Optional[Any] = None,
        interpolation_method: str = "nearest",
        use_parallel_sources: bool = False,
        max_workers: Optional[int] = None,
        **interpolation_kwargs,
    ) -> Union[VoxelGrid, AdaptiveResolutionGrid]:
        """
        Map signals from all data sources to the voxel grid.

        Args:
            model_id: Model ID
            voxel_grid: Voxel grid to map signals to
            sources: List of sources to include (['hatching', 'laser', 'ct', 'ispm'])
                    If None, includes all sources
            spatial_query: Optional spatial query to filter data
            temporal_query: Optional temporal query to filter data
            interpolation_method: Interpolation method ('nearest', 'linear', 'idw', 'gaussian_kde')
            use_parallel_sources: Whether to process sources in parallel (default: False)
            **interpolation_kwargs: Additional arguments for interpolation method

        Returns:
            Voxel grid with mapped signals
        """
        if sources is None:
            sources = ["hatching", "laser", "ct", "ispm"]

        logger.info(f"Mapping signals from sources {sources} to voxel grid for model {model_id}")

        # Store interpolation kwargs for use in mapping methods
        self._interpolation_kwargs = {
            "method": interpolation_method,
            **interpolation_kwargs,
        }

        if use_parallel_sources and len(sources) > 1:
            # Process sources in parallel
            from concurrent.futures import ThreadPoolExecutor, as_completed

            source_results = {}
            with ThreadPoolExecutor(max_workers=max_workers or len(sources)) as executor:
                futures = {}
                for source in sources:
                    future = executor.submit(
                        self._map_source_data,
                        source,
                        model_id,
                        voxel_grid,
                        spatial_query,
                        temporal_query,
                        interpolation_method,
                        **interpolation_kwargs,
                    )
                    futures[future] = source

                for future in as_completed(futures):
                    source = futures[future]
                    print(f"   Processing {source} data...", end=" ", flush=True)
                    try:
                        future.result()
                        print("✅")
                        source_results[source] = "success"
                    except Exception as e:
                        print("❌")
                        logger.error(f"Error mapping {source} data: {e}", exc_info=True)
                        source_results[source] = f"error: {str(e)}"
        else:
            # Sequential processing (original behavior)
            source_results = {}
            for i, source in enumerate(sources, 1):
                print(
                    f"   [{i}/{len(sources)}] Processing {source} data...",
                    end=" ",
                    flush=True,
                )
                try:
                    self._map_source_data(
                        source,
                        model_id,
                        voxel_grid,
                        spatial_query,
                        temporal_query,
                        interpolation_method,
                        **interpolation_kwargs,
                    )
                    print("✅")
                    source_results[source] = "success"
                except Exception as e:
                    print("❌")
                    logger.error(f"Error mapping {source} data: {e}", exc_info=True)
                    source_results[source] = f"error: {str(e)}"

        # Finalize voxel grid
        if hasattr(voxel_grid, "finalize"):
            voxel_grid.finalize()

        # Log available signals if the attribute exists
        if hasattr(voxel_grid, "available_signals"):
            logger.info(f"Signal mapping complete. Available signals: {voxel_grid.available_signals}")
        else:
            logger.info("Signal mapping complete.")

        return voxel_grid

    def _map_source_data(
        self,
        source: str,
        model_id: str,
        voxel_grid: Union[VoxelGrid, AdaptiveResolutionGrid],
        spatial_query: Optional[Any],
        temporal_query: Optional[Any],
            interpolation_method: str,
        **interpolation_kwargs,
    ):
        """Map data from a single source (helper for parallel processing)."""
        if source == "hatching":
            self._map_hatching_data(
                model_id,
                voxel_grid,
                spatial_query,
                temporal_query,
                interpolation_method,
                **interpolation_kwargs,
            )
        elif source == "laser":
            self._map_laser_data(
                model_id,
                voxel_grid,
                spatial_query,
                temporal_query,
                interpolation_method,
                **interpolation_kwargs,
            )
        elif source == "ct":
            self._map_ct_data(
                model_id,
                voxel_grid,
                spatial_query,
                interpolation_method,
                **interpolation_kwargs,
            )
        elif source == "ispm":
            self._map_ispm_data(
                model_id,
                voxel_grid,
                spatial_query,
                temporal_query,
                interpolation_method,
                **interpolation_kwargs,
            )
        else:
            logger.warning(f"Unknown source: {source}")

    def _map_hatching_data(
        self,
        model_id: str,
        voxel_grid: Union[VoxelGrid, AdaptiveResolutionGrid],
        spatial_query: Optional[Any],
        temporal_query: Optional[Any],
        interpolation_method: str = "nearest",
        **interpolation_kwargs,
    ):
        """Map hatching path data to voxel grid."""
        # Use hatching_client from unified_client
        if not hasattr(self.unified_client, "hatching_client") or not self.unified_client.hatching_client:
            logger.warning("Hatching client not available")
            return

        # Get hatching layers using the client's get_layers method
        try:
            layers = self.unified_client.hatching_client.get_layers(model_id)
            if not layers:
                logger.warning("No hatching layers found")
                return

            # Extract paths and signals from each layer
            all_paths: List[np.ndarray] = []
            all_signals: Dict[str, List[np.ndarray]] = {}

            for layer in layers:
                if "hatches" not in layer:
                    continue

                for hatch in layer["hatches"]:
                    if "points" in hatch and len(hatch["points"]) > 0:
                        path = np.array(hatch["points"])
                        all_paths.append(path)

                        # Extract signals
                        if "laser_power" in hatch:
                            if "laser_power" not in all_signals:
                                all_signals["laser_power"] = []
                            all_signals["laser_power"].append(np.full(len(path), hatch["laser_power"]))

                        if "scan_speed" in hatch:
                            if "scan_speed" not in all_signals:
                                all_signals["scan_speed"] = []
                            all_signals["scan_speed"].append(np.full(len(path), hatch["scan_speed"]))

            # Interpolate hatching paths to voxel grid
            if len(all_paths) > 0:
                # interpolate_hatching_paths expects VoxelGrid but works with AdaptiveResolutionGrid at runtime
                interpolate_hatching_paths(  # type: ignore[arg-type]
                    all_paths,
                    all_signals,
                    voxel_grid,  # type: ignore[arg-type]
                    interpolation_method=interpolation_method,
                    **interpolation_kwargs,
                )
                logger.info(f"Mapped {len(all_paths)} hatching paths to voxel grid")
        except Exception as e:
            logger.error(f"Error mapping hatching data: {e}", exc_info=True)

    def _map_laser_data(
        self,
        model_id: str,
        voxel_grid: Union[VoxelGrid, AdaptiveResolutionGrid],
        spatial_query: Optional[Any],
        temporal_query: Optional[Any],
        interpolation_method: str = "nearest",
        **interpolation_kwargs,
    ):
        """Map laser parameter data to voxel grid."""
        # Use laser_client from unified_client
        if not hasattr(self.unified_client, "laser_client") or not self.unified_client.laser_client:
            logger.warning("Laser client not available")
            return

        try:
            # Create query objects if not provided
            if spatial_query is None:
                try:
                    from ..query.base_query_client import SpatialQuery

                    spatial_query = SpatialQuery(component_id=model_id)
                except ImportError:
                    # Create a minimal SpatialQuery-like object if import fails
                    class SimpleSpatialQuery:
                        def __init__(
                            self,
                            component_id=None,
                            bbox_min=None,
                            bbox_max=None,
                            layer_range=None,
                        ):
                            self.component_id = component_id
                            self.bbox_min = bbox_min
                            self.bbox_max = bbox_max
                            self.layer_range = layer_range

                    spatial_query = SimpleSpatialQuery(component_id=model_id)

            # Query laser data
            laser_result = self.unified_client.laser_client.query(spatial=spatial_query, temporal=temporal_query)

            if not laser_result or laser_result.points is None or len(laser_result.points) == 0:
                logger.warning("No laser parameter data found")
                return

            # Extract points and signals from QueryResult
            points = np.array(laser_result.points)
            signals = {}

            if laser_result.signals:
                for signal_name, signal_values in laser_result.signals.items():
                    if signal_values is not None and len(signal_values) == len(points):
                        signals[signal_name] = np.array(signal_values)

            # Interpolate to voxel grid
            if len(points) > 0:
                # interpolate_to_voxels expects VoxelGrid but works with AdaptiveResolutionGrid at runtime
                interpolate_to_voxels(  # type: ignore[arg-type]
                    points,
                    signals,
                    voxel_grid,  # type: ignore[arg-type]
                    method=interpolation_method,
                    **interpolation_kwargs,
                )
                logger.info(f"Mapped {len(points)} laser parameter points to voxel grid")
        except Exception as e:
            logger.error(f"Error mapping laser data: {e}", exc_info=True)

    def _map_ct_data(
        self,
        model_id: str,
        voxel_grid: Union[VoxelGrid, AdaptiveResolutionGrid],
        spatial_query: Optional[Any],
        interpolation_method: str = "nearest",
        **interpolation_kwargs,
    ):
        """Map CT scan data to voxel grid."""
        # Use ct_client from unified_client
        if not hasattr(self.unified_client, "ct_client") or not self.unified_client.ct_client:
            logger.warning("CT client not available")
            return

        try:
            # Create query objects if not provided
            if spatial_query is None:
                try:
                    from ..query.base_query_client import SpatialQuery

                    spatial_query = SpatialQuery(component_id=model_id)
                except ImportError:
                    # Create a minimal SpatialQuery-like object if import fails
                    class SimpleSpatialQuery:
                        def __init__(
                            self,
                            component_id=None,
                            bbox_min=None,
                            bbox_max=None,
                            layer_range=None,
                        ):
                            self.component_id = component_id
                            self.bbox_min = bbox_min
                            self.bbox_max = bbox_max
                            self.layer_range = layer_range

                    spatial_query = SimpleSpatialQuery(component_id=model_id)

            # Query CT scan data
            ct_result = self.unified_client.ct_client.query(spatial=spatial_query)

            if not ct_result or ct_result.points is None or len(ct_result.points) == 0:
                logger.warning("No CT scan data found")
                return

            # Extract points and signals from QueryResult
            points = np.array(ct_result.points)
            signals = {}

            if ct_result.signals:
                for signal_name, signal_values in ct_result.signals.items():
                    if signal_values is not None and len(signal_values) == len(points):
                        signals[signal_name] = np.array(signal_values)

            # Interpolate to voxel grid
            if len(points) > 0:
                # interpolate_to_voxels expects VoxelGrid but works with AdaptiveResolutionGrid at runtime
                interpolate_to_voxels(  # type: ignore[arg-type]
                    points,
                    signals,
                    voxel_grid,  # type: ignore[arg-type]
                    method=interpolation_method,
                    **interpolation_kwargs,
                )
                logger.info(f"Mapped {len(points)} CT scan points to voxel grid")
        except Exception as e:
            logger.error(f"Error mapping CT data: {e}", exc_info=True)

    def _map_ispm_data(
        self,
        model_id: str,
        voxel_grid: Union[VoxelGrid, AdaptiveResolutionGrid],
        spatial_query: Optional[Any],
        temporal_query: Optional[Any],
        interpolation_method: str = "nearest",
        **interpolation_kwargs,
    ):
        """Map ISPM monitoring data to voxel grid."""
        # Use ispm_client from unified_client
        if not hasattr(self.unified_client, "ispm_client") or not self.unified_client.ispm_client:
            logger.warning("ISPM client not available")
            return

        try:
            # Create query objects if not provided
            if spatial_query is None:
                try:
                    from ..query.base_query_client import SpatialQuery

                    spatial_query = SpatialQuery(component_id=model_id)
                except ImportError:
                    # Create a minimal SpatialQuery-like object if import fails
                    class SimpleSpatialQuery:
                        def __init__(
                            self,
                            component_id=None,
                            bbox_min=None,
                            bbox_max=None,
                            layer_range=None,
                        ):
                            self.component_id = component_id
                            self.bbox_min = bbox_min
                            self.bbox_max = bbox_max
                            self.layer_range = layer_range

                    spatial_query = SimpleSpatialQuery(component_id=model_id)

            # Query ISPM data
            ispm_result = self.unified_client.ispm_client.query(spatial=spatial_query, temporal=temporal_query)

            if not ispm_result or ispm_result.points is None or len(ispm_result.points) == 0:
                logger.warning("No ISPM monitoring data found")
                return

            # Extract points and signals from QueryResult
            points = np.array(ispm_result.points)
            signals = {}

            if ispm_result.signals:
                for signal_name, signal_values in ispm_result.signals.items():
                    if signal_values is not None and len(signal_values) == len(points):
                        signals[signal_name] = np.array(signal_values)

            # Interpolate to voxel grid
            if len(points) > 0:
                # interpolate_to_voxels expects VoxelGrid but works with AdaptiveResolutionGrid at runtime
                interpolate_to_voxels(  # type: ignore[arg-type]
                    points,
                    signals,
                    voxel_grid,  # type: ignore[arg-type]
                    method=interpolation_method,
                    **interpolation_kwargs,
                )
                logger.info(f"Mapped {len(points)} ISPM monitoring points to voxel grid")
        except Exception as e:
            logger.error(f"Error mapping ISPM data: {e}", exc_info=True)

    def get_voxel_statistics(self, voxel_grid: Union[VoxelGrid, AdaptiveResolutionGrid]) -> Dict[str, Any]:
        """
        Get statistics about the voxel grid.

        Args:
            voxel_grid: Voxel grid to analyze

        Returns:
            Dictionary with statistics
        """
        if hasattr(voxel_grid, "get_statistics"):
            return voxel_grid.get_statistics()
        else:
            # Basic statistics
            return {
                "available_signals": (list(voxel_grid.available_signals) if hasattr(voxel_grid, "available_signals") else []),
                "resolution": getattr(voxel_grid, "resolution", None),
                "bbox_min": (tuple(voxel_grid.bbox_min) if hasattr(voxel_grid, "bbox_min") else None),
                "bbox_max": (tuple(voxel_grid.bbox_max) if hasattr(voxel_grid, "bbox_max") else None),
            }
