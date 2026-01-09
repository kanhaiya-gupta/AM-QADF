"""
Build Metadata Query Client

Query client for build file metadata, layer information, and component data.
Supports multi-component builds with component selection and isolation.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

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


@dataclass
class ComponentInfo:
    """Information about a component in a build."""

    component_id: str
    name: Optional[str]
    bbox_min: Tuple[float, float, float]
    bbox_max: Tuple[float, float, float]
    layer_count: int
    metadata: Dict[str, Any]


@dataclass
class BuildStyleInfo:
    """Information about a build style."""

    build_style_id: int
    laser_power: float
    laser_speed: float
    energy_density: float
    layer_indices: List[int]  # Which layers use this build style


class BuildMetadataClient(BaseQueryClient):
    """
    Query client for build file metadata.

    Provides access to:
    - Component information (multi-component builds)
    - Layer information
    - Build style information
    - Build platform metadata
    """

    def __init__(
        self,
        stl_part=None,
        generated_layers: Optional[List] = None,
        generated_models: Optional[List] = None,
        generated_build_styles: Optional[Dict] = None,
        component_id: Optional[str] = None,
    ):
        """
        Initialize build metadata client.

        Args:
            stl_part: pyslm.Part object
            generated_layers: List of generated layers
            generated_models: List of model objects
            generated_build_styles: Dictionary of build styles
            component_id: Optional component ID (for single component mode)
        """
        super().__init__(data_source="build_metadata")
        self.stl_part = stl_part
        self.generated_layers = generated_layers or []
        self.generated_models = generated_models or []
        self.generated_build_styles = generated_build_styles or {}
        self.component_id = component_id
        self._components: Dict[str, ComponentInfo] = {}
        self._build_styles: Dict[int, BuildStyleInfo] = {}

        # Initialize component and build style info
        self._update_component_info()
        self._update_build_style_info()

    def _update_component_info(self):
        """Update component information from data."""
        # For now, assume single component (can be extended for multi-component)
        if self.stl_part is not None:
            bbox = self.stl_part.boundingBox
            bbox_min = (bbox[0], bbox[1], bbox[2])
            bbox_max = (bbox[3], bbox[4], bbox[5])

            comp_id = self.component_id or "component_1"
            self._components[comp_id] = ComponentInfo(
                component_id=comp_id,
                name=getattr(self.stl_part, "name", None),
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                layer_count=len(self.generated_layers),
                metadata={
                    "origin": self.stl_part.origin,
                    "rotation": self.stl_part.rotation,
                },
            )

    def _update_build_style_info(self):
        """Update build style information from data."""
        # Map layers to build styles
        layers_per_style: Dict[int, List[int]] = {}

        for layer_idx, layer in enumerate(self.generated_layers):
            # Handle both real layer objects and mocks
            if hasattr(layer, "geometry") and layer.geometry:
                try:
                    bid = layer.geometry[0].bid
                    if bid not in layers_per_style:
                        layers_per_style[bid] = []
                    layers_per_style[bid].append(layer_idx)
                except (IndexError, AttributeError, TypeError):
                    # Skip if geometry structure is invalid
                    pass

        # Create BuildStyleInfo objects
        for bid, build_style in self.generated_build_styles.items():
            layer_indices = layers_per_style.get(bid, [])
            energy = build_style.laserPower / build_style.laserSpeed if build_style.laserSpeed > 0 else 0.0

            self._build_styles[bid] = BuildStyleInfo(
                build_style_id=bid,
                laser_power=build_style.laserPower,
                laser_speed=build_style.laserSpeed,
                energy_density=energy,
                layer_indices=layer_indices,
            )

    def query(
        self,
        spatial: Optional[SpatialQuery] = None,
        temporal: Optional[TemporalQuery] = None,
        signal_types: Optional[List[SignalType]] = None,
    ) -> QueryResult:
        """
        Query build metadata.

        Note: This client returns metadata rather than signal data.
        For signal data, use LaserParameterClient.

        Args:
            spatial: Spatial query parameters (component filtering)
            temporal: Temporal query parameters (layer range)
            signal_types: Not used for metadata queries

        Returns:
            QueryResult with metadata (points are layer centers, signals contain metadata)
        """
        self.validate_query(spatial, temporal)

        # Filter by component
        component_id = None
        if spatial and spatial.component_id:
            component_id = spatial.component_id

        if component_id and component_id not in self._components:
            return QueryResult(
                points=[],
                signals={},
                metadata={"error": f"Component {component_id} not found"},
            )

        # Get layer range
        layer_start = 0
        layer_end = len(self.generated_layers) - 1

        if temporal and temporal.layer_start is not None:
            layer_start = max(0, temporal.layer_start)
        if temporal and temporal.layer_end is not None:
            layer_end = min(len(self.generated_layers) - 1, temporal.layer_end)
        if spatial and spatial.layer_range:
            layer_start = max(layer_start, spatial.layer_range[0])
            layer_end = min(layer_end, spatial.layer_range[1])

        # Create points at layer centers (for visualization)
        points = []
        layer_info = []

        for layer_idx in range(layer_start, layer_end + 1):
            layer = self.generated_layers[layer_idx]
            z_height = float(layer.z) / 1000.0

            # Get layer center (approximate)
            if layer.geometry:
                # Use first geometry point as reference
                first_coord = (
                    layer.geometry[0].coords[0]
                    if hasattr(layer.geometry[0], "coords") and layer.geometry[0].coords
                    else (0.0, 0.0)
                )
                x, y = first_coord[0], first_coord[1]
            else:
                x, y = 0.0, 0.0

            points.append((x, y, z_height))

            # Get build style for this layer
            build_style_id = None
            if layer.geometry:
                build_style_id = layer.geometry[0].bid

            layer_info.append(
                {
                    "layer_index": layer_idx,
                    "z_height": z_height,
                    "build_style_id": build_style_id,
                    "geometry_count": len(layer.geometry),
                }
            )

        metadata = {
            "component_id": (component_id or list(self._components.keys())[0] if self._components else None),
            "layer_range": (layer_start, layer_end),
            "layer_info": layer_info,
            "total_layers": len(self.generated_layers),
            "build_styles": {
                bid: {
                    "power": bs.laser_power,
                    "speed": bs.laser_speed,
                    "energy": bs.energy_density,
                    "layers": bs.layer_indices,
                }
                for bid, bs in self._build_styles.items()
            },
        }

        return QueryResult(
            points=points,
            signals={},  # No signal data for metadata queries
            metadata=metadata,
        )

    def get_available_signals(self) -> List[SignalType]:
        """Metadata client doesn't provide signal data."""
        return []

    def get_bounding_box(
        self, component_id: Optional[str] = None
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """
        Get bounding box for a component or entire build.

        Args:
            component_id: Component ID (None = entire build)

        Returns:
            Tuple of (bbox_min, bbox_max)
        """
        if component_id:
            if component_id in self._components:
                comp = self._components[component_id]
                return (comp.bbox_min, comp.bbox_max)
            else:
                return ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

        # Return bounding box of all components
        if not self._components:
            return ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

        all_mins = [comp.bbox_min for comp in self._components.values()]
        all_maxs = [comp.bbox_max for comp in self._components.values()]

        bbox_min = (
            min(m[0] for m in all_mins),
            min(m[1] for m in all_mins),
            min(m[2] for m in all_mins),
        )
        bbox_max = (
            max(m[0] for m in all_maxs),
            max(m[1] for m in all_maxs),
            max(m[2] for m in all_maxs),
        )

        return (bbox_min, bbox_max)

    def list_components(self) -> List[str]:
        """
        List all component IDs in the build.

        Returns:
            List of component IDs
        """
        return list(self._components.keys())

    def get_component_info(self, component_id: str) -> Optional[ComponentInfo]:
        """
        Get information about a specific component.

        Args:
            component_id: Component ID

        Returns:
            ComponentInfo if found, None otherwise
        """
        return self._components.get(component_id)

    def get_build_styles(self) -> Dict[int, BuildStyleInfo]:
        """
        Get all build style information.

        Returns:
            Dictionary mapping build style ID to BuildStyleInfo
        """
        return self._build_styles.copy()

    def get_components(self) -> Dict[str, ComponentInfo]:
        """
        Get all component information.

        Returns:
            Dictionary mapping component ID to ComponentInfo
        """
        return self._components.copy()
