"""
Thermal Client

Query client for thermal field data.
Initially uses simulated thermal fields from energy density.
Future: Replace with real thermal sensor data.
"""

from typing import Optional, List, Tuple, Dict, Any
import numpy as np

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

    # Try to find and import base_query_client
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


class ThermalClient(BaseQueryClient):
    """
    Query client for thermal field data.

    Currently generates thermal fields from energy density.
    Future: Query real thermal sensor data from data warehouse.
    """

    def __init__(self, laser_client=None, thermal_generator=None):
        """
        Initialize thermal client.

        Args:
            laser_client: LaserParameterClient to get energy density
            thermal_generator: ThermalFieldGenerator instance (optional)
        """
        super().__init__()
        self.laser_client = laser_client

        # Import thermal generator
        try:
            from ..processing.signal_generation import ThermalFieldGenerator

            self.thermal_generator = thermal_generator or ThermalFieldGenerator()
        except ImportError:
            # Fallback import
            import sys
            from pathlib import Path

            current_file = Path(__file__).resolve()
            gen_path = current_file.parent.parent / "processing" / "signal_generation.py"

            if gen_path.exists():
                import importlib.util

                spec = importlib.util.spec_from_file_location("signal_generation", gen_path)
                gen_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(gen_module)
                self.thermal_generator = thermal_generator or gen_module.ThermalFieldGenerator()
            else:
                raise ImportError("Could not import ThermalFieldGenerator")

    def query(
        self,
        spatial: Optional[SpatialQuery] = None,
        temporal: Optional[TemporalQuery] = None,
        signal_types: Optional[List[SignalType]] = None,
    ) -> QueryResult:
        """
        Query thermal field data.

        Currently generates thermal field from energy density.
        Future: Query real sensor data.

        Args:
            spatial: Spatial query constraints
            temporal: Temporal query constraints
            signal_types: List of signal types to query

        Returns:
            QueryResult with thermal field data
        """
        if self.laser_client is None:
            raise ValueError("LaserParameterClient required to generate thermal field")

        # Query energy density from laser client
        energy_result = self.laser_client.query(spatial=spatial, temporal=temporal, signal_types=[SignalType.ENERGY])

        if "energy" not in energy_result.signals:
            return QueryResult(points=[], signals={}, metadata={"error": "No energy data available"})

        # Generate thermal field from energy density
        energy_values = energy_result.signals["energy"]
        points = energy_result.points

        # Convert energy to thermal field
        # Energy density is in J/mm³, convert to J/m³ for thermal generator
        energy_density_m3 = np.array(energy_values) * 1e9  # J/mm³ to J/m³

        # Generate thermal field
        # Use voxel size of 1mm (0.001m) for now
        voxel_size = 0.001  # meters
        temperature = self.thermal_generator.generate_thermal_field(
            energy_density_m3,
            voxel_size=voxel_size,
            apply_diffusion=True,
            apply_cooling=True,
            time_steps=1,
        )

        # Build result
        signals = {"temperature": (temperature.tolist() if isinstance(temperature, np.ndarray) else temperature)}

        # Add energy for reference
        signals["energy"] = energy_values

        return QueryResult(
            points=points,
            signals=signals,
            metadata={
                "source": "simulated",
                "generator": "ThermalFieldGenerator",
                "voxel_size": voxel_size,
            },
        )

    def get_available_signals(self) -> List[SignalType]:
        """
        Get list of available signal types.

        Returns:
            List of available SignalType enums
        """
        return [SignalType.THERMAL, SignalType.TEMPERATURE, SignalType.ENERGY]

    def get_bounding_box(
        self, component_id: Optional[str] = None
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """
        Get bounding box of thermal data.

        Args:
            component_id: Optional component ID to get bounding box for specific component

        Returns:
            Tuple of (bbox_min, bbox_max)
        """
        if self.laser_client is None:
            raise ValueError("LaserParameterClient required")

        return self.laser_client.get_bounding_box(component_id=component_id)

    def get_layer_count(self) -> int:
        """
        Get number of layers in thermal data.

        Returns:
            Number of layers
        """
        if self.laser_client is None:
            raise ValueError("LaserParameterClient required")

        if hasattr(self.laser_client, "get_layer_count"):
            return self.laser_client.get_layer_count()

        return 0
