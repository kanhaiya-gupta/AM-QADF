"""
AM-QADF Query Module

Query clients for accessing data from the NoSQL data warehouse.
Provides interfaces for querying different data sources:
- STL models
- Hatching paths
- Laser parameters
- CT scan data
- ISPM monitoring data
- Thermal data
- Build metadata
"""

from .base_query_client import (
    BaseQueryClient,
    QueryResult,
    SpatialQuery,
    TemporalQuery,
    SignalType,
)

from .stl_model_client import STLModelClient
from .hatching_client import HatchingClient
from .laser_parameter_client import LaserParameterClient
from .ct_scan_client import CTScanClient
from .in_situ_monitoring_client import InSituMonitoringClient
from .thermal_client import ThermalClient
from .build_metadata_client import BuildMetadataClient, ComponentInfo, BuildStyleInfo
from .unified_query_client import UnifiedQueryClient

__all__ = [
    # Base classes
    "BaseQueryClient",
    "QueryResult",
    "SpatialQuery",
    "TemporalQuery",
    "SignalType",
    # Query clients
    "STLModelClient",
    "HatchingClient",
    "LaserParameterClient",
    "CTScanClient",
    "InSituMonitoringClient",
    "ThermalClient",
    "BuildMetadataClient",
    "UnifiedQueryClient",
    # Data classes
    "ComponentInfo",
    "BuildStyleInfo",
]
