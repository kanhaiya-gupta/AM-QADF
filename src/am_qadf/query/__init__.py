"""
AM-QADF Query Module

Query clients for accessing data from the NoSQL data warehouse.
Provides interfaces for querying different data sources:
- STL models
- Hatching paths
- Laser parameters
- CT scan data
- ISPM monitoring data (Thermal, Optical)
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
from .laser_monitoring_client import LaserMonitoringClient
from .ct_scan_client import CTScanClient
from .ispm_thermal_client import ISPMThermalClient
from .ispm_optical_client import ISPMOpticalClient
from .ispm_acoustic_client import ISPMAcousticClient
from .ispm_strain_client import ISPMStrainClient
from .ispm_plume_client import ISPMPlumeClient
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
    "LaserMonitoringClient",
    "CTScanClient",
    "ISPMThermalClient",
    "ISPMOpticalClient",
    "ISPMAcousticClient",
    "ISPMStrainClient",
    "ISPMPlumeClient",
    "UnifiedQueryClient",
]
