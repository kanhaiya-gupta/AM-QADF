"""
Base Query Client

Abstract base class for query clients with standardized query interface.
Defines common query patterns: spatial, temporal, signal type queries.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class SignalType(Enum):
    """Types of signals available for querying."""

    POWER = "power"
    VELOCITY = "velocity"
    ENERGY = "energy"
    THERMAL = "thermal"
    TEMPERATURE = "temperature"  # Explicit temperature signal
    DENSITY = "density"
    STRESS = "stress"


@dataclass
class SpatialQuery:
    """Spatial query parameters."""

    bbox_min: Optional[Tuple[float, float, float]] = None  # (x_min, y_min, z_min) in mm
    bbox_max: Optional[Tuple[float, float, float]] = None  # (x_max, y_max, z_max) in mm
    component_id: Optional[str] = None  # Filter by component ID
    layer_range: Optional[Tuple[int, int]] = None  # (start_layer, end_layer)
    max_results: Optional[int] = None  # Maximum number of results to return (performance optimization)

    def is_empty(self) -> bool:
        """Check if query has no spatial constraints."""
        return self.bbox_min is None and self.bbox_max is None and self.component_id is None and self.layer_range is None


@dataclass
class TemporalQuery:
    """Temporal query parameters."""

    time_start: Optional[float] = None  # Start time (seconds or layer index)
    time_end: Optional[float] = None  # End time (seconds or layer index)
    layer_start: Optional[int] = None  # Start layer index
    layer_end: Optional[int] = None  # End layer index

    def is_empty(self) -> bool:
        """Check if query has no temporal constraints."""
        return self.time_start is None and self.time_end is None and self.layer_start is None and self.layer_end is None


@dataclass
class QueryResult:
    """Standardized query result format."""

    points: List[Tuple[float, float, float]]  # List of (x, y, z) coordinates in mm
    signals: Dict[str, List[float]]  # Dictionary mapping signal names to value arrays
    metadata: Dict[str, Any]  # Additional metadata (layer info, timestamps, etc.)
    component_id: Optional[str] = None  # Component ID if applicable

    def __post_init__(self):
        """Validate result data."""
        if len(self.points) > 0:
            num_points = len(self.points)
            for signal_name, values in self.signals.items():
                if len(values) != num_points:
                    raise ValueError(
                        f"Signal '{signal_name}' has {len(values)} values, "
                        f"but expected {num_points} (matching number of points)"
                    )


class BaseQueryClient(ABC):
    """
    Abstract base class for query clients.

    All query clients should inherit from this class and implement the query methods.
    This provides a standardized interface for querying different data sources.
    """

    def __init__(self, data_source: Optional[str] = None):
        """
        Initialize query client.

        Args:
            data_source: Optional identifier for the data source (file path, database connection, etc.)
        """
        self.data_source = data_source
        self._available_signals: List[SignalType] = []

    @abstractmethod
    def query(
        self,
        spatial: Optional[SpatialQuery] = None,
        temporal: Optional[TemporalQuery] = None,
        signal_types: Optional[List[SignalType]] = None,
    ) -> QueryResult:
        """
        Execute a query with spatial, temporal, and signal type filters.

        Args:
            spatial: Spatial query parameters (bounding box, component, layer range)
            temporal: Temporal query parameters (time range, layer range)
            signal_types: List of signal types to retrieve (None = all available)

        Returns:
            QueryResult with points, signals, and metadata
        """
        pass

    @abstractmethod
    def get_available_signals(self) -> List[SignalType]:
        """
        Get list of available signal types for this data source.

        Returns:
            List of available SignalType enums
        """
        pass

    @abstractmethod
    def get_bounding_box(
        self, component_id: Optional[str] = None
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """
        Get bounding box of the data.

        Args:
            component_id: Optional component ID to get bounding box for specific component

        Returns:
            Tuple of (bbox_min, bbox_max) where each is (x, y, z) in mm
        """
        pass

    def validate_query(self, spatial: Optional[SpatialQuery], temporal: Optional[TemporalQuery]) -> None:
        """
        Validate query parameters.

        Args:
            spatial: Spatial query parameters
            temporal: Temporal query parameters

        Raises:
            ValueError: If query parameters are invalid
        """
        if spatial and not spatial.is_empty():
            if spatial.bbox_min and spatial.bbox_max:
                if any(spatial.bbox_min[i] > spatial.bbox_max[i] for i in range(3)):
                    raise ValueError("bbox_min must be less than bbox_max in all dimensions")

        if temporal and not temporal.is_empty():
            if temporal.layer_start is not None and temporal.layer_end is not None:
                if temporal.layer_start > temporal.layer_end:
                    raise ValueError("layer_start must be <= layer_end")
            if temporal.time_start is not None and temporal.time_end is not None:
                if temporal.time_start > temporal.time_end:
                    raise ValueError("time_start must be <= time_end")

    def __repr__(self) -> str:
        """String representation of query client."""
        return f"{self.__class__.__name__}(data_source={self.data_source})"
