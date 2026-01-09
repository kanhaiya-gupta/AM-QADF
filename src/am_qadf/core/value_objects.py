"""
Core Value Objects

Immutable value objects for the AM-QADF framework.
These represent values that have no identity and are defined by their attributes.
"""

from typing import Optional, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class VoxelCoordinates:
    """
    Value object representing voxel coordinates for PBF-LB/M operations.

    This immutable object contains 3D coordinate information for voxels
    in PBF processes, including position, orientation, and metadata.
    """

    # Basic coordinates
    x: float  # mm
    y: float  # mm
    z: float  # mm

    # Voxel properties
    voxel_size: float = 0.1  # mm
    voxel_volume: Optional[float] = None  # mm³

    # Orientation (Euler angles in degrees)
    rotation_x: float = 0.0  # degrees
    rotation_y: float = 0.0  # degrees
    rotation_z: float = 0.0  # degrees

    # Voxel state
    is_solid: bool = True
    is_processed: bool = False
    is_defective: bool = False

    # Material properties
    material_density: Optional[float] = None  # g/cm³
    material_type: Optional[str] = None

    # Process information
    layer_number: Optional[int] = None
    scan_vector_id: Optional[str] = None
    processing_timestamp: Optional[datetime] = None

    # Quality metrics
    quality_score: Optional[float] = None  # 0-100
    temperature_peak: Optional[float] = None  # Celsius
    cooling_rate: Optional[float] = None  # K/s

    def __post_init__(self):
        """Calculate derived properties and validate."""
        if self.voxel_volume is None:
            object.__setattr__(self, "voxel_volume", self.voxel_size**3)
        self.validate()

    def validate(self) -> None:
        """Validate voxel coordinates."""
        # Coordinate validation - coordinates can be negative in 3D space
        # Only check for reasonable bounds (not infinite values)
        if not all(-1e6 <= coord <= 1e6 for coord in [self.x, self.y, self.z]):
            raise ValueError("Coordinates must be within reasonable bounds (-1e6 to 1e6)")

        # Voxel size validation
        if self.voxel_size <= 0:
            raise ValueError("Voxel size must be positive")

        # Rotation validation
        for rotation in [self.rotation_x, self.rotation_y, self.rotation_z]:
            if not -180 <= rotation <= 180:
                raise ValueError("Rotation angles must be between -180 and 180 degrees")

        # Material density validation
        if self.material_density is not None and self.material_density <= 0:
            raise ValueError("Material density must be positive")

        # Quality score validation
        if self.quality_score is not None and not 0 <= self.quality_score <= 100:
            raise ValueError("Quality score must be between 0 and 100")

        # Temperature validation
        if self.temperature_peak is not None and self.temperature_peak < 0:
            raise ValueError("Temperature cannot be negative")

        # Cooling rate validation
        if self.cooling_rate is not None and self.cooling_rate < 0:
            raise ValueError("Cooling rate cannot be negative")

    def get_coordinates(self) -> Tuple[float, float, float]:
        """Get coordinates as tuple."""
        return (self.x, self.y, self.z)

    def get_rotations(self) -> Tuple[float, float, float]:
        """Get rotations as tuple."""
        return (self.rotation_x, self.rotation_y, self.rotation_z)

    def distance_to(self, other: "VoxelCoordinates") -> float:
        """Calculate Euclidean distance to another voxel."""
        import math

        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)


@dataclass(frozen=True)
class QualityMetric:
    """
    Value object representing a quality metric.

    Immutable quality assessment value with metadata.
    """

    value: float
    metric_name: str
    unit: Optional[str] = None
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        """Validate quality metric after initialization."""
        self.validate()

    def validate(self) -> None:
        """Validate quality metric."""
        if not isinstance(self.value, (int, float)):
            raise ValueError("Quality metric value must be numeric")
        if not self.metric_name or not self.metric_name.strip():
            raise ValueError("Quality metric name cannot be empty")
