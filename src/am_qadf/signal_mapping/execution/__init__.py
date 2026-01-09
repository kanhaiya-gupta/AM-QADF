"""
Signal Mapping Execution Backends

Execution backends for signal mapping:
- Sequential: Default single-threaded execution
- Parallel: Multi-core parallel execution
- Spark: Distributed execution on Spark cluster
"""

from .sequential import (
    interpolate_to_voxels,
    interpolate_hatching_paths,
    INTERPOLATION_METHODS,
)

__all__ = [
    "interpolate_to_voxels",
    "interpolate_hatching_paths",
    "INTERPOLATION_METHODS",
]
