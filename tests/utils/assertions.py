"""
Custom assertion functions for AM-QADF tests.

Provides domain-specific assertion helpers.
"""

import numpy as np
from typing import Optional, Tuple


def assert_voxel_coordinates_valid(coords: np.ndarray, grid_dimensions: Tuple[int, int, int]) -> None:
    """
    Assert that voxel coordinates are valid.

    Args:
        coords: Voxel coordinates array (N, 3)
        grid_dimensions: Grid dimensions (nx, ny, nz)
    """
    assert coords.shape[1] == 3, "Coordinates must be 3D"
    assert np.all(coords >= 0), "All coordinates must be non-negative"
    assert np.all(coords[:, 0] < grid_dimensions[0]), "X coordinates out of bounds"
    assert np.all(coords[:, 1] < grid_dimensions[1]), "Y coordinates out of bounds"
    assert np.all(coords[:, 2] < grid_dimensions[2]), "Z coordinates out of bounds"


def assert_interpolation_result_valid(
    result: np.ndarray, expected_shape: Tuple[int, int, int], allow_nan: bool = False
) -> None:
    """
    Assert that an interpolation result is valid.

    Args:
        result: Interpolation result array
        expected_shape: Expected shape (nx, ny, nz)
        allow_nan: Whether NaN values are allowed
    """
    assert result.shape == expected_shape, f"Result shape {result.shape} != expected {expected_shape}"

    if not allow_nan:
        assert not np.any(np.isnan(result)), "Result contains NaN values"

    assert not np.any(np.isinf(result)), "Result contains Inf values"


def assert_fusion_result_valid(fused: np.ndarray, source_signals: dict, allow_nan: bool = False) -> None:
    """
    Assert that a fusion result is valid.

    Args:
        fused: Fused signal array
        source_signals: Dictionary of source signal arrays
        allow_nan: Whether NaN values are allowed
    """
    # Check shape matches source signals
    first_signal = list(source_signals.values())[0]
    assert fused.shape == first_signal.shape, f"Fused shape {fused.shape} != source shape {first_signal.shape}"

    # Check values are within reasonable range
    if not allow_nan:
        assert not np.any(np.isnan(fused)), "Fused signal contains NaN"

    assert not np.any(np.isinf(fused)), "Fused signal contains Inf"


def assert_quality_metric_valid(metric: float, metric_name: str, expected_range: Tuple[float, float] = (0.0, 1.0)) -> None:
    """
    Assert that a quality metric is valid.

    Args:
        metric: Quality metric value
        metric_name: Name of the metric (for error messages)
        expected_range: (min, max) expected range
    """
    min_val, max_val = expected_range
    assert min_val <= metric <= max_val, f"{metric_name} = {metric} not in range [{min_val}, {max_val}]"

    assert not np.isnan(metric), f"{metric_name} is NaN"
    assert not np.isinf(metric), f"{metric_name} is Inf"
