"""
Test helper functions for AM-QADF tests.

Common utilities used across test modules.
"""

import numpy as np
from typing import Tuple, Optional


def assert_array_close(
    actual: np.ndarray,
    expected: np.ndarray,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    msg: Optional[str] = None,
) -> None:
    """
    Assert that two arrays are close within tolerance.

    Args:
        actual: Actual array
        expected: Expected array
        rtol: Relative tolerance
        atol: Absolute tolerance
        msg: Optional error message
    """
    if msg is None:
        msg = f"Arrays not close: actual={actual}, expected={expected}"

    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=msg)


def assert_voxel_grid_valid(voxel_grid) -> None:
    """
    Assert that a voxel grid is valid.

    Args:
        voxel_grid: VoxelGrid instance to validate
    """
    assert voxel_grid is not None, "Voxel grid is None"
    assert voxel_grid.dimensions is not None, "Voxel grid dimensions are None"
    assert len(voxel_grid.dimensions) == 3, "Voxel grid must be 3D"
    assert all(d > 0 for d in voxel_grid.dimensions), "All dimensions must be positive"
    assert voxel_grid.resolution > 0, "Resolution must be positive"


def create_test_points(n_points: int = 10, bounds: Tuple[float, float] = (0.0, 10.0)) -> np.ndarray:
    """
    Create test 3D points.

    Args:
        n_points: Number of points to generate
        bounds: (min, max) bounds for each coordinate

    Returns:
        Array of shape (n_points, 3)
    """
    min_val, max_val = bounds
    return np.random.uniform(min_val, max_val, size=(n_points, 3))


def create_test_signal_array(
    shape: Tuple[int, int, int],
    signal_name: str = "test_signal",
    value_range: Tuple[float, float] = (0.0, 100.0),
) -> np.ndarray:
    """
    Create a test signal array.

    Args:
        shape: Shape of the array (nx, ny, nz)
        signal_name: Name of the signal (for logging)
        value_range: (min, max) range for values

    Returns:
        Signal array
    """
    min_val, max_val = value_range
    return np.random.uniform(min_val, max_val, size=shape)


def assert_signal_valid(
    signal: np.ndarray,
    expected_shape: Optional[Tuple[int, ...]] = None,
    allow_nan: bool = False,
) -> None:
    """
    Assert that a signal array is valid.

    Args:
        signal: Signal array to validate
        expected_shape: Expected shape (optional)
        allow_nan: Whether NaN values are allowed
    """
    assert signal is not None, "Signal is None"
    assert isinstance(signal, np.ndarray), "Signal must be a numpy array"

    if expected_shape is not None:
        assert signal.shape == expected_shape, f"Signal shape {signal.shape} != expected {expected_shape}"

    if not allow_nan:
        assert not np.any(np.isnan(signal)), "Signal contains NaN values"

    assert not np.any(np.isinf(signal)), "Signal contains Inf values"
