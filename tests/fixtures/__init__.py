"""
Test fixtures and data for AM-QADF tests.

Contains test data, mock objects, and fixture utilities.
"""

import pytest
from pathlib import Path

# Import voxel data fixtures
try:
    from .voxel_data import (
        load_small_voxel_grid,
        load_medium_voxel_grid,
        load_large_voxel_grid,
    )

    VOXEL_FIXTURES_AVAILABLE = True
except ImportError:
    VOXEL_FIXTURES_AVAILABLE = False


@pytest.fixture
def small_voxel_grid():
    """Pytest fixture for small voxel grid (10x10x10)."""
    if not VOXEL_FIXTURES_AVAILABLE:
        pytest.skip("Voxel fixtures not available")
    return load_small_voxel_grid()


@pytest.fixture
def medium_voxel_grid():
    """Pytest fixture for medium voxel grid (50x50x50)."""
    if not VOXEL_FIXTURES_AVAILABLE:
        pytest.skip("Voxel fixtures not available")
    return load_medium_voxel_grid()


@pytest.fixture
def large_voxel_grid():
    """Pytest fixture for large voxel grid (100x100x100)."""
    if not VOXEL_FIXTURES_AVAILABLE:
        pytest.skip("Voxel fixtures not available")
    return load_large_voxel_grid()


# Export commonly used fixtures
__all__ = [
    "small_voxel_grid",
    "medium_voxel_grid",
    "large_voxel_grid",
    "load_small_voxel_grid",
    "load_medium_voxel_grid",
    "load_large_voxel_grid",
]
