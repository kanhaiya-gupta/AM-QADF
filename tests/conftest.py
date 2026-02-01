"""
Shared pytest fixtures for AM-QADF tests.

This module provides common fixtures used across all test modules.
"""

import pytest
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
from unittest.mock import Mock, MagicMock
import sys

# Set matplotlib to use non-interactive backend for headless environments
# This must be done before any matplotlib imports
import matplotlib

matplotlib.use("Agg")

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ============================================================================
# Core Fixtures
# ============================================================================


@pytest.fixture
def sample_points_3d():
    """Generate sample 3D points for testing."""
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [0.5, 0.5, 0.5],
            [1.5, 1.5, 1.5],
        ]
    )


@pytest.fixture
def sample_values():
    """Generate sample signal values for testing."""
    return np.array([1.0, 2.0, 3.0, 1.5, 2.5])


@pytest.fixture
def sample_voxel_grid():
    """Create a test voxel grid."""
    try:
        from am_qadf.voxelization import VoxelGrid

        return VoxelGrid(origin=(0.0, 0.0, 0.0), resolution=1.0, dimensions=(10, 10, 10))
    except ImportError:
        pytest.skip("VoxelGrid not available")


@pytest.fixture
def small_voxel_grid():
    """Create a small test voxel grid (5x5x5)."""
    try:
        from am_qadf.voxelization import VoxelGrid

        return VoxelGrid(origin=(0.0, 0.0, 0.0), resolution=0.5, dimensions=(5, 5, 5))
    except ImportError:
        pytest.skip("VoxelGrid not available")


@pytest.fixture
def medium_voxel_grid():
    """Create a medium test voxel grid (20x20x20)."""
    try:
        from am_qadf.voxelization import VoxelGrid

        return VoxelGrid(origin=(0.0, 0.0, 0.0), resolution=0.5, dimensions=(20, 20, 20))
    except ImportError:
        pytest.skip("VoxelGrid not available")


# ============================================================================
# Signal Mapping Fixtures
# ============================================================================


@pytest.fixture
def sample_hatching_paths():
    """Generate sample hatching path data."""
    return {
        "points": np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        ),
        "signals": {
            "speed": np.array([100.0, 100.0, 100.0]),
            "power": np.array([200.0, 200.0, 200.0]),
        },
    }


@pytest.fixture
def sample_laser_points():
    """Generate sample laser parameter points."""
    return {
        "points": np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ]
        ),
        "signals": {
            "laser_power": np.array([250.0, 250.0]),
            "laser_speed": np.array([150.0, 150.0]),
            "energy_density": np.array([1.67, 1.67]),
        },
    }


@pytest.fixture
def sample_ct_points():
    """Generate sample CT scan points."""
    return {
        "points": np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.5],
                [1.0, 1.0, 1.0],
            ]
        ),
        "signals": {
            "density": np.array([0.95, 0.98, 0.97]),
            "porosity": np.array([0.05, 0.02, 0.03]),
        },
    }


# ============================================================================
# Mock Fixtures
# ============================================================================


@pytest.fixture
def mock_mongodb_client():
    """Create a mock MongoDB client.

    config.url and config.database must be real strings so C++ MongoDBQueryClient(uri, db_name)
    receives (str, str); MagicMock objects cause TypeError in pybind11.

    Uses fixed non-secret values only. Tests that call the C++ client skip when MongoDB is not
    available or not authenticated (no env file or credentials in repo).
    """
    mock_client = MagicMock()
    mock_client.db = MagicMock()
    mock_client.gridfs = MagicMock()
    mock_client.config = MagicMock()
    mock_client.config.url = "mongodb://localhost:27017"
    mock_client.config.database = "am_qadf_data"
    return mock_client


@pytest.fixture
def mock_query_client():
    """Create a mock query client."""
    mock_client = MagicMock()

    # Mock query methods
    mock_client.query_hatching_paths.return_value = {
        "points": np.array([[0, 0, 0], [1, 1, 1]]),
        "signals": {"speed": np.array([100, 100])},
    }

    mock_client.query_laser_parameters.return_value = {
        "points": np.array([[0, 0, 0]]),
        "signals": {"power": np.array([200])},
    }

    return mock_client


# ============================================================================
# Coordinate System Fixtures
# ============================================================================


@pytest.fixture
def sample_coordinate_systems():
    """Provide sample coordinate system configurations."""
    return {
        "build_platform": {
            "origin": (0.0, 0.0, 0.0),
            "axes": {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]},
        },
        "machine": {
            "origin": (100.0, 100.0, 0.0),
            "axes": {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]},
        },
    }


# ============================================================================
# Fusion Fixtures
# ============================================================================


@pytest.fixture
def sample_signal_arrays():
    """Generate sample signal arrays for fusion testing."""
    shape = (10, 10, 10)
    return {
        "laser_power": np.random.rand(*shape) * 300.0,
        "temperature": np.random.rand(*shape) * 1000.0,
        "density": np.random.rand(*shape) * 1.0,
    }


@pytest.fixture
def sample_quality_scores():
    """Generate sample quality scores for fusion."""
    return {"laser_power": 0.9, "temperature": 0.8, "density": 0.95}


# ============================================================================
# Analytics Fixtures
# ============================================================================


@pytest.fixture
def sample_analysis_data():
    """Generate sample data for analytics testing."""
    return {
        "parameters": {
            "laser_power": np.linspace(100, 300, 100),
            "laser_speed": np.linspace(50, 200, 100),
        },
        "outputs": {
            "density": np.random.rand(100) * 0.2 + 0.8,
            "porosity": np.random.rand(100) * 0.1,
        },
    }


# ============================================================================
# Performance Test Fixtures
# ============================================================================


@pytest.fixture
def large_point_cloud():
    """Generate a large point cloud for performance testing."""
    n_points = 100000
    return {
        "points": np.random.rand(n_points, 3) * 100.0,
        "values": np.random.rand(n_points) * 100.0,
    }


@pytest.fixture
def large_voxel_grid():
    """Create a large voxel grid for performance testing."""
    try:
        from am_qadf.voxelization import VoxelGrid

        return VoxelGrid(origin=(0.0, 0.0, 0.0), resolution=0.1, dimensions=(100, 100, 100))
    except ImportError:
        pytest.skip("VoxelGrid not available")


# ============================================================================
# Utility Fixtures
# ============================================================================


@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory for test files."""
    return tmp_path


@pytest.fixture
def tolerance():
    """Default numerical tolerance for assertions."""
    return 1e-6


@pytest.fixture
def rtol():
    """Default relative tolerance for assertions."""
    return 1e-5


@pytest.fixture
def atol():
    """Default absolute tolerance for assertions."""
    return 1e-8
