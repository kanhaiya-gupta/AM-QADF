"""
Voxel data fixtures for testing.

Provides pre-computed voxel grids in various sizes for use in tests.
"""

from pathlib import Path
import pickle

FIXTURE_DIR = Path(__file__).parent


def load_small_voxel_grid():
    """Load small voxel grid fixture (10x10x10)."""
    fixture_path = FIXTURE_DIR / "small_voxel_grid.pkl"
    if fixture_path.exists():
        with open(fixture_path, "rb") as f:
            return pickle.load(f)
    else:
        # Generate on-the-fly if not exists
        return generate_small_voxel_grid()


def load_medium_voxel_grid():
    """Load medium voxel grid fixture (50x50x50)."""
    fixture_path = FIXTURE_DIR / "medium_voxel_grid.pkl"
    if fixture_path.exists():
        with open(fixture_path, "rb") as f:
            return pickle.load(f)
    else:
        # Generate on-the-fly if not exists
        return generate_medium_voxel_grid()


def load_large_voxel_grid():
    """Load large voxel grid fixture (100x100x100)."""
    fixture_path = FIXTURE_DIR / "large_voxel_grid.pkl"
    if fixture_path.exists():
        with open(fixture_path, "rb") as f:
            return pickle.load(f)
    else:
        # Generate on-the-fly if not exists
        return generate_large_voxel_grid()


def generate_small_voxel_grid():
    """Generate small voxel grid (10x10x10)."""
    try:
        from am_qadf.voxelization.voxel_grid import VoxelGrid
        import numpy as np

        grid = VoxelGrid(bbox_min=(0.0, 0.0, 0.0), bbox_max=(10.0, 10.0, 10.0), resolution=1.0)

        # Add some sample signals
        np.random.seed(42)
        points = np.random.rand(100, 3) * 10.0

        # Add points to grid
        for point in points:
            grid.add_point(
                point[0],
                point[1],
                point[2],
                signals={
                    "laser_power": float(np.random.rand() * 300.0),
                    "temperature": float(np.random.rand() * 1000.0),
                },
            )

        # Finalize to aggregate signals
        grid.finalize()

        return grid
    except ImportError:
        raise ImportError("VoxelGrid not available. Cannot generate fixture.")


def generate_medium_voxel_grid():
    """Generate medium voxel grid (50x50x50)."""
    try:
        from am_qadf.voxelization.voxel_grid import VoxelGrid
        import numpy as np

        grid = VoxelGrid(bbox_min=(0.0, 0.0, 0.0), bbox_max=(50.0, 50.0, 50.0), resolution=1.0)

        # Add some sample signals
        np.random.seed(42)
        points = np.random.rand(1000, 3) * 50.0

        # Add points to grid
        for point in points:
            grid.add_point(
                point[0],
                point[1],
                point[2],
                signals={
                    "laser_power": float(np.random.rand() * 300.0),
                    "temperature": float(np.random.rand() * 1000.0),
                    "density": float(np.random.rand() * 1.0),
                },
            )

        # Finalize to aggregate signals
        grid.finalize()

        return grid
    except ImportError:
        raise ImportError("VoxelGrid not available. Cannot generate fixture.")


def generate_large_voxel_grid():
    """Generate large voxel grid (100x100x100)."""
    try:
        from am_qadf.voxelization.voxel_grid import VoxelGrid
        import numpy as np

        grid = VoxelGrid(bbox_min=(0.0, 0.0, 0.0), bbox_max=(100.0, 100.0, 100.0), resolution=1.0)

        # Add some sample signals
        np.random.seed(42)
        points = np.random.rand(5000, 3) * 100.0

        # Add points to grid
        for point in points:
            grid.add_point(
                point[0],
                point[1],
                point[2],
                signals={
                    "laser_power": float(np.random.rand() * 300.0),
                    "temperature": float(np.random.rand() * 1000.0),
                    "density": float(np.random.rand() * 1.0),
                    "velocity": float(np.random.rand() * 100.0),
                },
            )

        # Finalize to aggregate signals
        grid.finalize()

        return grid
    except ImportError:
        raise ImportError("VoxelGrid not available. Cannot generate fixture.")
