"""
Point cloud fixtures for testing.

Provides pre-computed point cloud data for hatching paths, laser points, and CT scans.
"""

import json
import numpy as np
from pathlib import Path

FIXTURE_DIR = Path(__file__).parent


def load_hatching_paths():
    """Load hatching paths fixture."""
    fixture_path = FIXTURE_DIR / "hatching_paths.json"
    if fixture_path.exists():
        with open(fixture_path, "r") as f:
            return json.load(f)
    else:
        # Generate on-the-fly if not exists
        return generate_hatching_paths()


def load_laser_points():
    """Load laser points fixture."""
    fixture_path = FIXTURE_DIR / "laser_points.json"
    if fixture_path.exists():
        with open(fixture_path, "r") as f:
            return json.load(f)
    else:
        # Generate on-the-fly if not exists
        return generate_laser_points()


def load_ct_points():
    """Load CT scan points fixture."""
    fixture_path = FIXTURE_DIR / "ct_points.json"
    if fixture_path.exists():
        with open(fixture_path, "r") as f:
            return json.load(f)
    else:
        # Generate on-the-fly if not exists
        return generate_ct_points()


def generate_hatching_paths():
    """Generate hatching paths fixture."""
    np.random.seed(42)

    layers = []
    for layer_idx in range(5):
        hatches = []
        for hatch_idx in range(10):
            # Generate hatch path points
            n_points = 50
            x_start = np.random.rand() * 100.0
            y_start = np.random.rand() * 100.0
            z = layer_idx * 0.03  # 30 micron layer height

            points = []
            for i in range(n_points):
                x = x_start + (i / n_points) * 10.0  # 10mm hatch length
                y = y_start + np.random.rand() * 0.1  # Small variation
                points.append([float(x), float(y), float(z)])

            hatch = {
                "start_point": points[0],
                "end_point": points[-1],
                "points": points,
                "laser_power": float(200.0 + np.random.rand() * 50.0),
                "scan_speed": float(1000.0 + np.random.rand() * 200.0),
                "energy_density": float(50.0 + np.random.rand() * 20.0),
                "laser_beam_width": 0.1,
                "hatch_spacing": 0.15,  # 0.15mm spacing allows gaps to be visible at 0.1mm voxel resolution
                "overlap_percentage": float(50.0 + np.random.rand() * 10.0),
                "hatch_type": "line",
                "scan_order": hatch_idx,
            }
            hatches.append(hatch)

        layer = {
            "model_id": "test_model_001",
            "layer_index": layer_idx,
            "layer_height": 0.03,
            "z_position": z,
            "hatches": hatches,
            "contours": [],
            "processing_time": "2024-01-01T00:00:00Z",
        }
        layers.append(layer)

    return layers


def generate_laser_points():
    """Generate laser points fixture."""
    np.random.seed(42)

    points = []
    for i in range(1000):
        point = {
            "model_id": "test_model_001",
            "layer_index": int(np.random.randint(0, 5)),
            "point_id": f"point_{i}",
            "spatial_coordinates": [
                float(np.random.rand() * 100.0),
                float(np.random.rand() * 100.0),
                float(np.random.rand() * 5.0 * 0.03),  # 5 layers
            ],
            "laser_power": float(200.0 + np.random.rand() * 50.0),
            "scan_speed": float(1000.0 + np.random.rand() * 200.0),
            "hatch_spacing": 0.1,
            "energy_density": float(50.0 + np.random.rand() * 20.0),
            "exposure_time": float(0.1 + np.random.rand() * 0.05),
            "timestamp": "2024-01-01T00:00:00Z",
            "region_type": np.random.choice(["contour", "hatch", "support"]),
        }
        points.append(point)

    return points


def generate_ct_points():
    """Generate CT scan points fixture."""
    np.random.seed(42)

    # Generate CT scan data structure
    ct_data = {
        "model_id": "test_model_001",
        "scan_id": "scan_001",
        "scan_timestamp": "2024-01-01T00:00:00Z",
        "voxel_grid": {
            "dimensions": [100, 100, 100],
            "spacing": [0.1, 0.1, 0.1],
            "origin": [0.0, 0.0, 0.0],
        },
        "points": [],
    }

    # Generate sample points (sparse representation)
    n_points = 500
    for i in range(n_points):
        point = {
            "x": float(np.random.rand() * 100.0),
            "y": float(np.random.rand() * 100.0),
            "z": float(np.random.rand() * 10.0),
            "density": float(4.0 + np.random.rand() * 0.5),  # Ti6Al4V density range
            "porosity": float(np.random.rand() * 0.05),  # 0-5% porosity
            "defect": np.random.rand() < 0.1,  # 10% chance of defect
        }
        ct_data["points"].append(point)

    # Add defect locations
    ct_data["defect_locations"] = [[p["x"], p["y"], p["z"]] for p in ct_data["points"] if p.get("defect", False)]

    return ct_data
