"""
Signal fixtures for testing.

Provides pre-computed signal arrays for use in tests.
"""

import numpy as np
from pathlib import Path

FIXTURE_DIR = Path(__file__).parent


def load_sample_signals():
    """Load sample signals fixture."""
    fixture_path = FIXTURE_DIR / "sample_signals.npz"
    if fixture_path.exists():
        return dict(np.load(fixture_path))
    else:
        # Generate on-the-fly if not exists
        return generate_sample_signals()


def generate_sample_signals():
    """Generate sample signals fixture."""
    np.random.seed(42)

    n_points = 1000

    signals = {
        "laser_power": np.random.rand(n_points) * 300.0,  # 0-300 W
        "scan_speed": np.random.rand(n_points) * 2000.0,  # 0-2000 mm/s
        "temperature": np.random.rand(n_points) * 1000.0,  # 0-1000 °C
        "density": np.random.rand(n_points) * 1.0 + 4.0,  # 4.0-5.0 g/cm³
        "porosity": np.random.rand(n_points) * 0.1,  # 0-10%
        "velocity": np.random.rand(n_points) * 100.0,  # 0-100 mm/s
        "energy_density": np.random.rand(n_points) * 100.0,  # 0-100 J/mm²
        "exposure_time": np.random.rand(n_points) * 0.2,  # 0-0.2 s
    }

    return signals
