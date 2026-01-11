"""
General validation test data generators.

Provides functions to generate various test datasets for validation testing.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json


def generate_validation_test_dataset(
    size: str = "small", include_edge_cases: bool = False, seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Generate complete validation test dataset.

    Args:
        size: Dataset size ('small', 'medium', 'large')
        include_edge_cases: Whether to include edge case scenarios
        seed: Random seed for reproducibility

    Returns:
        Dictionary with various test datasets
    """
    if seed is not None:
        np.random.seed(seed)

    size_configs = {
        "small": {"signal_shape": (20, 20, 5), "n_points": 100},
        "medium": {"signal_shape": (50, 50, 10), "n_points": 1000},
        "large": {"signal_shape": (100, 100, 20), "n_points": 10000},
    }

    config = size_configs.get(size, size_configs["medium"])

    dataset = {
        "framework_signal": np.random.rand(*config["signal_shape"]) * 100,
        "ground_truth_signal": np.random.rand(*config["signal_shape"]) * 100,
        "framework_coords": np.random.rand(config["n_points"], 3) * 10,
        "ground_truth_coords": np.random.rand(config["n_points"], 3) * 10,
        "framework_metrics": {
            "completeness": 0.9,
            "snr": 25.5,
            "alignment_accuracy": 0.95,
        },
        "mpm_metrics": {
            "completeness": 0.88,
            "snr": 24.8,
            "alignment_accuracy": 0.94,
        },
    }

    if include_edge_cases:
        dataset["edge_cases"] = {
            "empty_array": np.array([]),
            "single_element": np.array([1.0]),
            "with_nan": np.array([1.0, 2.0, np.nan, 4.0, 5.0]),
            "with_inf": np.array([1.0, 2.0, np.inf, 4.0, 5.0]),
            "all_zeros": np.zeros((10, 10)),
            "all_ones": np.ones((10, 10)),
        }

    return dataset


def generate_edge_case_data() -> Dict[str, Any]:
    """
    Generate edge case test data.

    Returns:
        Dictionary with various edge cases
    """
    return {
        "empty_array": np.array([]),
        "single_element_array": np.array([1.5]),
        "single_element_2d": np.array([[1.5]]),
        "with_nan": np.array([1.0, 2.0, np.nan, 4.0, 5.0, np.nan]),
        "all_nan": np.full(10, np.nan),
        "with_inf": np.array([1.0, 2.0, np.inf, 4.0, -np.inf, 5.0]),
        "all_zeros": np.zeros((10, 10, 5)),
        "all_ones": np.ones((10, 10, 5)),
        "very_large_values": np.array([1e10, 2e10, 3e10]),
        "very_small_values": np.array([1e-10, 2e-10, 3e-10]),
        "negative_values": np.array([-1.0, -2.0, -3.0, 1.0, 2.0]),
        "size_mismatch_1d": (np.array([1, 2, 3]), np.array([1, 2, 3, 4, 5])),
        "size_mismatch_2d": (np.random.rand(50, 50), np.random.rand(60, 60)),
        "size_mismatch_3d": (np.random.rand(10, 10, 5), np.random.rand(10, 10, 10)),
        "shape_mismatch": (np.random.rand(100, 3), np.random.rand(100, 2)),
    }


def generate_performance_test_data(operation_type: str = "signal_mapping", data_size: str = "large") -> Dict[str, Any]:
    """
    Generate performance test data for benchmarking.

    Args:
        operation_type: Type of operation to benchmark
        data_size: Size of test data ('small', 'medium', 'large', 'xlarge')

    Returns:
        Dictionary with performance test data
    """
    size_configs = {
        "small": {"signal_shape": (50, 50, 10), "n_points": 1000},
        "medium": {"signal_shape": (100, 100, 20), "n_points": 10000},
        "large": {"signal_shape": (200, 200, 50), "n_points": 100000},
        "xlarge": {"signal_shape": (500, 500, 100), "n_points": 1000000},
    }

    config = size_configs.get(data_size, size_configs["medium"])

    test_data = {
        "signal_mapping": {
            "points": np.random.rand(config["n_points"], 3) * 10,
            "signals": {
                "power": np.random.rand(config["n_points"]) * 300,
                "speed": np.random.rand(config["n_points"]) * 2000,
            },
            "voxel_grid_shape": config["signal_shape"],
        },
        "data_fusion": {
            "grid1": np.random.rand(*config["signal_shape"]),
            "grid2": np.random.rand(*config["signal_shape"]),
            "grid3": np.random.rand(*config["signal_shape"]),
        },
        "quality_assessment": {
            "voxel_data_shape": config["signal_shape"],
            "signals": ["signal1", "signal2", "signal3"],
        },
    }

    return test_data.get(operation_type, {})
