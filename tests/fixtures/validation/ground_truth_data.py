"""
Test fixtures for ground truth data generation.

Provides functions to generate synthetic ground truth data for validation testing.
"""

import numpy as np
from typing import Dict, Tuple, Optional
import pickle
from pathlib import Path


def generate_ground_truth_signal(
    shape: Tuple[int, ...] = (50, 50, 10), noise_level: float = 0.0, seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate synthetic ground truth signal.

    Args:
        shape: Shape of the signal array
        noise_level: Standard deviation of noise to add (0 = no noise)
        seed: Random seed for reproducibility

    Returns:
        Ground truth signal array
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate base signal with some structure
    signal = np.zeros(shape)
    if len(shape) == 3:
        # Add some 3D structure
        z_coords = np.arange(shape[2])[:, np.newaxis, np.newaxis]
        y_coords = np.arange(shape[1])[np.newaxis, :, np.newaxis]
        x_coords = np.arange(shape[0])[np.newaxis, np.newaxis, :]
        signal = 100 + 10 * np.sin(x_coords * 0.1) + 10 * np.cos(y_coords * 0.1) + 5 * np.sin(z_coords * 0.2)
    else:
        signal = np.random.rand(*shape) * 100

    # Add noise if specified
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, shape)
        signal = signal + noise

    return signal


def generate_ground_truth_coordinates(
    n_points: int = 1000,
    bounds: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    size: Tuple[float, float, float] = (10.0, 10.0, 10.0),
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate ground truth coordinates.

    Args:
        n_points: Number of coordinate points
        bounds: Minimum bounds (x_min, y_min, z_min)
        size: Size of bounding box (x_size, y_size, z_size)
        seed: Random seed for reproducibility

    Returns:
        Array of shape (n_points, 3) with coordinates
    """
    if seed is not None:
        np.random.seed(seed)

    coords = np.random.rand(n_points, 3)
    coords[:, 0] = coords[:, 0] * size[0] + bounds[0]
    coords[:, 1] = coords[:, 1] * size[1] + bounds[1]
    coords[:, 2] = coords[:, 2] * size[2] + bounds[2]

    return coords


def generate_ground_truth_quality_metrics(
    base_values: Optional[Dict[str, float]] = None, noise_level: float = 0.01, seed: Optional[int] = None
) -> Dict[str, float]:
    """
    Generate ground truth quality metrics.

    Args:
        base_values: Base metric values (defaults to typical quality metrics)
        noise_level: Relative noise level (0.01 = 1%)
        seed: Random seed for reproducibility

    Returns:
        Dictionary of quality metrics
    """
    if seed is not None:
        np.random.seed(seed)

    if base_values is None:
        base_values = {
            "overall_quality_score": 0.9,
            "data_quality_score": 0.85,
            "signal_quality_score": 0.92,
            "alignment_score": 0.88,
            "completeness_score": 0.95,
            "completeness": 0.90,
            "snr": 25.5,
            "alignment_accuracy": 0.95,
        }

    # Add noise to base values
    metrics = {}
    for key, value in base_values.items():
        noise = np.random.normal(0, value * noise_level)
        metrics[key] = float(np.clip(value + noise, 0.0, 1.0) if value <= 1.0 else value + noise)

    return metrics


def generate_ground_truth_with_noise(
    base_data: np.ndarray, noise_level: float = 0.02, seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate ground truth data with known noise level.

    Args:
        base_data: Base ground truth data
        noise_level: Standard deviation of noise (relative to data range)
        seed: Random seed for reproducibility

    Returns:
        Ground truth data with added noise
    """
    if seed is not None:
        np.random.seed(seed)

    data_range = np.max(base_data) - np.min(base_data)
    noise_std = data_range * noise_level
    noise = np.random.normal(0, noise_std, base_data.shape)

    return base_data + noise
