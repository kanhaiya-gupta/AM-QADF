"""
Test fixtures for process capability data.
"""

import numpy as np
from typing import Tuple


def generate_capable_process_data(
    n_samples: int = 100, usl: float = 12.0, lsl: float = 8.0, target: float = 10.0, cpk: float = 1.5, seed: int = 42
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """
    Generate data from a capable process.

    Args:
        n_samples: Number of samples
        usl: Upper specification limit
        lsl: Lower specification limit
        target: Target value
        cpk: Desired Cpk value
        seed: Random seed

    Returns:
        Tuple of (data array, (USL, LSL))
    """
    np.random.seed(seed)
    # Calculate std to achieve desired Cpk
    # Cpk = min[(USL - mean) / (3*sigma), (mean - LSL) / (3*sigma)]
    # For centered process: Cpk = (USL - LSL) / (6*sigma)
    sigma = (usl - lsl) / (6 * cpk)
    mean = target
    data = np.random.normal(mean, sigma, n_samples)
    return data, (usl, lsl)


def generate_incapable_process_data(
    n_samples: int = 100, usl: float = 12.0, lsl: float = 8.0, target: float = 10.0, cpk: float = 0.8, seed: int = 42
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """
    Generate data from an incapable process.

    Args:
        n_samples: Number of samples
        usl: Upper specification limit
        lsl: Lower specification limit
        target: Target value
        cpk: Desired Cpk value (< 1.0 for incapable)
        seed: Random seed

    Returns:
        Tuple of (data array, (USL, LSL))
    """
    np.random.seed(seed)
    # Calculate std to achieve desired Cpk
    sigma = (usl - lsl) / (6 * cpk)
    mean = target
    data = np.random.normal(mean, sigma, n_samples)
    return data, (usl, lsl)


def generate_shifted_process_data(
    n_samples: int = 100, usl: float = 12.0, lsl: float = 8.0, target: float = 10.0, shift: float = 1.5, seed: int = 42
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """
    Generate data from a shifted (off-center) process.

    Args:
        n_samples: Number of samples
        usl: Upper specification limit
        lsl: Lower specification limit
        target: Target value
        shift: Mean shift from target
        seed: Random seed

    Returns:
        Tuple of (data array, (USL, LSL))
    """
    np.random.seed(seed)
    # Use moderate variance
    sigma = (usl - lsl) / 8.0  # Moderate spread
    mean = target + shift  # Shifted mean
    data = np.random.normal(mean, sigma, n_samples)
    return data, (usl, lsl)


def generate_one_sided_spec_data(
    n_samples: int = 100, usl: float = 12.0, target: float = 10.0, seed: int = 42
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """
    Generate data for one-sided specification (USL only).

    Args:
        n_samples: Number of samples
        usl: Upper specification limit
        target: Target value
        seed: Random seed

    Returns:
        Tuple of (data array, (USL, -inf))
    """
    np.random.seed(seed)
    sigma = 1.0
    mean = target
    data = np.random.normal(mean, sigma, n_samples)
    return data, (usl, float("-inf"))
