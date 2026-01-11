"""
Test fixtures for baseline calculation data.
"""

import numpy as np
from typing import Dict


def generate_stable_baseline_data(n_samples: int = 200, mean: float = 10.0, std: float = 1.0, seed: int = 42) -> np.ndarray:
    """
    Generate stable baseline data for establishing control limits.

    Args:
        n_samples: Number of samples
        mean: Process mean
        std: Process standard deviation
        seed: Random seed

    Returns:
        Array of stable baseline data
    """
    np.random.seed(seed)
    return np.random.normal(mean, std, n_samples)


def generate_unstable_baseline_data(n_samples: int = 200, mean: float = 10.0, std: float = 1.0, seed: int = 42) -> np.ndarray:
    """
    Generate unstable baseline data (with multiple changes).

    Args:
        n_samples: Number of samples
        mean: Initial process mean
        std: Process standard deviation
        seed: Random seed

    Returns:
        Array of unstable baseline data
    """
    np.random.seed(seed)
    data = []
    segment_size = n_samples // 4

    # Four segments with different means
    means = [mean, mean + 1.0, mean - 0.5, mean + 0.5]
    for m in means:
        segment = np.random.normal(m, std, segment_size)
        data.extend(segment)

    # Fill remainder if needed
    if len(data) < n_samples:
        remainder = np.random.normal(mean, std, n_samples - len(data))
        data.extend(remainder)

    return np.array(data[:n_samples])


def generate_subgrouped_baseline_data(
    n_subgroups: int = 40,
    subgroup_size: int = 5,
    mean: float = 10.0,
    within_std: float = 0.5,
    between_std: float = 0.2,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate subgrouped baseline data with specified within and between variation.

    Args:
        n_subgroups: Number of subgroups
        subgroup_size: Size of each subgroup
        mean: Overall process mean
        within_std: Within-subgroup standard deviation
        between_std: Between-subgroup standard deviation
        seed: Random seed

    Returns:
        Flattened array of subgrouped data
    """
    np.random.seed(seed)
    data = []

    # Generate subgroup means with between-subgroup variation
    subgroup_means = np.random.normal(mean, between_std, n_subgroups)

    for sg_mean in subgroup_means:
        # Generate subgroup with within-subgroup variation
        subgroup = np.random.normal(sg_mean, within_std, subgroup_size)
        data.extend(subgroup)

    return np.array(data)
