"""
Test fixtures for control chart data.
"""

import numpy as np
from typing import Tuple, Dict


def generate_in_control_data(n_samples: int = 100, mean: float = 10.0, std: float = 1.0, seed: int = 42) -> np.ndarray:
    """
    Generate in-control process data (normal distribution).

    Args:
        n_samples: Number of samples
        mean: Process mean
        std: Process standard deviation
        seed: Random seed

    Returns:
        Array of in-control data
    """
    np.random.seed(seed)
    return np.random.normal(mean, std, n_samples)


def generate_out_of_control_data(
    n_samples: int = 100,
    mean: float = 10.0,
    std: float = 1.0,
    shift_at: int = 50,
    shift_magnitude: float = 5.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate out-of-control process data (with mean shift).

    Args:
        n_samples: Number of samples
        mean: Initial process mean
        std: Process standard deviation
        shift_at: Index where shift occurs
        shift_magnitude: Magnitude of mean shift
        seed: Random seed

    Returns:
        Array of out-of-control data
    """
    np.random.seed(seed)
    data = np.random.normal(mean, std, n_samples)
    data[shift_at:] += shift_magnitude
    return data


def generate_trend_data(
    n_samples: int = 100, start_mean: float = 10.0, trend: float = 0.1, std: float = 1.0, seed: int = 42
) -> np.ndarray:
    """
    Generate data with trend.

    Args:
        n_samples: Number of samples
        start_mean: Starting mean value
        trend: Linear trend per sample
        std: Process standard deviation
        seed: Random seed

    Returns:
        Array of trended data
    """
    np.random.seed(seed)
    trend_line = np.linspace(start_mean, start_mean + trend * (n_samples - 1), n_samples)
    noise = np.random.normal(0, std, n_samples)
    return trend_line + noise


def generate_cyclical_data(
    n_samples: int = 100, mean: float = 10.0, amplitude: float = 2.0, period: int = 20, std: float = 0.5, seed: int = 42
) -> np.ndarray:
    """
    Generate cyclical/seasonal data.

    Args:
        n_samples: Number of samples
        mean: Mean value
        amplitude: Amplitude of cycle
        period: Period of cycle
        std: Process standard deviation
        seed: Random seed

    Returns:
        Array of cyclical data
    """
    np.random.seed(seed)
    t = np.arange(n_samples)
    cycle = amplitude * np.sin(2 * np.pi * t / period)
    noise = np.random.normal(0, std, n_samples)
    return mean + cycle + noise


def generate_subgrouped_data(
    n_subgroups: int = 20, subgroup_size: int = 5, mean: float = 10.0, std: float = 1.0, seed: int = 42
) -> np.ndarray:
    """
    Generate subgrouped data for X-bar/R/S charts.

    Args:
        n_subgroups: Number of subgroups
        subgroup_size: Size of each subgroup
        mean: Process mean
        std: Process standard deviation
        seed: Random seed

    Returns:
        Flattened array of subgrouped data
    """
    np.random.seed(seed)
    data = []
    for _ in range(n_subgroups):
        subgroup = np.random.normal(mean, std, subgroup_size)
        data.extend(subgroup)
    return np.array(data)


def generate_xbar_r_chart_data() -> Dict[str, np.ndarray]:
    """
    Generate sample data for X-bar and R charts.

    Returns:
        Dictionary with 'data' key containing subgrouped data
    """
    data = generate_subgrouped_data(n_subgroups=25, subgroup_size=5, mean=10.0, std=1.0)
    return {"data": data, "subgroup_size": 5}


def generate_individual_mr_data() -> Dict[str, np.ndarray]:
    """
    Generate sample data for Individual and Moving Range charts.

    Returns:
        Dictionary with 'data' key containing individual measurements
    """
    # Mix in-control and out-of-control data
    in_control = generate_in_control_data(n_samples=80, mean=10.0, std=1.0, seed=42)
    ooc = generate_out_of_control_data(n_samples=20, mean=15.0, std=1.5, shift_at=0, shift_magnitude=0, seed=43)
    data = np.concatenate([in_control, ooc])
    return {"data": data}
