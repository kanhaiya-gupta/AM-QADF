"""
Test fixtures for MPM comparison data generation.

Provides functions to generate synthetic MPM system data for comparison testing.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from .ground_truth_data import generate_ground_truth_quality_metrics, generate_ground_truth_signal


def generate_mpm_metrics(
    framework_metrics: Dict[str, float],
    correlation: float = 0.9,
    bias: float = 0.0,
    noise_level: float = 0.01,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """
    Generate MPM metrics correlated with framework metrics.

    Args:
        framework_metrics: Framework-generated quality metrics
        correlation: Target correlation (0-1)
        bias: Systematic bias to add
        noise_level: Relative noise level
        seed: Random seed for reproducibility

    Returns:
        Dictionary of MPM quality metrics
    """
    if seed is not None:
        np.random.seed(seed)

    mpm_metrics = {}
    for key, value in framework_metrics.items():
        # Add correlation: correlated component + uncorrelated component
        correlated_component = value * correlation
        uncorrelated_component = np.random.normal(0, value * noise_level) * (1 - correlation)
        bias_component = bias
        noise_component = np.random.normal(0, value * noise_level)

        mpm_value = correlated_component + uncorrelated_component + bias_component + noise_component

        # Clip to valid range if metric is between 0 and 1
        if 0.0 <= value <= 1.0:
            mpm_value = np.clip(mpm_value, 0.0, 1.0)
        else:
            mpm_value = max(0, mpm_value)  # Ensure non-negative

        mpm_metrics[key] = float(mpm_value)

    return mpm_metrics


def generate_mpm_arrays(
    framework_array: np.ndarray,
    correlation: float = 0.9,
    bias: float = 0.0,
    noise_level: float = 0.02,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate MPM output arrays correlated with framework arrays.

    Args:
        framework_array: Framework output array
        correlation: Target correlation (0-1)
        bias: Systematic bias to add
        noise_level: Relative noise level
        seed: Random seed for reproducibility

    Returns:
        MPM output array
    """
    if seed is not None:
        np.random.seed(seed)

    # Correlated component
    correlated = framework_array * correlation

    # Uncorrelated noise
    array_std = np.std(framework_array)
    noise = np.random.normal(0, array_std * noise_level * (1 - correlation), framework_array.shape)

    # Bias
    bias_array = np.full_like(framework_array, bias)

    mpm_array = correlated + noise + bias_array

    return mpm_array


def generate_mpm_with_correlation(
    framework_values: np.ndarray, target_correlation: float = 0.85, seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate MPM values with specific correlation to framework values.

    Args:
        framework_values: Framework output values
        target_correlation: Target correlation coefficient
        seed: Random seed for reproducibility

    Returns:
        MPM values with specified correlation
    """
    if seed is not None:
        np.random.seed(seed)

    # Normalize framework values
    framework_std = np.std(framework_values)
    framework_mean = np.mean(framework_values)
    framework_normalized = (framework_values - framework_mean) / (framework_std + 1e-10)

    # Generate correlated component
    correlated = framework_normalized * target_correlation

    # Generate uncorrelated component
    uncorrelated_std = np.sqrt(1 - target_correlation**2)
    uncorrelated = np.random.normal(0, uncorrelated_std, framework_values.shape)

    # Combine and denormalize
    mpm_normalized = correlated + uncorrelated
    mpm_values = mpm_normalized * framework_std + framework_mean

    return mpm_values


def generate_mpm_with_drift(framework_values: np.ndarray, drift_rate: float = 0.001, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate MPM values with systematic drift.

    Args:
        framework_values: Framework output values
        drift_rate: Drift rate (per sample)
        seed: Random seed for reproducibility

    Returns:
        MPM values with drift
    """
    if seed is not None:
        np.random.seed(seed)

    # Create drift pattern
    n_samples = len(framework_values)
    drift = np.linspace(0, drift_rate * n_samples, n_samples)

    # Add drift to framework values
    mpm_values = framework_values + drift

    return mpm_values


def generate_mpm_comparison_dataset(
    framework_data: Dict[str, any], correlation: float = 0.9, bias: float = 0.0, seed: Optional[int] = None
) -> Dict[str, any]:
    """
    Generate complete MPM comparison dataset from framework data.

    Args:
        framework_data: Framework output data (dict with metrics and/or arrays)
        correlation: Target correlation
        bias: Systematic bias
        seed: Random seed

    Returns:
        MPM comparison dataset matching framework structure
    """
    if seed is not None:
        np.random.seed(seed)

    mpm_data = {}

    for key, value in framework_data.items():
        if isinstance(value, np.ndarray):
            mpm_data[key] = generate_mpm_arrays(value, correlation=correlation, bias=bias, seed=seed)
        elif isinstance(value, (int, float)):
            # Single value - apply correlation and bias
            mpm_data[key] = value * correlation + bias + np.random.normal(0, abs(value) * 0.01)
        elif isinstance(value, dict):
            # Nested dictionary
            mpm_data[key] = generate_mpm_comparison_dataset(value, correlation, bias, seed)
        else:
            mpm_data[key] = value  # Keep as is

    return mpm_data
