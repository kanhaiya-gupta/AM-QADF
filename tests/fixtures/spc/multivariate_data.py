"""
Test fixtures for multivariate SPC data.
"""

import numpy as np
from typing import Tuple


def generate_multivariate_in_control_data(
    n_samples: int = 100, n_variables: int = 5, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate in-control multivariate process data.

    Args:
        n_samples: Number of samples
        n_variables: Number of variables
        seed: Random seed

    Returns:
        Tuple of (data array (n_samples x n_variables), correlation matrix)
    """
    np.random.seed(seed)

    # Generate correlation matrix (positive definite)
    corr = np.eye(n_variables)
    for i in range(n_variables):
        for j in range(i + 1, n_variables):
            corr[i, j] = corr[j, i] = 0.5 ** abs(i - j)  # Decaying correlation

    # Generate multivariate normal data
    mean = np.zeros(n_variables)
    cov = np.outer(np.ones(n_variables), np.ones(n_variables)) * 0.5 + np.eye(n_variables) * 0.5
    cov = np.dot(cov, corr)
    cov = np.dot(cov, cov.T)  # Ensure positive definite

    data = np.random.multivariate_normal(mean, cov, n_samples)

    return data, cov


def generate_multivariate_out_of_control_data(
    n_samples: int = 100, n_variables: int = 5, shift_at: int = 50, shift_magnitude: float = 3.0, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate multivariate out-of-control data (with mean shift).

    Args:
        n_samples: Number of samples
        n_variables: Number of variables
        shift_at: Index where shift occurs
        shift_magnitude: Magnitude of mean shift
        seed: Random seed

    Returns:
        Tuple of (data array, correlation matrix)
    """
    np.random.seed(seed)

    # Generate correlation matrix
    corr = np.eye(n_variables)
    for i in range(n_variables):
        for j in range(i + 1, n_variables):
            corr[i, j] = corr[j, i] = 0.5 ** abs(i - j)

    mean = np.zeros(n_variables)
    cov = np.outer(np.ones(n_variables), np.ones(n_variables)) * 0.5 + np.eye(n_variables) * 0.5
    cov = np.dot(cov, corr)
    cov = np.dot(cov, cov.T)

    # Generate in-control data
    data = np.random.multivariate_normal(mean, cov, n_samples)

    # Apply shift
    shift_vector = np.ones(n_variables) * shift_magnitude
    data[shift_at:] += shift_vector

    return data, cov


def generate_high_dimensional_data(
    n_samples: int = 100, n_variables: int = 20, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate high-dimensional multivariate data.

    Args:
        n_samples: Number of samples
        n_variables: Number of variables
        seed: Random seed

    Returns:
        Tuple of (data array, correlation matrix)
    """
    np.random.seed(seed)

    # Generate sparse correlation matrix
    corr = np.eye(n_variables)
    for i in range(n_variables):
        for j in range(i + 1, min(i + 3, n_variables)):  # Only nearby variables correlated
            corr[i, j] = corr[j, i] = 0.3

    mean = np.zeros(n_variables)
    cov = np.eye(n_variables)
    for i in range(n_variables):
        for j in range(i, min(i + 3, n_variables)):
            cov[i, j] = cov[j, i] = 0.5 * (0.5 ** abs(i - j))

    data = np.random.multivariate_normal(mean, cov, n_samples)

    return data, cov
