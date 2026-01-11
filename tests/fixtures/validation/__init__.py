"""
Test fixtures for validation module.

Provides ground truth data, MPM comparison data, and validation test datasets.
"""

from .ground_truth_data import (
    generate_ground_truth_signal,
    generate_ground_truth_coordinates,
    generate_ground_truth_quality_metrics,
    generate_ground_truth_with_noise,
)

from .mpm_comparison_data import (
    generate_mpm_metrics,
    generate_mpm_arrays,
    generate_mpm_with_correlation,
    generate_mpm_with_drift,
    generate_mpm_comparison_dataset,
)

from .validation_test_data import (
    generate_validation_test_dataset,
    generate_edge_case_data,
    generate_performance_test_data,
)

__all__ = [
    # Ground truth data
    "generate_ground_truth_signal",
    "generate_ground_truth_coordinates",
    "generate_ground_truth_quality_metrics",
    "generate_ground_truth_with_noise",
    # MPM comparison data
    "generate_mpm_metrics",
    "generate_mpm_arrays",
    "generate_mpm_with_correlation",
    "generate_mpm_with_drift",
    "generate_mpm_comparison_dataset",
    # Validation test data
    "generate_validation_test_dataset",
    "generate_edge_case_data",
    "generate_performance_test_data",
]
