"""
Accuracy Validation

Validation utilities for comparing framework outputs against ground truth data.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging
from datetime import datetime

# Optional sklearn dependency
try:
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("sklearn not available, using numpy-based implementations")

logger = logging.getLogger(__name__)

# Fallback implementations if sklearn not available
if not SKLEARN_AVAILABLE:

    def mean_squared_error(y_true, y_pred):
        return float(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))

    def mean_absolute_error(y_true, y_pred):
        return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))

    def r2_score(y_true, y_pred):
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)
        ss_res = np.sum((y_true_arr - y_pred_arr) ** 2)
        ss_tot = np.sum((y_true_arr - np.mean(y_true_arr)) ** 2)
        return float(1 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0


@dataclass
class AccuracyValidationResult:
    """Result of accuracy validation."""

    signal_name: str
    rmse: float  # Root Mean Square Error
    mae: float  # Mean Absolute Error
    r2_score: float
    max_error: float
    within_tolerance: bool
    ground_truth_size: int
    validated_points: int
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """Whether validation passed (alias for within_tolerance)."""
        return self.within_tolerance

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "signal_name": self.signal_name,
            "rmse": self.rmse,
            "mae": self.mae,
            "r2_score": self.r2_score,
            "max_error": self.max_error,
            "within_tolerance": self.within_tolerance,
            "ground_truth_size": self.ground_truth_size,
            "validated_points": self.validated_points,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class AccuracyValidator:
    """
    Validator for accuracy validation against ground truth.

    Provides:
    - Signal mapping accuracy validation
    - Spatial alignment validation
    - Temporal alignment validation
    - Quality metrics validation
    """

    def __init__(self, max_acceptable_error: float = 0.1, tolerance_percent: float = 5.0):
        """
        Initialize accuracy validator.

        Args:
            max_acceptable_error: Maximum acceptable error (absolute)
            tolerance_percent: Tolerance as percentage of ground truth value
        """
        self.max_acceptable_error = max_acceptable_error
        self.tolerance_percent = tolerance_percent

    def validate_signal_mapping(
        self, mapped_data: np.ndarray, ground_truth: np.ndarray, signal_name: str = "unknown"
    ) -> AccuracyValidationResult:
        """
        Validate signal mapping accuracy.

        Args:
            mapped_data: Framework-mapped signal data
            ground_truth: Ground truth signal data
            signal_name: Name of the signal being validated

        Returns:
            AccuracyValidationResult
        """
        return self._validate_arrays(mapped_data, ground_truth, signal_name)

    def validate_spatial_alignment(
        self, framework_coords: np.ndarray, ground_truth_coords: np.ndarray
    ) -> AccuracyValidationResult:
        """
        Validate spatial alignment accuracy.

        Args:
            framework_coords: Framework-calculated coordinates
            ground_truth_coords: Ground truth coordinates

        Returns:
            AccuracyValidationResult
        """
        # Calculate Euclidean distances
        if framework_coords.shape != ground_truth_coords.shape:
            logger.warning("Coordinate shape mismatch, attempting to align")
            min_shape = tuple(min(s1, s2) for s1, s2 in zip(framework_coords.shape, ground_truth_coords.shape))
            framework_coords = framework_coords[: min_shape[0], : min_shape[1]]
            ground_truth_coords = ground_truth_coords[: min_shape[0], : min_shape[1]]

        distances = np.sqrt(np.sum((framework_coords - ground_truth_coords) ** 2, axis=-1))
        distances_flat = distances.flatten()

        return self._validate_from_errors(distances_flat, np.zeros_like(distances_flat), "spatial_alignment")

    def validate_temporal_alignment(
        self, framework_times: np.ndarray, ground_truth_times: np.ndarray
    ) -> AccuracyValidationResult:
        """
        Validate temporal alignment accuracy.

        Args:
            framework_times: Framework-calculated timestamps
            ground_truth_times: Ground truth timestamps

        Returns:
            AccuracyValidationResult
        """
        time_differences = np.abs(framework_times.flatten() - ground_truth_times.flatten())
        return self._validate_from_errors(time_differences, np.zeros_like(time_differences), "temporal_alignment")

    def validate_quality_metrics(
        self, framework_metrics: Dict[str, float], ground_truth_metrics: Dict[str, float]
    ) -> Dict[str, AccuracyValidationResult]:
        """
        Validate quality metrics accuracy.

        Args:
            framework_metrics: Framework-calculated quality metrics
            ground_truth_metrics: Ground truth quality metrics

        Returns:
            Dictionary mapping metric names to AccuracyValidationResult
        """
        results = {}
        common_metrics = set(framework_metrics.keys()) & set(ground_truth_metrics.keys())

        for metric_name in common_metrics:
            fw_val = np.array([framework_metrics[metric_name]])
            gt_val = np.array([ground_truth_metrics[metric_name]])
            results[metric_name] = self._validate_arrays(fw_val, gt_val, metric_name)

        return results

    def _validate_arrays(self, predicted: np.ndarray, actual: np.ndarray, signal_name: str) -> AccuracyValidationResult:
        """Internal method to validate arrays."""
        # Flatten arrays
        predicted_flat = predicted.flatten()
        actual_flat = actual.flatten()

        # Handle size mismatches
        min_len = min(len(predicted_flat), len(actual_flat))
        if min_len == 0:
            logger.warning(f"Empty arrays for signal {signal_name}")
            return AccuracyValidationResult(
                signal_name=signal_name,
                rmse=float("inf"),
                mae=float("inf"),
                r2_score=-1.0,
                max_error=float("inf"),
                within_tolerance=False,
                ground_truth_size=0,
                validated_points=0,
                metadata={"error": "Empty arrays"},
            )

        predicted_flat = predicted_flat[:min_len]
        actual_flat = actual_flat[:min_len]

        # Remove NaN values
        valid_mask = ~(np.isnan(predicted_flat) | np.isnan(actual_flat))
        predicted_valid = predicted_flat[valid_mask]
        actual_valid = actual_flat[valid_mask]

        if len(predicted_valid) == 0:
            logger.warning(f"No valid values for signal {signal_name}")
            return AccuracyValidationResult(
                signal_name=signal_name,
                rmse=float("inf"),
                mae=float("inf"),
                r2_score=-1.0,
                max_error=float("inf"),
                within_tolerance=False,
                ground_truth_size=len(actual_flat),
                validated_points=0,
                metadata={"error": "No valid values"},
            )

        # Calculate error metrics
        errors = predicted_valid - actual_valid
        rmse = np.sqrt(mean_squared_error(actual_valid, predicted_valid))
        mae = mean_absolute_error(actual_valid, predicted_valid)

        # Calculate R² score
        try:
            r2 = r2_score(actual_valid, predicted_valid)
            if np.isnan(r2):
                r2 = -1.0
        except Exception as e:
            logger.warning(f"R² score calculation failed: {e}")
            r2 = -1.0

        max_error = np.max(np.abs(errors))

        # Check if within tolerance
        relative_errors = np.abs(errors / (np.abs(actual_valid) + 1e-10)) * 100
        mean_relative_error = np.mean(relative_errors)
        within_tolerance = rmse <= self.max_acceptable_error and mean_relative_error <= self.tolerance_percent

        return AccuracyValidationResult(
            signal_name=signal_name,
            rmse=float(rmse),
            mae=float(mae),
            r2_score=float(r2),
            max_error=float(max_error),
            within_tolerance=within_tolerance,
            ground_truth_size=len(actual_flat),
            validated_points=len(predicted_valid),
            metadata={
                "mean_relative_error_percent": float(mean_relative_error),
                "std_error": float(np.std(errors)),
                "median_error": float(np.median(errors)),
            },
        )

    def _validate_from_errors(self, errors: np.ndarray, reference: np.ndarray, signal_name: str) -> AccuracyValidationResult:
        """Validate from pre-calculated errors."""
        valid_mask = ~np.isnan(errors)
        errors_valid = errors[valid_mask]

        if len(errors_valid) == 0:
            return AccuracyValidationResult(
                signal_name=signal_name,
                rmse=float("inf"),
                mae=float("inf"),
                r2_score=-1.0,
                max_error=float("inf"),
                within_tolerance=False,
                ground_truth_size=len(reference),
                validated_points=0,
                metadata={"error": "No valid errors"},
            )

        rmse = np.sqrt(np.mean(errors_valid**2))
        mae = np.mean(np.abs(errors_valid))
        max_error = np.max(np.abs(errors_valid))

        # R² not applicable for error-only validation
        r2 = -1.0

        within_tolerance = rmse <= self.max_acceptable_error and mae <= self.max_acceptable_error

        return AccuracyValidationResult(
            signal_name=signal_name,
            rmse=float(rmse),
            mae=float(mae),
            r2_score=r2,
            max_error=float(max_error),
            within_tolerance=within_tolerance,
            ground_truth_size=len(reference),
            validated_points=len(errors_valid),
            metadata={
                "std_error": float(np.std(errors_valid)),
                "median_error": float(np.median(errors_valid)),
            },
        )

    def calculate_rmse(self, predicted: np.ndarray, actual: np.ndarray) -> float:
        """Calculate Root Mean Square Error."""
        return float(np.sqrt(mean_squared_error(actual.flatten(), predicted.flatten())))

    def calculate_mae(self, predicted: np.ndarray, actual: np.ndarray) -> float:
        """Calculate Mean Absolute Error."""
        return float(mean_absolute_error(actual.flatten(), predicted.flatten()))

    def calculate_r2_score(self, predicted: np.ndarray, actual: np.ndarray) -> float:
        """Calculate R² score."""
        try:
            r2 = r2_score(actual.flatten(), predicted.flatten())
            return float(r2) if not np.isnan(r2) else -1.0
        except Exception:
            return -1.0

    def validate_within_tolerance(self, errors: np.ndarray, tolerance: Optional[float] = None) -> bool:
        """
        Check if errors are within tolerance.

        Args:
            errors: Array of errors
            tolerance: Tolerance threshold (uses max_acceptable_error if None)

        Returns:
            True if all errors are within tolerance
        """
        if tolerance is None:
            tolerance = self.max_acceptable_error

        return bool(np.all(np.abs(errors) <= tolerance))
