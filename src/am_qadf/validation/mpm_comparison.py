"""
MPM System Comparison

Comparison utilities for comparing framework outputs with MPM (Melt Pool Monitoring)
system native outputs.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging
from datetime import datetime
from scipy.stats import pearsonr, spearmanr

logger = logging.getLogger(__name__)


@dataclass
class MPMComparisonResult:
    """Result of MPM system comparison."""

    metric_name: str
    framework_value: float
    mpm_value: float
    correlation: float
    difference: float
    relative_error: float
    is_valid: bool
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def validated_points(self) -> int:
        """Number of validated points (from metadata)."""
        return self.metadata.get("valid_points", 0)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "metric_name": self.metric_name,
            "framework_value": self.framework_value,
            "mpm_value": self.mpm_value,
            "correlation": self.correlation,
            "difference": self.difference,
            "relative_error": self.relative_error,
            "is_valid": self.is_valid,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class MPMComparisonEngine:
    """
    Engine for comparing framework outputs with MPM system outputs.

    Provides:
    - Quality metric comparison
    - Anomaly detection comparison
    - Correlation analysis
    - Alignment validation
    """

    def __init__(self, correlation_threshold: float = 0.85, max_relative_error: float = 0.1):
        """
        Initialize MPM comparison engine.

        Args:
            correlation_threshold: Minimum correlation for validation (default: 0.85)
            max_relative_error: Maximum relative error for validation (default: 0.1 = 10%)
        """
        self.correlation_threshold = correlation_threshold
        self.max_relative_error = max_relative_error

    def compare_metric(self, metric_name: str, framework_value: float, mpm_value: float) -> MPMComparisonResult:
        """
        Compare a single metric between framework and MPM.

        Args:
            metric_name: Name of the metric
            framework_value: Framework-calculated value
            mpm_value: MPM system value

        Returns:
            MPMComparisonResult
        """
        difference = abs(framework_value - mpm_value)
        relative_error = (difference / abs(mpm_value) * 100) if mpm_value != 0 else float("inf")

        # Correlation is 1.0 for single value comparison (perfect match)
        correlation = 1.0 if difference == 0 else 0.0

        is_valid = relative_error <= (self.max_relative_error * 100) and correlation >= self.correlation_threshold

        return MPMComparisonResult(
            metric_name=metric_name,
            framework_value=framework_value,
            mpm_value=mpm_value,
            correlation=correlation,
            difference=difference,
            relative_error=relative_error,
            is_valid=is_valid,
            metadata={},
        )

    def compare_quality_metrics(
        self, framework_metrics: Dict[str, float], mpm_metrics: Dict[str, float]
    ) -> Dict[str, MPMComparisonResult]:
        """
        Compare quality metrics between framework and MPM.

        Args:
            framework_metrics: Dictionary of framework quality metrics
            mpm_metrics: Dictionary of MPM quality metrics

        Returns:
            Dictionary mapping metric names to MPMComparisonResult
        """
        results = {}
        common_metrics = set(framework_metrics.keys()) & set(mpm_metrics.keys())

        for metric_name in common_metrics:
            results[metric_name] = self.compare_metric(metric_name, framework_metrics[metric_name], mpm_metrics[metric_name])

        return results

    def compare_arrays(self, metric_name: str, framework_array: np.ndarray, mpm_array: np.ndarray) -> MPMComparisonResult:
        """
        Compare arrays between framework and MPM.

        Args:
            metric_name: Name of the metric
            framework_array: Framework output array
            mpm_array: MPM output array

        Returns:
            MPMComparisonResult with correlation and error metrics
        """
        # Flatten arrays for comparison
        framework_flat = framework_array.flatten()
        mpm_flat = mpm_array.flatten()

        # Handle size mismatches
        min_len = min(len(framework_flat), len(mpm_flat))
        if min_len == 0:
            logger.warning(f"Empty arrays for metric {metric_name}")
            return MPMComparisonResult(
                metric_name=metric_name,
                framework_value=0.0,
                mpm_value=0.0,
                correlation=0.0,
                difference=0.0,
                relative_error=0.0,
                is_valid=False,
                metadata={"error": "Empty arrays"},
            )

        framework_flat = framework_flat[:min_len]
        mpm_flat = mpm_flat[:min_len]

        # Remove NaN values
        valid_mask = ~(np.isnan(framework_flat) | np.isnan(mpm_flat))
        framework_valid = framework_flat[valid_mask]
        mpm_valid = mpm_flat[valid_mask]

        if len(framework_valid) == 0:
            logger.warning(f"No valid values for metric {metric_name}")
            return MPMComparisonResult(
                metric_name=metric_name,
                framework_value=0.0,
                mpm_value=0.0,
                correlation=0.0,
                difference=0.0,
                relative_error=0.0,
                is_valid=False,
                metadata={"error": "No valid values"},
            )

        # Calculate statistics
        framework_mean = np.mean(framework_valid)
        mpm_mean = np.mean(mpm_valid)

        # Calculate correlation
        try:
            correlation, p_value = pearsonr(framework_valid, mpm_valid)
            correlation = float(correlation) if not np.isnan(correlation) else 0.0
        except Exception as e:
            logger.warning(f"Correlation calculation failed for {metric_name}: {e}")
            correlation = 0.0

        # Calculate difference metrics
        differences = np.abs(framework_valid - mpm_valid)
        mean_difference = np.mean(differences)
        relative_errors = np.abs(differences / (np.abs(mpm_valid) + 1e-10)) * 100
        mean_relative_error = np.mean(relative_errors)

        is_valid = correlation >= self.correlation_threshold and mean_relative_error <= (self.max_relative_error * 100)

        return MPMComparisonResult(
            metric_name=metric_name,
            framework_value=float(framework_mean),
            mpm_value=float(mpm_mean),
            correlation=correlation,
            difference=float(mean_difference),
            relative_error=float(mean_relative_error),
            is_valid=is_valid,
            metadata={
                "p_value": float(p_value) if "p_value" in locals() else None,
                "valid_points": len(framework_valid),
                "total_points": len(framework_flat),
                "rmse": float(np.sqrt(np.mean(differences**2))),
                "mae": float(mean_difference),
            },
        )

    def calculate_correlation(self, framework_values: np.ndarray, mpm_values: np.ndarray, method: str = "pearson") -> float:
        """
        Calculate correlation between framework and MPM values.

        Args:
            framework_values: Framework output values
            mpm_values: MPM output values
            method: Correlation method ('pearson' or 'spearman')

        Returns:
            Correlation coefficient (0 to 1)
        """
        # Flatten and align
        framework_flat = framework_values.flatten()
        mpm_flat = mpm_values.flatten()

        min_len = min(len(framework_flat), len(mpm_flat))
        framework_flat = framework_flat[:min_len]
        mpm_flat = mpm_flat[:min_len]

        # Remove NaN values
        valid_mask = ~(np.isnan(framework_flat) | np.isnan(mpm_flat))
        framework_valid = framework_flat[valid_mask]
        mpm_valid = mpm_flat[valid_mask]

        if len(framework_valid) < 2:
            return 0.0

        try:
            if method.lower() == "pearson":
                correlation, _ = pearsonr(framework_valid, mpm_valid)
            elif method.lower() == "spearman":
                correlation, _ = spearmanr(framework_valid, mpm_valid)
            else:
                logger.warning(f"Unknown correlation method: {method}, using pearson")
                correlation, _ = pearsonr(framework_valid, mpm_valid)

            return float(correlation) if not np.isnan(correlation) else 0.0
        except Exception as e:
            logger.error(f"Correlation calculation failed: {e}")
            return 0.0

    def compare_all_metrics(
        self, framework_data: Any, mpm_data: Any, metrics: Optional[List[str]] = None
    ) -> Dict[str, MPMComparisonResult]:
        """
        Compare all available metrics between framework and MPM data.

        Args:
            framework_data: Framework-generated data (dict, object, or array)
            mpm_data: MPM system data (dict, object, or array)
            metrics: List of metric names to compare. If None, compares all available.

        Returns:
            Dictionary mapping metric names to MPMComparisonResult
        """
        results = {}

        # Handle dictionary inputs
        if isinstance(framework_data, dict) and isinstance(mpm_data, dict):
            if metrics is None:
                metrics = list(set(framework_data.keys()) & set(mpm_data.keys()))

            for metric_name in metrics:
                if metric_name in framework_data and metric_name in mpm_data:
                    fw_val = framework_data[metric_name]
                    mpm_val = mpm_data[metric_name]

                    if isinstance(fw_val, (int, float)) and isinstance(mpm_val, (int, float)):
                        results[metric_name] = self.compare_metric(metric_name, float(fw_val), float(mpm_val))
                    elif isinstance(fw_val, np.ndarray) and isinstance(mpm_val, np.ndarray):
                        results[metric_name] = self.compare_arrays(metric_name, fw_val, mpm_val)
                    else:
                        logger.warning(f"Cannot compare metric {metric_name}: unsupported types")

        # Handle array inputs
        elif isinstance(framework_data, np.ndarray) and isinstance(mpm_data, np.ndarray):
            metric_name = metrics[0] if metrics else "array_comparison"
            results[metric_name] = self.compare_arrays(metric_name, framework_data, mpm_data)

        else:
            logger.warning("Unsupported data types for comparison")

        return results
