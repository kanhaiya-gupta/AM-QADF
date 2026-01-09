"""
Cross-Validation Strategies for Anomaly Detection

Provides various cross-validation strategies for robust performance estimation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass
import logging
from sklearn.model_selection import KFold, TimeSeriesSplit

from ..core.base_detector import (
    BaseAnomalyDetector,
    AnomalyDetectionResult,
    AnomalyDetectionConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class CVResult:
    """Result from a single cross-validation fold."""

    fold: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    metrics: Dict[str, float]
    detector: BaseAnomalyDetector


class AnomalyDetectionCV:
    """
    Cross-validation framework for anomaly detection.

    Provides various CV strategies including k-fold, time-series, and spatial CV.
    """

    def __init__(self, n_splits: int = 5, random_state: Optional[int] = None):
        """
        Initialize CV framework.

        Args:
            n_splits: Number of CV folds
            random_state: Random seed
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.results: List[CVResult] = []

    def k_fold_cv(
        self,
        detector: BaseAnomalyDetector,
        data: Union[np.ndarray, list],
        labels: Optional[np.ndarray] = None,
        shuffle: bool = True,
    ) -> List[CVResult]:
        """
        Perform k-fold cross-validation.

        Args:
            detector: Anomaly detector to evaluate
            data: Data for CV
            labels: Optional ground truth labels
            shuffle: Whether to shuffle data before splitting

        Returns:
            List of CVResult objects
        """
        kf = KFold(n_splits=self.n_splits, shuffle=shuffle, random_state=self.random_state)
        results = []

        # Convert to array if needed
        if isinstance(data, list):
            data = np.array(data)

        for fold, (train_idx, test_idx) in enumerate(kf.split(data)):
            # Split data
            train_data = data[train_idx]
            test_data = data[test_idx]

            # Fit detector
            detector_copy = self._clone_detector(detector)
            detector_copy.fit(train_data)

            # Predict on test set
            test_results = detector_copy.predict(test_data)

            # Calculate metrics if labels available
            metrics = {}
            if labels is not None:
                test_labels = labels[test_idx]
                y_pred = np.array([r.is_anomaly for r in test_results])
                y_scores = np.array([r.anomaly_score for r in test_results])

                from .metrics import calculate_classification_metrics

                metrics_obj = calculate_classification_metrics(test_labels, y_pred, y_scores)
                metrics = {
                    "precision": metrics_obj.precision,
                    "recall": metrics_obj.recall,
                    "f1_score": metrics_obj.f1_score,
                    "accuracy": metrics_obj.accuracy,
                    "roc_auc": metrics_obj.roc_auc or 0.0,
                }

            result = CVResult(
                fold=fold,
                train_indices=train_idx,
                test_indices=test_idx,
                metrics=metrics,
                detector=detector_copy,
            )
            results.append(result)

        self.results = results
        return results

    def time_series_cv(
        self,
        detector: BaseAnomalyDetector,
        data: Union[np.ndarray, list],
        labels: Optional[np.ndarray] = None,
    ) -> List[CVResult]:
        """
        Perform time-series cross-validation (respects temporal ordering).

        Args:
            detector: Anomaly detector to evaluate
            data: Data for CV (should be ordered by time)
            labels: Optional ground truth labels

        Returns:
            List of CVResult objects
        """
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        results = []

        # Convert to array if needed
        if isinstance(data, list):
            data = np.array(data)

        for fold, (train_idx, test_idx) in enumerate(tscv.split(data)):
            # Split data
            train_data = data[train_idx]
            test_data = data[test_idx]

            # Fit detector
            detector_copy = self._clone_detector(detector)
            detector_copy.fit(train_data)

            # Predict on test set
            test_results = detector_copy.predict(test_data)

            # Calculate metrics if labels available
            metrics = {}
            if labels is not None:
                test_labels = labels[test_idx]
                y_pred = np.array([r.is_anomaly for r in test_results])
                y_scores = np.array([r.anomaly_score for r in test_results])

                from .metrics import calculate_classification_metrics

                metrics_obj = calculate_classification_metrics(test_labels, y_pred, y_scores)
                metrics = {
                    "precision": metrics_obj.precision,
                    "recall": metrics_obj.recall,
                    "f1_score": metrics_obj.f1_score,
                    "accuracy": metrics_obj.accuracy,
                    "roc_auc": metrics_obj.roc_auc or 0.0,
                }

            result = CVResult(
                fold=fold,
                train_indices=train_idx,
                test_indices=test_idx,
                metrics=metrics,
                detector=detector_copy,
            )
            results.append(result)

        self.results = results
        return results

    def spatial_cv(
        self,
        detector: BaseAnomalyDetector,
        data: Union[np.ndarray, list],
        spatial_coords: np.ndarray,
        labels: Optional[np.ndarray] = None,
        n_regions: int = 5,
    ) -> List[CVResult]:
        """
        Perform spatial cross-validation (respects spatial dependencies).

        Args:
            detector: Anomaly detector to evaluate
            data: Data for CV
            spatial_coords: Spatial coordinates (x, y, z) for each sample
            labels: Optional ground truth labels
            n_regions: Number of spatial regions for splitting

        Returns:
            List of CVResult objects
        """
        results = []

        # Convert to array if needed
        if isinstance(data, list):
            data = np.array(data)

        # Create spatial regions (simple grid-based splitting)
        coords_min = np.min(spatial_coords, axis=0)
        coords_max = np.max(spatial_coords, axis=0)
        coords_range = coords_max - coords_min

        # Simple grid-based spatial splitting
        n_per_region = len(data) // n_regions

        for fold in range(n_regions):
            # Create spatial mask (every nth region)
            region_mask = np.zeros(len(data), dtype=bool)
            for i in range(fold, len(data), n_regions):
                region_mask[i] = True

            train_mask = ~region_mask
            test_mask = region_mask

            train_data = data[train_mask]
            test_data = data[test_mask]

            # Fit detector
            detector_copy = self._clone_detector(detector)
            detector_copy.fit(train_data)

            # Predict on test set
            test_results = detector_copy.predict(test_data)

            # Calculate metrics if labels available
            metrics = {}
            if labels is not None:
                test_labels = labels[test_mask]
                y_pred = np.array([r.is_anomaly for r in test_results])
                y_scores = np.array([r.anomaly_score for r in test_results])

                from .metrics import calculate_classification_metrics

                metrics_obj = calculate_classification_metrics(test_labels, y_pred, y_scores)
                metrics = {
                    "precision": metrics_obj.precision,
                    "recall": metrics_obj.recall,
                    "f1_score": metrics_obj.f1_score,
                    "accuracy": metrics_obj.accuracy,
                    "roc_auc": metrics_obj.roc_auc or 0.0,
                }

            result = CVResult(
                fold=fold,
                train_indices=np.where(train_mask)[0],
                test_indices=np.where(test_mask)[0],
                metrics=metrics,
                detector=detector_copy,
            )
            results.append(result)

        self.results = results
        return results

    def _clone_detector(self, detector: BaseAnomalyDetector) -> BaseAnomalyDetector:
        """Create a copy of the detector for CV."""
        import copy

        return copy.deepcopy(detector)

    def get_summary_metrics(self) -> Dict[str, Tuple[float, float]]:
        """
        Get summary statistics across all CV folds.

        Returns:
            Dictionary mapping metric names to (mean, std) tuples
        """
        if not self.results:
            return {}

        # Collect all metrics
        all_metrics = {}
        for result in self.results:
            for metric_name, value in result.metrics.items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(value)

        # Calculate mean and std
        summary = {}
        for metric_name, values in all_metrics.items():
            summary[metric_name] = (np.mean(values), np.std(values))

        return summary


# Convenience functions
def k_fold_cv(
    detector: BaseAnomalyDetector,
    data: Union[np.ndarray, list],
    labels: Optional[np.ndarray] = None,
    n_splits: int = 5,
) -> List[CVResult]:
    """Convenience function for k-fold CV."""
    cv = AnomalyDetectionCV(n_splits=n_splits)
    return cv.k_fold_cv(detector, data, labels)


def time_series_cv(
    detector: BaseAnomalyDetector,
    data: Union[np.ndarray, list],
    labels: Optional[np.ndarray] = None,
    n_splits: int = 5,
) -> List[CVResult]:
    """Convenience function for time-series CV."""
    cv = AnomalyDetectionCV(n_splits=n_splits)
    return cv.time_series_cv(detector, data, labels)


def spatial_cv(
    detector: BaseAnomalyDetector,
    data: Union[np.ndarray, list],
    spatial_coords: np.ndarray,
    labels: Optional[np.ndarray] = None,
    n_regions: int = 5,
) -> List[CVResult]:
    """Convenience function for spatial CV."""
    cv = AnomalyDetectionCV(n_splits=n_regions)
    return cv.spatial_cv(detector, data, spatial_coords, labels, n_regions)
