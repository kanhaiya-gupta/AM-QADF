"""
Base Anomaly Detector for Phase 11: Anomaly Detection in Fused Multimodal Data

This module provides the base classes and data structures for all anomaly detection
methods in the PBF-LB/M process analysis pipeline.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
from datetime import datetime

from .types import AnomalyType

logger = logging.getLogger(__name__)


@dataclass
class AnomalyDetectionResult:
    """Result of anomaly detection for a single data point or collection."""

    # Identification
    voxel_index: Tuple[int, int, int]  # Voxel grid index
    voxel_coordinates: Tuple[float, float, float]  # World coordinates (x, y, z)

    # Detection results
    is_anomaly: bool  # Whether this is detected as an anomaly
    anomaly_score: float  # Anomaly score (higher = more anomalous)
    confidence: float  # Detection confidence (0-1)

    # Method information
    detector_name: str  # Name of detection method
    anomaly_type: Optional[AnomalyType] = None  # Type of anomaly
    detection_timestamp: datetime = field(default_factory=datetime.now)

    # Additional metadata
    features: Dict[str, float] = field(default_factory=dict)  # Feature values used
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata


@dataclass
class AnomalyDetectionConfig:
    """Configuration for anomaly detection."""

    # General parameters
    threshold: float = 0.5  # Anomaly score threshold
    confidence_threshold: float = 0.7  # Minimum confidence for detection

    # Preprocessing
    normalize_features: bool = True
    handle_missing: str = "mean"  # "mean", "median", "drop", "zero"
    remove_outliers: bool = False

    # Method-specific parameters (can be overridden)
    method_params: Dict[str, Any] = field(default_factory=dict)

    # Performance
    parallel_processing: bool = False
    max_workers: int = 4
    batch_size: int = 1000


class BaseAnomalyDetector(ABC):
    """
    Base class for all anomaly detection methods.

    All anomaly detectors should inherit from this class and implement
    the required abstract methods.
    """

    def __init__(self, config: Optional[AnomalyDetectionConfig] = None):
        """
        Initialize the anomaly detector.

        Args:
            config: Configuration for the detector
        """
        self.config = config or AnomalyDetectionConfig()
        self.name = self.__class__.__name__
        self.is_fitted = False
        self.feature_names: List[str] = []

        logger.info(f"Initialized {self.name} detector")

    @abstractmethod
    def fit(
        self,
        data: Union[np.ndarray, pd.DataFrame, Dict[Tuple[int, int, int], Any]],
        labels: Optional[np.ndarray] = None,
    ) -> "BaseAnomalyDetector":
        """
        Fit the detector on normal data.

        Args:
            data: Training data (normal data only)
            labels: Optional ground truth labels (for supervised methods)

        Returns:
            Self for method chaining
        """
        pass

    @abstractmethod
    def predict(self, data: Union[np.ndarray, pd.DataFrame, Dict[Tuple[int, int, int], Any]]) -> List[AnomalyDetectionResult]:
        """
        Detect anomalies in data.

        Args:
            data: Data to detect anomalies in

        Returns:
            List of AnomalyDetectionResult objects
        """
        pass

    def predict_scores(self, data: Union[np.ndarray, pd.DataFrame, Dict[Tuple[int, int, int], Any]]) -> np.ndarray:
        """
        Get anomaly scores without full result objects.

        Args:
            data: Data to score

        Returns:
            Array of anomaly scores
        """
        results = self.predict(data)
        return np.array([r.anomaly_score for r in results])

    def _preprocess_data(self, data: Union[np.ndarray, pd.DataFrame, Dict[Tuple[int, int, int], Any]]) -> np.ndarray:
        """
        Preprocess data for anomaly detection.

        Args:
            data: Input data in various formats

        Returns:
            Preprocessed numpy array
        """
        # Convert to numpy array
        if isinstance(data, dict):
            # Convert fused voxel data dict to array
            array_data = self._dict_to_array(data)
        elif isinstance(data, pd.DataFrame):
            array_data = data.values
        else:
            array_data = np.asarray(data)

        # Handle missing values
        if self.config.handle_missing == "mean":
            array_data = self._fill_missing_mean(array_data)
        elif self.config.handle_missing == "median":
            array_data = self._fill_missing_median(array_data)
        elif self.config.handle_missing == "zero":
            array_data = np.nan_to_num(array_data, nan=0.0)
        elif self.config.handle_missing == "drop":
            array_data = array_data[~np.isnan(array_data).any(axis=1)]

        # Normalize if requested
        if self.config.normalize_features and not self.is_fitted:
            # Store normalization parameters during fit and normalize
            self._mean = np.nanmean(array_data, axis=0)
            self._std = np.nanstd(array_data, axis=0)
            self._std[self._std == 0] = 1.0  # Avoid division by zero
            array_data = (array_data - self._mean) / self._std
        elif self.config.normalize_features and self.is_fitted:
            # Apply stored normalization
            if not hasattr(self, "_mean") or not hasattr(self, "_std"):
                # Fallback: compute on the fly if not set (shouldn't happen in normal flow)
                self._mean = np.nanmean(array_data, axis=0)
                self._std = np.nanstd(array_data, axis=0)
                self._std[self._std == 0] = 1.0
            array_data = (array_data - self._mean) / self._std

        return array_data

    def _dict_to_array(self, data: Dict[Tuple[int, int, int], Any]) -> np.ndarray:
        """
        Convert fused voxel data dictionary to numpy array.

        Args:
            data: Dictionary mapping voxel indices to FusedVoxelData or similar

        Returns:
            Numpy array of feature vectors
        """
        if not data:
            return np.array([])

        # Extract features from first item to determine structure
        first_item = next(iter(data.values()))

        # Try to extract features from FusedVoxelData or similar structure
        if hasattr(first_item, "__dict__"):
            # Extract numeric features
            features = []
            for key, value in first_item.__dict__.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    features.append(key)

            if not self.feature_names:
                self.feature_names = features

            # Build array
            array_data = []
            indices = []
            for idx, item in data.items():
                row = [getattr(item, f, 0.0) for f in self.feature_names]
                array_data.append(row)
                indices.append(idx)

            self._voxel_indices = indices
            return np.array(array_data)
        else:
            # Assume it's already a feature vector
            return np.array(list(data.values()))

    def _fill_missing_mean(self, data: np.ndarray) -> np.ndarray:
        """Fill missing values with column means."""
        col_means = np.nanmean(data, axis=0)
        nan_indices = np.isnan(data)
        data = data.copy()
        for col_idx in range(data.shape[1]):
            data[nan_indices[:, col_idx], col_idx] = col_means[col_idx]
        return data

    def _fill_missing_median(self, data: np.ndarray) -> np.ndarray:
        """Fill missing values with column medians."""
        col_medians = np.nanmedian(data, axis=0)
        nan_indices = np.isnan(data)
        data = data.copy()
        for col_idx in range(data.shape[1]):
            data[nan_indices[:, col_idx], col_idx] = col_medians[col_idx]
        return data

    def _calculate_confidence(self, scores: np.ndarray, threshold: float) -> np.ndarray:
        """
        Calculate confidence scores based on distance from threshold.

        Higher confidence for scores further from threshold (more certain).
        Lower confidence for scores near threshold (less certain).

        Args:
            scores: Anomaly scores
            threshold: Detection threshold

        Returns:
            Confidence scores (0-1)
        """
        # Confidence increases with distance from threshold
        # Scores at threshold have lowest confidence, scores far from threshold have highest
        distances = np.abs(scores - threshold)
        max_distance = np.max(distances) if len(distances) > 0 else 1.0
        if max_distance > 0:
            # Normalize distances to [0, 1] and use as confidence
            # Closer to threshold (smaller distance) = lower confidence
            # Further from threshold (larger distance) = higher confidence
            confidence = distances / max_distance
        else:
            # All scores are at threshold, give minimum confidence
            confidence = np.zeros_like(scores)
        return np.clip(confidence, 0.0, 1.0)

    def _create_results(
        self,
        scores: np.ndarray,
        indices: Optional[List[Tuple[int, int, int]]] = None,
        coordinates: Optional[List[Tuple[float, float, float]]] = None,
        features: Optional[List[Dict[str, float]]] = None,
    ) -> List[AnomalyDetectionResult]:
        """
        Create AnomalyDetectionResult objects from scores.

        Args:
            scores: Anomaly scores
            indices: Voxel indices (optional)
            coordinates: World coordinates (optional)
            features: Feature dictionaries (optional)

        Returns:
            List of AnomalyDetectionResult objects
        """
        results = []
        is_anomaly = scores >= self.config.threshold
        confidence = self._calculate_confidence(scores, self.config.threshold)

        for i, score in enumerate(scores):
            idx = indices[i] if indices and i < len(indices) else (0, 0, 0)
            coords = coordinates[i] if coordinates and i < len(coordinates) else (0.0, 0.0, 0.0)
            feat = features[i] if features and i < len(features) else {}

            result = AnomalyDetectionResult(
                voxel_index=idx,
                voxel_coordinates=coords,
                is_anomaly=bool(is_anomaly[i]),
                anomaly_score=float(score),
                confidence=float(confidence[i]),
                detector_name=self.name,
                features=feat,
            )
            results.append(result)

        return results
