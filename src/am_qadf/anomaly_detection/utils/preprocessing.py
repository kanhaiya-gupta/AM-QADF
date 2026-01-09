"""
Data Preprocessing for Anomaly Detection

This module provides preprocessing utilities for preparing fused multimodal data
for anomaly detection algorithms.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing."""

    # Normalization
    normalization_method: str = "standard"  # "standard", "robust", "minmax", "none"

    # Missing data handling
    missing_data_strategy: str = "mean"  # "mean", "median", "mode", "drop", "interpolate"

    # Outlier handling
    remove_outliers: bool = False
    outlier_method: str = "iqr"  # "iqr", "zscore", "isolation_forest"
    outlier_threshold: float = 3.0

    # Feature selection
    feature_selection: bool = False
    feature_selection_method: str = "variance"  # "variance", "correlation", "mutual_info"
    min_variance: float = 0.01

    # Dimensionality reduction
    reduce_dimensions: bool = False
    reduction_method: str = "pca"  # "pca", "ica", "tsne"
    n_components: int = 10


class DataPreprocessor:
    """
    Data preprocessor for anomaly detection.

    Handles normalization, missing data, outliers, and feature engineering
    for fused multimodal voxel data.
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Initialize the preprocessor.

        Args:
            config: Preprocessing configuration
        """
        self.config = config or PreprocessingConfig()
        self.scaler = None
        self.feature_selector = None
        self.is_fitted = False

        logger.info("DataPreprocessor initialized")

    def fit_transform(
        self,
        data: Union[np.ndarray, pd.DataFrame, Dict[Tuple[int, int, int], Any]],
        return_metadata: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Fit preprocessor and transform data.

        Args:
            data: Input data
            return_metadata: Whether to return preprocessing metadata

        Returns:
            Transformed data (and metadata if requested)
        """
        self.fit(data)
        transformed = self.transform(data)

        if return_metadata:
            metadata = {
                "scaler": self.scaler,
                "feature_names": getattr(self, "feature_names", []),
                "n_features": transformed.shape[1] if len(transformed.shape) > 1 else 1,
                "n_samples": transformed.shape[0],
            }
            return transformed, metadata
        return transformed

    def fit(self, data: Union[np.ndarray, pd.DataFrame, Dict[Tuple[int, int, int], Any]]):
        """
        Fit preprocessor on data.

        Args:
            data: Training data
        """
        # Convert to array
        array_data = self._to_array(data)

        # Handle missing data
        array_data = self._handle_missing_data(array_data, fit=True)

        # Remove outliers if requested
        if self.config.remove_outliers:
            array_data = self._remove_outliers(array_data, fit=True)

        # Fit scaler
        if self.config.normalization_method == "standard":
            self.scaler = StandardScaler()
        elif self.config.normalization_method == "robust":
            self.scaler = RobustScaler()
        elif self.config.normalization_method == "minmax":
            self.scaler = MinMaxScaler()
        else:
            self.scaler = None

        if self.scaler:
            self.scaler.fit(array_data)

        self.is_fitted = True
        logger.info(f"Preprocessor fitted on {array_data.shape[0]} samples, {array_data.shape[1]} features")

    def transform(self, data: Union[np.ndarray, pd.DataFrame, Dict[Tuple[int, int, int], Any]]) -> np.ndarray:
        """
        Transform data using fitted preprocessor.

        Args:
            data: Data to transform

        Returns:
            Transformed data
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        # Convert to array
        array_data = self._to_array(data)

        # Handle missing data
        array_data = self._handle_missing_data(array_data, fit=False)

        # Remove outliers if requested
        if self.config.remove_outliers:
            array_data = self._remove_outliers(array_data, fit=False)

        # Normalize
        if self.scaler:
            array_data = self.scaler.transform(array_data)
            # Clip minmax normalization to [0, 1] to handle floating point precision issues
            if self.config.normalization_method == "minmax":
                array_data = np.clip(array_data, 0, 1)

        return array_data

    def _to_array(self, data: Union[np.ndarray, pd.DataFrame, Dict[Tuple[int, int, int], Any]]) -> np.ndarray:
        """Convert various data formats to numpy array."""
        if isinstance(data, dict):
            return self._dict_to_array(data)
        elif isinstance(data, pd.DataFrame):
            return data.values
        else:
            return np.asarray(data)

    def _dict_to_array(self, data: Dict[Tuple[int, int, int], Any]) -> np.ndarray:
        """Convert fused voxel data dictionary to array."""
        if not data:
            return np.array([])

        first_item = next(iter(data.values()))

        # Extract numeric features
        if hasattr(first_item, "__dict__"):
            features = []
            for key, value in first_item.__dict__.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    features.append(key)

            if not hasattr(self, "feature_names"):
                self.feature_names = features

            array_data = []
            for item in data.values():
                row = [getattr(item, f, 0.0) for f in self.feature_names]
                array_data.append(row)

            return np.array(array_data)
        else:
            return np.array(list(data.values()))

    def _handle_missing_data(self, data: np.ndarray, fit: bool = False) -> np.ndarray:
        """Handle missing data according to strategy."""
        if not np.isnan(data).any():
            return data

        data = data.copy()

        if self.config.missing_data_strategy == "mean":
            if fit:
                self._missing_means = np.nanmean(data, axis=0)
            for col_idx in range(data.shape[1]):
                nan_mask = np.isnan(data[:, col_idx])
                if fit:
                    data[nan_mask, col_idx] = self._missing_means[col_idx]
                else:
                    data[nan_mask, col_idx] = self._missing_means[col_idx]

        elif self.config.missing_data_strategy == "median":
            if fit:
                self._missing_medians = np.nanmedian(data, axis=0)
            for col_idx in range(data.shape[1]):
                nan_mask = np.isnan(data[:, col_idx])
                if fit:
                    data[nan_mask, col_idx] = self._missing_medians[col_idx]
                else:
                    data[nan_mask, col_idx] = self._missing_medians[col_idx]

        elif self.config.missing_data_strategy == "drop":
            data = data[~np.isnan(data).any(axis=1)]

        elif self.config.missing_data_strategy == "zero":
            data = np.nan_to_num(data, nan=0.0)

        return data

    def _remove_outliers(self, data: np.ndarray, fit: bool = False) -> np.ndarray:
        """Remove outliers according to method."""
        if self.config.outlier_method == "iqr":
            return self._remove_outliers_iqr(data)
        elif self.config.outlier_method == "zscore":
            return self._remove_outliers_zscore(data)
        else:
            return data  # Other methods require sklearn

    def _remove_outliers_iqr(self, data: np.ndarray) -> np.ndarray:
        """Remove outliers using IQR method."""
        Q1 = np.percentile(data, 25, axis=0)
        Q3 = np.percentile(data, 75, axis=0)
        IQR = Q3 - Q1
        lower_bound = Q1 - self.config.outlier_threshold * IQR
        upper_bound = Q3 + self.config.outlier_threshold * IQR

        mask = np.all((data >= lower_bound) & (data <= upper_bound), axis=1)
        return data[mask]

    def _remove_outliers_zscore(self, data: np.ndarray) -> np.ndarray:
        """Remove outliers using Z-score method."""
        z_scores = np.abs(stats.zscore(data, axis=0, nan_policy="omit"))
        mask = np.all(z_scores < self.config.outlier_threshold, axis=1)
        return data[mask]


def extract_features_from_fused_data(
    fused_data: Dict[Tuple[int, int, int], Any],
) -> Tuple[np.ndarray, List[Tuple[int, int, int]], List[str]]:
    """
    Extract feature matrix from fused voxel data.

    Args:
        fused_data: Dictionary mapping voxel indices to FusedVoxelData objects

    Returns:
        Tuple of (feature_matrix, voxel_indices, feature_names)
    """
    if not fused_data:
        return np.array([]), [], []

    first_item = next(iter(fused_data.values()))

    # Extract numeric feature names
    feature_names = []
    if hasattr(first_item, "__dict__"):
        for key, value in first_item.__dict__.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                feature_names.append(key)

    # Build feature matrix
    features = []
    indices = []
    for idx, item in fused_data.items():
        row = [getattr(item, f, 0.0) for f in feature_names]
        features.append(row)
        indices.append(idx)

    return np.array(features), indices, feature_names
