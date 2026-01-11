"""
Model Performance Tracking for PBF-LB/M Systems

This module provides model performance tracking and monitoring capabilities
for prediction models over time.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging
from collections import deque
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ModelPerformanceMetrics:
    """Model performance tracking metrics."""

    model_id: str
    model_type: str
    version: str
    training_date: datetime
    performance_metrics: Dict[str, float]  # R², RMSE, MAE, accuracy, etc.
    validation_metrics: Dict[str, float] = field(default_factory=dict)  # Cross-validation metrics
    feature_importance: Dict[str, float] = field(default_factory=dict)  # Feature importance
    drift_score: float = 0.0  # Model drift score (0-1, higher = more drift)
    last_evaluated: datetime = field(default_factory=datetime.now)
    evaluation_count: int = 0  # Number of times model evaluated
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelPerformanceTracker:
    """
    Track and monitor model performance over time.

    This class provides capabilities to evaluate, track, and monitor model
    performance over time, detecting degradation and drift.
    """

    def __init__(self, model_id: str, model_registry: "ModelRegistry", history_size: int = 100):
        """
        Initialize performance tracker for specific model.

        Args:
            model_id: Model ID to track
            model_registry: ModelRegistry instance
            history_size: Maximum number of performance history entries to keep
        """
        self.model_id = model_id
        self.model_registry = model_registry
        self.history_size = history_size
        self.performance_history: deque = deque(maxlen=history_size)

        # Try to load existing history
        self._load_history()

        logger.info(f"Model Performance Tracker initialized for model {model_id}")

    def _get_history_file(self) -> Path:
        """Get history file path for this model."""
        if self.model_registry.storage_path is None:
            storage_path = Path("models")
        else:
            storage_path = Path(self.model_registry.storage_path)
        return storage_path / self.model_id / "performance_history.json"

    def _load_history(self) -> None:
        """Load performance history from file."""
        history_file = self._get_history_file()
        if history_file.exists():
            try:
                with open(history_file, "r") as f:
                    data = json.load(f)
                    for entry in data:
                        # Convert datetime strings back to datetime objects
                        entry["training_date"] = datetime.fromisoformat(entry["training_date"])
                        entry["last_evaluated"] = datetime.fromisoformat(entry["last_evaluated"])
                        self.performance_history.append(ModelPerformanceMetrics(**entry))
                logger.debug(f"Loaded {len(self.performance_history)} performance history entries")
            except Exception as e:
                logger.warning(f"Error loading performance history: {e}")
                self.performance_history.clear()

    def _save_history(self) -> None:
        """Save performance history to file."""
        history_file = self._get_history_file()
        try:
            history_file.parent.mkdir(parents=True, exist_ok=True)
            data = []
            for entry in self.performance_history:
                entry_dict = {
                    "model_id": entry.model_id,
                    "model_type": entry.model_type,
                    "version": entry.version,
                    "training_date": entry.training_date.isoformat(),
                    "performance_metrics": entry.performance_metrics,
                    "validation_metrics": entry.validation_metrics,
                    "feature_importance": entry.feature_importance,
                    "drift_score": entry.drift_score,
                    "last_evaluated": entry.last_evaluated.isoformat(),
                    "evaluation_count": entry.evaluation_count,
                    "metadata": entry.metadata,
                }
                data.append(entry_dict)

            with open(history_file, "w") as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved {len(self.performance_history)} performance history entries")
        except Exception as e:
            logger.error(f"Error saving performance history: {e}")

    def evaluate_model_performance(
        self,
        model: Any,
        test_data: pd.DataFrame,
        quality_target: str,
        evaluation_date: datetime = None,
        feature_names: Optional[List[str]] = None,
    ) -> ModelPerformanceMetrics:
        """
        Evaluate model performance on test data.

        Args:
            model: Trained model object (must have predict method)
            test_data: DataFrame with test data
            quality_target: Name of quality target variable
            evaluation_date: Date of evaluation (uses current time if None)
            feature_names: Feature names (optional)

        Returns:
            ModelPerformanceMetrics with current performance
        """
        try:
            if evaluation_date is None:
                evaluation_date = datetime.now()

            # Get model metadata from registry
            try:
                _, model_version = self.model_registry.load_model(self.model_id)
            except Exception:
                # Model not in registry, create basic metadata
                model_version = None
                logger.warning(f"Model {self.model_id} not found in registry, using basic metadata")

            if model_version is None:
                model_type = type(model).__name__
                version = "unknown"
                training_date = datetime.now()
            else:
                model_type = model_version.model_type
                version = model_version.version
                training_date = model_version.training_date

            # Prepare test data
            # Try to get expected number of features from model
            expected_n_features = None
            if hasattr(model, "n_features_in_"):
                expected_n_features = model.n_features_in_
            elif hasattr(model, "feature_names_in_"):
                expected_n_features = len(model.feature_names_in_)
                # Use model's feature names if available
                if feature_names is None and hasattr(model, "feature_names_in_"):
                    feature_names = list(model.feature_names_in_)

            # If no feature names provided, use all columns except target
            if feature_names is None:
                feature_names = [col for col in test_data.columns if col != quality_target]

            # Extract numeric features only from original DataFrame, respecting model's expected features
            numeric_features = []
            numeric_data = []

            for feat in feature_names:
                if feat in test_data.columns:
                    col_data = pd.to_numeric(test_data[feat], errors="coerce")
                    if not col_data.isna().all():  # At least some valid numeric values
                        numeric_features.append(feat)
                        numeric_data.append(col_data.values)

            if len(numeric_features) == 0:
                raise ValueError(f"No numeric features found in test data. Available features: {feature_names}")

            # Check if we have the expected number of features
            if expected_n_features is not None and len(numeric_features) != expected_n_features:
                # Try to match by position or use first N features
                if len(numeric_features) > expected_n_features:
                    logger.warning(
                        f"Test data has {len(numeric_features)} features but model expects {expected_n_features}. "
                        f"Using first {expected_n_features} features: {numeric_features[:expected_n_features]}"
                    )
                    numeric_features = numeric_features[:expected_n_features]
                    numeric_data = numeric_data[:expected_n_features]
                elif len(numeric_features) < expected_n_features:
                    raise ValueError(
                        f"Model expects {expected_n_features} features but only {len(numeric_features)} "
                        f"numeric features found: {numeric_features}"
                    )

            X_test = np.column_stack(numeric_data) if len(numeric_data) > 1 else numeric_data[0].reshape(-1, 1)

            # Convert target to numeric
            try:
                y_test = pd.to_numeric(test_data[quality_target], errors="coerce").values
            except (ValueError, TypeError):
                raise ValueError(f"Target variable '{quality_target}' cannot be converted to numeric")

            # Handle missing values
            valid_mask = ~(np.isnan(X_test).any(axis=1) | np.isnan(y_test))
            X_test = X_test[valid_mask]
            y_test = y_test[valid_mask]

            if len(X_test) == 0:
                raise ValueError("No valid test data after handling missing values")

            # Ensure X_test is 2D
            if X_test.ndim == 1:
                X_test = X_test.reshape(-1, 1)

            # Make predictions
            y_pred = model.predict(X_test)

            # Determine if classification or regression
            is_classification = len(np.unique(y_test)) <= 10 and np.all(np.mod(y_test, 1) == 0)

            # Calculate performance metrics
            if is_classification:
                # Classification metrics
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

                y_pred_binary = (y_pred > 0.5).astype(int) if y_pred.dtype == float else y_pred.astype(int)
                y_test_binary = y_test.astype(int)

                performance_metrics = {
                    "accuracy": float(accuracy_score(y_test_binary, y_pred_binary)),
                    "precision": float(precision_score(y_test_binary, y_pred_binary, zero_division=0)),
                    "recall": float(recall_score(y_test_binary, y_pred_binary, zero_division=0)),
                    "f1_score": float(f1_score(y_test_binary, y_pred_binary, zero_division=0)),
                }

                # Try to get ROC AUC if probabilities available
                if hasattr(model, "predict_proba"):
                    try:
                        y_pred_proba = model.predict_proba(X_test)[:, 1]
                        performance_metrics["roc_auc"] = float(roc_auc_score(y_test_binary, y_pred_proba))
                    except Exception:
                        pass
            else:
                # Regression metrics
                from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

                performance_metrics = {
                    "r2_score": float(r2_score(y_test, y_pred)),
                    "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                    "mae": float(mean_absolute_error(y_test, y_pred)),
                }

            # Get feature importance if available
            feature_importance = {}
            if hasattr(model, "feature_importances_") and feature_names:
                feature_importance = {
                    name: float(importance) for name, importance in zip(feature_names, model.feature_importances_)
                }

            # Calculate drift score (simplified - compare data distributions)
            drift_score = 0.0  # Will be calculated by calculate_drift_score method

            # Create performance metrics
            metrics = ModelPerformanceMetrics(
                model_id=self.model_id,
                model_type=model_type,
                version=version,
                training_date=training_date,
                performance_metrics=performance_metrics,
                validation_metrics={},  # Can be added separately
                feature_importance=feature_importance,
                drift_score=drift_score,
                last_evaluated=evaluation_date,
                evaluation_count=len(self.performance_history) + 1,
                metadata={"test_samples": len(X_test)},
            )

            # Add to history
            self.track_performance_history(metrics)

            logger.info(
                f"Evaluated model {self.model_id} performance: "
                f"{list(performance_metrics.keys())[0]}={list(performance_metrics.values())[0]:.3f}"
            )
            return metrics

        except Exception as e:
            logger.error(f"Error evaluating model performance: {e}")
            raise

    def track_performance_history(self, metrics: ModelPerformanceMetrics) -> None:
        """Add performance metrics to history."""
        self.performance_history.append(metrics)
        self._save_history()
        logger.debug(f"Added performance metrics to history (count: {len(self.performance_history)})")

    def detect_performance_degradation(
        self, threshold: float = 0.1, metric_name: Optional[str] = None  # 10% degradation threshold
    ) -> Tuple[bool, float]:
        """
        Detect if model performance has degraded significantly.

        Args:
            threshold: Degradation threshold (0-1, e.g., 0.1 = 10% degradation)
            metric_name: Specific metric to check (checks first metric if None)

        Returns:
            Tuple of (degradation_detected, degradation_percentage)
        """
        if len(self.performance_history) < 2:
            return False, 0.0

        # Get baseline (first evaluation) and latest evaluation
        baseline = self.performance_history[0]
        latest = self.performance_history[-1]

        # Select metric to check
        if metric_name is None:
            # Use first metric available
            if baseline.performance_metrics:
                metric_name = list(baseline.performance_metrics.keys())[0]
            else:
                return False, 0.0

        if metric_name not in baseline.performance_metrics or metric_name not in latest.performance_metrics:
            logger.warning(f"Metric {metric_name} not available in performance history")
            return False, 0.0

        baseline_value = baseline.performance_metrics[metric_name]
        latest_value = latest.performance_metrics[metric_name]

        # Determine if higher is better (for most metrics except RMSE, MAE, MSE)
        higher_is_better = metric_name.lower() not in ["rmse", "mae", "mse", "error"]

        if baseline_value == 0:
            return False, 0.0

        # Calculate degradation
        if higher_is_better:
            degradation = (baseline_value - latest_value) / abs(baseline_value)
        else:
            degradation = (latest_value - baseline_value) / abs(baseline_value)

        degradation_detected = degradation > threshold

        if degradation_detected:
            logger.warning(
                f"Performance degradation detected: {metric_name} degraded by "
                f"{degradation*100:.1f}% (threshold: {threshold*100:.1f}%)"
            )

        return degradation_detected, degradation

    def get_performance_trend(self, metric_name: str) -> Dict[str, Any]:
        """
        Get performance trend for specific metric.

        Args:
            metric_name: Name of metric to analyze

        Returns:
            Dictionary with trend direction, slope, etc.
        """
        if len(self.performance_history) < 2:
            return {
                "metric_name": metric_name,
                "trend": "insufficient_data",
                "slope": 0.0,
                "data_points": len(self.performance_history),
            }

        # Extract metric values over time
        values = []
        timestamps = []

        for entry in self.performance_history:
            if metric_name in entry.performance_metrics:
                values.append(entry.performance_metrics[metric_name])
                timestamps.append(entry.last_evaluated.timestamp())

        if len(values) < 2:
            return {"metric_name": metric_name, "trend": "insufficient_data", "slope": 0.0, "data_points": len(values)}

        # Calculate linear trend
        timestamps = np.array(timestamps)
        values = np.array(values)

        # Normalize timestamps to 0-1 range
        if timestamps.max() > timestamps.min():
            timestamps_norm = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min())
        else:
            timestamps_norm = np.ones_like(timestamps) * 0.5

        # Linear regression
        slope = np.polyfit(timestamps_norm, values, 1)[0]

        # Determine trend direction
        if abs(slope) < 0.001:
            trend = "stable"
        elif slope > 0:
            trend = "improving"
        else:
            trend = "degrading"

        # Calculate average change per evaluation
        avg_change = (values[-1] - values[0]) / len(values) if len(values) > 1 else 0.0

        return {
            "metric_name": metric_name,
            "trend": trend,
            "slope": float(slope),
            "avg_change_per_eval": float(avg_change),
            "initial_value": float(values[0]),
            "latest_value": float(values[-1]),
            "data_points": len(values),
        }

    def calculate_drift_score(self, current_data: pd.DataFrame, training_data: pd.DataFrame) -> float:
        """
        Calculate model drift score (data distribution change).

        Simple implementation using statistical distance between data distributions.

        Args:
            current_data: Current data distribution
            training_data: Original training data distribution

        Returns:
            Drift score (0-1, higher = more drift)
        """
        try:
            # Simple drift detection: compare means and standard deviations
            common_columns = set(current_data.columns) & set(training_data.columns)

            if len(common_columns) == 0:
                return 1.0  # Maximum drift if no common columns

            drift_scores = []

            for col in common_columns:
                try:
                    current_col = current_data[col].dropna().values
                    training_col = training_data[col].dropna().values

                    if len(current_col) == 0 or len(training_col) == 0:
                        continue

                    # Calculate normalized difference in means and stds
                    current_mean = np.mean(current_col)
                    training_mean = np.mean(training_col)
                    current_std = np.std(current_col)
                    training_std = np.std(training_col)

                    # Normalize by training std (or use absolute difference if std is 0)
                    if training_std > 0:
                        mean_diff = abs(current_mean - training_mean) / training_std
                        std_diff = abs(current_std - training_std) / (training_std + 1e-10)
                    else:
                        mean_diff = abs(current_mean - training_mean)
                        std_diff = abs(current_std - training_std)

                    # Combine into single drift score for this feature (0-1 scale)
                    feature_drift = min(1.0, (mean_diff + std_diff) / 2.0)
                    drift_scores.append(feature_drift)

                except Exception as e:
                    logger.debug(f"Error calculating drift for column {col}: {e}")
                    continue

            if len(drift_scores) == 0:
                return 0.0

            # Average drift across all features
            overall_drift = np.mean(drift_scores)

            return float(min(1.0, max(0.0, overall_drift)))

        except Exception as e:
            logger.error(f"Error calculating drift score: {e}")
            return 0.0

    def get_performance_history(self) -> List[Dict[str, Any]]:
        """Get performance history as list of dictionaries."""
        return [
            {
                "model_id": entry.model_id,
                "last_evaluated": entry.last_evaluated.isoformat(),
                "performance_metrics": entry.performance_metrics,
                "drift_score": entry.drift_score,
                "evaluation_count": entry.evaluation_count,
            }
            for entry in self.performance_history
        ]
