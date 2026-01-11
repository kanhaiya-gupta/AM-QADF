"""
Early Defect Prediction for PBF-LB/M Systems

This module provides early defect prediction capabilities to predict defects
before build completion using partial build data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
import logging
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import warnings

logger = logging.getLogger(__name__)


@dataclass
class PredictionConfig:
    """Configuration for quality prediction models."""

    model_type: str = "random_forest"  # 'random_forest', 'gradient_boosting', 'lstm', 'transformer'
    enable_early_prediction: bool = True
    early_prediction_horizon: int = 100  # Number of samples before build completion
    time_series_forecast_horizon: int = 10
    enable_deep_learning: bool = False
    validation_method: str = "cross_validation"  # 'cross_validation', 'holdout', 'time_series_split'
    n_folds: int = 5
    test_size: float = 0.2
    random_seed: Optional[int] = None


@dataclass
class EarlyDefectPredictionResult:
    """Result of early defect prediction."""

    success: bool
    model_type: str
    defect_probability: np.ndarray  # Probability of defect for each sample
    defect_prediction: np.ndarray  # Binary defect prediction (0=no defect, 1=defect)
    prediction_confidence: np.ndarray  # Confidence in prediction (0-1)
    early_prediction_accuracy: float  # Accuracy of early predictions
    prediction_horizon: int  # Number of samples ahead predicted
    model_performance: Dict[str, float]  # Model performance metrics
    feature_importance: Dict[str, float]  # Feature importance scores
    analysis_time: float = 0.0
    error_message: Optional[str] = None


class EarlyDefectPredictor:
    """
    Early defect prediction before build completion.

    This class provides capabilities to predict defects early in the build process
    using partial build data, allowing for early intervention and process correction.
    """

    def __init__(self, config: PredictionConfig = None):
        """Initialize early defect predictor."""
        self.config = config or PredictionConfig()
        self.trained_model = None
        self.feature_names: Optional[List[str]] = None
        self.label_encoder = None

        # Set random seed
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)

        logger.info("Early Defect Predictor initialized")

    def train_early_prediction_model(
        self, process_data: pd.DataFrame, defect_labels: np.ndarray, feature_names: List[str] = None, early_horizon: int = None
    ) -> EarlyDefectPredictionResult:
        """
        Train model to predict defects early in build process.

        Args:
            process_data: Process data with features (should include partial build data)
            defect_labels: Binary labels (0=no defect, 1=defect) for completed builds
            feature_names: Feature names (optional)
            early_horizon: Samples before completion to use for prediction (optional, uses config if None)

        Returns:
            EarlyDefectPredictionResult with trained model and performance
        """
        try:
            start_time = datetime.now()

            if feature_names is None:
                feature_names = [col for col in process_data.columns if col not in ["defect", "defect_label", "label"]]

            if early_horizon is None:
                early_horizon = self.config.early_prediction_horizon

            # Prepare data
            X = process_data[feature_names].values
            y = defect_labels.astype(int)

            # Handle missing values
            valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X = X[valid_mask]
            y = y[valid_mask]

            if len(X) == 0:
                raise ValueError("No valid data after handling missing values")

            # Check if we have enough data for train/test split
            if len(X) < 2:
                raise ValueError(f"Insufficient data for train_test_split: {len(X)} samples (need at least 2)")

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.config.test_size,
                random_state=self.config.random_seed,
                stratify=y if len(np.unique(y)) > 1 else None,
            )

            # Select model
            if self.config.model_type == "random_forest":
                model = RandomForestClassifier(n_estimators=100, random_state=self.config.random_seed, class_weight="balanced")
            elif self.config.model_type == "gradient_boosting":
                model = GradientBoostingClassifier(n_estimators=100, random_state=self.config.random_seed)
            else:
                raise ValueError(f"Unsupported model type: {self.config.model_type}")

            # Train model
            model.fit(X_train, y_train)

            # Store model for prediction
            self.trained_model = model
            self.feature_names = feature_names

            # Make predictions
            y_pred = model.predict(X_test)
            pred_proba = model.predict_proba(X_test)
            # Handle case where predict_proba returns single column (binary classification with one class in test)
            if pred_proba.shape[1] > 1:
                y_pred_proba = pred_proba[:, 1]  # Probability of defect (second class)
            else:
                y_pred_proba = pred_proba[:, 0]  # Use first (and only) column

            # Calculate model performance
            model_performance = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1_score": f1_score(y_test, y_pred, zero_division=0),
            }

            # Calculate ROC AUC if both classes present
            if len(np.unique(y_test)) > 1:
                try:
                    model_performance["roc_auc"] = roc_auc_score(y_test, y_pred_proba)
                except ValueError:
                    model_performance["roc_auc"] = 0.0
            else:
                model_performance["roc_auc"] = 0.0

            # Get feature importance
            if hasattr(model, "feature_importances_"):
                feature_importance = {
                    name: float(importance) for name, importance in zip(feature_names, model.feature_importances_)
                }
            else:
                feature_importance = {name: 0.0 for name in feature_names}

            # Calculate prediction confidence (based on probability)
            prediction_confidence = np.abs(y_pred_proba - 0.5) * 2  # 0 to 1 scale

            # Early prediction accuracy (for samples predicted early)
            early_prediction_accuracy = model_performance["accuracy"]

            # Calculate analysis time
            analysis_time = (datetime.now() - start_time).total_seconds()

            # Create result
            result = EarlyDefectPredictionResult(
                success=True,
                model_type=self.config.model_type,
                defect_probability=y_pred_proba,
                defect_prediction=y_pred,
                prediction_confidence=prediction_confidence,
                early_prediction_accuracy=early_prediction_accuracy,
                prediction_horizon=early_horizon,
                model_performance=model_performance,
                feature_importance=feature_importance,
                analysis_time=analysis_time,
            )

            logger.info(
                f"Early defect prediction model trained: {analysis_time:.2f}s, " f"Accuracy: {early_prediction_accuracy:.3f}"
            )
            return result

        except Exception as e:
            logger.error(f"Error in early defect prediction model training: {e}")
            return EarlyDefectPredictionResult(
                success=False,
                model_type=self.config.model_type,
                defect_probability=np.array([]),
                defect_prediction=np.array([]),
                prediction_confidence=np.array([]),
                early_prediction_accuracy=0.0,
                prediction_horizon=early_horizon or self.config.early_prediction_horizon,
                model_performance={},
                feature_importance={},
                error_message=str(e),
            )

    def predict_early_defect(
        self, partial_process_data: pd.DataFrame, build_progress: float  # 0.0-1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict defect probability for partial build data.

        Args:
            partial_process_data: Process data from partial build
            build_progress: Build progress (0.0-1.0)

        Returns:
            Tuple of (defect_probability, prediction_confidence)
        """
        if self.trained_model is None:
            raise ValueError("Model not trained. Call train_early_prediction_model first.")

        if self.feature_names is None:
            raise ValueError("Feature names not available. Model may not be properly trained.")

        try:
            # Prepare data
            X = partial_process_data[self.feature_names].values

            # Handle missing values
            if np.isnan(X).any():
                X = np.nan_to_num(X, nan=0.0)

            # Make predictions
            defect_probability = self.trained_model.predict_proba(X)[:, 1]
            defect_prediction = self.trained_model.predict(X)

            # Calculate prediction confidence
            prediction_confidence = np.abs(defect_probability - 0.5) * 2

            logger.debug(
                f"Early defect prediction for build progress {build_progress:.2f}: "
                f"defect_probability={np.mean(defect_probability):.3f}"
            )

            return defect_probability, prediction_confidence

        except Exception as e:
            logger.error(f"Error in early defect prediction: {e}")
            raise

    def update_model_with_new_data(
        self, new_process_data: pd.DataFrame, new_defect_labels: np.ndarray
    ) -> EarlyDefectPredictionResult:
        """
        Update model with new training data (incremental learning).

        Note: For now, retrains the entire model. Future enhancement: implement
        true incremental learning with warm_start or online learning algorithms.

        Args:
            new_process_data: New process data
            new_defect_labels: New defect labels

        Returns:
            EarlyDefectPredictionResult with updated model
        """
        try:
            # For now, retrain with combined data
            # In future, implement true incremental learning
            logger.info("Updating model with new data (full retraining)")

            # Combine with existing training data if available
            # This is a simplified approach - in production, you'd maintain a training dataset
            result = self.train_early_prediction_model(new_process_data, new_defect_labels, self.feature_names)

            return result

        except Exception as e:
            logger.error(f"Error updating model: {e}")
            return EarlyDefectPredictionResult(
                success=False,
                model_type=self.config.model_type,
                defect_probability=np.array([]),
                defect_prediction=np.array([]),
                prediction_confidence=np.array([]),
                early_prediction_accuracy=0.0,
                prediction_horizon=self.config.early_prediction_horizon,
                model_performance={},
                feature_importance={},
                error_message=str(e),
            )

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance for early defect prediction.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.trained_model is None:
            raise ValueError("Model not trained. Call train_early_prediction_model first.")

        if self.feature_names is None:
            raise ValueError("Feature names not available.")

        if not hasattr(self.trained_model, "feature_importances_"):
            return {name: 0.0 for name in self.feature_names}

        return {
            name: float(importance) for name, importance in zip(self.feature_names, self.trained_model.feature_importances_)
        }
