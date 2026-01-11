"""
Prediction Validation for PBF-LB/M Systems

This module provides validation workflows for prediction models including
cross-validation, experimental validation, and model performance evaluation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
import logging
from datetime import datetime
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit, cross_val_score, cross_validate
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import warnings

logger = logging.getLogger(__name__)


@dataclass
class OptimizationValidationResult:
    """Result of optimization validation."""

    success: bool
    validation_method: str  # 'experimental', 'cross_validation', 'simulation'
    optimized_parameters: Dict[str, float]  # Validated optimal parameters
    predicted_objective: Union[float, List[float]]  # Predicted objective value(s)
    experimental_objective: Optional[Union[float, List[float]]] = None  # Experimental objective value(s) if available
    validation_error: Optional[float] = None  # Error between predicted and experimental
    validation_metrics: Dict[str, float] = field(default_factory=dict)  # Validation metrics
    experimental_data: Optional[pd.DataFrame] = None  # Experimental validation data
    validation_time: float = 0.0
    error_message: Optional[str] = None


class PredictionValidator:
    """
    Validation workflows for prediction models.

    This class provides comprehensive validation capabilities for prediction models
    including cross-validation, experimental validation, and prediction interval calculation.
    """

    def __init__(self, config: "PredictionConfig" = None):
        """Initialize prediction validator."""
        from .early_defect_predictor import PredictionConfig

        self.config = config or PredictionConfig()

        # Set random seed
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)

        logger.info("Prediction Validator initialized")

    def cross_validate_model(
        self,
        predictor: Union["QualityPredictor", "EarlyDefectPredictor"],
        process_data: pd.DataFrame,
        quality_target: str,
        n_folds: int = None,
        validation_method: str = None,  # 'kfold', 'stratified', 'time_series_split'
    ) -> Dict[str, float]:
        """
        Perform cross-validation on prediction model.

        Args:
            predictor: Trained predictor model (QualityPredictor or EarlyDefectPredictor)
            process_data: DataFrame containing process data
            quality_target: Name of quality target variable
            n_folds: Number of folds (uses config if None)
            validation_method: Cross-validation method (uses config if None)

        Returns:
            Dictionary with mean and std of performance metrics across folds
        """
        try:
            start_time = datetime.now()

            if n_folds is None:
                n_folds = self.config.n_folds

            if validation_method is None:
                validation_method = "kfold"

            # Prepare data
            feature_names = [col for col in process_data.columns if col != quality_target]
            X = process_data[feature_names].values
            y = process_data[quality_target].values

            # Handle missing values
            valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X = X[valid_mask]
            y = y[valid_mask]

            # Select cross-validation strategy
            if validation_method == "kfold":
                cv = KFold(n_splits=n_folds, shuffle=True, random_state=self.config.random_seed)
            elif validation_method == "stratified":
                # For classification problems
                if len(np.unique(y)) > 1 and len(np.unique(y)) <= 10:
                    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.config.random_seed)
                else:
                    logger.warning("Cannot use stratified CV for regression or too many classes, using KFold")
                    cv = KFold(n_splits=n_folds, shuffle=True, random_state=self.config.random_seed)
            elif validation_method == "time_series_split":
                cv = TimeSeriesSplit(n_splits=n_folds)
            else:
                raise ValueError(f"Unsupported validation method: {validation_method}")

            # Determine if classification or regression
            is_classification = isinstance(predictor, "EarlyDefectPredictor") or (
                len(np.unique(y)) <= 10 and predictor.__class__.__name__ == "EarlyDefectPredictor"
            )

            # Select appropriate scoring metrics
            if is_classification:
                scoring = {
                    "accuracy": "accuracy",
                    "precision": "precision",
                    "recall": "recall",
                    "f1": "f1",
                    "roc_auc": "roc_auc",
                }
            else:
                scoring = {
                    "r2": "r2",
                    "neg_mean_squared_error": "neg_mean_squared_error",
                    "neg_mean_absolute_error": "neg_mean_absolute_error",
                }

            # Perform cross-validation
            # Note: This is a simplified approach. For full implementation, we'd need to
            # train a new model in each fold. For now, we assume the predictor has a
            # train/test method that we can call.

            # Manual cross-validation
            cv_scores = {key: [] for key in scoring.keys()}

            for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # Train model on fold
                train_df = pd.DataFrame(X_train, columns=feature_names)
                train_df[quality_target] = y_train

                # Train predictor (assuming it has a train method)
                if hasattr(predictor, "train_early_prediction_model"):
                    # For EarlyDefectPredictor
                    result = predictor.train_early_prediction_model(
                        train_df, y_train.astype(int) if is_classification else None, feature_names
                    )
                    y_pred = result.defect_prediction if hasattr(result, "defect_prediction") else None
                    y_pred_proba = result.defect_probability if hasattr(result, "defect_probability") else None
                elif hasattr(predictor, "analyze_quality_prediction"):
                    # For QualityPredictor
                    result = predictor.analyze_quality_prediction(train_df, quality_target, feature_names)
                    y_pred = result.quality_predictions if hasattr(result, "quality_predictions") else None
                else:
                    logger.warning(f"Predictor type {type(predictor)} not supported for cross-validation")
                    continue

                if y_pred is None or len(y_pred) == 0:
                    continue

                # Calculate scores for this fold
                if is_classification:
                    cv_scores["accuracy"].append(accuracy_score(y_test, y_pred))
                    cv_scores["precision"].append(precision_score(y_test, y_pred, zero_division=0))
                    cv_scores["recall"].append(recall_score(y_test, y_pred, zero_division=0))
                    cv_scores["f1"].append(f1_score(y_test, y_pred, zero_division=0))
                    if y_pred_proba is not None and len(np.unique(y_test)) > 1:
                        try:
                            cv_scores["roc_auc"].append(roc_auc_score(y_test, y_pred_proba))
                        except ValueError:
                            cv_scores["roc_auc"].append(0.0)
                else:
                    cv_scores["r2"].append(r2_score(y_test, y_pred))
                    cv_scores["neg_mean_squared_error"].append(-mean_squared_error(y_test, y_pred))
                    cv_scores["neg_mean_absolute_error"].append(-mean_absolute_error(y_test, y_pred))

            # Calculate mean and std across folds
            cv_results = {}
            for metric_name, scores in cv_scores.items():
                if len(scores) > 0:
                    cv_results[f"{metric_name}_mean"] = float(np.mean(scores))
                    cv_results[f"{metric_name}_std"] = float(np.std(scores))
                else:
                    cv_results[f"{metric_name}_mean"] = 0.0
                    cv_results[f"{metric_name}_std"] = 0.0

            analysis_time = (datetime.now() - start_time).total_seconds()
            cv_results["n_folds"] = n_folds
            cv_results["validation_method"] = validation_method
            cv_results["analysis_time"] = analysis_time

            logger.info(f"Cross-validation completed: {analysis_time:.2f}s, " f"{n_folds} folds, method: {validation_method}")
            return cv_results

        except Exception as e:
            logger.error(f"Error in cross-validation: {e}")
            return {
                "error": str(e),
                "n_folds": n_folds or self.config.n_folds,
                "validation_method": validation_method or "kfold",
            }

    def validate_with_experimental_data(
        self,
        predictor: Union["QualityPredictor", "EarlyDefectPredictor"],
        predicted_data: pd.DataFrame,
        experimental_data: pd.DataFrame,
        quality_target: str,
    ) -> OptimizationValidationResult:
        """
        Validate predictions against experimental data.

        Args:
            predictor: Trained predictor model
            predicted_data: Data used for predictions
            experimental_data: Experimental validation data with actual values
            quality_target: Name of quality target variable

        Returns:
            OptimizationValidationResult with validation metrics
        """
        # Validate inputs first (before try block so ValueError is raised, not caught)
        if quality_target not in experimental_data.columns:
            raise ValueError(f"Quality target '{quality_target}' not found in experimental data")

        try:
            start_time = datetime.now()

            # Make predictions on predicted_data
            # Check predict_quality first (common case for QualityPredictor)
            if hasattr(predictor, "predict_quality"):
                # For QualityPredictor
                predicted_values = predictor.predict_quality(predicted_data)
            elif hasattr(predictor, "predict_early_defect"):
                # For EarlyDefectPredictor
                # predict_early_defect returns EarlyDefectPredictionResult, not tuple
                result = predictor.predict_early_defect(predicted_data, build_progress=1.0, train_model=False)
                if hasattr(result, "defect_probability"):
                    predicted_values = result.defect_probability
                else:
                    # Fallback: try to extract from result object
                    predicted_values = (
                        np.array([result.defect_probability]) if hasattr(result, "defect_probability") else np.array([])
                    )
            else:
                raise ValueError(f"Predictor type {type(predictor)} not supported for validation")

            # Get experimental values
            experimental_values = experimental_data[quality_target].values

            # Ensure same length
            min_len = min(len(predicted_values), len(experimental_values))
            predicted_values = predicted_values[:min_len]
            experimental_values = experimental_values[:min_len]

            # Calculate validation metrics
            validation_error = np.mean(np.abs(predicted_values - experimental_values))
            rmse = np.sqrt(mean_squared_error(experimental_values, predicted_values))
            mae = mean_absolute_error(experimental_values, predicted_values)

            # For classification, calculate additional metrics
            if hasattr(predictor, "predict_early_defect"):
                # Binary classification
                predicted_binary = (predicted_values > 0.5).astype(int)
                experimental_binary = (experimental_values > 0.5).astype(int)

                accuracy = accuracy_score(experimental_binary, predicted_binary)
                precision = precision_score(experimental_binary, predicted_binary, zero_division=0)
                recall = recall_score(experimental_binary, predicted_binary, zero_division=0)
                f1 = f1_score(experimental_binary, predicted_binary, zero_division=0)

                validation_metrics = {
                    "accuracy": float(accuracy),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1_score": float(f1),
                    "validation_error": float(validation_error),
                    "rmse": float(rmse),
                    "mae": float(mae),
                }
            else:
                # Regression
                r2 = r2_score(experimental_values, predicted_values)
                validation_metrics = {
                    "r2_score": float(r2),
                    "rmse": float(rmse),
                    "mae": float(mae),
                    "validation_error": float(validation_error),
                }

            validation_time = (datetime.now() - start_time).total_seconds()

            result = OptimizationValidationResult(
                success=True,
                validation_method="experimental",
                optimized_parameters={},  # Not applicable for prediction validation
                predicted_objective=float(np.mean(predicted_values)),
                experimental_objective=float(np.mean(experimental_values)),
                validation_error=float(validation_error),
                validation_metrics=validation_metrics,
                experimental_data=experimental_data,
                validation_time=validation_time,
            )

            logger.info(
                f"Experimental validation completed: {validation_time:.2f}s, " f"Validation error: {validation_error:.4f}"
            )
            return result

        except Exception as e:
            logger.error(f"Error in experimental validation: {e}")
            return OptimizationValidationResult(
                success=False,
                validation_method="experimental",
                optimized_parameters={},
                predicted_objective=0.0,
                validation_error=None,
                validation_metrics={},
                error_message=str(e),
            )

    def calculate_prediction_intervals(
        self, predictions: np.ndarray, confidence_level: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate prediction intervals for uncertainty quantification.

        Args:
            predictions: Array of predictions
            confidence_level: Confidence level (default: 0.95)

        Returns:
            Tuple of (lower_bound, upper_bound) arrays
        """
        try:
            # Simple approach: use prediction variance/uncertainty
            # In practice, this would use model uncertainty estimation
            mean_pred = np.mean(predictions)
            std_pred = np.std(predictions)

            # Z-score for confidence level
            try:
                from scipy import stats

                z_score = stats.norm.ppf((1 + confidence_level) / 2)
            except ImportError:
                # Fallback: approximate z-score for common confidence levels
                z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
                z_score = z_scores.get(confidence_level, 1.96)

            lower_bound = mean_pred - z_score * std_pred
            upper_bound = mean_pred + z_score * std_pred

            # Expand to full array length
            lower_bound = np.full(len(predictions), lower_bound)
            upper_bound = np.full(len(predictions), upper_bound)

            return lower_bound, upper_bound

        except Exception as e:
            logger.warning(f"Error calculating prediction intervals: {e}, using simple bounds")
            # Fallback: use min/max as bounds
            lower_bound = np.full(len(predictions), np.min(predictions))
            upper_bound = np.full(len(predictions), np.max(predictions))
            return lower_bound, upper_bound
