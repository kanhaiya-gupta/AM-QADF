"""
Quality Analysis for PBF-LB/M Systems

This module provides specialized quality analysis capabilities for PBF-LB/M
additive manufacturing systems, including quality prediction, quality control,
and quality optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import logging
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import warnings

logger = logging.getLogger(__name__)


@dataclass
class QualityAnalysisConfig:
    """Configuration for quality analysis."""

    # Model parameters
    model_type: str = "random_forest"  # "random_forest", "gradient_boosting"
    test_size: float = 0.2
    random_seed: Optional[int] = None

    # Quality thresholds
    quality_threshold: float = 0.8
    defect_threshold: float = 0.1

    # Analysis parameters
    confidence_level: float = 0.95


@dataclass
class QualityAnalysisResult:
    """Result of quality analysis."""

    success: bool
    method: str
    quality_metrics: Dict[str, float]
    quality_predictions: np.ndarray
    quality_classifications: np.ndarray
    model_performance: Dict[str, float]
    analysis_time: float = 0.0
    error_message: Optional[str] = None


class QualityAnalyzer:
    """
    Quality analyzer for PBF-LB/M systems.

    This class provides specialized quality analysis capabilities including
    quality prediction, quality control, and quality optimization for
    PBF-LB/M additive manufacturing.
    """

    def __init__(self, config: QualityAnalysisConfig = None):
        """Initialize the quality analyzer."""
        self.config = config or QualityAnalysisConfig()
        self.analysis_cache: Dict[str, Any] = {}
        self.trained_model = None
        self.feature_names: Optional[List[str]] = None

        # Set random seed
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)

        logger.info("Quality Analyzer initialized")

    def analyze_quality_prediction(
        self,
        process_data: pd.DataFrame,
        quality_target: str,
        feature_names: List[str] = None,
    ) -> QualityAnalysisResult:
        """
        Perform quality prediction analysis.

        Args:
            process_data: DataFrame containing process data
            quality_target: Name of quality target variable
            feature_names: List of feature names (optional)

        Returns:
            QualityAnalysisResult: Quality prediction analysis results
        """
        try:
            start_time = datetime.now()

            if feature_names is None:
                feature_names = [col for col in process_data.columns if col != quality_target]

            # Prepare data
            X = process_data[feature_names].values
            y = process_data[quality_target].values

            # Handle missing values
            valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X = X[valid_mask]
            y = y[valid_mask]

            # Check if we have enough data for train/test split
            if len(X) < 2:
                raise ValueError(f"Insufficient data for train_test_split: {len(X)} samples (need at least 2)")

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.config.test_size,
                random_state=self.config.random_seed,
            )

            # Select model
            if self.config.model_type == "random_forest":
                model = RandomForestRegressor(random_state=self.config.random_seed)
            elif self.config.model_type == "gradient_boosting":
                model = GradientBoostingRegressor(random_state=self.config.random_seed)
            else:
                raise ValueError(f"Unsupported model type: {self.config.model_type}")

            # Train model
            model.fit(X_train, y_train)

            # Store model for prediction
            self.trained_model = model
            self.feature_names = feature_names

            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate model performance
            model_performance = {
                "r2_score": r2_score(y_test, y_pred),
                "mse": mean_squared_error(y_test, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            }

            # Calculate quality metrics
            quality_metrics = {
                "mean_quality": np.mean(y_pred),
                "std_quality": np.std(y_pred),
                "min_quality": np.min(y_pred),
                "max_quality": np.max(y_pred),
                "quality_range": np.max(y_pred) - np.min(y_pred),
            }

            # Classify quality
            quality_classifications = self._classify_quality(y_pred)

            # Calculate analysis time
            analysis_time = (datetime.now() - start_time).total_seconds()

            # Create result
            result = QualityAnalysisResult(
                success=True,
                method="QualityPrediction",
                quality_metrics=quality_metrics,
                quality_predictions=y_pred,
                quality_classifications=quality_classifications,
                model_performance=model_performance,
                analysis_time=analysis_time,
            )

            # Cache result
            self._cache_result("quality_prediction", result)

            logger.info(f"Quality prediction analysis completed: {analysis_time:.2f}s")
            return result

        except Exception as e:
            logger.error(f"Error in quality prediction analysis: {e}")
            return QualityAnalysisResult(
                success=False,
                method="QualityPrediction",
                quality_metrics={},
                quality_predictions=np.array([]),
                quality_classifications=np.array([]),
                model_performance={},
                error_message=str(e),
            )

    def _classify_quality(self, quality_values: np.ndarray) -> np.ndarray:
        """Classify quality values into categories."""
        classifications = np.zeros(len(quality_values), dtype=int)

        # High quality: >= quality_threshold
        classifications[quality_values >= self.config.quality_threshold] = 2

        # Medium quality: defect_threshold <= quality < quality_threshold
        medium_mask = (quality_values >= self.config.defect_threshold) & (quality_values < self.config.quality_threshold)
        classifications[medium_mask] = 1

        # Low quality (defects): < defect_threshold
        classifications[quality_values < self.config.defect_threshold] = 0

        return classifications

    def _cache_result(self, method: str, result: QualityAnalysisResult):
        """Cache analysis result."""
        cache_key = f"{method}_{hash(str(result.quality_metrics))}"
        self.analysis_cache[cache_key] = result

    def get_cached_result(self, method: str) -> Optional[QualityAnalysisResult]:
        """Get cached analysis result."""
        cache_key = f"{method}_default"
        return self.analysis_cache.get(cache_key)

    def clear_cache(self):
        """Clear analysis cache."""
        self.analysis_cache.clear()

    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get analysis statistics."""
        return {
            "cache_size": len(self.analysis_cache),
            "config": {
                "model_type": self.config.model_type,
                "test_size": self.config.test_size,
                "quality_threshold": self.config.quality_threshold,
                "defect_threshold": self.config.defect_threshold,
            },
        }

    def analyze_quality_control(
        self,
        process_data: pd.DataFrame,
        quality_target: str,
        feature_names: List[str] = None,
    ) -> QualityAnalysisResult:
        """
        Perform quality control analysis.

        Args:
            process_data: DataFrame containing process data
            quality_target: Name of quality target variable
            feature_names: List of feature names (optional)

        Returns:
            QualityAnalysisResult: Quality control analysis results
        """
        # Use quality prediction as base, then add control metrics
        result = self.analyze_quality_prediction(process_data, quality_target, feature_names)

        if result.success:
            # Add quality control metrics
            quality_metrics = result.quality_metrics.copy()
            quality_metrics.update(
                {
                    "quality_control_score": np.mean(result.quality_classifications == 2),  # High quality ratio
                    "defect_rate": np.mean(result.quality_classifications == 0),  # Defect rate
                    "in_control_rate": np.mean(result.quality_classifications >= 1),  # In control rate
                }
            )

            result.quality_metrics = quality_metrics
            result.method = "QualityControl"

        return result


class QualityPredictor(QualityAnalyzer):
    """
    Enhanced specialized quality predictor with early defect detection,
    time-series forecasting, and validation capabilities.
    """

    def __init__(self, config: QualityAnalysisConfig = None):
        super().__init__(config)
        self.method_name = "QualityPredictor"
        self._early_defect_predictor = None
        self._time_series_predictor = None
        self._prediction_validator = None

    def predict(
        self,
        process_data: pd.DataFrame,
        quality_target: str,
        feature_names: List[str] = None,
    ) -> QualityAnalysisResult:
        """Predict quality from process data."""
        return self.analyze_quality_prediction(process_data, quality_target, feature_names)

    def predict_quality(self, new_data: pd.DataFrame, feature_names: List[str] = None) -> np.ndarray:
        """
        Predict quality for new data using trained model.

        Args:
            new_data: DataFrame containing new process data
            feature_names: List of feature names (optional, uses stored if None)

        Returns:
            Array of quality predictions
        """
        if self.trained_model is None:
            raise ValueError("Model not trained. Call analyze_quality_prediction first.")

        if feature_names is None:
            feature_names = self.feature_names

        if feature_names is None:
            raise ValueError("Feature names must be provided or model must be trained with feature names.")

        # Prepare data
        X = new_data[feature_names].values

        # Make predictions
        predictions = self.trained_model.predict(X)

        return predictions

    def predict_early_defect(
        self,
        partial_process_data: pd.DataFrame,
        build_progress: float,
        defect_labels: Optional[np.ndarray] = None,
        feature_names: List[str] = None,
        train_model: bool = True,
    ):
        """
        Predict defects early in build using partial build data.

        Args:
            partial_process_data: Process data from partial build
            build_progress: Build progress (0.0-1.0)
            defect_labels: Binary labels (0=no defect, 1=defect) for training (if train_model=True)
            feature_names: Feature names (optional)
            train_model: Whether to train model first (requires defect_labels)

        Returns:
            EarlyDefectPredictionResult with predictions
        """
        try:
            from .prediction.early_defect_predictor import EarlyDefectPredictor, PredictionConfig

            if self._early_defect_predictor is None or train_model:
                if defect_labels is None:
                    raise ValueError("defect_labels required when train_model=True")

                # Initialize predictor with config matching current config
                pred_config = PredictionConfig(
                    model_type=self.config.model_type if hasattr(self.config, "model_type") else "random_forest",
                    random_seed=self.config.random_seed,
                )
                self._early_defect_predictor = EarlyDefectPredictor(pred_config)

                # Train model
                result = self._early_defect_predictor.train_early_prediction_model(
                    partial_process_data, defect_labels, feature_names
                )
                return result
            else:
                # Use existing model to predict
                defect_probability, prediction_confidence = self._early_defect_predictor.predict_early_defect(
                    partial_process_data, build_progress
                )
                # Return a simple result structure
                from .prediction.early_defect_predictor import EarlyDefectPredictionResult

                return EarlyDefectPredictionResult(
                    success=True,
                    model_type=self._early_defect_predictor.config.model_type,
                    defect_probability=defect_probability,
                    defect_prediction=(defect_probability > 0.5).astype(int),
                    prediction_confidence=prediction_confidence,
                    early_prediction_accuracy=0.0,  # Not available without labels
                    prediction_horizon=self._early_defect_predictor.config.early_prediction_horizon,
                    model_performance={},
                    feature_importance=self._early_defect_predictor.get_feature_importance(),
                )
        except ImportError as e:
            logger.error(f"Error importing early defect predictor: {e}")
            raise ValueError("Early defect prediction module not available")
        except Exception as e:
            logger.error(f"Error in early defect prediction: {e}")
            from .prediction.early_defect_predictor import EarlyDefectPredictionResult

            return EarlyDefectPredictionResult(
                success=False,
                model_type="unknown",
                defect_probability=np.array([]),
                defect_prediction=np.array([]),
                prediction_confidence=np.array([]),
                early_prediction_accuracy=0.0,
                prediction_horizon=100,
                model_performance={},
                feature_importance={},
                error_message=str(e),
            )

    def forecast_quality_timeseries(
        self, historical_quality: np.ndarray, forecast_horizon: int = 10, model_type: str = "arima"
    ):
        """
        Forecast quality using time-series model.

        Args:
            historical_quality: Historical quality values (1D array)
            forecast_horizon: Number of steps ahead to forecast
            model_type: Type of time-series model ('arima', 'exponential_smoothing', 'moving_average', 'prophet')

        Returns:
            TimeSeriesPredictionResult with forecast and confidence intervals
        """
        try:
            from .prediction.time_series_predictor import TimeSeriesPredictor, PredictionConfig

            if self._time_series_predictor is None:
                pred_config = PredictionConfig(
                    time_series_forecast_horizon=forecast_horizon, random_seed=self.config.random_seed
                )
                self._time_series_predictor = TimeSeriesPredictor(pred_config)

            result = self._time_series_predictor.forecast_quality_metric(historical_quality, forecast_horizon, model_type)
            return result

        except ImportError as e:
            logger.error(f"Error importing time-series predictor: {e}")
            raise ValueError("Time-series prediction module not available")
        except Exception as e:
            logger.error(f"Error in time-series forecasting: {e}")
            from .prediction.time_series_predictor import TimeSeriesPredictionResult

            return TimeSeriesPredictionResult(
                success=False,
                model_type=model_type,
                forecast=np.array([]),
                forecast_lower_bound=np.array([]),
                forecast_upper_bound=np.array([]),
                forecast_horizon=forecast_horizon,
                historical_data=historical_quality,
                model_performance={},
                error_message=str(e),
            )

    def cross_validate(
        self, process_data: pd.DataFrame, quality_target: str, n_folds: int = 5, validation_method: str = "kfold"
    ) -> Dict[str, float]:
        """
        Perform cross-validation on quality prediction model.

        Args:
            process_data: DataFrame containing process data
            quality_target: Name of quality target variable
            n_folds: Number of folds for cross-validation
            validation_method: Cross-validation method ('kfold', 'stratified', 'time_series_split')

        Returns:
            Dictionary with mean and std of performance metrics across folds
        """
        try:
            from .prediction.prediction_validator import PredictionValidator, PredictionConfig

            if self._prediction_validator is None:
                pred_config = PredictionConfig(n_folds=n_folds, random_seed=self.config.random_seed)
                self._prediction_validator = PredictionValidator(pred_config)

            cv_results = self._prediction_validator.cross_validate_model(
                self, process_data, quality_target, n_folds, validation_method
            )
            return cv_results

        except ImportError as e:
            logger.error(f"Error importing prediction validator: {e}")
            raise ValueError("Prediction validation module not available")
        except Exception as e:
            logger.error(f"Error in cross-validation: {e}")
            return {"error": str(e), "n_folds": n_folds, "validation_method": validation_method}
