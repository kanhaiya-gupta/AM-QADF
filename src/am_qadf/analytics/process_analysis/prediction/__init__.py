"""
AM-QADF Prediction Module

Prediction capabilities for quality and defect prediction in additive manufacturing.
Includes early defect prediction, time-series forecasting, and deep learning models.
"""

from .early_defect_predictor import (
    PredictionConfig,
    EarlyDefectPredictionResult,
    EarlyDefectPredictor,
)

from .time_series_predictor import (
    TimeSeriesPredictionResult,
    TimeSeriesPredictor,
)

from .prediction_validator import (
    OptimizationValidationResult,
    PredictionValidator,
)

__all__ = [
    # Configuration
    "PredictionConfig",
    # Results
    "EarlyDefectPredictionResult",
    "TimeSeriesPredictionResult",
    "OptimizationValidationResult",
    # Predictors
    "EarlyDefectPredictor",
    "TimeSeriesPredictor",
    "PredictionValidator",
    # Will be added as more modules are implemented:
    # "DeepLearningPredictor",
]
