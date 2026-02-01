"""
Unit tests for PredictionValidator.

Tests for prediction validation workflows including cross-validation and experimental validation.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from am_qadf.analytics.process_analysis.prediction.prediction_validator import (
    OptimizationValidationResult,
    PredictionValidator,
)


class TestOptimizationValidationResult:
    """Test suite for OptimizationValidationResult dataclass."""

    @pytest.mark.unit
    def test_result_creation_simulation(self):
        """Test creating OptimizationValidationResult for simulation validation."""
        result = OptimizationValidationResult(
            success=True,
            validation_method="simulation",
            optimized_parameters={"param1": 1.0, "param2": 2.0},
            predicted_objective=0.5,
            experimental_objective=0.52,
            validation_error=0.02,
            validation_metrics={"validation_error": 0.02, "relative_error": 0.04},
            validation_time=1.5,
        )

        assert result.success is True
        assert result.validation_method == "simulation"
        assert len(result.optimized_parameters) == 2
        assert result.predicted_objective == 0.5
        assert result.experimental_objective == 0.52
        assert result.validation_error == 0.02
        assert "validation_error" in result.validation_metrics

    @pytest.mark.unit
    def test_result_creation_experimental(self):
        """Test creating OptimizationValidationResult for experimental validation."""
        experimental_data = pd.DataFrame({"quality": [0.8, 0.85, 0.9]})

        result = OptimizationValidationResult(
            success=True,
            validation_method="experimental",
            optimized_parameters={"param1": 1.0},
            predicted_objective=0.85,
            experimental_objective=0.83,
            validation_error=0.02,
            validation_metrics={"rmse": 0.05, "mae": 0.02},
            experimental_data=experimental_data,
            validation_time=2.0,
        )

        assert result.success is True
        assert result.validation_method == "experimental"
        assert result.experimental_data is not None


class TestPredictionValidator:
    """Test suite for PredictionValidator class."""

    @pytest.fixture
    def validator(self):
        """Create PredictionValidator instance."""
        from am_qadf.analytics.process_analysis.prediction.early_defect_predictor import PredictionConfig

        config = PredictionConfig(random_seed=42)
        return PredictionValidator(config)

    @pytest.fixture
    def sample_process_data(self):
        """Create sample process data."""
        np.random.seed(42)
        n_samples = 100
        data = {
            "feature_0": np.random.randn(n_samples),
            "feature_1": np.random.randn(n_samples),
            "feature_2": np.random.randn(n_samples),
            "quality": np.random.uniform(0.5, 1.0, n_samples),
        }
        return pd.DataFrame(data)

    @pytest.mark.unit
    def test_validator_initialization(self, validator):
        """Test PredictionValidator initialization."""
        assert validator.config is not None

    @pytest.mark.unit
    def test_cross_validate_model_kfold(self, validator, sample_process_data):
        """Test cross-validation with k-fold."""
        from am_qadf.analytics.process_analysis.quality_analysis import QualityPredictor, QualityAnalysisConfig

        config = QualityAnalysisConfig(random_seed=42)
        predictor = QualityPredictor(config)

        result = validator.cross_validate_model(
            predictor, sample_process_data, quality_target="quality", n_folds=5, validation_method="kfold"
        )

        assert isinstance(result, dict)
        assert result.get("n_folds") == 5
        assert result.get("validation_method") == "kfold"
        # Should have some metrics
        assert "n_folds" in result

    @pytest.mark.unit
    def test_cross_validate_model_time_series_split(self, validator, sample_process_data):
        """Test cross-validation with time series split."""
        from am_qadf.analytics.process_analysis.quality_analysis import QualityPredictor, QualityAnalysisConfig

        config = QualityAnalysisConfig(random_seed=42)
        predictor = QualityPredictor(config)

        result = validator.cross_validate_model(
            predictor, sample_process_data, quality_target="quality", n_folds=3, validation_method="time_series_split"
        )

        assert isinstance(result, dict)
        assert result.get("n_folds") == 3
        assert result.get("validation_method") == "time_series_split"

    @pytest.mark.unit
    def test_validate_with_experimental_data_success(self, validator, sample_process_data):
        """Test experimental validation with successful validation."""
        from am_qadf.analytics.process_analysis.quality_analysis import QualityPredictor, QualityAnalysisConfig

        config = QualityAnalysisConfig(random_seed=42)
        predictor = QualityPredictor(config)

        # Train predictor first
        predictor.analyze_quality_prediction(sample_process_data, "quality")

        # Split data for prediction and experimental
        predicted_data = sample_process_data.iloc[:20].copy()
        experimental_data = pd.DataFrame({"quality": sample_process_data["quality"].iloc[:20].values})

        result = validator.validate_with_experimental_data(
            predictor, predicted_data, experimental_data, quality_target="quality"
        )

        assert result.success is True
        assert result.validation_method == "experimental"
        assert result.validation_error is not None
        assert len(result.validation_metrics) > 0

    @pytest.mark.unit
    def test_validate_with_experimental_data_missing_target(self, validator, sample_process_data):
        """Test experimental validation with missing quality target."""
        from am_qadf.analytics.process_analysis.quality_analysis import QualityPredictor, QualityAnalysisConfig

        config = QualityAnalysisConfig(random_seed=42)
        predictor = QualityPredictor(config)

        # Train predictor first
        predictor.analyze_quality_prediction(sample_process_data, "quality")

        predicted_data = sample_process_data.iloc[:20].copy()

        experimental_data = pd.DataFrame({"other_metric": np.random.uniform(0.7, 0.9, 20)})

        with pytest.raises(ValueError, match="Quality target 'quality' not found"):
            validator.validate_with_experimental_data(predictor, predicted_data, experimental_data, quality_target="quality")

    @pytest.mark.unit
    def test_calculate_prediction_intervals(self, validator):
        """Test prediction interval calculation."""
        predictions = np.array([0.7, 0.8, 0.9, 0.85, 0.75])

        lower_bound, upper_bound = validator.calculate_prediction_intervals(predictions, confidence_level=0.95)

        assert len(lower_bound) == len(predictions)
        assert len(upper_bound) == len(predictions)
        assert np.all(lower_bound <= upper_bound)
        # Bounds are based on mean and std, so should encompass the data
        assert lower_bound[0] <= np.mean(predictions) <= upper_bound[0]

    @pytest.mark.unit
    def test_calculate_prediction_intervals_different_confidence(self, validator):
        """Test prediction intervals with different confidence levels."""
        predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        lower_90, upper_90 = validator.calculate_prediction_intervals(predictions, confidence_level=0.90)
        lower_95, upper_95 = validator.calculate_prediction_intervals(predictions, confidence_level=0.95)

        # 95% intervals should be wider than 90%
        width_90 = (upper_90 - lower_90).mean()
        width_95 = (upper_95 - lower_95).mean()
        assert width_95 >= width_90
