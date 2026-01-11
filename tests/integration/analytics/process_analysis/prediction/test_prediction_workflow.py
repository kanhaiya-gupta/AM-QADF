"""
Integration tests for prediction workflows.

Tests integration of early defect prediction, time-series forecasting, and validation.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import shutil
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch

try:
    from am_qadf.analytics.process_analysis.prediction.early_defect_predictor import (
        EarlyDefectPredictor,
        PredictionConfig,
        EarlyDefectPredictionResult,
    )
    from am_qadf.analytics.process_analysis.prediction.time_series_predictor import (
        TimeSeriesPredictor,
        TimeSeriesPredictionResult,
    )
    from am_qadf.analytics.process_analysis.prediction.prediction_validator import (
        PredictionValidator,
        OptimizationValidationResult,
    )
    from am_qadf.analytics.process_analysis.quality_analysis import QualityPredictor, QualityAnalysisConfig
except ImportError as e:
    pytest.skip(f"Prediction modules not available: {e}", allow_module_level=True)


class TestPredictionWorkflow:
    """Integration tests for prediction workflows."""

    @pytest.fixture
    def sample_process_data(self):
        """Create sample process data for testing."""
        np.random.seed(42)
        n_samples = 200
        n_features = 10

        data = {f"feature_{i}": np.random.randn(n_samples) for i in range(n_features)}
        # Add some temporal structure
        data["timestamp"] = pd.date_range(start="2024-01-01", periods=n_samples, freq="H")
        data["build_id"] = [f"build_{i//50}" for i in range(n_samples)]

        # Create quality target with some relationship to features
        quality = 0.7 + 0.1 * data["feature_0"] + 0.05 * data["feature_1"] + np.random.randn(n_samples) * 0.1
        quality = np.clip(quality, 0.0, 1.0)
        data["quality"] = quality

        # Create defect labels (1 if quality < 0.6)
        data["defect_label"] = (quality < 0.6).astype(int)

        return pd.DataFrame(data)

    @pytest.fixture
    def prediction_config(self):
        """Create prediction configuration."""
        return PredictionConfig(
            model_type="random_forest",
            early_prediction_horizon=50,
            time_series_forecast_horizon=10,
            validation_method="kfold",
            n_folds=5,
            random_seed=42,
        )

    @pytest.fixture
    def early_defect_predictor(self, prediction_config):
        """Create EarlyDefectPredictor instance."""
        return EarlyDefectPredictor(prediction_config)

    @pytest.fixture
    def time_series_predictor(self, prediction_config):
        """Create TimeSeriesPredictor instance."""
        return TimeSeriesPredictor(prediction_config)

    @pytest.fixture
    def prediction_validator(self, prediction_config):
        """Create PredictionValidator instance."""
        return PredictionValidator(prediction_config)

    @pytest.mark.integration
    def test_early_defect_prediction_workflow(self, early_defect_predictor, sample_process_data):
        """Test complete early defect prediction workflow."""
        # Prepare training data
        feature_names = [
            col for col in sample_process_data.columns if col not in ["timestamp", "build_id", "quality", "defect_label"]
        ]

        # Group by build_id for training
        builds = sample_process_data.groupby("build_id")
        process_data_list = []
        defect_labels_list = []

        for build_id, build_data in builds:
            if len(build_data) >= 50:  # Minimum samples per build
                process_data_list.append(build_data[feature_names])
                # Use final quality to determine defect
                defect_labels_list.append(build_data["defect_label"].iloc[-1])

        if len(process_data_list) == 0:
            pytest.skip("Insufficient build data for testing")

        # Combine process data
        combined_process_data = pd.concat(process_data_list, ignore_index=True)
        defect_labels = np.array(defect_labels_list)

        # Train early prediction model
        result = early_defect_predictor.train_early_prediction_model(
            combined_process_data[: len(defect_labels) * 10],  # Ensure enough samples
            np.repeat(defect_labels, 10)[: len(combined_process_data)],  # Repeat labels
            feature_names=feature_names,
            early_horizon=50,
        )

        assert result.success is True
        assert result.early_prediction_accuracy >= 0.0
        assert len(result.model_performance) > 0
        assert early_defect_predictor.trained_model is not None

        # Test prediction on partial build data
        partial_data = sample_process_data[feature_names].iloc[:30]
        defect_prob, confidence = early_defect_predictor.predict_early_defect(partial_data, build_progress=0.3)

        assert len(defect_prob) > 0
        assert len(confidence) > 0
        assert np.all((defect_prob >= 0.0) & (defect_prob <= 1.0))
        assert np.all((confidence >= 0.0) & (confidence <= 1.0))

    @pytest.mark.integration
    def test_time_series_forecasting_workflow(self, time_series_predictor, sample_process_data):
        """Test complete time-series forecasting workflow."""
        # Extract quality time-series
        quality_series = sample_process_data["quality"].values

        # Test with moving average (most reliable)
        result = time_series_predictor.forecast_quality_metric(
            quality_series, forecast_horizon=10, model_type="moving_average"
        )

        assert result.success is True
        assert len(result.forecast) == 10
        assert len(result.forecast_lower_bound) == 10
        assert len(result.forecast_upper_bound) == 10
        assert np.all(result.forecast_lower_bound <= result.forecast_upper_bound)

        # Test anomaly detection
        actual_values = quality_series[-10:]  # Use last 10 as actual
        anomalies = time_series_predictor.detect_anomalies_in_forecast(result, actual_values)

        assert len(anomalies) == 10
        assert anomalies.dtype == bool

    @pytest.mark.integration
    def test_process_parameter_forecasting(self, time_series_predictor, sample_process_data):
        """Test forecasting specific process parameters."""
        # Create parameter history
        parameter_history = pd.DataFrame(
            {
                "temperature": 800
                + 50 * np.sin(np.arange(len(sample_process_data)) * 0.1)
                + np.random.randn(len(sample_process_data)) * 10,
                "power": 200
                + 20 * np.cos(np.arange(len(sample_process_data)) * 0.1)
                + np.random.randn(len(sample_process_data)) * 5,
            }
        )

        result = time_series_predictor.forecast_process_parameter(
            parameter_history, parameter_name="temperature", forecast_horizon=10
        )

        assert result.success is True
        assert len(result.forecast) == 10

    @pytest.mark.integration
    def test_prediction_validation_workflow(self, prediction_validator, sample_process_data):
        """Test prediction validation workflow."""
        # Create quality predictor
        quality_config = QualityAnalysisConfig(random_seed=42)
        quality_predictor = QualityPredictor(quality_config)

        # Train quality predictor
        feature_names = [
            col for col in sample_process_data.columns if col not in ["timestamp", "build_id", "quality", "defect_label"]
        ]

        quality_result = quality_predictor.analyze_quality_prediction(
            sample_process_data, quality_target="quality", feature_names=feature_names
        )

        assert quality_result.success is True

        # Perform cross-validation
        cv_result = prediction_validator.cross_validate_model(
            quality_predictor, sample_process_data, quality_target="quality", n_folds=5, validation_method="kfold"
        )

        assert isinstance(cv_result, dict)
        assert cv_result.get("n_folds") == 5
        assert cv_result.get("validation_method") == "kfold"

        # Perform experimental validation
        predicted_data = sample_process_data.iloc[:20].copy()
        experimental_data = pd.DataFrame({"quality": sample_process_data["quality"].iloc[:20].values})

        validation_result = prediction_validator.validate_with_experimental_data(
            quality_predictor, predicted_data, experimental_data, quality_target="quality"
        )

        assert validation_result.success is True
        assert validation_result.validation_method == "experimental"
        assert validation_result.validation_error is not None
        assert len(validation_result.validation_metrics) > 0

    @pytest.mark.integration
    def test_prediction_intervals_calculation(self, prediction_validator):
        """Test prediction interval calculation."""
        predictions = np.array([0.7, 0.8, 0.9, 0.85, 0.75, 0.82, 0.88])

        lower_bound, upper_bound = prediction_validator.calculate_prediction_intervals(predictions, confidence_level=0.95)

        assert len(lower_bound) == len(predictions)
        assert len(upper_bound) == len(predictions)
        assert np.all(lower_bound <= upper_bound)
        assert np.all(lower_bound <= predictions)
        assert np.all(upper_bound >= predictions)

    @pytest.mark.integration
    def test_end_to_end_prediction_workflow(
        self, early_defect_predictor, time_series_predictor, prediction_validator, sample_process_data
    ):
        """Test end-to-end prediction workflow combining all components."""
        # Step 1: Train early defect prediction
        feature_names = [
            col for col in sample_process_data.columns if col not in ["timestamp", "build_id", "quality", "defect_label"]
        ]

        # Simplified training with available data
        defect_result = early_defect_predictor.train_early_prediction_model(
            sample_process_data[feature_names],
            sample_process_data["defect_label"].values,
            feature_names=feature_names[:5],  # Use subset for faster training
        )

        assert defect_result.success is True

        # Step 2: Forecast quality time-series
        quality_series = sample_process_data["quality"].values
        forecast_result = time_series_predictor.forecast_quality_metric(
            quality_series, forecast_horizon=5, model_type="moving_average"
        )

        assert forecast_result.success is True

        # Step 3: Validate predictions
        quality_config = QualityAnalysisConfig(random_seed=42)
        quality_predictor = QualityPredictor(quality_config)

        quality_predictor.analyze_quality_prediction(
            sample_process_data, quality_target="quality", feature_names=feature_names[:5]
        )

        cv_result = prediction_validator.cross_validate_model(
            quality_predictor, sample_process_data, quality_target="quality", n_folds=3, validation_method="kfold"
        )

        assert isinstance(cv_result, dict)
        assert "n_folds" in cv_result

        # Step 4: Calculate prediction intervals
        test_predictions = np.array([0.75, 0.80, 0.85, 0.78, 0.82])
        lower, upper = prediction_validator.calculate_prediction_intervals(test_predictions, confidence_level=0.95)

        assert len(lower) == len(test_predictions)
        assert len(upper) == len(test_predictions)
