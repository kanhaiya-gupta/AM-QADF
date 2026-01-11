"""
Unit tests for EarlyDefectPredictor.

Tests for early defect prediction models and configuration.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from am_qadf.analytics.process_analysis.prediction.early_defect_predictor import (
    PredictionConfig,
    EarlyDefectPredictionResult,
    EarlyDefectPredictor,
)


class TestPredictionConfig:
    """Test suite for PredictionConfig dataclass."""

    @pytest.mark.unit
    def test_config_creation_default(self):
        """Test creating PredictionConfig with default values."""
        config = PredictionConfig()

        assert config.model_type == "random_forest"
        assert config.enable_early_prediction is True
        assert config.early_prediction_horizon == 100
        assert config.time_series_forecast_horizon == 10
        assert config.enable_deep_learning is False
        assert config.validation_method == "cross_validation"
        assert config.n_folds == 5
        assert config.test_size == 0.2
        assert config.random_seed is None

    @pytest.mark.unit
    def test_config_creation_custom(self):
        """Test creating PredictionConfig with custom values."""
        config = PredictionConfig(
            model_type="gradient_boosting",
            enable_early_prediction=False,
            early_prediction_horizon=200,
            time_series_forecast_horizon=20,
            enable_deep_learning=True,
            validation_method="holdout",
            n_folds=10,
            test_size=0.3,
            random_seed=42,
        )

        assert config.model_type == "gradient_boosting"
        assert config.enable_early_prediction is False
        assert config.early_prediction_horizon == 200
        assert config.time_series_forecast_horizon == 20
        assert config.enable_deep_learning is True
        assert config.validation_method == "holdout"
        assert config.n_folds == 10
        assert config.test_size == 0.3
        assert config.random_seed == 42


class TestEarlyDefectPredictionResult:
    """Test suite for EarlyDefectPredictionResult dataclass."""

    @pytest.mark.unit
    def test_result_creation(self):
        """Test creating EarlyDefectPredictionResult."""
        result = EarlyDefectPredictionResult(
            success=True,
            model_type="random_forest",
            defect_probability=np.array([0.1, 0.3, 0.8, 0.9]),
            defect_prediction=np.array([0, 0, 1, 1]),
            prediction_confidence=np.array([0.8, 0.7, 0.7, 0.8]),
            early_prediction_accuracy=0.85,
            prediction_horizon=100,
            model_performance={"accuracy": 0.85, "precision": 0.9, "recall": 0.8},
            feature_importance={"feature1": 0.5, "feature2": 0.3, "feature3": 0.2},
            analysis_time=1.5,
        )

        assert result.success is True
        assert result.model_type == "random_forest"
        assert len(result.defect_probability) == 4
        assert len(result.defect_prediction) == 4
        assert result.early_prediction_accuracy == 0.85
        assert result.prediction_horizon == 100
        assert "accuracy" in result.model_performance
        assert len(result.feature_importance) == 3
        assert result.analysis_time == 1.5


class TestEarlyDefectPredictor:
    """Test suite for EarlyDefectPredictor class."""

    @pytest.fixture
    def predictor(self):
        """Create EarlyDefectPredictor instance."""
        config = PredictionConfig(random_seed=42)
        return EarlyDefectPredictor(config)

    @pytest.fixture
    def sample_process_data(self):
        """Create sample process data."""
        np.random.seed(42)
        n_samples = 100
        n_features = 5

        data = {f"feature_{i}": np.random.randn(n_samples) for i in range(n_features)}
        return pd.DataFrame(data)

    @pytest.fixture
    def sample_defect_labels(self):
        """Create sample defect labels."""
        np.random.seed(42)
        # Binary labels: 0 = no defect, 1 = defect
        labels = np.random.choice([0, 1], size=100, p=[0.7, 0.3])
        return labels

    @pytest.mark.unit
    def test_predictor_initialization(self, predictor):
        """Test EarlyDefectPredictor initialization."""
        assert predictor.config is not None
        assert predictor.trained_model is None
        assert predictor.feature_names is None

    @pytest.mark.unit
    def test_train_early_prediction_model_success(self, predictor, sample_process_data, sample_defect_labels):
        """Test training early prediction model successfully."""
        result = predictor.train_early_prediction_model(
            sample_process_data, sample_defect_labels, feature_names=None, early_horizon=50
        )

        assert result.success is True
        assert result.model_type == "random_forest"
        assert len(result.defect_probability) > 0
        assert len(result.defect_prediction) > 0
        assert result.early_prediction_accuracy >= 0.0
        assert result.prediction_horizon == 50
        assert "accuracy" in result.model_performance
        assert len(result.feature_importance) > 0
        assert predictor.trained_model is not None

    @pytest.mark.unit
    def test_train_early_prediction_model_with_feature_names(self, predictor, sample_process_data, sample_defect_labels):
        """Test training with specified feature names."""
        feature_names = ["feature_0", "feature_1", "feature_2"]

        result = predictor.train_early_prediction_model(sample_process_data, sample_defect_labels, feature_names=feature_names)

        assert result.success is True
        assert predictor.feature_names == feature_names

    @pytest.mark.unit
    def test_train_early_prediction_model_empty_data(self, predictor):
        """Test training with empty data."""
        empty_data = pd.DataFrame()
        empty_labels = np.array([])

        result = predictor.train_early_prediction_model(empty_data, empty_labels)

        assert result.success is False
        assert result.error_message is not None

    @pytest.mark.unit
    def test_predict_early_defect_no_model(self, predictor, sample_process_data):
        """Test prediction without trained model."""
        with pytest.raises(ValueError, match="Model not trained"):
            predictor.predict_early_defect(sample_process_data, build_progress=0.5)

    @pytest.mark.unit
    def test_predict_early_defect_success(self, predictor, sample_process_data, sample_defect_labels):
        """Test early defect prediction with trained model."""
        # Train model first
        predictor.train_early_prediction_model(sample_process_data, sample_defect_labels)

        # Use partial data for prediction
        partial_data = sample_process_data.iloc[:20]

        defect_probability, prediction_confidence = predictor.predict_early_defect(partial_data, build_progress=0.5)

        assert len(defect_probability) == len(partial_data)
        assert len(prediction_confidence) == len(partial_data)
        assert np.all((defect_probability >= 0.0) & (defect_probability <= 1.0))
        assert np.all((prediction_confidence >= 0.0) & (prediction_confidence <= 1.0))

    @pytest.mark.unit
    def test_update_model_with_new_data(self, predictor, sample_process_data, sample_defect_labels):
        """Test updating model with new data."""
        # Train initial model
        initial_result = predictor.train_early_prediction_model(sample_process_data.iloc[:50], sample_defect_labels[:50])

        assert initial_result.success is True

        # Update with new data
        new_data = sample_process_data.iloc[50:]
        new_labels = sample_defect_labels[50:]

        updated_result = predictor.update_model_with_new_data(new_data, new_labels)

        assert updated_result.success is True
        assert predictor.trained_model is not None

    @pytest.mark.unit
    def test_get_feature_importance_no_model(self, predictor):
        """Test getting feature importance without trained model."""
        with pytest.raises(ValueError, match="Model not trained"):
            predictor.get_feature_importance()

    @pytest.mark.unit
    def test_get_feature_importance_success(self, predictor, sample_process_data, sample_defect_labels):
        """Test getting feature importance with trained model."""
        # Train model first
        predictor.train_early_prediction_model(sample_process_data, sample_defect_labels)

        feature_importance = predictor.get_feature_importance()

        assert isinstance(feature_importance, dict)
        assert len(feature_importance) > 0
        # Check that importance values are non-negative
        assert all(v >= 0.0 for v in feature_importance.values())

    @pytest.mark.unit
    def test_predictor_different_model_types(self, sample_process_data, sample_defect_labels):
        """Test predictor with different model types."""
        # Test Random Forest
        config_rf = PredictionConfig(model_type="random_forest", random_seed=42)
        predictor_rf = EarlyDefectPredictor(config_rf)
        result_rf = predictor_rf.train_early_prediction_model(sample_process_data, sample_defect_labels)
        assert result_rf.success is True
        assert result_rf.model_type == "random_forest"

        # Test Gradient Boosting
        config_gb = PredictionConfig(model_type="gradient_boosting", random_seed=42)
        predictor_gb = EarlyDefectPredictor(config_gb)
        result_gb = predictor_gb.train_early_prediction_model(sample_process_data, sample_defect_labels)
        assert result_gb.success is True
        assert result_gb.model_type == "gradient_boosting"

    @pytest.mark.unit
    def test_predictor_invalid_model_type(self):
        """Test predictor with invalid model type."""
        config = PredictionConfig(model_type="invalid_model", random_seed=42)
        predictor = EarlyDefectPredictor(config)

        sample_data = pd.DataFrame({"feature_0": np.random.randn(50)})
        sample_labels = np.random.choice([0, 1], size=50)

        result = predictor.train_early_prediction_model(sample_data, sample_labels)
        assert result.success is False
        assert result.error_message is not None
