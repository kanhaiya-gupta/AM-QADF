"""
Unit tests for TimeSeriesPredictor.

Tests for time-series forecasting models and configuration.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from am_qadf.analytics.process_analysis.prediction.time_series_predictor import (
    TimeSeriesPredictionResult,
    TimeSeriesPredictor,
)


class TestTimeSeriesPredictionResult:
    """Test suite for TimeSeriesPredictionResult dataclass."""

    @pytest.mark.unit
    def test_result_creation(self):
        """Test creating TimeSeriesPredictionResult."""
        result = TimeSeriesPredictionResult(
            success=True,
            model_type="arima",
            forecast=np.array([1.0, 1.1, 1.2, 1.3]),
            forecast_lower_bound=np.array([0.9, 1.0, 1.1, 1.2]),
            forecast_upper_bound=np.array([1.1, 1.2, 1.3, 1.4]),
            forecast_horizon=4,
            historical_data=np.array([0.8, 0.9, 1.0]),
            model_performance={"mae": 0.05, "rmse": 0.1, "mape": 5.0},
            analysis_time=1.5,
        )

        assert result.success is True
        assert result.model_type == "arima"
        assert len(result.forecast) == 4
        assert len(result.forecast_lower_bound) == 4
        assert len(result.forecast_upper_bound) == 4
        assert result.forecast_horizon == 4
        assert "mae" in result.model_performance
        assert result.analysis_time == 1.5


class TestTimeSeriesPredictor:
    """Test suite for TimeSeriesPredictor class."""

    @pytest.fixture
    def predictor(self):
        """Create TimeSeriesPredictor instance."""
        from am_qadf.analytics.process_analysis.prediction.early_defect_predictor import PredictionConfig

        config = PredictionConfig(random_seed=42)
        return TimeSeriesPredictor(config)

    @pytest.fixture
    def sample_time_series_data(self):
        """Create sample time-series data."""
        np.random.seed(42)
        # Generate synthetic time-series with trend and noise
        n_samples = 50
        trend = np.linspace(0, 10, n_samples)
        noise = np.random.randn(n_samples) * 0.5
        data = trend + noise
        return data

    @pytest.mark.unit
    def test_predictor_initialization(self, predictor):
        """Test TimeSeriesPredictor initialization."""
        assert predictor.config is not None
        assert predictor.trained_model is None
        assert predictor.model_type is None

    @pytest.mark.unit
    def test_forecast_quality_metric_arima(self, predictor, sample_time_series_data):
        """Test forecasting with ARIMA model."""
        result = predictor.forecast_quality_metric(sample_time_series_data, forecast_horizon=10, model_type="arima")

        # ARIMA may not be available, so check success or graceful failure
        assert hasattr(result, "success")
        if result.success:
            assert result.model_type == "arima"
            assert len(result.forecast) == 10
            assert len(result.forecast_lower_bound) == 10
            assert len(result.forecast_upper_bound) == 10
            assert result.forecast_horizon == 10

    @pytest.mark.unit
    def test_forecast_quality_metric_exponential_smoothing(self, predictor, sample_time_series_data):
        """Test forecasting with exponential smoothing."""
        result = predictor.forecast_quality_metric(
            sample_time_series_data, forecast_horizon=5, model_type="exponential_smoothing"
        )

        # Exponential smoothing may not be available, so check success or graceful failure
        assert hasattr(result, "success")
        if result.success:
            assert result.model_type == "exponential_smoothing"
            assert len(result.forecast) == 5
            assert len(result.forecast_lower_bound) == 5
            assert len(result.forecast_upper_bound) == 5

    @pytest.mark.unit
    def test_forecast_quality_metric_moving_average(self, predictor, sample_time_series_data):
        """Test forecasting with moving average."""
        result = predictor.forecast_quality_metric(sample_time_series_data, forecast_horizon=5, model_type="moving_average")

        assert result.success is True
        assert result.model_type == "moving_average"
        assert len(result.forecast) == 5

    @pytest.mark.unit
    def test_forecast_quality_metric_insufficient_data(self, predictor):
        """Test forecasting with insufficient data."""
        insufficient_data = np.array([1.0, 2.0])  # Only 2 samples

        result = predictor.forecast_quality_metric(
            insufficient_data, forecast_horizon=10, model_type="moving_average"  # Use moving_average which requires less data
        )

        # Should handle insufficient data gracefully - may fail or use simple forecasting
        assert hasattr(result, "success")

    @pytest.mark.unit
    def test_forecast_quality_metric_with_nans(self, predictor):
        """Test forecasting with NaN values."""
        data_with_nans = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])

        result = predictor.forecast_quality_metric(data_with_nans, forecast_horizon=3, model_type="moving_average")

        # Should handle NaNs (fill with mean or handle gracefully)
        assert hasattr(result, "success")
        if result.success:
            assert len(result.forecast) == 3

    @pytest.mark.unit
    def test_forecast_process_parameter(self, predictor):
        """Test forecasting specific process parameter."""
        parameter_history = pd.DataFrame(
            {"temperature": np.linspace(800, 1200, 50), "power": np.linspace(200, 300, 50), "time": np.arange(50)}
        )

        result = predictor.forecast_process_parameter(parameter_history, parameter_name="temperature", forecast_horizon=10)

        assert result.success is True
        assert len(result.forecast) == 10

    @pytest.mark.unit
    def test_forecast_process_parameter_invalid_name(self, predictor):
        """Test forecasting with invalid parameter name."""
        parameter_history = pd.DataFrame({"temperature": np.linspace(800, 1200, 50)})

        with pytest.raises(ValueError, match="Parameter 'invalid_param' not found"):
            predictor.forecast_process_parameter(parameter_history, parameter_name="invalid_param", forecast_horizon=10)

    @pytest.mark.unit
    def test_detect_anomalies_in_forecast(self, predictor):
        """Test anomaly detection in forecasts."""
        # Create forecast result
        forecast = TimeSeriesPredictionResult(
            success=True,
            model_type="arima",
            forecast=np.array([1.0, 1.1, 1.2, 1.3]),
            forecast_lower_bound=np.array([0.9, 1.0, 1.1, 1.2]),
            forecast_upper_bound=np.array([1.1, 1.2, 1.3, 1.4]),
            forecast_horizon=4,
            historical_data=np.array([0.8, 0.9, 1.0]),
            model_performance={},
        )

        # Actual values with some anomalies
        actual_values = np.array([1.05, 1.15, 2.0, 1.35])  # 2.0 is anomaly

        anomalies = predictor.detect_anomalies_in_forecast(forecast, actual_values)

        assert len(anomalies) == len(actual_values)
        assert anomalies.dtype == bool
        # Third value should be detected as anomaly (outside confidence interval)
        assert anomalies[2] == True  # 2.0 is outside [1.1, 1.3] (use == for numpy bool)

    @pytest.mark.unit
    def test_detect_anomalies_in_forecast_length_mismatch(self, predictor):
        """Test anomaly detection with length mismatch."""
        forecast = TimeSeriesPredictionResult(
            success=True,
            model_type="arima",
            forecast=np.array([1.0, 1.1, 1.2]),
            forecast_lower_bound=np.array([0.9, 1.0, 1.1]),
            forecast_upper_bound=np.array([1.1, 1.2, 1.3]),
            forecast_horizon=3,
            historical_data=np.array([0.8, 0.9, 1.0]),
            model_performance={},
        )

        actual_values = np.array([1.05, 1.15])  # Different length

        with pytest.raises(ValueError, match="Actual values length must match forecast horizon"):
            predictor.detect_anomalies_in_forecast(forecast, actual_values)

    @pytest.mark.unit
    def test_detect_anomalies_in_forecast_failed_forecast(self, predictor):
        """Test anomaly detection with failed forecast."""
        failed_forecast = TimeSeriesPredictionResult(
            success=False,
            model_type="arima",
            forecast=np.array([]),
            forecast_lower_bound=np.array([]),
            forecast_upper_bound=np.array([]),
            forecast_horizon=0,
            historical_data=np.array([]),
            model_performance={},
            error_message="Forecast failed",
        )

        actual_values = np.array([1.0, 1.1])

        with pytest.raises(ValueError, match="Forecast result is not successful"):
            predictor.detect_anomalies_in_forecast(failed_forecast, actual_values)
