"""
Time-Series Forecasting for PBF-LB/M Systems

This module provides time-series forecasting capabilities for quality metrics
and process parameters in additive manufacturing.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

# Try to import statsmodels for ARIMA
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose

    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("statsmodels not available. Install with: pip install statsmodels")

# Try to import Prophet
try:
    from prophet import Prophet

    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class TimeSeriesPredictionResult:
    """Result of time-series forecasting."""

    success: bool
    model_type: str
    forecast: np.ndarray  # Forecasted values
    forecast_lower_bound: np.ndarray  # Lower bound of forecast confidence interval
    forecast_upper_bound: np.ndarray  # Upper bound of forecast confidence interval
    forecast_horizon: int  # Forecast horizon
    historical_data: np.ndarray  # Historical data used for forecasting
    model_performance: Dict[str, float]  # Model performance metrics (MAPE, RMSE, etc.)
    trend_components: Optional[np.ndarray] = None  # Trend components (if decomposed)
    seasonality_components: Optional[np.ndarray] = None  # Seasonality components (if decomposed)
    analysis_time: float = 0.0
    error_message: Optional[str] = None


class TimeSeriesPredictor:
    """
    Time-series forecasting for quality metrics.

    This class provides time-series forecasting capabilities for quality metrics
    and process parameters using various models (ARIMA, exponential smoothing, etc.).
    """

    def __init__(self, config: "PredictionConfig" = None):
        """Initialize time-series predictor."""
        from .early_defect_predictor import PredictionConfig

        self.config = config or PredictionConfig()
        self.trained_model = None
        self.model_type = None

        # Set random seed
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)

        logger.info("Time-Series Predictor initialized")

    def forecast_quality_metric(
        self,
        historical_data: np.ndarray,
        forecast_horizon: int = None,
        model_type: str = "arima",  # 'arima', 'lstm', 'prophet', 'exponential_smoothing', 'moving_average'
    ) -> TimeSeriesPredictionResult:
        """
        Forecast quality metric using time-series model.

        Args:
            historical_data: Historical time-series data (1D array)
            forecast_horizon: Number of steps ahead to forecast (uses config if None)
            model_type: Type of time-series model

        Returns:
            TimeSeriesPredictionResult with forecast and confidence intervals
        """
        try:
            start_time = datetime.now()

            if forecast_horizon is None:
                forecast_horizon = self.config.time_series_forecast_horizon

            # Handle missing values
            historical_data = np.array(historical_data)
            historical_data = np.nan_to_num(historical_data, nan=np.nanmean(historical_data))

            if len(historical_data) < 10:
                raise ValueError("Insufficient historical data (minimum 10 samples required)")

            # Store model type
            self.model_type = model_type

            # Forecast based on model type
            if model_type == "arima":
                forecast, lower_bound, upper_bound, trend, seasonality = self._forecast_arima(
                    historical_data, forecast_horizon
                )
            elif model_type == "exponential_smoothing":
                forecast, lower_bound, upper_bound, trend, seasonality = self._forecast_exponential_smoothing(
                    historical_data, forecast_horizon
                )
            elif model_type == "moving_average":
                forecast, lower_bound, upper_bound, trend, seasonality = self._forecast_moving_average(
                    historical_data, forecast_horizon
                )
            elif model_type == "prophet":
                forecast, lower_bound, upper_bound, trend, seasonality = self._forecast_prophet(
                    historical_data, forecast_horizon
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            # Calculate model performance (using last portion of historical data as test)
            test_size = min(forecast_horizon, len(historical_data) // 5)
            if test_size > 0:
                train_data = historical_data[:-test_size]
                test_data = historical_data[-test_size:]
                # Simple backward forecast for performance evaluation
                train_forecast, _, _, _, _ = self._forecast_simple(train_data, test_size)
                mae = mean_absolute_error(test_data, train_forecast)
                rmse = np.sqrt(mean_squared_error(test_data, train_forecast))
                mape = np.mean(np.abs((test_data - train_forecast) / (test_data + 1e-10))) * 100

                model_performance = {"mae": float(mae), "rmse": float(rmse), "mape": float(mape)}
            else:
                model_performance = {"mae": 0.0, "rmse": 0.0, "mape": 0.0}

            # Calculate analysis time
            analysis_time = (datetime.now() - start_time).total_seconds()

            # Create result
            result = TimeSeriesPredictionResult(
                success=True,
                model_type=model_type,
                forecast=forecast,
                forecast_lower_bound=lower_bound,
                forecast_upper_bound=upper_bound,
                forecast_horizon=forecast_horizon,
                historical_data=historical_data,
                model_performance=model_performance,
                trend_components=trend,
                seasonality_components=seasonality,
                analysis_time=analysis_time,
            )

            logger.info(
                f"Time-series forecast completed: {analysis_time:.2f}s, "
                f"Horizon: {forecast_horizon}, MAPE: {model_performance['mape']:.2f}%"
            )
            return result

        except Exception as e:
            logger.error(f"Error in time-series forecasting: {e}")
            return TimeSeriesPredictionResult(
                success=False,
                model_type=model_type,
                forecast=np.array([]),
                forecast_lower_bound=np.array([]),
                forecast_upper_bound=np.array([]),
                forecast_horizon=forecast_horizon or self.config.time_series_forecast_horizon,
                historical_data=historical_data if "historical_data" in locals() else np.array([]),
                model_performance={},
                error_message=str(e),
            )

    def _forecast_arima(
        self, data: np.ndarray, horizon: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Forecast using ARIMA model."""
        if not STATSMODELS_AVAILABLE:
            logger.warning("statsmodels not available, falling back to simple forecasting")
            return self._forecast_simple(data, horizon)

        try:
            # Fit ARIMA model (auto-select order with AIC)
            # Start with simple (1,1,1) order
            model = ARIMA(data, order=(1, 1, 1))
            fitted_model = model.fit()

            # Forecast
            forecast_result = fitted_model.forecast(steps=horizon)
            forecast = forecast_result.values

            # Get confidence intervals (95%)
            conf_int = fitted_model.get_forecast(steps=horizon).conf_int()
            lower_bound = conf_int.iloc[:, 0].values
            upper_bound = conf_int.iloc[:, 1].values

            # Decompose for trend and seasonality (if enough data)
            trend = None
            seasonality = None
            if len(data) > 50:
                try:
                    decomp = seasonal_decompose(data, period=min(12, len(data) // 2), model="additive")
                    trend = decomp.trend.values
                    seasonality = decomp.seasonal.values
                except:
                    pass

            self.trained_model = fitted_model

            return forecast, lower_bound, upper_bound, trend, seasonality

        except Exception as e:
            logger.warning(f"ARIMA forecasting failed: {e}, falling back to simple forecasting")
            return self._forecast_simple(data, horizon)

    def _forecast_exponential_smoothing(
        self, data: np.ndarray, horizon: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Forecast using exponential smoothing."""
        # Simple exponential smoothing
        alpha = 0.3  # Smoothing parameter
        forecast = []
        last_value = data[-1]

        for i in range(horizon):
            # Simple exponential smoothing formula
            next_value = alpha * last_value + (1 - alpha) * last_value
            forecast.append(next_value)
            last_value = next_value

        forecast = np.array(forecast)

        # Simple confidence intervals based on historical variance
        std = np.std(data)
        lower_bound = forecast - 1.96 * std
        upper_bound = forecast + 1.96 * std

        return forecast, lower_bound, upper_bound, None, None

    def _forecast_moving_average(
        self, data: np.ndarray, horizon: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Forecast using moving average."""
        # Simple moving average
        window_size = min(10, len(data) // 2)
        if window_size < 2:
            window_size = 2

        last_average = np.mean(data[-window_size:])
        forecast = np.full(horizon, last_average)

        # Confidence intervals based on historical variance
        std = np.std(data)
        lower_bound = forecast - 1.96 * std
        upper_bound = forecast + 1.96 * std

        return forecast, lower_bound, upper_bound, None, None

    def _forecast_prophet(
        self, data: np.ndarray, horizon: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Forecast using Prophet (if available)."""
        if not PROPHET_AVAILABLE:
            logger.warning("Prophet not available, falling back to simple forecasting")
            return self._forecast_simple(data, horizon)

        try:
            # Prepare data for Prophet (requires DataFrame with 'ds' and 'y' columns)
            df = pd.DataFrame({"ds": pd.date_range(start="2020-01-01", periods=len(data), freq="D"), "y": data})

            # Fit Prophet model
            model = Prophet()
            model.fit(df)

            # Create future dataframe
            future = model.make_future_dataframe(periods=horizon)

            # Forecast
            forecast_df = model.predict(future)

            # Extract forecast values
            forecast = forecast_df["yhat"].values[-horizon:]
            lower_bound = forecast_df["yhat_lower"].values[-horizon:]
            upper_bound = forecast_df["yhat_upper"].values[-horizon:]

            # Extract trend and seasonality
            trend = forecast_df["trend"].values[-horizon:]
            seasonality = forecast_df["yearly"].values[-horizon:] if "yearly" in forecast_df.columns else None

            self.trained_model = model

            return forecast, lower_bound, upper_bound, trend, seasonality

        except Exception as e:
            logger.warning(f"Prophet forecasting failed: {e}, falling back to simple forecasting")
            return self._forecast_simple(data, horizon)

    def _forecast_simple(
        self, data: np.ndarray, horizon: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Simple forecasting fallback (linear trend extrapolation)."""
        # Simple linear trend extrapolation
        if len(data) < 2:
            # If insufficient data, use constant value
            forecast = np.full(horizon, data[-1] if len(data) > 0 else 0.0)
        else:
            # Linear trend
            x = np.arange(len(data))
            coeffs = np.polyfit(x[-min(20, len(data)) :], data[-min(20, len(data)) :], 1)
            trend_line = np.poly1d(coeffs)

            future_x = np.arange(len(data), len(data) + horizon)
            forecast = trend_line(future_x)

        # Confidence intervals based on historical variance
        std = np.std(data)
        lower_bound = forecast - 1.96 * std
        upper_bound = forecast + 1.96 * std

        return forecast, lower_bound, upper_bound, None, None

    def forecast_process_parameter(
        self, parameter_history: pd.DataFrame, parameter_name: str, forecast_horizon: int = None
    ) -> TimeSeriesPredictionResult:
        """
        Forecast specific process parameter.

        Args:
            parameter_history: DataFrame with parameter history
            parameter_name: Name of parameter to forecast
            forecast_horizon: Number of steps ahead to forecast

        Returns:
            TimeSeriesPredictionResult with forecast
        """
        if parameter_name not in parameter_history.columns:
            raise ValueError(f"Parameter '{parameter_name}' not found in history data")

        historical_data = parameter_history[parameter_name].values

        return self.forecast_quality_metric(
            historical_data, forecast_horizon=forecast_horizon, model_type="arima"  # Default to ARIMA
        )

    def detect_anomalies_in_forecast(self, forecast: TimeSeriesPredictionResult, actual_values: np.ndarray) -> np.ndarray:
        """
        Detect anomalies by comparing forecast to actual values.

        Args:
            forecast: TimeSeriesPredictionResult with forecast
            actual_values: Actual values to compare against

        Returns:
            Array of anomaly flags (True = anomaly detected)
        """
        if not forecast.success:
            raise ValueError("Forecast result is not successful")

        if len(actual_values) != len(forecast.forecast):
            raise ValueError("Actual values length must match forecast horizon")

        # Anomaly if actual value is outside confidence interval
        anomalies = (actual_values < forecast.forecast_lower_bound) | (actual_values > forecast.forecast_upper_bound)

        # Additional check: if difference from forecast is > 3 standard deviations
        residuals = actual_values - forecast.forecast
        std_residuals = np.std(residuals)
        if std_residuals > 0:
            z_scores = np.abs(residuals) / std_residuals
            anomalies = anomalies | (z_scores > 3.0)

        return anomalies.astype(bool)
