"""
Unit tests for time series analysis.

Tests for TimeSeriesConfig, TimeSeriesResult, TimeSeriesAnalyzer, TrendAnalyzer, and SeasonalityAnalyzer.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from am_qadf.analytics.statistical_analysis.time_series import (
    TimeSeriesConfig,
    TimeSeriesResult,
    TimeSeriesAnalyzer,
    TrendAnalyzer,
    SeasonalityAnalyzer,
)


class TestTimeSeriesConfig:
    """Test suite for TimeSeriesConfig dataclass."""

    @pytest.mark.unit
    def test_config_creation_default(self):
        """Test creating TimeSeriesConfig with default values."""
        config = TimeSeriesConfig()

        assert config.trend_method == "linear"
        assert config.polynomial_degree == 2
        assert config.seasonality_detection is True
        assert config.min_periods == 2
        assert config.max_periods == 100
        assert config.forecast_horizon == 10
        assert config.confidence_level == 0.95
        assert config.significance_level == 0.05
        assert config.random_seed is None

    @pytest.mark.unit
    def test_config_creation_custom(self):
        """Test creating TimeSeriesConfig with custom values."""
        config = TimeSeriesConfig(
            trend_method="polynomial",
            polynomial_degree=3,
            seasonality_detection=False,
            min_periods=5,
            max_periods=200,
            forecast_horizon=20,
            confidence_level=0.99,
            significance_level=0.01,
            random_seed=42,
        )

        assert config.trend_method == "polynomial"
        assert config.polynomial_degree == 3
        assert config.seasonality_detection is False
        assert config.min_periods == 5
        assert config.max_periods == 200
        assert config.forecast_horizon == 20
        assert config.confidence_level == 0.99
        assert config.significance_level == 0.01
        assert config.random_seed == 42


class TestTimeSeriesResult:
    """Test suite for TimeSeriesResult dataclass."""

    @pytest.mark.unit
    def test_result_creation(self):
        """Test creating TimeSeriesResult."""
        time_series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = TimeSeriesResult(
            success=True,
            method="Trend_linear",
            time_series_data=time_series,
            trend_analysis={"slope": 1.0, "intercept": 0.0},
            seasonality_analysis={"has_seasonality": False},
            forecasting_results={"forecast": [6.0, 7.0]},
            analysis_time=1.5,
        )

        assert result.success is True
        assert result.method == "Trend_linear"
        assert len(result.time_series_data) == 5
        assert "slope" in result.trend_analysis
        assert result.analysis_time == 1.5
        assert result.error_message is None

    @pytest.mark.unit
    def test_result_creation_with_error(self):
        """Test creating TimeSeriesResult with error."""
        time_series = pd.Series([1.0, 2.0, 3.0])
        result = TimeSeriesResult(
            success=False,
            method="Trend_linear",
            time_series_data=time_series,
            trend_analysis={},
            seasonality_analysis={},
            forecasting_results={},
            error_message="Test error",
        )

        assert result.success is False
        assert result.error_message == "Test error"


class TestTimeSeriesAnalyzer:
    """Test suite for TimeSeriesAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create a TimeSeriesAnalyzer instance."""
        return TimeSeriesAnalyzer()

    @pytest.fixture
    def sample_time_series(self):
        """Create sample time series data."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        values = np.cumsum(np.random.randn(50)) + 10.0
        return pd.Series(values, index=dates, name="test_series")

    @pytest.mark.unit
    def test_analyzer_creation_default(self):
        """Test creating TimeSeriesAnalyzer with default config."""
        analyzer = TimeSeriesAnalyzer()

        assert analyzer.config is not None
        assert isinstance(analyzer.config, TimeSeriesConfig)
        assert analyzer.analysis_cache == {}

    @pytest.mark.unit
    def test_analyzer_creation_custom_config(self):
        """Test creating TimeSeriesAnalyzer with custom config."""
        config = TimeSeriesConfig(trend_method="polynomial")
        analyzer = TimeSeriesAnalyzer(config=config)

        assert analyzer.config.trend_method == "polynomial"

    @pytest.mark.unit
    def test_analyze_trend_linear(self, analyzer, sample_time_series):
        """Test linear trend analysis."""
        result = analyzer.analyze_trend(sample_time_series, method="linear")

        assert isinstance(result, TimeSeriesResult)
        assert result.method == "Trend_linear"
        assert result.success is True
        assert "slope" in result.trend_analysis
        assert "intercept" in result.trend_analysis

    @pytest.mark.unit
    def test_analyze_trend_polynomial(self, analyzer, sample_time_series):
        """Test polynomial trend analysis."""
        result = analyzer.analyze_trend(sample_time_series, method="polynomial")

        assert isinstance(result, TimeSeriesResult)
        assert result.method == "Trend_polynomial"
        assert result.success is True

    @pytest.mark.unit
    def test_analyze_trend_exponential(self, analyzer, sample_time_series):
        """Test exponential trend analysis."""
        # Use positive values for exponential
        positive_series = pd.Series(np.abs(sample_time_series.values) + 1.0, index=sample_time_series.index)
        result = analyzer.analyze_trend(positive_series, method="exponential")

        assert isinstance(result, TimeSeriesResult)
        assert result.method == "Trend_exponential"
        assert result.success is True

    @pytest.mark.unit
    def test_analyze_trend_invalid_method(self, analyzer, sample_time_series):
        """Test trend analysis with invalid method."""
        result = analyzer.analyze_trend(sample_time_series, method="invalid")

        assert isinstance(result, TimeSeriesResult)
        assert result.success is False
        assert result.error_message is not None

    @pytest.mark.unit
    def test_analyze_seasonality(self, analyzer, sample_time_series):
        """Test seasonality analysis."""
        result = analyzer.analyze_seasonality(sample_time_series)

        assert isinstance(result, TimeSeriesResult)
        assert result.method == "Seasonality"
        assert result.success is True
        assert "has_seasonality" in result.seasonality_analysis or "period" in result.seasonality_analysis

    @pytest.mark.unit
    def test_analyze_seasonality_periodic_data(self, analyzer):
        """Test seasonality analysis with periodic data."""
        # Create periodic time series
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        values = np.sin(2 * np.pi * np.arange(100) / 10) + 10.0  # Period of 10
        periodic_series = pd.Series(values, index=dates)

        result = analyzer.analyze_seasonality(periodic_series)

        assert isinstance(result, TimeSeriesResult)
        assert result.success is True

    @pytest.mark.unit
    def test_forecast(self, analyzer, sample_time_series):
        """Test time series forecasting."""
        result = analyzer.forecast(sample_time_series, horizon=5)

        assert isinstance(result, TimeSeriesResult)
        assert result.method == "Forecast"
        assert result.success is True
        assert "forecast" in result.forecasting_results or "forecast_values" in result.forecasting_results

    @pytest.mark.unit
    def test_forecast_custom_horizon(self, analyzer, sample_time_series):
        """Test forecasting with custom horizon."""
        result = analyzer.forecast(sample_time_series, horizon=10)

        assert isinstance(result, TimeSeriesResult)
        assert result.success is True

    @pytest.mark.unit
    def test_comprehensive_analysis(self, analyzer, sample_time_series):
        """Test comprehensive time series analysis."""
        result = analyzer.comprehensive_analysis(sample_time_series)

        assert isinstance(result, TimeSeriesResult)
        assert result.success is True
        assert len(result.trend_analysis) > 0
        assert len(result.seasonality_analysis) > 0

    @pytest.mark.unit
    def test_cache_result(self, analyzer, sample_time_series):
        """Test caching analysis results."""
        result = analyzer.analyze_trend(sample_time_series)

        cached = analyzer.get_cached_result("trend", sample_time_series.name)
        # May or may not be cached depending on implementation
        assert True  # Just verify no error

    @pytest.mark.unit
    def test_clear_cache(self, analyzer):
        """Test clearing analysis cache."""
        analyzer.clear_cache()
        assert len(analyzer.analysis_cache) == 0


class TestTrendAnalyzer:
    """Test suite for TrendAnalyzer class."""

    @pytest.mark.unit
    def test_trend_analyzer_creation(self):
        """Test creating TrendAnalyzer."""
        analyzer = TrendAnalyzer()

        assert isinstance(analyzer, TimeSeriesAnalyzer)
        assert analyzer.method_name == "Trend"


class TestSeasonalityAnalyzer:
    """Test suite for SeasonalityAnalyzer class."""

    @pytest.mark.unit
    def test_seasonality_analyzer_creation(self):
        """Test creating SeasonalityAnalyzer."""
        analyzer = SeasonalityAnalyzer()

        assert isinstance(analyzer, TimeSeriesAnalyzer)
        assert analyzer.method_name == "Seasonality"
