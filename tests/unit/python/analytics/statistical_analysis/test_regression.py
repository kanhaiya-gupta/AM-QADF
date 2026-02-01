"""
Unit tests for regression analysis.

Tests for RegressionConfig, RegressionResult, RegressionAnalyzer, LinearRegression, and PolynomialRegression.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from am_qadf.analytics.statistical_analysis.regression import (
    RegressionConfig,
    RegressionResult,
    RegressionAnalyzer,
    LinearRegression,
    PolynomialRegression,
)


class TestRegressionConfig:
    """Test suite for RegressionConfig dataclass."""

    @pytest.mark.unit
    def test_config_creation_default(self):
        """Test creating RegressionConfig with default values."""
        config = RegressionConfig()

        assert config.model_type == "linear"
        assert config.polynomial_degree == 2
        assert config.regularization_alpha == 1.0
        assert config.l1_ratio == 0.5
        assert config.cv_folds == 5
        assert config.scoring == "r2"
        assert config.confidence_level == 0.95
        assert config.significance_level == 0.05
        assert config.random_seed is None

    @pytest.mark.unit
    def test_config_creation_custom(self):
        """Test creating RegressionConfig with custom values."""
        config = RegressionConfig(
            model_type="ridge",
            polynomial_degree=3,
            regularization_alpha=0.5,
            l1_ratio=0.7,
            cv_folds=10,
            scoring="neg_mean_squared_error",
            confidence_level=0.99,
            significance_level=0.01,
            random_seed=42,
        )

        assert config.model_type == "ridge"
        assert config.polynomial_degree == 3
        assert config.regularization_alpha == 0.5
        assert config.l1_ratio == 0.7
        assert config.cv_folds == 10
        assert config.scoring == "neg_mean_squared_error"
        assert config.confidence_level == 0.99
        assert config.significance_level == 0.01
        assert config.random_seed == 42


class TestRegressionResult:
    """Test suite for RegressionResult dataclass."""

    @pytest.mark.unit
    def test_result_creation(self):
        """Test creating RegressionResult."""
        result = RegressionResult(
            success=True,
            method="LinearRegression",
            feature_names=["feature1", "feature2"],
            target_name="target",
            model_coefficients={"intercept": 1.0, "feature1": 0.5, "feature2": 0.3},
            model_metrics={"r2": 0.9, "mse": 0.1},
            predictions=np.array([1.0, 2.0, 3.0]),
            residuals=np.array([0.1, -0.1, 0.0]),
            confidence_intervals={"target": (0.5, 1.5)},
            analysis_time=1.5,
        )

        assert result.success is True
        assert result.method == "LinearRegression"
        assert len(result.feature_names) == 2
        assert result.target_name == "target"
        assert len(result.model_coefficients) == 3
        assert len(result.predictions) == 3
        assert result.analysis_time == 1.5
        assert result.error_message is None

    @pytest.mark.unit
    def test_result_creation_with_error(self):
        """Test creating RegressionResult with error."""
        result = RegressionResult(
            success=False,
            method="LinearRegression",
            feature_names=[],
            target_name="target",
            model_coefficients={},
            model_metrics={},
            predictions=np.array([]),
            residuals=np.array([]),
            confidence_intervals={},
            error_message="Test error",
        )

        assert result.success is False
        assert result.error_message == "Test error"


class TestRegressionAnalyzer:
    """Test suite for RegressionAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create a RegressionAnalyzer instance."""
        return RegressionAnalyzer()

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 100
        X = pd.DataFrame(
            {
                "feature1": np.random.randn(n_samples),
                "feature2": np.random.randn(n_samples),
            }
        )
        y = pd.Series(2 * X["feature1"] + 3 * X["feature2"] + np.random.randn(n_samples) * 0.1)
        return X, y

    @pytest.mark.unit
    def test_analyzer_creation_default(self):
        """Test creating RegressionAnalyzer with default config."""
        analyzer = RegressionAnalyzer()

        assert analyzer.config is not None
        assert isinstance(analyzer.config, RegressionConfig)
        assert analyzer.analysis_cache == {}

    @pytest.mark.unit
    def test_analyzer_creation_custom_config(self):
        """Test creating RegressionAnalyzer with custom config."""
        config = RegressionConfig(model_type="ridge")
        analyzer = RegressionAnalyzer(config=config)

        assert analyzer.config.model_type == "ridge"

    @pytest.mark.unit
    def test_analyze_linear_regression(self, analyzer, sample_data):
        """Test linear regression analysis."""
        X, y = sample_data
        result = analyzer.analyze_linear_regression(X, y)

        assert isinstance(result, RegressionResult)
        assert result.method == "LinearRegression"
        assert result.success is True
        assert len(result.feature_names) == 2
        assert len(result.predictions) == len(y)
        assert "r2" in result.model_metrics or "r2_score" in result.model_metrics

    @pytest.mark.unit
    def test_analyze_linear_regression_specific_features(self, analyzer, sample_data):
        """Test linear regression with specific features."""
        X, y = sample_data
        result = analyzer.analyze_linear_regression(X, y, feature_names=["feature1"])

        assert isinstance(result, RegressionResult)
        assert result.success is True
        assert len(result.feature_names) == 1

    @pytest.mark.unit
    def test_analyze_polynomial_regression(self, analyzer, sample_data):
        """Test polynomial regression analysis."""
        X, y = sample_data
        result = analyzer.analyze_polynomial_regression(X, y, degree=2)

        assert isinstance(result, RegressionResult)
        assert result.method == "PolynomialRegression"
        assert result.success is True
        assert len(result.predictions) == len(y)

    @pytest.mark.unit
    def test_analyze_polynomial_regression_custom_degree(self, analyzer, sample_data):
        """Test polynomial regression with custom degree."""
        X, y = sample_data
        result = analyzer.analyze_polynomial_regression(X, y, degree=3)

        assert isinstance(result, RegressionResult)
        assert result.success is True

    @pytest.mark.unit
    def test_analyze_ridge_regression(self, analyzer, sample_data):
        """Test ridge regression analysis."""
        X, y = sample_data
        result = analyzer.analyze_ridge_regression(X, y, alpha=0.5)

        assert isinstance(result, RegressionResult)
        assert result.method == "RidgeRegression"
        assert result.success is True

    @pytest.mark.unit
    def test_analyze_lasso_regression(self, analyzer, sample_data):
        """Test lasso regression analysis."""
        X, y = sample_data
        result = analyzer.analyze_lasso_regression(X, y, alpha=0.5)

        assert isinstance(result, RegressionResult)
        assert result.method == "LassoRegression"
        assert result.success is True

    @pytest.mark.unit
    def test_analyze_elastic_net_regression(self, analyzer, sample_data):
        """Test elastic net regression analysis."""
        X, y = sample_data
        result = analyzer.analyze_elastic_net_regression(X, y, alpha=0.5, l1_ratio=0.5)

        assert isinstance(result, RegressionResult)
        assert result.method == "ElasticNetRegression"
        assert result.success is True

    @pytest.mark.unit
    def test_analyze_linear_regression_error_handling(self, analyzer):
        """Test error handling in linear regression."""
        empty_X = pd.DataFrame()
        empty_y = pd.Series([])

        result = analyzer.analyze_linear_regression(empty_X, empty_y)

        assert isinstance(result, RegressionResult)
        assert result.success is False
        assert result.error_message is not None

    @pytest.mark.unit
    def test_cache_result(self, analyzer):
        """Test caching analysis results."""
        result = RegressionResult(
            success=True,
            method="test",
            feature_names=["feature1"],
            target_name="target",
            model_coefficients={},
            model_metrics={},
            predictions=np.array([1.0]),
            residuals=np.array([0.0]),
            confidence_intervals={},
        )

        analyzer._cache_result("test_method", result)

        cached = analyzer.get_cached_result("test_method", ["feature1"], "target")
        assert cached is not None

    @pytest.mark.unit
    def test_get_cached_result_none(self, analyzer):
        """Test getting cached result when none exists."""
        cached = analyzer.get_cached_result("nonexistent", ["feature1"], "target")
        assert cached is None

    @pytest.mark.unit
    def test_clear_cache(self, analyzer):
        """Test clearing analysis cache."""
        result = RegressionResult(
            success=True,
            method="test",
            feature_names=["feature1"],
            target_name="target",
            model_coefficients={},
            model_metrics={},
            predictions=np.array([1.0]),
            residuals=np.array([0.0]),
            confidence_intervals={},
        )

        analyzer._cache_result("test_method", result)
        assert len(analyzer.analysis_cache) > 0

        analyzer.clear_cache()
        assert len(analyzer.analysis_cache) == 0


class TestLinearRegression:
    """Test suite for LinearRegression class."""

    @pytest.mark.unit
    def test_linear_regression_creation(self):
        """Test creating LinearRegression."""
        analyzer = LinearRegression()

        assert isinstance(analyzer, RegressionAnalyzer)
        assert analyzer.method_name == "Linear"


class TestPolynomialRegression:
    """Test suite for PolynomialRegression class."""

    @pytest.mark.unit
    def test_polynomial_regression_creation(self):
        """Test creating PolynomialRegression."""
        analyzer = PolynomialRegression()

        assert isinstance(analyzer, RegressionAnalyzer)
        assert analyzer.method_name == "Polynomial"
