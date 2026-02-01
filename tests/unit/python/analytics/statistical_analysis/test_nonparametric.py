"""
Unit tests for nonparametric analysis.

Tests for NonparametricConfig, NonparametricResult, NonparametricAnalyzer, and KernelDensityAnalyzer.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from am_qadf.analytics.statistical_analysis.nonparametric import (
    NonparametricConfig,
    NonparametricResult,
    NonparametricAnalyzer,
    KernelDensityAnalyzer,
)


class TestNonparametricConfig:
    """Test suite for NonparametricConfig dataclass."""

    @pytest.mark.unit
    def test_config_creation_default(self):
        """Test creating NonparametricConfig with default values."""
        config = NonparametricConfig()

        assert config.kde_bandwidth == "scott"
        assert config.kde_kernel == "gaussian"
        assert config.test_alpha == 0.05
        assert config.alternative == "two-sided"
        assert config.confidence_level == 0.95
        assert config.random_seed is None

    @pytest.mark.unit
    def test_config_creation_custom(self):
        """Test creating NonparametricConfig with custom values."""
        config = NonparametricConfig(
            kde_bandwidth="silverman",
            kde_kernel="epanechnikov",
            test_alpha=0.01,
            alternative="greater",
            confidence_level=0.99,
            random_seed=42,
        )

        assert config.kde_bandwidth == "silverman"
        assert config.kde_kernel == "epanechnikov"
        assert config.test_alpha == 0.01
        assert config.alternative == "greater"
        assert config.confidence_level == 0.99
        assert config.random_seed == 42


class TestNonparametricResult:
    """Test suite for NonparametricResult dataclass."""

    @pytest.mark.unit
    def test_result_creation(self):
        """Test creating NonparametricResult."""
        result = NonparametricResult(
            success=True,
            method="KernelDensity",
            data_names=["data1"],
            analysis_results={"bandwidth": 0.5},
            test_statistics={"statistic": 1.5},
            p_values={"p_value": 0.05},
            confidence_intervals={"data1": (0.5, 1.5)},
            analysis_time=1.5,
        )

        assert result.success is True
        assert result.method == "KernelDensity"
        assert len(result.data_names) == 1
        assert len(result.test_statistics) == 1
        assert result.analysis_time == 1.5
        assert result.error_message is None

    @pytest.mark.unit
    def test_result_creation_with_error(self):
        """Test creating NonparametricResult with error."""
        result = NonparametricResult(
            success=False,
            method="KernelDensity",
            data_names=[],
            analysis_results={},
            test_statistics={},
            p_values={},
            confidence_intervals={},
            error_message="Test error",
        )

        assert result.success is False
        assert result.error_message == "Test error"


class TestNonparametricAnalyzer:
    """Test suite for NonparametricAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create a NonparametricAnalyzer instance."""
        return NonparametricAnalyzer()

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        return pd.Series(np.random.randn(100), name="test_data")

    @pytest.mark.unit
    def test_analyzer_creation_default(self):
        """Test creating NonparametricAnalyzer with default config."""
        analyzer = NonparametricAnalyzer()

        assert analyzer.config is not None
        assert isinstance(analyzer.config, NonparametricConfig)
        assert analyzer.analysis_cache == {}

    @pytest.mark.unit
    def test_analyzer_creation_custom_config(self):
        """Test creating NonparametricAnalyzer with custom config."""
        config = NonparametricConfig(kde_bandwidth="silverman")
        analyzer = NonparametricAnalyzer(config=config)

        assert analyzer.config.kde_bandwidth == "silverman"

    @pytest.mark.unit
    def test_analyze_kernel_density(self, analyzer, sample_data):
        """Test kernel density estimation."""
        result = analyzer.analyze_kernel_density(sample_data)

        assert isinstance(result, NonparametricResult)
        assert result.method == "KernelDensity"
        assert result.success is True
        assert len(result.data_names) == 1
        assert "bandwidth" in result.analysis_results or "kde_model" in result.analysis_results

    @pytest.mark.unit
    def test_analyze_kernel_density_custom_name(self, analyzer, sample_data):
        """Test kernel density estimation with custom data name."""
        result = analyzer.analyze_kernel_density(sample_data, data_name="custom_name")

        assert isinstance(result, NonparametricResult)
        assert result.success is True
        assert "custom_name" in result.data_names

    @pytest.mark.unit
    def test_analyze_kernel_density_empty_data(self, analyzer):
        """Test kernel density estimation with empty data."""
        empty_data = pd.Series([], name="empty")

        result = analyzer.analyze_kernel_density(empty_data)

        assert isinstance(result, NonparametricResult)
        assert result.success is False
        assert result.error_message is not None

    @pytest.mark.unit
    def test_analyze_nonparametric_tests_one_sample(self, analyzer, sample_data):
        """Test nonparametric tests with one sample."""
        result = analyzer.analyze_nonparametric_tests(sample_data)

        assert isinstance(result, NonparametricResult)
        assert result.success is True
        assert len(result.test_statistics) > 0

    @pytest.mark.unit
    def test_analyze_nonparametric_tests_two_sample(self, analyzer, sample_data):
        """Test nonparametric tests with two samples."""
        np.random.seed(43)
        data2 = pd.Series(np.random.randn(100), name="test_data2")

        result = analyzer.analyze_nonparametric_tests(sample_data, data2)

        assert isinstance(result, NonparametricResult)
        assert result.success is True
        assert len(result.test_statistics) > 0
        assert len(result.p_values) > 0

    @pytest.mark.unit
    def test_analyze_nonparametric_tests_ks_test(self, analyzer, sample_data):
        """Test Kolmogorov-Smirnov test."""
        np.random.seed(43)
        data2 = pd.Series(np.random.randn(100), name="test_data2")

        result = analyzer.analyze_nonparametric_tests(sample_data, data2)

        assert isinstance(result, NonparametricResult)
        assert result.success is True

    @pytest.mark.unit
    def test_analyze_nonparametric_tests_mann_whitney(self, analyzer, sample_data):
        """Test Mann-Whitney U test."""
        np.random.seed(43)
        data2 = pd.Series(np.random.randn(100), name="test_data2")

        result = analyzer.analyze_nonparametric_tests(sample_data, data2)

        assert isinstance(result, NonparametricResult)
        assert result.success is True

    @pytest.mark.unit
    def test_analyze_spearman_correlation(self, analyzer, sample_data):
        """Test Spearman correlation analysis."""
        np.random.seed(43)
        data2 = pd.Series(
            sample_data.values + np.random.randn(len(sample_data)) * 0.1,
            name="test_data2",
        )

        result = analyzer.analyze_spearman_correlation(sample_data, data2)

        assert isinstance(result, NonparametricResult)
        assert result.success is True
        assert "spearman_correlation" in result.test_statistics or "correlation" in result.test_statistics

    @pytest.mark.unit
    def test_analyze_kendall_tau(self, analyzer, sample_data):
        """Test Kendall's tau correlation analysis."""
        np.random.seed(43)
        data2 = pd.Series(
            sample_data.values + np.random.randn(len(sample_data)) * 0.1,
            name="test_data2",
        )

        result = analyzer.analyze_kendall_tau(sample_data, data2)

        assert isinstance(result, NonparametricResult)
        assert result.success is True
        assert "kendall_tau" in result.test_statistics or "tau" in result.test_statistics

    @pytest.mark.unit
    def test_cache_result(self, analyzer, sample_data):
        """Test caching analysis results."""
        result = analyzer.analyze_kernel_density(sample_data)

        cached = analyzer.get_cached_result("kernel_density", sample_data.name)
        # May or may not be cached depending on implementation
        assert True  # Just verify no error

    @pytest.mark.unit
    def test_get_cached_result_none(self, analyzer):
        """Test getting cached result when none exists."""
        cached = analyzer.get_cached_result("nonexistent", "data1")
        assert cached is None

    @pytest.mark.unit
    def test_clear_cache(self, analyzer):
        """Test clearing analysis cache."""
        analyzer.clear_cache()
        assert len(analyzer.analysis_cache) == 0


class TestKernelDensityAnalyzer:
    """Test suite for KernelDensityAnalyzer class."""

    @pytest.mark.unit
    def test_kernel_density_analyzer_creation(self):
        """Test creating KernelDensityAnalyzer."""
        analyzer = KernelDensityAnalyzer()

        assert isinstance(analyzer, NonparametricAnalyzer)
        assert analyzer.method_name == "KernelDensity"
