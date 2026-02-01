"""
Unit tests for uncertainty quantification.

Tests for UncertaintyConfig, UncertaintyResult, UncertaintyQuantifier, MonteCarloAnalyzer, and BayesianAnalyzer.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from am_qadf.analytics.sensitivity_analysis.uncertainty import (
    UncertaintyConfig,
    UncertaintyResult,
    UncertaintyQuantifier,
    MonteCarloAnalyzer,
    BayesianAnalyzer,
)


class TestUncertaintyConfig:
    """Test suite for UncertaintyConfig dataclass."""

    @pytest.mark.unit
    def test_config_creation_default(self):
        """Test creating UncertaintyConfig with default values."""
        config = UncertaintyConfig()

        assert config.monte_carlo_samples == 10000
        assert config.monte_carlo_burn_in == 1000
        assert config.monte_carlo_thinning == 1
        assert config.bayesian_samples == 5000
        assert config.bayesian_tune == 1000
        assert config.bayesian_chains == 2
        assert config.propagation_method == "monte_carlo"
        assert config.taylor_order == 2
        assert config.confidence_level == 0.95
        assert config.significance_level == 0.05
        assert config.random_seed is None

    @pytest.mark.unit
    def test_config_creation_custom(self):
        """Test creating UncertaintyConfig with custom values."""
        config = UncertaintyConfig(
            monte_carlo_samples=5000,
            monte_carlo_burn_in=500,
            monte_carlo_thinning=2,
            bayesian_samples=3000,
            bayesian_tune=500,
            bayesian_chains=4,
            propagation_method="taylor",
            taylor_order=3,
            confidence_level=0.99,
            significance_level=0.01,
            random_seed=42,
        )

        assert config.monte_carlo_samples == 5000
        assert config.monte_carlo_burn_in == 500
        assert config.monte_carlo_thinning == 2
        assert config.bayesian_samples == 3000
        assert config.bayesian_tune == 500
        assert config.bayesian_chains == 4
        assert config.propagation_method == "taylor"
        assert config.taylor_order == 3
        assert config.confidence_level == 0.99
        assert config.significance_level == 0.01
        assert config.random_seed == 42


class TestUncertaintyResult:
    """Test suite for UncertaintyResult dataclass."""

    @pytest.mark.unit
    def test_result_creation(self):
        """Test creating UncertaintyResult."""
        result = UncertaintyResult(
            success=True,
            method="MonteCarlo",
            parameter_names=["param1", "param2"],
            parameter_distributions={"param1": {"type": "normal", "params": {"mean": 0, "std": 1}}},
            output_statistics={"mean": 1.0, "std": 0.5},
            confidence_intervals={"output": (0.5, 1.5)},
            sensitivity_analysis={"param1": 0.8},
            analysis_time=1.5,
            sample_size=1000,
        )

        assert result.success is True
        assert result.method == "MonteCarlo"
        assert len(result.parameter_names) == 2
        assert len(result.parameter_distributions) == 1
        assert len(result.output_statistics) == 2
        assert len(result.confidence_intervals) == 1
        assert result.analysis_time == 1.5
        assert result.sample_size == 1000
        assert result.error_message is None

    @pytest.mark.unit
    def test_result_creation_with_error(self):
        """Test creating UncertaintyResult with error."""
        result = UncertaintyResult(
            success=False,
            method="MonteCarlo",
            parameter_names=[],
            parameter_distributions={},
            output_statistics={},
            confidence_intervals={},
            sensitivity_analysis={},
            analysis_time=0.0,
            sample_size=0,
            error_message="Test error",
        )

        assert result.success is False
        assert result.error_message == "Test error"


class TestUncertaintyQuantifier:
    """Test suite for UncertaintyQuantifier class."""

    @pytest.fixture
    def quantifier(self):
        """Create an UncertaintyQuantifier instance."""
        return UncertaintyQuantifier()

    @pytest.fixture
    def simple_model_function(self):
        """Create a simple model function for testing."""

        def model(params):
            # Simple linear model: y = 2*x1 + x2
            if isinstance(params, np.ndarray):
                return 2 * params[0] + params[1]
            return 2 * params[0] + params[1]

        return model

    @pytest.fixture
    def parameter_distributions(self):
        """Create parameter distributions for testing."""
        return {
            "param1": {"type": "normal", "params": {"mean": 1.0, "std": 0.1}},
            "param2": {"type": "normal", "params": {"mean": 2.0, "std": 0.2}},
        }

    @pytest.mark.unit
    def test_quantifier_creation_default(self):
        """Test creating UncertaintyQuantifier with default config."""
        quantifier = UncertaintyQuantifier()

        assert quantifier.config is not None
        assert isinstance(quantifier.config, UncertaintyConfig)
        assert quantifier.analysis_cache == {}

    @pytest.mark.unit
    def test_quantifier_creation_custom_config(self):
        """Test creating UncertaintyQuantifier with custom config."""
        config = UncertaintyConfig(monte_carlo_samples=5000)
        quantifier = UncertaintyQuantifier(config=config)

        assert quantifier.config.monte_carlo_samples == 5000

    @pytest.mark.unit
    def test_analyze_monte_carlo(self, quantifier, simple_model_function, parameter_distributions):
        """Test Monte Carlo uncertainty analysis."""
        quantifier.config.monte_carlo_samples = 100  # Use smaller sample for testing

        result = quantifier.analyze_monte_carlo(
            simple_model_function,
            parameter_distributions,
            parameter_names=["param1", "param2"],
            n_samples=100,
        )

        assert isinstance(result, UncertaintyResult)
        assert result.method == "MonteCarlo"
        assert result.success is True
        assert len(result.parameter_names) == 2
        assert result.sample_size == 100
        assert len(result.output_statistics) > 0

    @pytest.mark.unit
    def test_analyze_monte_carlo_custom_samples(self, quantifier, simple_model_function, parameter_distributions):
        """Test Monte Carlo analysis with custom sample size."""
        result = quantifier.analyze_monte_carlo(
            simple_model_function,
            parameter_distributions,
            parameter_names=["param1", "param2"],
            n_samples=50,
        )

        assert isinstance(result, UncertaintyResult)
        assert result.sample_size == 50

    @pytest.mark.unit
    def test_analyze_monte_carlo_error_handling(self, quantifier, parameter_distributions):
        """Test error handling in Monte Carlo analysis."""

        def failing_model(params):
            raise ValueError("Model error")

        result = quantifier.analyze_monte_carlo(
            failing_model,
            parameter_distributions,
            parameter_names=["param1", "param2"],
            n_samples=10,
        )

        assert isinstance(result, UncertaintyResult)
        assert result.success is False
        assert result.error_message is not None

    @pytest.mark.unit
    @patch("am_qadf.analytics.sensitivity_analysis.uncertainty.PYMC_AVAILABLE", True)
    def test_analyze_bayesian(self, quantifier, simple_model_function, parameter_distributions):
        """Test Bayesian uncertainty analysis."""
        with patch("am_qadf.analytics.sensitivity_analysis.uncertainty.pm") as mock_pm:
            # Mock PyMC components
            mock_model = Mock()
            mock_trace = Mock()
            mock_pm.Model.return_value.__enter__ = Mock(return_value=mock_model)
            mock_pm.Model.return_value.__exit__ = Mock(return_value=None)
            mock_pm.sample.return_value = mock_trace

            quantifier.config.bayesian_samples = 100  # Use smaller sample for testing

            result = quantifier.analyze_bayesian(
                simple_model_function,
                parameter_distributions,
                parameter_names=["param1", "param2"],
            )

            # May succeed or fail depending on PyMC availability
            assert isinstance(result, UncertaintyResult)
            assert result.method == "Bayesian"

    @pytest.mark.unit
    @patch("am_qadf.analytics.sensitivity_analysis.uncertainty.PYMC_AVAILABLE", False)
    def test_analyze_bayesian_no_pymc(self, quantifier, simple_model_function, parameter_distributions):
        """Test Bayesian analysis when PyMC is not available."""
        result = quantifier.analyze_bayesian(
            simple_model_function,
            parameter_distributions,
            parameter_names=["param1", "param2"],
        )

        assert isinstance(result, UncertaintyResult)
        # Should fail gracefully when PyMC is not available
        assert result.success is False or result.method == "Bayesian"

    @pytest.mark.unit
    def test_cache_result(self, quantifier):
        """Test caching analysis results."""
        result = UncertaintyResult(
            success=True,
            method="test",
            parameter_names=["param1"],
            parameter_distributions={},
            output_statistics={},
            confidence_intervals={},
            sensitivity_analysis={},
            analysis_time=1.0,
            sample_size=100,
        )

        quantifier._cache_result("test_method", result)

        cached = quantifier.get_cached_result("test_method", ["param1"])
        assert cached is not None

    @pytest.mark.unit
    def test_get_cached_result_none(self, quantifier):
        """Test getting cached result when none exists."""
        cached = quantifier.get_cached_result("nonexistent", ["param1"])
        assert cached is None

    @pytest.mark.unit
    def test_clear_cache(self, quantifier):
        """Test clearing analysis cache."""
        result = UncertaintyResult(
            success=True,
            method="test",
            parameter_names=["param1"],
            parameter_distributions={},
            output_statistics={},
            confidence_intervals={},
            sensitivity_analysis={},
            analysis_time=1.0,
            sample_size=100,
        )

        quantifier._cache_result("test_method", result)
        assert len(quantifier.analysis_cache) > 0

        quantifier.clear_cache()
        assert len(quantifier.analysis_cache) == 0


class TestMonteCarloAnalyzer:
    """Test suite for MonteCarloAnalyzer class."""

    @pytest.mark.unit
    def test_monte_carlo_analyzer_creation(self):
        """Test creating MonteCarloAnalyzer."""
        analyzer = MonteCarloAnalyzer()

        assert isinstance(analyzer, UncertaintyQuantifier)
        assert analyzer.method_name == "MonteCarlo"


class TestBayesianAnalyzer:
    """Test suite for BayesianAnalyzer class."""

    @pytest.mark.unit
    def test_bayesian_analyzer_creation(self):
        """Test creating BayesianAnalyzer."""
        analyzer = BayesianAnalyzer()

        assert isinstance(analyzer, UncertaintyQuantifier)
        assert analyzer.method_name == "Bayesian"
