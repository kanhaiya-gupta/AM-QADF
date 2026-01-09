"""
Unit tests for local sensitivity analysis.

Tests for LocalSensitivityConfig, LocalSensitivityResult, LocalSensitivityAnalyzer, and DerivativeAnalyzer.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from am_qadf.analytics.sensitivity_analysis.local_analysis import (
    LocalSensitivityConfig,
    LocalSensitivityResult,
    LocalSensitivityAnalyzer,
    DerivativeAnalyzer,
)


class TestLocalSensitivityConfig:
    """Test suite for LocalSensitivityConfig dataclass."""

    @pytest.mark.unit
    def test_config_creation_default(self):
        """Test creating LocalSensitivityConfig with default values."""
        config = LocalSensitivityConfig()

        assert config.perturbation_size == 1e-6
        assert config.finite_difference_method == "central"
        assert config.step_size == 1e-6
        assert config.num_diff_order == 1
        assert config.num_diff_accuracy == 2
        assert config.confidence_level == 0.95
        assert config.significance_level == 0.05

    @pytest.mark.unit
    def test_config_creation_custom(self):
        """Test creating LocalSensitivityConfig with custom values."""
        config = LocalSensitivityConfig(
            perturbation_size=1e-5,
            finite_difference_method="forward",
            step_size=1e-5,
            num_diff_order=2,
            num_diff_accuracy=4,
            confidence_level=0.99,
            significance_level=0.01,
        )

        assert config.perturbation_size == 1e-5
        assert config.finite_difference_method == "forward"
        assert config.step_size == 1e-5
        assert config.num_diff_order == 2
        assert config.num_diff_accuracy == 4
        assert config.confidence_level == 0.99
        assert config.significance_level == 0.01


class TestLocalSensitivityResult:
    """Test suite for LocalSensitivityResult dataclass."""

    @pytest.mark.unit
    def test_result_creation(self):
        """Test creating LocalSensitivityResult."""
        nominal_point = {"param1": 1.0, "param2": 2.0}
        result = LocalSensitivityResult(
            success=True,
            method="Derivatives",
            parameter_names=["param1", "param2"],
            nominal_point=nominal_point,
            sensitivity_gradients={"param1": 0.5, "param2": 0.3},
            sensitivity_elasticities={"param1": 0.5, "param2": 0.6},
            analysis_time=1.5,
        )

        assert result.success is True
        assert result.method == "Derivatives"
        assert len(result.parameter_names) == 2
        assert result.nominal_point == nominal_point
        assert len(result.sensitivity_gradients) == 2
        assert len(result.sensitivity_elasticities) == 2
        assert result.analysis_time == 1.5
        assert result.error_message is None

    @pytest.mark.unit
    def test_result_creation_with_error(self):
        """Test creating LocalSensitivityResult with error."""
        result = LocalSensitivityResult(
            success=False,
            method="Derivatives",
            parameter_names=[],
            nominal_point={},
            sensitivity_gradients={},
            sensitivity_elasticities={},
            analysis_time=0.0,
            error_message="Test error",
        )

        assert result.success is False
        assert result.error_message == "Test error"


class TestLocalSensitivityAnalyzer:
    """Test suite for LocalSensitivityAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create a LocalSensitivityAnalyzer instance."""
        return LocalSensitivityAnalyzer()

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
    def nominal_point(self):
        """Create nominal point for testing."""
        return {"param1": 1.0, "param2": 2.0}

    @pytest.mark.unit
    def test_analyzer_creation_default(self):
        """Test creating LocalSensitivityAnalyzer with default config."""
        analyzer = LocalSensitivityAnalyzer()

        assert analyzer.config is not None
        assert isinstance(analyzer.config, LocalSensitivityConfig)
        assert analyzer.analysis_cache == {}

    @pytest.mark.unit
    def test_analyzer_creation_custom_config(self):
        """Test creating LocalSensitivityAnalyzer with custom config."""
        config = LocalSensitivityConfig(perturbation_size=1e-5)
        analyzer = LocalSensitivityAnalyzer(config=config)

        assert analyzer.config.perturbation_size == 1e-5

    @pytest.mark.unit
    def test_analyze_derivatives(self, analyzer, simple_model_function, nominal_point):
        """Test derivative-based local sensitivity analysis."""
        result = analyzer.analyze_derivatives(simple_model_function, nominal_point, parameter_names=["param1", "param2"])

        assert isinstance(result, LocalSensitivityResult)
        assert result.method == "Derivatives"
        assert result.success is True
        assert len(result.parameter_names) == 2
        assert result.nominal_point == nominal_point
        assert len(result.sensitivity_gradients) == 2
        assert len(result.sensitivity_elasticities) == 2

    @pytest.mark.unit
    def test_analyze_perturbation(self, analyzer, simple_model_function, nominal_point):
        """Test perturbation-based local sensitivity analysis."""
        result = analyzer.analyze_perturbation(simple_model_function, nominal_point, parameter_names=["param1", "param2"])

        assert isinstance(result, LocalSensitivityResult)
        assert result.method == "Perturbation"
        assert result.success is True
        assert len(result.sensitivity_gradients) == 2

    @pytest.mark.unit
    def test_analyze_perturbation_custom_size(self, analyzer, simple_model_function, nominal_point):
        """Test perturbation analysis with custom perturbation size."""
        result = analyzer.analyze_perturbation(
            simple_model_function,
            nominal_point,
            parameter_names=["param1", "param2"],
            perturbation_size=1e-4,
        )

        assert isinstance(result, LocalSensitivityResult)
        assert result.success is True

    @pytest.mark.unit
    def test_analyze_central_differences(self, analyzer, simple_model_function, nominal_point):
        """Test central difference local sensitivity analysis."""
        result = analyzer.analyze_central_differences(
            simple_model_function, nominal_point, parameter_names=["param1", "param2"]
        )

        assert isinstance(result, LocalSensitivityResult)
        assert result.method == "CentralDifferences"
        assert result.success is True
        assert len(result.sensitivity_gradients) == 2

    @pytest.mark.unit
    def test_analyze_central_differences_custom_step(self, analyzer, simple_model_function, nominal_point):
        """Test central difference analysis with custom step size."""
        result = analyzer.analyze_central_differences(
            simple_model_function,
            nominal_point,
            parameter_names=["param1", "param2"],
            step_size=1e-5,
        )

        assert isinstance(result, LocalSensitivityResult)
        assert result.success is True

    @pytest.mark.unit
    def test_analyze_automatic_differentiation(self, analyzer, simple_model_function, nominal_point):
        """Test automatic differentiation local sensitivity analysis."""
        result = analyzer.analyze_automatic_differentiation(
            simple_model_function, nominal_point, parameter_names=["param1", "param2"]
        )

        assert isinstance(result, LocalSensitivityResult)
        assert result.method == "AutomaticDifferentiation"
        assert result.success is True
        assert len(result.sensitivity_gradients) == 2

    @pytest.mark.unit
    def test_analyze_derivatives_error_handling(self, analyzer, nominal_point):
        """Test error handling in derivative analysis."""

        def failing_model(params):
            raise ValueError("Model error")

        result = analyzer.analyze_derivatives(failing_model, nominal_point, parameter_names=["param1", "param2"])

        assert isinstance(result, LocalSensitivityResult)
        assert result.success is False
        assert result.error_message is not None

    @pytest.mark.unit
    def test_analyze_perturbation_error_handling(self, analyzer, nominal_point):
        """Test error handling in perturbation analysis."""

        def failing_model(params):
            raise ValueError("Model error")

        result = analyzer.analyze_perturbation(failing_model, nominal_point, parameter_names=["param1", "param2"])

        assert isinstance(result, LocalSensitivityResult)
        assert result.success is False
        assert result.error_message is not None

    @pytest.mark.unit
    def test_cache_result(self, analyzer):
        """Test caching analysis results."""
        result = LocalSensitivityResult(
            success=True,
            method="test",
            parameter_names=["param1"],
            nominal_point={"param1": 1.0},
            sensitivity_gradients={"param1": 0.5},
            sensitivity_elasticities={"param1": 0.5},
            analysis_time=1.0,
        )

        analyzer._cache_result("test_method", result)

        cached = analyzer.get_cached_result("test_method", ["param1"])
        assert cached is not None

    @pytest.mark.unit
    def test_get_cached_result_none(self, analyzer):
        """Test getting cached result when none exists."""
        cached = analyzer.get_cached_result("nonexistent", ["param1"])
        assert cached is None

    @pytest.mark.unit
    def test_clear_cache(self, analyzer):
        """Test clearing analysis cache."""
        result = LocalSensitivityResult(
            success=True,
            method="test",
            parameter_names=["param1"],
            nominal_point={"param1": 1.0},
            sensitivity_gradients={"param1": 0.5},
            sensitivity_elasticities={"param1": 0.5},
            analysis_time=1.0,
        )

        analyzer._cache_result("test_method", result)
        assert len(analyzer.analysis_cache) > 0

        analyzer.clear_cache()
        assert len(analyzer.analysis_cache) == 0

    @pytest.mark.unit
    def test_get_analysis_statistics(self, analyzer):
        """Test getting analysis statistics."""
        stats = analyzer.get_analysis_statistics()

        assert isinstance(stats, dict)
        assert "cache_size" in stats
        assert "config" in stats
        assert "perturbation_size" in stats["config"]


class TestDerivativeAnalyzer:
    """Test suite for DerivativeAnalyzer class."""

    @pytest.mark.unit
    def test_derivative_analyzer_creation(self):
        """Test creating DerivativeAnalyzer."""
        analyzer = DerivativeAnalyzer()

        assert isinstance(analyzer, LocalSensitivityAnalyzer)
        assert analyzer.method_name == "Derivatives"

    @pytest.mark.unit
    def test_derivative_analyzer_analyze(self):
        """Test DerivativeAnalyzer analyze method."""
        analyzer = DerivativeAnalyzer()

        def model(params):
            return 2 * params[0] + params[1]

        nominal_point = {"param1": 1.0, "param2": 2.0}

        result = analyzer.analyze(model, nominal_point, parameter_names=["param1", "param2"])

        assert isinstance(result, LocalSensitivityResult)
        assert result.method == "Derivatives"
