"""
Unit tests for parameter analysis.

Tests for ParameterAnalysisConfig, ParameterAnalysisResult, ParameterAnalyzer, and ProcessParameterOptimizer.
"""

import pytest
import numpy as np
import pandas as pd
from am_qadf.analytics.process_analysis.parameter_analysis import (
    ParameterAnalysisConfig,
    ParameterAnalysisResult,
    ParameterAnalyzer,
    ProcessParameterOptimizer,
)


class TestParameterAnalysisConfig:
    """Test suite for ParameterAnalysisConfig dataclass."""

    @pytest.mark.unit
    def test_config_creation_default(self):
        """Test creating ParameterAnalysisConfig with default values."""
        config = ParameterAnalysisConfig()

        assert config.optimization_method == "differential_evolution"
        assert config.max_iterations == 1000
        assert config.tolerance == 1e-6
        assert config.interaction_threshold == 0.3
        assert config.correlation_method == "pearson"
        assert config.confidence_level == 0.95
        assert config.significance_level == 0.05
        assert config.random_seed is None

    @pytest.mark.unit
    def test_config_creation_custom(self):
        """Test creating ParameterAnalysisConfig with custom values."""
        config = ParameterAnalysisConfig(
            optimization_method="minimize",
            max_iterations=500,
            tolerance=1e-5,
            interaction_threshold=0.5,
            correlation_method="spearman",
            random_seed=42,
        )

        assert config.optimization_method == "minimize"
        assert config.max_iterations == 500
        assert config.tolerance == 1e-5
        assert config.interaction_threshold == 0.5
        assert config.correlation_method == "spearman"
        assert config.random_seed == 42


class TestParameterAnalysisResult:
    """Test suite for ParameterAnalysisResult dataclass."""

    @pytest.mark.unit
    def test_result_creation(self):
        """Test creating ParameterAnalysisResult."""
        result = ParameterAnalysisResult(
            success=True,
            method="test_method",
            parameter_names=["param1", "param2"],
            optimal_parameters={"param1": 1.0, "param2": 2.0},
            optimal_value=0.5,
            parameter_interactions={"param1": {"param2": 0.3}},
            parameter_importance={"param1": 0.6, "param2": 0.4},
            analysis_time=1.5,
        )

        assert result.success is True
        assert result.method == "test_method"
        assert len(result.parameter_names) == 2
        assert result.optimal_parameters["param1"] == 1.0
        assert result.optimal_value == 0.5
        assert result.analysis_time == 1.5


class TestParameterAnalyzer:
    """Test suite for ParameterAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create a ParameterAnalyzer instance."""
        return ParameterAnalyzer()

    @pytest.fixture
    def simple_objective(self):
        """Simple objective function for testing."""

        def objective(params):
            # Simple quadratic function: (x-1)^2 + (y-2)^2
            x = params.get("x", 0)
            y = params.get("y", 0)
            return (x - 1) ** 2 + (y - 2) ** 2

        return objective

    @pytest.mark.unit
    def test_analyzer_creation_default(self):
        """Test creating ParameterAnalyzer with default config."""
        analyzer = ParameterAnalyzer()

        assert analyzer.config is not None
        assert analyzer.analysis_cache == {}

    @pytest.mark.unit
    def test_analyzer_creation_custom(self):
        """Test creating ParameterAnalyzer with custom config."""
        config = ParameterAnalysisConfig(max_iterations=500, random_seed=42)
        analyzer = ParameterAnalyzer(config)

        assert analyzer.config.max_iterations == 500
        assert analyzer.config.random_seed == 42

    @pytest.mark.unit
    def test_analyze_parameter_optimization_differential_evolution(self, analyzer, simple_objective):
        """Test parameter optimization with differential evolution."""
        parameter_bounds = {"x": (-5.0, 5.0), "y": (-5.0, 5.0)}

        result = analyzer.analyze_parameter_optimization(
            simple_objective,
            parameter_bounds,
            optimization_method="differential_evolution",
        )

        assert isinstance(result, ParameterAnalysisResult)
        assert result.success is True
        assert "x" in result.optimal_parameters
        assert "y" in result.optimal_parameters
        # Optimal value should be close to 0 (minimum of (x-1)^2 + (y-2)^2)
        assert result.optimal_value < 1.0

    @pytest.mark.unit
    def test_analyze_parameter_optimization_minimize(self, analyzer, simple_objective):
        """Test parameter optimization with minimize method."""
        parameter_bounds = {"x": (-5.0, 5.0), "y": (-5.0, 5.0)}

        result = analyzer.analyze_parameter_optimization(simple_objective, parameter_bounds, optimization_method="minimize")

        assert isinstance(result, ParameterAnalysisResult)
        assert result.success is True
        assert "x" in result.optimal_parameters
        assert "y" in result.optimal_parameters

    @pytest.mark.unit
    def test_analyze_parameter_interactions(self, analyzer):
        """Test parameter interaction analysis."""
        # Create sample data
        data = pd.DataFrame(
            {
                "param1": np.random.randn(100),
                "param2": np.random.randn(100),
                "output": np.random.randn(100),
            }
        )

        interactions = analyzer.analyze_parameter_interactions(
            data, parameter_names=["param1", "param2"], output_name="output"
        )

        assert isinstance(interactions, dict)
        assert "param1" in interactions or "param2" in interactions

    @pytest.mark.unit
    def test_analyze_parameter_importance(self, analyzer):
        """Test parameter importance analysis."""
        # Create sample data
        data = pd.DataFrame(
            {
                "param1": np.random.randn(100),
                "param2": np.random.randn(100),
                "output": np.random.randn(100),
            }
        )

        importance = analyzer.analyze_parameter_importance(data, parameter_names=["param1", "param2"], output_name="output")

        assert isinstance(importance, dict)
        assert len(importance) > 0


class TestProcessParameterOptimizer:
    """Test suite for ProcessParameterOptimizer class."""

    @pytest.fixture
    def optimizer(self):
        """Create a ProcessParameterOptimizer instance."""
        return ProcessParameterOptimizer()

    @pytest.fixture
    def simple_objective(self):
        """Simple objective function for testing."""

        def objective(params):
            x = params.get("x", 0)
            y = params.get("y", 0)
            return (x - 1) ** 2 + (y - 2) ** 2

        return objective

    @pytest.mark.unit
    def test_optimizer_creation(self, optimizer):
        """Test creating ProcessParameterOptimizer."""
        assert optimizer is not None
        assert optimizer.config is not None

    @pytest.mark.unit
    def test_optimize_process_parameters(self, optimizer, simple_objective):
        """Test optimizing process parameters."""
        parameter_bounds = {"x": (-5.0, 5.0), "y": (-5.0, 5.0)}

        result = optimizer.optimize_process_parameters(simple_objective, parameter_bounds)

        assert isinstance(result, ParameterAnalysisResult)
        assert result.success is True
        assert "x" in result.optimal_parameters
        assert "y" in result.optimal_parameters
