"""
Unit tests for process optimization.

Tests for OptimizationConfig, OptimizationResult, ProcessOptimizer, and MultiObjectiveOptimizer.
"""

import pytest
import numpy as np
import pandas as pd
from am_qadf.analytics.process_analysis.optimization import (
    OptimizationConfig,
    OptimizationResult,
    ProcessOptimizer,
    MultiObjectiveOptimizer,
)


class TestOptimizationConfig:
    """Test suite for OptimizationConfig dataclass."""

    @pytest.mark.unit
    def test_config_creation_default(self):
        """Test creating OptimizationConfig with default values."""
        config = OptimizationConfig()

        assert config.optimization_method == "differential_evolution"
        assert config.max_iterations == 1000
        assert config.population_size == 50
        assert config.tolerance == 1e-6
        assert config.n_objectives == 2
        assert config.pareto_front_size == 100
        assert config.random_seed is None

    @pytest.mark.unit
    def test_config_creation_custom(self):
        """Test creating OptimizationConfig with custom values."""
        config = OptimizationConfig(
            optimization_method="minimize",
            max_iterations=500,
            population_size=100,
            tolerance=1e-5,
            n_objectives=3,
            pareto_front_size=200,
            random_seed=42,
        )

        assert config.optimization_method == "minimize"
        assert config.max_iterations == 500
        assert config.population_size == 100
        assert config.tolerance == 1e-5
        assert config.n_objectives == 3
        assert config.pareto_front_size == 200
        assert config.random_seed == 42


class TestOptimizationResult:
    """Test suite for OptimizationResult dataclass."""

    @pytest.mark.unit
    def test_result_creation_single_objective(self):
        """Test creating OptimizationResult for single-objective optimization."""
        result = OptimizationResult(
            success=True,
            method="test_method",
            parameter_names=["param1", "param2"],
            optimal_parameters={"param1": 1.0, "param2": 2.0},
            optimal_values=0.5,
            analysis_time=1.5,
        )

        assert result.success is True
        assert result.method == "test_method"
        assert len(result.parameter_names) == 2
        assert result.optimal_parameters["param1"] == 1.0
        assert isinstance(result.optimal_values, float)
        assert result.optimal_values == 0.5
        assert result.analysis_time == 1.5

    @pytest.mark.unit
    def test_result_creation_multi_objective(self):
        """Test creating OptimizationResult for multi-objective optimization."""
        pareto_front = pd.DataFrame({"obj1": [1.0, 2.0, 3.0], "obj2": [3.0, 2.0, 1.0]})

        result = OptimizationResult(
            success=True,
            method="test_method",
            parameter_names=["param1", "param2"],
            optimal_parameters={"param1": 1.0, "param2": 2.0},
            optimal_values=[0.5, 0.6],
            pareto_front=pareto_front,
            analysis_time=1.5,
        )

        assert result.success is True
        assert isinstance(result.optimal_values, list)
        assert len(result.optimal_values) == 2
        assert result.pareto_front is not None
        assert len(result.pareto_front) == 3


class TestProcessOptimizer:
    """Test suite for ProcessOptimizer class."""

    @pytest.fixture
    def optimizer(self):
        """Create a ProcessOptimizer instance."""
        return ProcessOptimizer()

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
    def test_optimizer_creation_default(self):
        """Test creating ProcessOptimizer with default config."""
        optimizer = ProcessOptimizer()

        assert optimizer.config is not None
        assert optimizer.analysis_cache == {}

    @pytest.mark.unit
    def test_optimizer_creation_custom(self):
        """Test creating ProcessOptimizer with custom config."""
        config = OptimizationConfig(max_iterations=500, random_seed=42)
        optimizer = ProcessOptimizer(config)

        assert optimizer.config.max_iterations == 500
        assert optimizer.config.random_seed == 42

    @pytest.mark.unit
    def test_optimize_single_objective_differential_evolution(self, optimizer, simple_objective):
        """Test single-objective optimization with differential evolution."""
        parameter_bounds = {"x": (-5.0, 5.0), "y": (-5.0, 5.0)}

        result = optimizer.optimize_single_objective(
            simple_objective,
            parameter_bounds,
            optimization_method="differential_evolution",
        )

        assert isinstance(result, OptimizationResult)
        assert result.success is True
        assert "x" in result.optimal_parameters
        assert "y" in result.optimal_parameters
        assert isinstance(result.optimal_values, float)
        # Optimal value should be close to 0 (minimum of (x-1)^2 + (y-2)^2)
        assert result.optimal_values < 1.0

    @pytest.mark.unit
    def test_optimize_single_objective_minimize(self, optimizer, simple_objective):
        """Test single-objective optimization with minimize method."""
        parameter_bounds = {"x": (-5.0, 5.0), "y": (-5.0, 5.0)}

        result = optimizer.optimize_single_objective(simple_objective, parameter_bounds, optimization_method="minimize")

        assert isinstance(result, OptimizationResult)
        assert result.success is True
        assert "x" in result.optimal_parameters
        assert "y" in result.optimal_parameters

    @pytest.mark.unit
    def test_optimize_multi_objective(self, optimizer):
        """Test multi-objective optimization."""

        def multi_objective(params):
            x = params.get("x", 0)
            y = params.get("y", 0)
            # Two objectives: minimize x^2 and y^2
            return [x**2, y**2]

        parameter_bounds = {"x": (-5.0, 5.0), "y": (-5.0, 5.0)}

        result = optimizer.optimize_multi_objective(multi_objective, parameter_bounds, n_objectives=2)

        assert isinstance(result, OptimizationResult)
        assert result.success is True
        assert isinstance(result.optimal_values, list)
        assert len(result.optimal_values) == 2


class TestMultiObjectiveOptimizer:
    """Test suite for MultiObjectiveOptimizer class."""

    @pytest.fixture
    def optimizer(self):
        """Create a MultiObjectiveOptimizer instance."""
        return MultiObjectiveOptimizer()

    @pytest.mark.unit
    def test_optimizer_creation(self, optimizer):
        """Test creating MultiObjectiveOptimizer."""
        assert optimizer is not None
        assert optimizer.config is not None

    @pytest.mark.unit
    def test_optimize_pareto_front(self, optimizer):
        """Test optimizing Pareto front."""

        def multi_objective(params):
            x = params.get("x", 0)
            y = params.get("y", 0)
            return [x**2, y**2]

        parameter_bounds = {"x": (-5.0, 5.0), "y": (-5.0, 5.0)}

        result = optimizer.optimize_pareto_front(multi_objective, parameter_bounds, n_objectives=2)

        assert isinstance(result, OptimizationResult)
        assert result.success is True
        # For multi-objective, we expect a Pareto front
        assert result.pareto_front is not None or isinstance(result.optimal_values, list)
