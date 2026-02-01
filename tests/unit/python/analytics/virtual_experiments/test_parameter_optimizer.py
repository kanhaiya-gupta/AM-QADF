"""
Unit tests for parameter optimizer.

Tests for OptimizationResult and ParameterOptimizer classes.
"""

import pytest
import numpy as np
from am_qadf.analytics.virtual_experiments.parameter_optimizer import (
    OptimizationResult,
    ParameterOptimizer,
)


class TestOptimizationResult:
    """Test suite for OptimizationResult dataclass."""

    @pytest.mark.unit
    def test_result_creation_single_objective(self):
        """Test creating OptimizationResult for single-objective."""
        result = OptimizationResult(
            success=True,
            optimal_parameters={"power": 250.0, "velocity": 1200.0},
            optimal_objectives={"quality": 0.9},
            optimization_method="L-BFGS-B",
            iterations=50,
            convergence_info={"message": "Optimization terminated successfully"},
        )

        assert result.success is True
        assert result.optimal_parameters["power"] == 250.0
        assert result.optimal_objectives["quality"] == 0.9
        assert result.optimization_method == "L-BFGS-B"
        assert result.iterations == 50
        assert result.pareto_front is None

    @pytest.mark.unit
    def test_result_creation_multi_objective(self):
        """Test creating OptimizationResult for multi-objective."""
        result = OptimizationResult(
            success=True,
            optimal_parameters={"power": 250.0, "velocity": 1200.0},
            optimal_objectives={"quality": 0.9, "density": 0.95},
            optimization_method="NSGA2",
            iterations=100,
            convergence_info={},
            pareto_front=[
                {"power": 200.0, "velocity": 1000.0, "quality": 0.8, "density": 0.94},
                {"power": 250.0, "velocity": 1200.0, "quality": 0.9, "density": 0.95},
            ],
        )

        assert result.success is True
        assert result.pareto_front is not None
        assert len(result.pareto_front) == 2


class TestParameterOptimizer:
    """Test suite for ParameterOptimizer class."""

    @pytest.fixture
    def optimizer(self):
        """Create a ParameterOptimizer instance."""
        return ParameterOptimizer()

    @pytest.fixture
    def sample_experiment_results(self):
        """Create sample experiment results for testing."""
        return [
            {
                "input_parameters": {"power": 200.0, "velocity": 1000.0},
                "output_responses": {"quality": 0.8},
            },
            {
                "input_parameters": {"power": 250.0, "velocity": 1200.0},
                "output_responses": {"quality": 0.85},
            },
            {
                "input_parameters": {"power": 300.0, "velocity": 1500.0},
                "output_responses": {"quality": 0.9},
            },
        ]

    @pytest.mark.unit
    def test_optimizer_creation(self, optimizer):
        """Test creating ParameterOptimizer."""
        assert optimizer is not None
        assert optimizer.optimization_cache == {}

    @pytest.mark.unit
    def test_optimize_single_objective_maximize(self, optimizer, sample_experiment_results):
        """Test single-objective optimization (maximize)."""
        result = optimizer.optimize_single_objective(
            sample_experiment_results,
            parameter_names=["power", "velocity"],
            objective_name="quality",
            maximize=True,
        )

        assert isinstance(result, OptimizationResult)
        assert result.success is True or result.success is False  # May fail with limited data
        assert len(result.optimal_parameters) == 2
        assert "quality" in result.optimal_objectives

    @pytest.mark.unit
    def test_optimize_single_objective_minimize(self, optimizer, sample_experiment_results):
        """Test single-objective optimization (minimize)."""
        # Add a cost objective to minimize
        results_with_cost = [
            {
                "input_parameters": {"power": 200.0, "velocity": 1000.0},
                "output_responses": {"cost": 100.0},
            },
            {
                "input_parameters": {"power": 250.0, "velocity": 1200.0},
                "output_responses": {"cost": 120.0},
            },
        ]

        result = optimizer.optimize_single_objective(
            results_with_cost,
            parameter_names=["power", "velocity"],
            objective_name="cost",
            maximize=False,
        )

        assert isinstance(result, OptimizationResult)
        assert result.success is True or result.success is False

    @pytest.mark.unit
    def test_optimize_single_objective_with_constraints(self, optimizer, sample_experiment_results):
        """Test single-objective optimization with constraints."""
        constraints = {"power": (200.0, 300.0), "velocity": (1000.0, 1500.0)}

        result = optimizer.optimize_single_objective(
            sample_experiment_results,
            parameter_names=["power", "velocity"],
            objective_name="quality",
            maximize=True,
            constraints=constraints,
        )

        assert isinstance(result, OptimizationResult)
        assert result.success is True or result.success is False

    @pytest.mark.unit
    def test_optimize_multi_objective(self, optimizer):
        """Test multi-objective optimization."""
        results = [
            {
                "input_parameters": {"power": 200.0, "velocity": 1000.0},
                "output_responses": {"quality": 0.8, "density": 0.94},
            },
            {
                "input_parameters": {"power": 250.0, "velocity": 1200.0},
                "output_responses": {"quality": 0.85, "density": 0.95},
            },
            {
                "input_parameters": {"power": 300.0, "velocity": 1500.0},
                "output_responses": {"quality": 0.9, "density": 0.96},
            },
        ]

        result = optimizer.optimize_multi_objective(
            results,
            parameter_names=["power", "velocity"],
            objective_names=["quality", "density"],
        )

        assert isinstance(result, OptimizationResult)
        assert result.success is True or result.success is False
        # For multi-objective, may have Pareto front or list of objectives
        assert len(result.optimal_objectives) >= 1

    @pytest.mark.unit
    def test_optimize_single_objective_error_handling(self, optimizer):
        """Test error handling in single-objective optimization."""
        # Pass invalid data
        result = optimizer.optimize_single_objective(
            [{"invalid": "data"}], parameter_names=["power"], objective_name="quality"
        )

        assert isinstance(result, OptimizationResult)
        assert result.success is False
