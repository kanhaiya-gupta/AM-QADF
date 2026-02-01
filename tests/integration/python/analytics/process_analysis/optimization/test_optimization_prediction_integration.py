"""
Integration tests for optimization with prediction integration.

Tests integration of process optimization with prediction models and validation.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch

try:
    from am_qadf.analytics.process_analysis.optimization import ProcessOptimizer, OptimizationConfig, OptimizationResult
    from am_qadf.analytics.process_analysis.quality_analysis import QualityPredictor, QualityAnalysisConfig
    from am_qadf.analytics.process_analysis.prediction.prediction_validator import (
        PredictionValidator,
        OptimizationValidationResult,
    )
    from am_qadf.analytics.process_analysis.model_tracking.model_registry import ModelRegistry
    import tempfile
    import shutil
except ImportError as e:
    pytest.skip(f"Optimization or prediction modules not available: {e}", allow_module_level=True)


class TestOptimizationPredictionIntegration:
    """Integration tests for optimization with prediction integration."""

    @pytest.fixture
    def sample_process_data(self):
        """Create sample process data for optimization."""
        np.random.seed(42)
        n_samples = 100

        data = {
            "laser_power": np.random.uniform(200, 300, n_samples),
            "scan_speed": np.random.uniform(800, 1200, n_samples),
            "layer_thickness": np.random.uniform(0.02, 0.04, n_samples),
            "hatch_spacing": np.random.uniform(0.08, 0.12, n_samples),
        }

        # Create quality as function of parameters
        quality = (
            0.7
            + 0.001 * data["laser_power"]
            + 0.0002 * data["scan_speed"]
            - 10 * data["layer_thickness"]
            - 5 * data["hatch_spacing"]
            + np.random.randn(n_samples) * 0.05
        )
        quality = np.clip(quality, 0.0, 1.0)
        data["quality"] = quality

        return pd.DataFrame(data)

    @pytest.fixture
    def optimizer(self):
        """Create ProcessOptimizer instance."""
        config = OptimizationConfig(
            optimization_method="differential_evolution", max_iterations=50, population_size=10, random_seed=42
        )
        return ProcessOptimizer(config)

    @pytest.fixture
    def quality_predictor(self, sample_process_data):
        """Create and train QualityPredictor."""
        config = QualityAnalysisConfig(random_seed=42)
        predictor = QualityPredictor(config)

        predictor.analyze_quality_prediction(
            sample_process_data,
            quality_target="quality",
            feature_names=["laser_power", "scan_speed", "layer_thickness", "hatch_spacing"],
        )

        return predictor

    @pytest.fixture
    def temp_registry_dir(self):
        """Create temporary directory for registry."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.integration
    def test_optimization_with_quality_predictor(self, optimizer, quality_predictor):
        """Test optimization using quality predictor as objective function."""

        # Define objective function using trained predictor
        def objective_function(params):
            """Objective: maximize quality (minimize negative quality)."""
            param_df = pd.DataFrame([params])
            quality_pred = quality_predictor.predict_quality(param_df)
            return -quality_pred[0]  # Negative for minimization

        # Define parameter bounds
        parameter_bounds = {
            "laser_power": (200.0, 300.0),
            "scan_speed": (800.0, 1200.0),
            "layer_thickness": (0.02, 0.04),
            "hatch_spacing": (0.08, 0.12),
        }

        # Perform optimization
        result = optimizer.optimize_single_objective(objective_function, parameter_bounds)

        assert result.success is True
        assert len(result.optimal_parameters) == 4
        assert all(key in result.optimal_parameters for key in parameter_bounds.keys())
        assert result.optimal_values < 0  # Negative quality (to be maximized)

        # Verify optimal parameters are within bounds
        for param_name, (min_val, max_val) in parameter_bounds.items():
            assert min_val <= result.optimal_parameters[param_name] <= max_val

    @pytest.mark.integration
    def test_optimization_with_validation(self, optimizer, quality_predictor, sample_process_data):
        """Test optimization with prediction validation."""
        from am_qadf.analytics.process_analysis.prediction.early_defect_predictor import PredictionConfig
        from am_qadf.analytics.process_analysis.prediction.prediction_validator import PredictionValidator

        # Create validator
        pred_config = PredictionConfig(random_seed=42)
        validator = PredictionValidator(pred_config)

        # Define objective function
        def objective_function(params):
            param_df = pd.DataFrame([params])
            quality_pred = quality_predictor.predict_quality(param_df)
            return -quality_pred[0]

        parameter_bounds = {
            "laser_power": (200.0, 300.0),
            "scan_speed": (800.0, 1200.0),
            "layer_thickness": (0.02, 0.04),
            "hatch_spacing": (0.08, 0.12),
        }

        # Optimize
        opt_result = optimizer.optimize_single_objective(objective_function, parameter_bounds)

        assert opt_result.success is True

        # Validate optimized parameters
        optimized_df = pd.DataFrame([opt_result.optimal_parameters])
        experimental_df = pd.DataFrame({"quality": sample_process_data["quality"].iloc[:1].values})

        validation_result = validator.validate_with_experimental_data(
            quality_predictor, optimized_df, experimental_df, quality_target="quality"
        )

        assert validation_result.success is True
        assert validation_result.validation_method == "experimental"

    @pytest.mark.integration
    def test_multi_objective_optimization(self, optimizer):
        """Test multi-objective optimization workflow."""

        # Define multiple objectives (quality and efficiency)
        def objective_functions(params):
            # Objective 1: Maximize quality (minimize negative)
            quality = 0.7 + 0.001 * params["laser_power"] - 10 * params["layer_thickness"]
            quality = np.clip(quality, 0.0, 1.0)

            # Objective 2: Minimize energy consumption (proportional to power/speed)
            energy = params["laser_power"] / params["scan_speed"]

            return [-quality, energy]  # Negative quality for minimization

        parameter_bounds = {"laser_power": (200.0, 300.0), "scan_speed": (800.0, 1200.0), "layer_thickness": (0.02, 0.04)}

        result = optimizer.optimize_multi_objective(objective_functions, parameter_bounds, n_objectives=2)

        assert result.success is True
        assert result.pareto_front is not None
        assert len(result.pareto_front) > 0

    @pytest.mark.integration
    def test_constrained_optimization(self, optimizer, quality_predictor):
        """Test constrained optimization workflow."""

        # Define objective function
        def objective_function(params):
            param_df = pd.DataFrame([params])
            quality_pred = quality_predictor.predict_quality(param_df)
            return -quality_pred[0]

        # Define parameter bounds
        parameter_bounds = {
            "laser_power": (200.0, 300.0),
            "scan_speed": (800.0, 1200.0),
            "layer_thickness": (0.02, 0.04),
            "hatch_spacing": (0.08, 0.12),
        }

        # Define constraints (e.g., energy constraint)
        def energy_constraint(params):
            # Energy = power / speed, should be <= 0.3
            return (params["laser_power"] / params["scan_speed"]) - 0.3

        constraints = [energy_constraint]

        result = optimizer.optimize_with_constraints(
            objective_function, parameter_bounds, constraints, constraint_method="penalty"
        )

        assert result.success is True
        assert len(result.optimal_parameters) == 4

        # Verify constraint is satisfied (or at least attempted)
        energy = result.optimal_parameters["laser_power"] / result.optimal_parameters["scan_speed"]
        # Constraint should be satisfied (with some tolerance for penalty method)
        assert energy <= 0.35  # Allow small violation for penalty method

    @pytest.mark.integration
    def test_optimization_with_model_registry(self, optimizer, quality_predictor, temp_registry_dir):
        """Test optimization with model registry integration."""
        registry = ModelRegistry(storage_path=temp_registry_dir)

        # Register the quality predictor model
        if hasattr(quality_predictor, "trained_model") and quality_predictor.trained_model is not None:
            model_id = registry.register_model(
                model=quality_predictor.trained_model,
                model_type="QualityPredictor",
                version="1.0",
                metadata={"feature_names": ["laser_power", "scan_speed", "layer_thickness", "hatch_spacing"]},
                performance_metrics=quality_predictor.analyze_quality_prediction(
                    pd.DataFrame(
                        {
                            "laser_power": [250],
                            "scan_speed": [1000],
                            "layer_thickness": [0.03],
                            "hatch_spacing": [0.10],
                            "quality": [0.8],
                        }
                    ),
                    "quality",
                ).model_performance,
            )

            assert model_id is not None

            # Use model from registry for optimization
            loaded_model, model_version = registry.load_model(model_id)

            def objective_function(params):
                param_df = pd.DataFrame([params])
                predictions = loaded_model.predict(param_df[model_version.metadata.get("feature_names", [])])
                return -predictions[0]

            parameter_bounds = {
                "laser_power": (200.0, 300.0),
                "scan_speed": (800.0, 1200.0),
                "layer_thickness": (0.02, 0.04),
                "hatch_spacing": (0.08, 0.12),
            }

            result = optimizer.optimize_single_objective(objective_function, parameter_bounds)

            assert result.success is True

    @pytest.mark.integration
    def test_end_to_end_optimization_workflow(self, optimizer, quality_predictor, sample_process_data):
        """Test end-to-end optimization workflow."""

        # Step 1: Define optimization problem
        def objective_function(params):
            param_df = pd.DataFrame([params])
            quality_pred = quality_predictor.predict_quality(param_df)
            return -quality_pred[0]

        parameter_bounds = {
            "laser_power": (200.0, 300.0),
            "scan_speed": (800.0, 1200.0),
            "layer_thickness": (0.02, 0.04),
            "hatch_spacing": (0.08, 0.12),
        }

        # Step 2: Perform optimization
        opt_result = optimizer.optimize_single_objective(objective_function, parameter_bounds)

        assert opt_result.success is True

        # Step 3: Validate optimized parameters
        from am_qadf.analytics.process_analysis.prediction.early_defect_predictor import PredictionConfig
        from am_qadf.analytics.process_analysis.prediction.prediction_validator import PredictionValidator

        pred_config = PredictionConfig(random_seed=42)
        validator = PredictionValidator(pred_config)

        optimized_df = pd.DataFrame([opt_result.optimal_parameters])
        experimental_df = pd.DataFrame({"quality": [0.85]})  # Simulated experimental result

        validation_result = validator.validate_with_experimental_data(
            quality_predictor, optimized_df, experimental_df, quality_target="quality"
        )

        assert validation_result.success is True

        # Step 4: Calculate prediction intervals for uncertainty
        optimal_quality = -opt_result.optimal_values
        test_predictions = np.array([optimal_quality] * 5)

        lower, upper = validator.calculate_prediction_intervals(test_predictions, confidence_level=0.95)

        assert len(lower) == len(test_predictions)
        assert len(upper) == len(test_predictions)
