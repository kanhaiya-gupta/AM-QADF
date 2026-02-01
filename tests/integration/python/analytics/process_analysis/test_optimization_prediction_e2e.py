"""
End-to-end integration tests for Process Optimization and Prediction.

Tests complete workflows combining prediction, optimization, model tracking, and validation.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import shutil
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch

try:
    from am_qadf.analytics.process_analysis.quality_analysis import QualityPredictor, QualityAnalysisConfig
    from am_qadf.analytics.process_analysis.prediction.early_defect_predictor import EarlyDefectPredictor, PredictionConfig
    from am_qadf.analytics.process_analysis.prediction.time_series_predictor import TimeSeriesPredictor
    from am_qadf.analytics.process_analysis.prediction.prediction_validator import PredictionValidator
    from am_qadf.analytics.process_analysis.optimization import ProcessOptimizer, OptimizationConfig
    from am_qadf.analytics.process_analysis.model_tracking.model_registry import ModelRegistry
    from am_qadf.analytics.process_analysis.model_tracking.performance_tracker import ModelPerformanceTracker
except ImportError as e:
    pytest.skip(f"Required modules not available: {e}", allow_module_level=True)


class TestOptimizationPredictionE2E:
    """End-to-end integration tests for optimization and prediction workflows."""

    @pytest.fixture
    def sample_process_data(self):
        """Create comprehensive sample process data."""
        np.random.seed(42)
        n_samples = 200

        # Generate process parameters
        data = {
            "laser_power": np.random.uniform(200, 300, n_samples),
            "scan_speed": np.random.uniform(800, 1200, n_samples),
            "layer_thickness": np.random.uniform(0.02, 0.04, n_samples),
            "hatch_spacing": np.random.uniform(0.08, 0.12, n_samples),
            "temperature": np.random.uniform(800, 1200, n_samples),
            "build_time": np.random.uniform(100, 500, n_samples),
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

        # Create defect labels
        data["defect_label"] = (quality < 0.6).astype(int)

        # Add temporal structure
        data["timestamp"] = pd.date_range(start="2024-01-01", periods=n_samples, freq="H")
        data["build_id"] = [f"build_{i//50}" for i in range(n_samples)]

        return pd.DataFrame(data)

    @pytest.fixture
    def temp_registry_dir(self):
        """Create temporary directory for registry."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.integration
    def test_complete_prediction_optimization_workflow(self, sample_process_data, temp_registry_dir):
        """Test complete workflow: prediction -> optimization -> validation -> model tracking."""
        feature_names = ["laser_power", "scan_speed", "layer_thickness", "hatch_spacing"]

        # Step 1: Train quality predictor
        quality_config = QualityAnalysisConfig(random_seed=42)
        quality_predictor = QualityPredictor(quality_config)

        quality_result = quality_predictor.analyze_quality_prediction(
            sample_process_data, quality_target="quality", feature_names=feature_names
        )

        assert quality_result.success is True

        # Step 2: Register model in registry
        registry = ModelRegistry(storage_path=temp_registry_dir)

        model_id = registry.register_model(
            model=quality_predictor.trained_model,
            model_type="RandomForestRegressor",
            version="1.0",
            metadata={"feature_names": feature_names},
            performance_metrics=quality_result.model_performance,
        )

        assert model_id is not None

        # Step 3: Train early defect predictor
        pred_config = PredictionConfig(random_seed=42, early_prediction_horizon=50)
        early_predictor = EarlyDefectPredictor(pred_config)

        defect_result = early_predictor.train_early_prediction_model(
            sample_process_data[feature_names], sample_process_data["defect_label"].values, feature_names=feature_names
        )

        assert defect_result.success is True

        # Step 4: Forecast quality time-series
        time_series_predictor = TimeSeriesPredictor(pred_config)
        quality_series = sample_process_data["quality"].values

        forecast_result = time_series_predictor.forecast_quality_metric(
            quality_series, forecast_horizon=10, model_type="moving_average"
        )

        assert forecast_result.success is True

        # Step 5: Perform optimization using quality predictor
        optimizer_config = OptimizationConfig(
            optimization_method="differential_evolution", max_iterations=50, population_size=10, random_seed=42
        )
        optimizer = ProcessOptimizer(optimizer_config)

        def objective_function(params):
            param_df = pd.DataFrame([params])
            quality_pred = quality_predictor.predict_quality(param_df)
            return -quality_pred[0]  # Negative for minimization

        parameter_bounds = {
            "laser_power": (200.0, 300.0),
            "scan_speed": (800.0, 1200.0),
            "layer_thickness": (0.02, 0.04),
            "hatch_spacing": (0.08, 0.12),
        }

        opt_result = optimizer.optimize_single_objective(objective_function, parameter_bounds)

        assert opt_result.success is True
        assert len(opt_result.optimal_parameters) == 4

        # Step 6: Validate optimized parameters
        validator = PredictionValidator(pred_config)

        optimized_df = pd.DataFrame([opt_result.optimal_parameters])
        experimental_df = pd.DataFrame({"quality": [0.85]})  # Simulated experimental result

        validation_result = validator.validate_with_experimental_data(
            quality_predictor, optimized_df, experimental_df, quality_target="quality"
        )

        assert validation_result.success is True

        # Step 7: Track model performance over time
        tracker = ModelPerformanceTracker(model_id=model_id, model_registry=registry, history_size=10)

        # Create test data
        test_data = sample_process_data.iloc[:30].copy()

        # Evaluate performance
        metrics = tracker.evaluate_model_performance(
            model=quality_predictor.trained_model, test_data=test_data, quality_target="quality"
        )

        assert (
            isinstance(metrics, type(tracker.performance_history[0]) if len(tracker.performance_history) > 0 else None)
            or metrics is not None
        )

        # Step 8: Calculate drift score
        training_df = sample_process_data[feature_names].iloc[:100]
        current_df = test_data[feature_names]

        drift_score = tracker.calculate_drift_score(current_df, training_df)

        assert 0.0 <= drift_score <= 1.0

        # Step 9: Check performance trend
        # Add more evaluations for trend analysis
        for _ in range(3):
            tracker.evaluate_model_performance(
                model=quality_predictor.trained_model, test_data=test_data, quality_target="quality"
            )

        trend = tracker.get_performance_trend("r2_score" if "r2_score" in metrics.performance_metrics else "mae")
        assert isinstance(trend, dict)

        # Step 10: Verify all components integrated successfully
        # Model registry
        loaded_model, model_version = registry.load_model(model_id)
        assert loaded_model is not None
        assert model_version.model_id == model_id

        # Performance tracking
        history = tracker.get_performance_history()
        assert len(history) >= 1

        # Prediction validation
        assert validation_result.validation_error is not None
        assert len(validation_result.validation_metrics) > 0

    @pytest.mark.integration
    def test_multi_objective_optimization_with_predictions(self, sample_process_data):
        """Test multi-objective optimization with quality and efficiency predictions."""
        feature_names = ["laser_power", "scan_speed", "layer_thickness", "hatch_spacing"]

        # Train quality predictor
        quality_config = QualityAnalysisConfig(random_seed=42)
        quality_predictor = QualityPredictor(quality_config)

        quality_predictor.analyze_quality_prediction(
            sample_process_data, quality_target="quality", feature_names=feature_names
        )

        # Define multi-objective function
        def objective_functions(params):
            # Objective 1: Maximize quality
            param_df = pd.DataFrame([params])
            quality_pred = quality_predictor.predict_quality(param_df)

            # Objective 2: Minimize energy consumption (power/speed)
            energy = params["laser_power"] / params["scan_speed"]

            return [-quality_pred[0], energy]  # Negative quality for minimization

        # Perform multi-objective optimization
        optimizer_config = OptimizationConfig(
            optimization_method="weighted_sum", max_iterations=50, population_size=10, pareto_front_size=20, random_seed=42
        )
        optimizer = ProcessOptimizer(optimizer_config)

        parameter_bounds = {
            "laser_power": (200.0, 300.0),
            "scan_speed": (800.0, 1200.0),
            "layer_thickness": (0.02, 0.04),
            "hatch_spacing": (0.08, 0.12),
        }

        result = optimizer.optimize_multi_objective(objective_functions, parameter_bounds, n_objectives=2)

        assert result.success is True
        assert result.pareto_front is not None
        assert len(result.pareto_front) > 0

    @pytest.mark.integration
    def test_prediction_model_lifecycle(self, sample_process_data, temp_registry_dir):
        """Test complete model lifecycle: training -> registration -> tracking -> retraining."""
        feature_names = ["laser_power", "scan_speed", "layer_thickness", "hatch_spacing"]

        # Step 1: Initial model training
        quality_config = QualityAnalysisConfig(random_seed=42)
        quality_predictor = QualityPredictor(quality_config)

        initial_result = quality_predictor.analyze_quality_prediction(
            sample_process_data.iloc[:100], quality_target="quality", feature_names=feature_names
        )

        assert initial_result.success is True

        # Step 2: Register initial model
        registry = ModelRegistry(storage_path=temp_registry_dir)

        model_id_v1 = registry.register_model(
            model=quality_predictor.trained_model,
            model_type="RandomForestRegressor",
            version="1.0",
            metadata={"feature_names": feature_names},
            performance_metrics=initial_result.model_performance,
        )

        # Step 3: Track performance
        tracker = ModelPerformanceTracker(model_id=model_id_v1, model_registry=registry)

        test_data = sample_process_data.iloc[100:130].copy()

        # Initial performance evaluation
        metrics_v1 = tracker.evaluate_model_performance(
            model=quality_predictor.trained_model, test_data=test_data, quality_target="quality"
        )

        assert metrics_v1 is not None

        # Step 4: Retrain with new data (simulating model update)
        updated_result = quality_predictor.analyze_quality_prediction(
            sample_process_data, quality_target="quality", feature_names=feature_names  # Use all data
        )

        assert updated_result.success is True

        # Step 5: Register updated model as new version
        model_id_v2 = registry.register_model(
            model=quality_predictor.trained_model,
            model_type="RandomForestRegressor",
            version="2.0",
            metadata={"feature_names": feature_names},
            performance_metrics=updated_result.model_performance,
        )

        # Step 6: Compare model versions
        comparison = registry.compare_models(model_id_v1, model_id_v2)
        assert isinstance(comparison, dict)

        # Step 7: List models
        all_models = registry.list_models()
        assert len(all_models) >= 2

        # Verify both versions exist
        model_ids = [m["model_id"] for m in all_models]
        assert model_id_v1 in model_ids
        assert model_id_v2 in model_ids
