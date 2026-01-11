"""
Integration tests for model tracking workflows.

Tests integration of model registry, performance tracking, and monitoring.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import shutil
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch

try:
    from am_qadf.analytics.process_analysis.model_tracking.model_registry import ModelRegistry, ModelVersion
    from am_qadf.analytics.process_analysis.model_tracking.performance_tracker import (
        ModelPerformanceTracker,
        ModelPerformanceMetrics,
    )
    from am_qadf.analytics.process_analysis.model_tracking.model_monitor import ModelMonitor, ModelMonitoringConfig
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
except ImportError as e:
    pytest.skip(f"Model tracking modules not available: {e}", allow_module_level=True)


class TestModelTrackingWorkflow:
    """Integration tests for model tracking workflows."""

    @pytest.fixture
    def temp_registry_dir(self):
        """Create temporary directory for registry."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def registry(self, temp_registry_dir):
        """Create ModelRegistry instance."""
        return ModelRegistry(storage_path=temp_registry_dir)

    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n_samples = 100
        X = np.random.rand(n_samples, 5)
        y = 0.5 + 0.3 * X[:, 0] + 0.2 * X[:, 1] + np.random.randn(n_samples) * 0.1
        return X, y

    @pytest.fixture
    def sample_test_data(self):
        """Create sample test data."""
        np.random.seed(123)
        n_samples = 30
        X = np.random.rand(n_samples, 5)
        y = 0.5 + 0.3 * X[:, 0] + 0.2 * X[:, 1] + np.random.randn(n_samples) * 0.1
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(5)])
        df["quality"] = y
        return df

    @pytest.mark.integration
    def test_model_registration_and_retrieval(self, registry, sample_training_data):
        """Test model registration and retrieval workflow."""
        X_train, y_train = sample_training_data

        # Train model
        model1 = RandomForestRegressor(random_state=42, n_estimators=10)
        model1.fit(X_train, y_train)

        # Register model
        model_id1 = registry.register_model(
            model=model1,
            model_type="RandomForestRegressor",
            version="1.0",
            metadata={"feature_names": [f"feature_{i}" for i in range(5)]},
            performance_metrics={"r2_score": 0.85, "rmse": 0.1},
        )

        assert model_id1 is not None
        assert model_id1 in registry._models

        # Load model
        loaded_model, model_version = registry.load_model(model_id1)

        assert loaded_model is not None
        assert model_version.model_id == model_id1
        assert model_version.model_type == "RandomForestRegressor"
        assert model_version.version == "1.0"

        # Verify model works
        X_test = np.random.rand(5, 5)
        predictions = loaded_model.predict(X_test)
        assert len(predictions) == 5

    @pytest.mark.integration
    def test_model_versioning_workflow(self, registry, sample_training_data):
        """Test model versioning workflow."""
        X_train, y_train = sample_training_data

        # Register version 1.0
        model1 = RandomForestRegressor(random_state=42, n_estimators=10)
        model1.fit(X_train, y_train)

        model_id1 = registry.register_model(
            model=model1, model_type="RandomForestRegressor", version="1.0", performance_metrics={"r2_score": 0.85}
        )

        # Register version 2.0 with improved performance
        model2 = RandomForestRegressor(random_state=42, n_estimators=20)
        model2.fit(X_train, y_train)

        model_id2 = registry.register_model(
            model=model2, model_type="RandomForestRegressor", version="2.0", performance_metrics={"r2_score": 0.90}
        )

        # List models by version
        v1_models = registry.list_models(version="1.0")
        v2_models = registry.list_models(version="2.0")

        assert len(v1_models) >= 1
        assert len(v2_models) >= 1
        assert all(m["version"] == "1.0" for m in v1_models)
        assert all(m["version"] == "2.0" for m in v2_models)

        # Compare models
        comparison = registry.compare_models(model_id1, model_id2)
        assert isinstance(comparison, dict)

    @pytest.mark.integration
    def test_performance_tracking_workflow(self, registry, sample_training_data, sample_test_data):
        """Test performance tracking workflow."""
        X_train, y_train = sample_training_data

        # Train and register model
        model = RandomForestRegressor(random_state=42, n_estimators=10)
        model.fit(X_train, y_train)

        model_id = registry.register_model(
            model=model,
            model_type="RandomForestRegressor",
            version="1.0",
            metadata={"feature_names": [f"feature_{i}" for i in range(5)]},
            performance_metrics={"r2_score": 0.85},
        )

        # Create performance tracker
        tracker = ModelPerformanceTracker(model_id=model_id, model_registry=registry, history_size=10)

        # Evaluate performance multiple times
        for i in range(5):
            metrics = tracker.evaluate_model_performance(model=model, test_data=sample_test_data, quality_target="quality")

            assert isinstance(metrics, ModelPerformanceMetrics)
            assert metrics.model_id == model_id
            assert len(metrics.performance_metrics) > 0

        # Check history
        history = tracker.get_performance_history()
        assert len(history) == 5
        assert all("performance_metrics" in entry for entry in history)

    @pytest.mark.integration
    def test_performance_degradation_detection(self, registry, sample_training_data, sample_test_data):
        """Test performance degradation detection."""
        X_train, y_train = sample_training_data

        # Train and register model
        model = RandomForestRegressor(random_state=42, n_estimators=10)
        model.fit(X_train, y_train)

        model_id = registry.register_model(
            model=model, model_type="RandomForestRegressor", version="1.0", performance_metrics={"r2_score": 0.85}
        )

        tracker = ModelPerformanceTracker(model_id=model_id, model_registry=registry, history_size=10)

        # Evaluate multiple times (performance should be stable)
        for _ in range(6):
            tracker.evaluate_model_performance(model=model, test_data=sample_test_data, quality_target="quality")

        # Check for degradation
        detected, degradation_pct = tracker.detect_performance_degradation(
            metric_name="r2_score", threshold=0.1, min_evaluations=5
        )

        assert isinstance(detected, bool)
        # For stable model, degradation should be minimal
        assert degradation_pct is not None or not detected

    @pytest.mark.integration
    def test_drift_detection_workflow(self, registry, sample_training_data, sample_test_data):
        """Test data drift detection workflow."""
        X_train, y_train = sample_training_data

        # Train and register model
        model = RandomForestRegressor(random_state=42, n_estimators=10)
        model.fit(X_train, y_train)

        model_id = registry.register_model(model=model, model_type="RandomForestRegressor", version="1.0")

        tracker = ModelPerformanceTracker(model_id=model_id, model_registry=registry)

        # Create training data DataFrame
        training_df = pd.DataFrame(X_train, columns=[f"feature_{i}" for i in range(5)])

        # Test with similar distribution (low drift)
        drift_score_low = tracker.calculate_drift_score(sample_test_data[[f"feature_{i}" for i in range(5)]], training_df)

        assert 0.0 <= drift_score_low <= 1.0

        # Test with shifted distribution (higher drift)
        shifted_data = sample_test_data[[f"feature_{i}" for i in range(5)]].copy()
        shifted_data["feature_0"] += 2.0  # Significant shift

        drift_score_high = tracker.calculate_drift_score(shifted_data, training_df)

        assert 0.0 <= drift_score_high <= 1.0
        # Shifted data should generally have higher drift
        assert drift_score_high >= drift_score_low

    @pytest.mark.integration
    def test_model_comparison_workflow(self, registry, sample_training_data):
        """Test model comparison workflow."""
        X_train, y_train = sample_training_data

        # Register multiple models
        models = []
        model_ids = []

        for i, model_type in enumerate([RandomForestRegressor, GradientBoostingRegressor]):
            model = model_type(random_state=42, n_estimators=10)
            model.fit(X_train, y_train)

            model_id = registry.register_model(
                model=model, model_type=model_type.__name__, version=f"1.{i}", performance_metrics={"r2_score": 0.8 + i * 0.05}
            )

            models.append(model)
            model_ids.append(model_id)

        # List all models
        all_models = registry.list_models()
        assert len(all_models) >= 2

        # Compare two models
        comparison = registry.compare_models(model_ids[0], model_ids[1])
        assert isinstance(comparison, dict)
        assert "model_id1" in comparison or "model1" in comparison

    @pytest.mark.integration
    def test_end_to_end_model_tracking(self, registry, sample_training_data, sample_test_data):
        """Test end-to-end model tracking workflow."""
        X_train, y_train = sample_training_data

        # Step 1: Train and register model
        model = RandomForestRegressor(random_state=42, n_estimators=10)
        model.fit(X_train, y_train)

        model_id = registry.register_model(
            model=model,
            model_type="RandomForestRegressor",
            version="1.0",
            metadata={"feature_names": [f"feature_{i}" for i in range(5)]},
            performance_metrics={"r2_score": 0.85, "rmse": 0.1},
        )

        # Step 2: Create tracker
        tracker = ModelPerformanceTracker(model_id=model_id, model_registry=registry, history_size=10)

        # Step 3: Track performance over time
        for _ in range(3):
            metrics = tracker.evaluate_model_performance(model=model, test_data=sample_test_data, quality_target="quality")
            assert metrics is not None

        # Step 4: Get performance trend
        trend = tracker.get_performance_trend("r2_score")
        assert isinstance(trend, dict)

        # Step 5: Calculate drift
        training_df = pd.DataFrame(X_train, columns=[f"feature_{i}" for i in range(5)])
        drift_score = tracker.calculate_drift_score(sample_test_data[[f"feature_{i}" for i in range(5)]], training_df)
        assert 0.0 <= drift_score <= 1.0

        # Step 6: Check for degradation
        detected, degradation_pct = tracker.detect_performance_degradation(
            metric_name="r2_score", threshold=0.15, min_evaluations=3
        )
        assert isinstance(detected, bool)
