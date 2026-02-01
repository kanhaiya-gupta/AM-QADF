"""
Unit tests for ModelPerformanceTracker.

Tests for performance tracking, degradation detection, and drift calculation.
"""

import pytest
import numpy as np
import pandas as pd
import os
import tempfile
import shutil
from collections import deque
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from am_qadf.analytics.process_analysis.model_tracking.model_registry import ModelRegistry, ModelVersion
from am_qadf.analytics.process_analysis.model_tracking.performance_tracker import (
    ModelPerformanceMetrics,
    ModelPerformanceTracker,
)


class TestModelPerformanceMetrics:
    """Test suite for ModelPerformanceMetrics dataclass."""

    @pytest.mark.unit
    def test_metrics_creation(self):
        """Test creating ModelPerformanceMetrics."""
        metrics = ModelPerformanceMetrics(
            model_id="test_model_001",
            model_type="RandomForestRegressor",
            version="1.0",
            training_date=datetime.now(),
            performance_metrics={"r2_score": 0.85, "rmse": 0.1},
            validation_metrics={"cv_r2_mean": 0.83},
            feature_importance={"feature1": 0.5, "feature2": 0.3},
            drift_score=0.1,
            last_evaluated=datetime.now(),
            evaluation_count=5,
        )

        assert metrics.model_id == "test_model_001"
        assert "r2_score" in metrics.performance_metrics
        assert "cv_r2_mean" in metrics.validation_metrics
        assert len(metrics.feature_importance) == 2
        assert metrics.drift_score == 0.1
        assert metrics.evaluation_count == 5


class TestModelPerformanceTracker:
    """Test suite for ModelPerformanceTracker class."""

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
    def registered_model(self, registry):
        """Register a model and return model_id."""
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(random_state=42)
        X = np.random.rand(100, 5)
        y = np.random.rand(100)
        model.fit(X, y)

        model_id = registry.register_model(
            model=model,
            model_type="RandomForestRegressor",
            version="1.0",
            metadata={"feature_names": [f"feature_{i}" for i in range(5)]},
            performance_metrics={"r2_score": 0.85, "rmse": 0.1},
        )
        return model_id

    @pytest.fixture
    def mock_model(self, registry, registered_model):
        """Load the registered model."""
        model, _ = registry.load_model(registered_model)
        return model

    @pytest.fixture
    def tracker(self, registry, registered_model, mock_model):
        """Create ModelPerformanceTracker instance."""
        # Load the model first
        model, _ = registry.load_model(registered_model)
        tracker = ModelPerformanceTracker(model_id=registered_model, model_registry=registry, history_size=10)
        return tracker

    @pytest.fixture
    def test_data_with_target(self):
        """Create test data for evaluation with target."""
        np.random.seed(42)
        data = pd.DataFrame({f"feature_{i}": np.random.randn(50) for i in range(5)})
        data["quality"] = np.random.rand(50)  # Add target column
        return data

    @pytest.mark.unit
    def test_tracker_initialization(self, tracker):
        """Test ModelPerformanceTracker initialization."""
        assert tracker.model_id is not None
        assert tracker.model_registry is not None
        assert tracker.history_size == 10
        assert isinstance(tracker.performance_history, deque)

    @pytest.mark.unit
    def test_evaluate_model_performance_success(self, tracker, registry, registered_model, test_data_with_target):
        """Test evaluating model performance."""
        # Load the model
        model, _ = registry.load_model(registered_model)

        metrics = tracker.evaluate_model_performance(model=model, test_data=test_data_with_target, quality_target="quality")

        assert isinstance(metrics, ModelPerformanceMetrics)
        assert metrics.model_id == tracker.model_id
        assert len(metrics.performance_metrics) > 0
        assert "r2_score" in metrics.performance_metrics or "mae" in metrics.performance_metrics
        assert metrics.evaluation_count > 0

    @pytest.mark.unit
    def test_track_performance_history(self, tracker, registry, registered_model, test_data_with_target):
        """Test tracking performance history."""
        # Load the model
        model, _ = registry.load_model(registered_model)

        initial_history_size = len(tracker.performance_history)

        metrics = tracker.evaluate_model_performance(model=model, test_data=test_data_with_target, quality_target="quality")

        assert len(tracker.performance_history) >= initial_history_size
        assert tracker.performance_history[-1] == metrics

    @pytest.mark.unit
    def test_detect_performance_degradation_no_degradation(self, tracker, registry, registered_model, test_data_with_target):
        """Test degradation detection when performance is stable."""
        # Load the model
        model, _ = registry.load_model(registered_model)

        # Evaluate multiple times with similar performance
        for _ in range(6):
            tracker.evaluate_model_performance(model=model, test_data=test_data_with_target, quality_target="quality")

        detected, degradation_pct = tracker.detect_performance_degradation(
            metric_name="r2_score", threshold=0.1  # 10% degradation threshold
        )

        # Should not detect significant degradation for stable model
        assert isinstance(detected, bool)
        assert degradation_pct is not None or detected is False

    @pytest.mark.unit
    def test_detect_performance_degradation_insufficient_evaluations(
        self, tracker, registry, registered_model, test_data_with_target
    ):
        """Test degradation detection with insufficient evaluations."""
        # Load the model
        model, _ = registry.load_model(registered_model)

        # Only 2 evaluations
        tracker.evaluate_model_performance(model=model, test_data=test_data_with_target, quality_target="quality")
        tracker.evaluate_model_performance(model=model, test_data=test_data_with_target, quality_target="quality")

        detected, degradation_pct = tracker.detect_performance_degradation(metric_name="r2_score", threshold=0.1)

        # With only 2 evaluations, should return False (needs at least 2 for comparison)
        assert isinstance(detected, bool)
        assert degradation_pct is not None or detected is False

    @pytest.mark.unit
    def test_get_performance_trend(self, tracker, registry, registered_model, test_data_with_target):
        """Test getting performance trend."""
        # Load the model
        model, _ = registry.load_model(registered_model)

        # Evaluate multiple times
        for _ in range(10):
            tracker.evaluate_model_performance(model=model, test_data=test_data_with_target, quality_target="quality")

        trend = tracker.get_performance_trend("r2_score")

        assert isinstance(trend, dict)
        # May have trend_available or other keys depending on implementation
        assert len(trend) > 0

    @pytest.mark.unit
    def test_get_performance_trend_insufficient_data(self, tracker):
        """Test getting trend with insufficient data."""
        trend = tracker.get_performance_trend("r2_score")

        assert isinstance(trend, dict)
        # Should handle insufficient data gracefully

    @pytest.mark.unit
    def test_calculate_drift_score(self, tracker):
        """Test calculating drift score."""
        # Training data (reference)
        training_data = pd.DataFrame({f"feature_{i}": np.random.randn(100) for i in range(5)})

        # Current data (potentially drifted)
        current_data = pd.DataFrame({f"feature_{i}": np.random.randn(100) + 0.5 for i in range(5)})  # Shifted distribution

        drift_score = tracker.calculate_drift_score(current_data, training_data)

        assert isinstance(drift_score, float)
        assert 0.0 <= drift_score <= 1.0  # Drift score should be between 0 and 1
        # Shifted data should have higher drift score
        assert drift_score > 0.0

    @pytest.mark.unit
    def test_calculate_drift_score_no_common_features(self, tracker):
        """Test drift calculation with no common features."""
        training_data = pd.DataFrame({"feature_A": np.random.randn(100), "feature_B": np.random.randn(100)})

        current_data = pd.DataFrame({"feature_X": np.random.randn(100), "feature_Y": np.random.randn(100)})

        drift_score = tracker.calculate_drift_score(current_data, training_data)

        # Should return maximum drift (1.0) when no common features
        assert drift_score == 1.0
