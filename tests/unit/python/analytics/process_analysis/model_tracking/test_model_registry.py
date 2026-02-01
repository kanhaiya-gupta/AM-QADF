"""
Unit tests for ModelRegistry.

Tests for model versioning, storage, and retrieval.
"""

import pytest
import numpy as np
import pandas as pd
import os
import tempfile
import shutil
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from am_qadf.analytics.process_analysis.model_tracking.model_registry import (
    ModelVersion,
    ModelRegistry,
)


class TestModelVersion:
    """Test suite for ModelVersion dataclass."""

    @pytest.mark.unit
    def test_model_version_creation(self):
        """Test creating ModelVersion."""
        version = ModelVersion(
            model_id="test_model_001",
            model_type="RandomForestRegressor",
            version="1.0",
            training_date=datetime.now(),
            metadata={"feature_names": ["feature1", "feature2"]},
            performance_metrics={"r2_score": 0.85, "rmse": 0.1},
            storage_path="/path/to/model.joblib",
        )

        assert version.model_id == "test_model_001"
        assert version.model_type == "RandomForestRegressor"
        assert version.version == "1.0"
        assert "feature_names" in version.metadata
        assert "r2_score" in version.performance_metrics
        assert version.storage_path == "/path/to/model.joblib"


class TestModelRegistry:
    """Test suite for ModelRegistry class."""

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
    def mock_model(self):
        """Create mock model for testing."""
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(random_state=42)
        # Train on dummy data
        X = np.random.rand(10, 3)
        y = np.random.rand(10)
        model.fit(X, y)
        return model

    @pytest.mark.unit
    def test_registry_initialization(self, registry):
        """Test ModelRegistry initialization."""
        assert registry.storage_path is not None
        assert os.path.exists(registry.storage_path)
        assert isinstance(registry._models, dict)

    @pytest.mark.unit
    def test_register_model_success(self, registry, mock_model):
        """Test registering a model successfully."""
        model_id = registry.register_model(
            model=mock_model,
            model_type="RandomForestRegressor",
            version="1.0",
            metadata={"feature_names": ["f1", "f2", "f3"]},
            performance_metrics={"r2_score": 0.85},
        )

        assert model_id is not None
        assert model_id in registry._models
        assert registry._models[model_id].model_type == "RandomForestRegressor"
        assert registry._models[model_id].version == "1.0"
        assert registry._models[model_id].performance_metrics["r2_score"] == 0.85
        # Check that model file exists
        assert os.path.exists(registry._models[model_id].file_path)

    @pytest.mark.unit
    def test_load_model_success(self, registry, mock_model):
        """Test loading a registered model."""
        model_id = registry.register_model(
            model=mock_model,
            model_type="RandomForestRegressor",
            version="1.0",
            metadata={"feature_names": ["f1", "f2", "f3"]},
            performance_metrics={"r2_score": 0.85},
        )

        loaded_model, model_version = registry.load_model(model_id)

        assert loaded_model is not None
        assert model_version.model_id == model_id
        # Verify model can make predictions
        X_test = np.random.rand(5, 3)
        predictions = loaded_model.predict(X_test)
        assert len(predictions) == 5

    @pytest.mark.unit
    def test_load_model_not_found(self, registry):
        """Test loading non-existent model."""
        with pytest.raises(ValueError, match="Model nonexistent not found"):
            registry.load_model("nonexistent")

    @pytest.mark.unit
    def test_list_models_all(self, registry, mock_model):
        """Test listing all models."""
        # Register multiple models
        id1 = registry.register_model(
            mock_model, "RandomForestRegressor", "1.0", metadata={}, performance_metrics={"r2_score": 0.85}
        )
        id2 = registry.register_model(
            mock_model, "GradientBoostingRegressor", "1.0", metadata={}, performance_metrics={"r2_score": 0.90}
        )

        models = registry.list_models()

        assert len(models) >= 2
        model_ids = [m["model_id"] for m in models]
        assert id1 in model_ids
        assert id2 in model_ids

    @pytest.mark.unit
    def test_list_models_filter_by_type(self, registry, mock_model):
        """Test listing models filtered by type."""
        id1 = registry.register_model(
            mock_model, "RandomForestRegressor", "1.0", metadata={}, performance_metrics={"r2_score": 0.85}
        )
        id2 = registry.register_model(
            mock_model, "GradientBoostingRegressor", "1.0", metadata={}, performance_metrics={"r2_score": 0.90}
        )

        models = registry.list_models(model_type="RandomForestRegressor")

        assert len(models) >= 1
        assert all(m["model_type"] == "RandomForestRegressor" for m in models)
        assert id1 in [m["model_id"] for m in models]

    @pytest.mark.unit
    def test_list_models_filter_by_version(self, registry, mock_model):
        """Test listing models filtered by version."""
        id1 = registry.register_model(
            mock_model, "RandomForestRegressor", "1.0", metadata={}, performance_metrics={"r2_score": 0.85}
        )
        id2 = registry.register_model(
            mock_model, "RandomForestRegressor", "2.0", metadata={}, performance_metrics={"r2_score": 0.90}
        )

        models = registry.list_models(version="1.0")

        assert len(models) >= 1
        assert all(m["version"] == "1.0" for m in models)
        assert id1 in [m["model_id"] for m in models]

    @pytest.mark.unit
    def test_compare_models(self, registry, mock_model):
        """Test comparing two models."""
        id1 = registry.register_model(
            mock_model, "RandomForestRegressor", "1.0", metadata={}, performance_metrics={"r2_score": 0.85, "rmse": 0.1}
        )
        id2 = registry.register_model(
            mock_model, "RandomForestRegressor", "2.0", metadata={}, performance_metrics={"r2_score": 0.90, "rmse": 0.08}
        )

        comparison = registry.compare_models(id1, id2)

        assert "model1_id" in comparison or "model_id1" in comparison or "model1" in comparison
        assert "model2_id" in comparison or "model_id2" in comparison or "model2" in comparison
        # Comparison returns metrics differences
        assert isinstance(comparison, dict)

    @pytest.mark.unit
    def test_compare_models_invalid_ids(self, registry):
        """Test comparing models with invalid IDs."""
        with pytest.raises(ValueError, match="Model nonexistent1 not found"):
            registry.compare_models("nonexistent1", "nonexistent2")

    @pytest.mark.unit
    def test_delete_model_success(self, registry, mock_model):
        """Test deleting a model successfully."""
        model_id = registry.register_model(
            mock_model, "RandomForestRegressor", "1.0", metadata={}, performance_metrics={"r2_score": 0.85}
        )

        # Verify model exists
        assert model_id in registry._models
        model_path = registry._models[model_id].file_path

        success = registry.delete_model(model_id)

        assert success is True
        assert model_id not in registry._models
        # Model file should be deleted
        assert not os.path.exists(model_path)

    @pytest.mark.unit
    def test_delete_model_not_found(self, registry):
        """Test deleting non-existent model."""
        success = registry.delete_model("nonexistent")

        assert success is False

    @pytest.mark.unit
    def test_registry_persistence(self, temp_registry_dir):
        """Test that registry persists across instances."""
        # Create first registry and register model
        registry1 = ModelRegistry(storage_path=temp_registry_dir)
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(random_state=42)
        model.fit(np.random.rand(10, 3), np.random.rand(10))

        model_id = registry1.register_model(
            model, "RandomForestRegressor", "1.0", metadata={}, performance_metrics={"r2_score": 0.85}
        )

        # Create new registry instance (simulates reload)
        registry2 = ModelRegistry(storage_path=temp_registry_dir)

        # Verify model is still in registry
        assert model_id in registry2._models
        assert registry2._models[model_id].performance_metrics["r2_score"] == 0.85
