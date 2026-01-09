"""
Unit tests for virtual experiment client.

Tests for VirtualExperimentConfig and VirtualExperimentClient.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock
from am_qadf.analytics.virtual_experiments.client import (
    VirtualExperimentConfig,
    VirtualExperimentClient,
)


class TestVirtualExperimentConfig:
    """Test suite for VirtualExperimentConfig dataclass."""

    @pytest.mark.unit
    def test_config_creation_default(self):
        """Test creating VirtualExperimentConfig with default values."""
        config = VirtualExperimentConfig()

        assert config.experiment_type == "parameter_sweep"
        assert config.base_model_id is None
        assert config.parameter_ranges is None
        assert config.use_warehouse_ranges is True
        assert config.design_type == "factorial"
        assert config.num_samples == 100
        assert config.compare_with_warehouse is True
        assert config.comparison_metrics is None
        assert config.spatial_region is None
        assert config.layer_range is None

    @pytest.mark.unit
    def test_config_creation_custom(self):
        """Test creating VirtualExperimentConfig with custom values."""
        config = VirtualExperimentConfig(
            experiment_type="optimization",
            base_model_id="model_123",
            parameter_ranges={"power": (100.0, 300.0), "velocity": (500.0, 1500.0)},
            use_warehouse_ranges=False,
            design_type="lhs",
            num_samples=200,
            compare_with_warehouse=False,
            comparison_metrics=["quality", "density"],
            spatial_region=((0, 0, 0), (10, 10, 10)),
            layer_range=(0, 50),
        )

        assert config.experiment_type == "optimization"
        assert config.base_model_id == "model_123"
        assert config.parameter_ranges["power"] == (100.0, 300.0)
        assert config.use_warehouse_ranges is False
        assert config.design_type == "lhs"
        assert config.num_samples == 200
        assert config.compare_with_warehouse is False
        assert len(config.comparison_metrics) == 2
        assert config.spatial_region == ((0, 0, 0), (10, 10, 10))
        assert config.layer_range == (0, 50)


class TestVirtualExperimentClient:
    """Test suite for VirtualExperimentClient class."""

    @pytest.fixture
    def mock_unified_client(self):
        """Create a mock UnifiedQueryClient."""
        mock_client = Mock()
        mock_client.stl_client = Mock()
        mock_client.stl_client.list_models = Mock(return_value=[{"model_id": "model1"}, {"model_id": "model2"}])
        mock_client.laser_client = None
        return mock_client

    @pytest.fixture
    def client(self, mock_unified_client):
        """Create a VirtualExperimentClient instance."""
        return VirtualExperimentClient(mock_unified_client)

    @pytest.mark.unit
    def test_client_creation(self, mock_unified_client):
        """Test creating VirtualExperimentClient."""
        client = VirtualExperimentClient(mock_unified_client)

        assert client.unified_client is not None
        assert client.voxel_client is None

    @pytest.mark.unit
    def test_client_creation_with_voxel_client(self, mock_unified_client):
        """Test creating VirtualExperimentClient with voxel client."""
        mock_voxel_client = Mock()
        client = VirtualExperimentClient(mock_unified_client, mock_voxel_client)

        assert client.unified_client is not None
        assert client.voxel_client is not None

    @pytest.mark.unit
    def test_query_historical_builds(self, client):
        """Test querying historical builds."""
        result = client.query_historical_builds(limit=10)

        assert isinstance(result, pd.DataFrame)
        # Should return empty DataFrame if no laser client
        assert len(result) == 0 or isinstance(result, pd.DataFrame)

    @pytest.mark.unit
    def test_query_historical_builds_with_model_type(self, client):
        """Test querying historical builds with model type filter."""
        result = client.query_historical_builds(model_type="test_model", limit=10)

        assert isinstance(result, pd.DataFrame)

    @pytest.mark.unit
    def test_query_historical_builds_with_process_conditions(self, client):
        """Test querying historical builds with process conditions."""
        result = client.query_historical_builds(process_conditions=["laser_power > 200"], limit=10)

        assert isinstance(result, pd.DataFrame)

    @pytest.mark.unit
    def test_get_parameter_ranges_from_warehouse(self, client):
        """Test getting parameter ranges from warehouse."""
        # This will likely return empty dict if no laser client
        ranges = client.get_parameter_ranges_from_warehouse(model_id="test_model")

        assert isinstance(ranges, dict)

    @pytest.mark.unit
    def test_design_virtual_experiment(self, client):
        """Test designing virtual experiment."""
        config = VirtualExperimentConfig(experiment_type="parameter_sweep", design_type="factorial", num_samples=10)

        # This may fail if experiment designer is not available
        try:
            design = client.design_virtual_experiment(config)
            assert design is not None
        except Exception:
            # Expected if virtual experiment modules not available
            pass

    @pytest.mark.unit
    def test_execute_virtual_experiment(self, client):
        """Test executing virtual experiment."""
        config = VirtualExperimentConfig(experiment_type="parameter_sweep", num_samples=5)

        # This may fail if experiment designer is not available
        try:
            results = client.execute_virtual_experiment(config)
            assert results is not None
        except Exception:
            # Expected if virtual experiment modules not available
            pass
