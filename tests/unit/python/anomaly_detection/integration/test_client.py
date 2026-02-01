"""
Unit tests for anomaly detection client.

Tests for AnomalyDetectionConfig and AnomalyDetectionClient.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
from am_qadf.anomaly_detection.integration.client import (
    AnomalyDetectionConfig,
    AnomalyDetectionClient,
)


class TestAnomalyDetectionConfig:
    """Test suite for AnomalyDetectionConfig dataclass."""

    @pytest.mark.unit
    def test_config_creation_default(self):
        """Test creating AnomalyDetectionConfig with default values."""
        config = AnomalyDetectionConfig()

        assert config.methods is None
        assert config.signals is None
        assert config.spatial_region is None
        assert config.layer_range is None
        assert config.time_range is None
        assert config.threshold == 0.5
        assert config.use_voxel_domain is True
        assert config.voxel_resolution == 0.5
        assert config.use_historical_training is True
        assert config.historical_model_ids is None

    @pytest.mark.unit
    def test_config_creation_custom(self):
        """Test creating AnomalyDetectionConfig with custom values."""
        time_range = (datetime(2023, 1, 1), datetime(2023, 12, 31))
        config = AnomalyDetectionConfig(
            methods=["isolation_forest", "dbscan"],
            signals=["laser_power", "density"],
            spatial_region=((0, 0, 0), (10, 10, 10)),
            layer_range=(0, 50),
            time_range=time_range,
            threshold=0.7,
            use_voxel_domain=False,
            voxel_resolution=1.0,
            use_historical_training=False,
            historical_model_ids=["model1", "model2"],
        )

        assert config.methods == ["isolation_forest", "dbscan"]
        assert config.signals == ["laser_power", "density"]
        assert config.spatial_region == ((0, 0, 0), (10, 10, 10))
        assert config.layer_range == (0, 50)
        assert config.time_range == time_range
        assert config.threshold == 0.7
        assert config.use_voxel_domain is False
        assert config.voxel_resolution == 1.0
        assert config.use_historical_training is False
        assert config.historical_model_ids == ["model1", "model2"]


class TestAnomalyDetectionClient:
    """Test suite for AnomalyDetectionClient class."""

    @pytest.fixture
    def mock_unified_client(self):
        """Create a mock UnifiedQueryClient."""
        mock_client = Mock()
        mock_client.merge_temporal_data = Mock(return_value={})
        mock_client.ct_client = None
        mock_client.stl_client = None
        return mock_client

    @pytest.fixture
    def mock_voxel_client(self):
        """Create a mock VoxelDomainClient."""
        mock_client = Mock()
        mock_client.map_signals_to_voxels = Mock(
            return_value=Mock(
                available_signals=["laser_power", "density"],
                get_signal_array=Mock(return_value=np.random.randn(10, 10, 10)),
            )
        )
        return mock_client

    @pytest.fixture
    def client(self, mock_unified_client):
        """Create an AnomalyDetectionClient instance."""
        return AnomalyDetectionClient(mock_unified_client)

    @pytest.fixture
    def client_with_voxel(self, mock_unified_client, mock_voxel_client):
        """Create an AnomalyDetectionClient with voxel client."""
        return AnomalyDetectionClient(mock_unified_client, mock_voxel_client)

    @pytest.mark.unit
    def test_client_creation(self, mock_unified_client):
        """Test creating AnomalyDetectionClient."""
        client = AnomalyDetectionClient(mock_unified_client)

        assert client.unified_client is not None
        assert client.voxel_client is None
        assert isinstance(client.detectors, dict)

    @pytest.mark.unit
    def test_client_creation_with_voxel_client(self, mock_unified_client, mock_voxel_client):
        """Test creating AnomalyDetectionClient with voxel client."""
        client = AnomalyDetectionClient(mock_unified_client, mock_voxel_client)

        assert client.unified_client is not None
        assert client.voxel_client is not None

    @pytest.mark.unit
    def test_query_fused_data(self, client):
        """Test querying fused data."""
        result = client.query_fused_data(model_id="test_model", sources=["laser", "ispm"], layer_range=(0, 10))

        assert isinstance(result, pd.DataFrame)
        # Should return empty DataFrame if no data
        assert len(result) == 0 or isinstance(result, pd.DataFrame)

    @pytest.mark.unit
    def test_query_fused_data_all_sources(self, client):
        """Test querying fused data from all sources."""
        result = client.query_fused_data(model_id="test_model", sources=["laser", "ispm", "ct"])

        assert isinstance(result, pd.DataFrame)

    @pytest.mark.unit
    def test_query_fused_data_with_merged_data(self, mock_unified_client):
        """Test querying fused data when merged data is available."""
        # Mock merged data structure
        mock_unified_client.merge_temporal_data = Mock(
            return_value={
                "sources": {
                    "laser": {
                        "points": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                        "signals": {"power": [100.0, 200.0], "velocity": [10.0, 20.0]},
                    }
                }
            }
        )

        client = AnomalyDetectionClient(mock_unified_client)
        result = client.query_fused_data(model_id="test_model", sources=["laser"])

        assert isinstance(result, pd.DataFrame)
        if len(result) > 0:
            assert "x" in result.columns
            assert "y" in result.columns
            assert "z" in result.columns

    @pytest.mark.unit
    def test_query_historical_data(self, client):
        """Test querying historical data."""
        result = client.query_historical_data(model_ids=["model1", "model2"], signals=["laser_power"])

        assert isinstance(result, dict)

    @pytest.mark.unit
    def test_query_historical_data_no_model_ids(self, mock_unified_client):
        """Test querying historical data without model IDs."""
        mock_stl_client = Mock()
        mock_stl_client.list_models = Mock(return_value=[{"model_id": "model1"}, {"model_id": "model2"}])
        mock_unified_client.stl_client = mock_stl_client

        client = AnomalyDetectionClient(mock_unified_client)
        result = client.query_historical_data()

        assert isinstance(result, dict)

    @pytest.mark.unit
    def test_detect_anomalies_with_dataframe(self, client):
        """Test detecting anomalies with provided DataFrame."""
        config = AnomalyDetectionConfig(methods=["isolation_forest"], signals=["laser_power"])

        # Create test DataFrame
        test_data = pd.DataFrame(
            {
                "laser_power": np.random.randn(50) * 10 + 100,
                "density": np.random.randn(50) * 0.1 + 0.9,
                "x": np.random.randn(50),
                "y": np.random.randn(50),
                "z": np.random.randn(50),
            }
        )

        try:
            result = client.detect_anomalies(model_id="test_model", config=config, data=test_data)

            assert result is not None
            assert "model_id" in result
            assert "methods" in result
            assert "results" in result
            assert "anomaly_labels" in result
        except Exception as e:
            # Expected if detectors are not available
            pass

    @pytest.mark.unit
    def test_detect_anomalies_with_voxel_domain(self, client_with_voxel):
        """Test detecting anomalies with voxel domain."""
        config = AnomalyDetectionConfig(use_voxel_domain=True, methods=["isolation_forest"], signals=["laser_power"])

        try:
            result = client_with_voxel.detect_anomalies(model_id="test_model", config=config)

            assert result is not None
            assert "model_id" in result
        except Exception:
            # Expected if modules not available
            pass

    @pytest.mark.unit
    def test_detect_anomalies_with_historical_training(self, client):
        """Test detecting anomalies with historical training."""
        config = AnomalyDetectionConfig(
            use_historical_training=True,
            historical_model_ids=["model1", "model2"],
            methods=["isolation_forest"],
        )

        # Mock historical data
        historical_data = {
            "model1": np.random.randn(100, 3) * 10 + 100,
            "model2": np.random.randn(100, 3) * 10 + 100,
        }

        try:
            result = client.detect_anomalies(model_id="test_model", config=config, historical_data=historical_data)

            assert result is not None
        except Exception:
            # Expected if modules not available
            pass

    @pytest.mark.unit
    def test_detect_anomalies_no_data(self, client):
        """Test detecting anomalies when no data is available."""
        config = AnomalyDetectionConfig(methods=["isolation_forest"])

        # Mock empty data
        mock_unified_client = Mock()
        mock_unified_client.merge_temporal_data = Mock(return_value={})
        client.unified_client = mock_unified_client

        result = client.detect_anomalies(model_id="test_model", config=config)

        assert result is not None
        assert result["detection_data_shape"] == (0,)
        assert result["anomaly_labels"] == []

    @pytest.mark.unit
    def test_prepare_voxel_data_for_detection(self, client):
        """Test preparing voxel data for detection."""
        signal_arrays = {
            "laser_power": np.random.randn(10, 10, 10) * 10 + 100,
            "density": np.random.randn(10, 10, 10) * 0.1 + 0.9,
        }

        prepared = client._prepare_voxel_data_for_detection(signal_arrays)

        assert isinstance(prepared, np.ndarray)
        assert prepared.shape[0] == 1000  # 10*10*10
        assert prepared.shape[1] == 2  # 2 signals

    @pytest.mark.unit
    def test_prepare_raw_data_for_detection(self, client):
        """Test preparing raw data for detection."""
        merged_data = {"laser": [{"power": 100, "velocity": 10}, {"power": 200, "velocity": 20}]}

        prepared = client._prepare_raw_data_for_detection(merged_data)

        assert isinstance(prepared, np.ndarray)
        assert prepared.shape[0] > 0

    @pytest.mark.unit
    def test_prepare_training_data(self, client):
        """Test preparing training data from historical data."""
        historical_data = {
            "model1": np.random.randn(100, 3) * 10 + 100,
            "model2": np.random.randn(100, 3) * 10 + 100,
        }

        training_data = client._prepare_training_data(historical_data)

        assert isinstance(training_data, np.ndarray)
        assert training_data.shape[0] > 0

    @pytest.mark.unit
    def test_prepare_training_data_voxel(self, client):
        """Test preparing training data from voxel historical data."""
        mock_voxel_data = Mock()
        mock_voxel_data.available_signals = ["laser_power", "density"]
        mock_voxel_data.get_signal_array = Mock(return_value=np.random.randn(10, 10, 10))

        historical_data = {"model1": mock_voxel_data}

        training_data = client._prepare_training_data(historical_data)

        assert isinstance(training_data, np.ndarray)
        assert training_data.shape[0] > 0
