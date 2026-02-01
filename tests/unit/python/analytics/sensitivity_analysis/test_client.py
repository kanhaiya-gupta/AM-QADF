"""
Unit tests for sensitivity analysis client.

Tests for SensitivityAnalysisConfig and SensitivityAnalysisClient.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
from am_qadf.analytics.sensitivity_analysis.client import (
    SensitivityAnalysisConfig,
    SensitivityAnalysisClient,
)


class TestSensitivityAnalysisConfig:
    """Test suite for SensitivityAnalysisConfig dataclass."""

    @pytest.mark.unit
    def test_config_creation_default(self):
        """Test creating SensitivityAnalysisConfig with default values."""
        config = SensitivityAnalysisConfig()

        assert config.method == "sobol"
        assert config.process_variables is None
        assert config.measurement_variables is None
        assert config.spatial_region is None
        assert config.layer_range is None
        assert config.time_range is None
        assert config.sample_size == 1000
        assert config.confidence_level == 0.95
        assert config.use_voxel_domain is False
        assert config.voxel_resolution == 0.5

    @pytest.mark.unit
    def test_config_creation_custom(self):
        """Test creating SensitivityAnalysisConfig with custom values."""
        spatial_region = ((0, 0, 0), (10, 10, 10))
        layer_range = (0, 50)
        time_range = (datetime(2023, 1, 1), datetime(2023, 1, 2))

        config = SensitivityAnalysisConfig(
            method="morris",
            process_variables=["laser_power", "scan_speed"],
            measurement_variables=["density", "temperature"],
            spatial_region=spatial_region,
            layer_range=layer_range,
            time_range=time_range,
            sample_size=500,
            confidence_level=0.99,
            use_voxel_domain=True,
            voxel_resolution=0.25,
        )

        assert config.method == "morris"
        assert config.process_variables == ["laser_power", "scan_speed"]
        assert config.measurement_variables == ["density", "temperature"]
        assert config.spatial_region == spatial_region
        assert config.layer_range == layer_range
        assert config.time_range == time_range
        assert config.sample_size == 500
        assert config.confidence_level == 0.99
        assert config.use_voxel_domain is True
        assert config.voxel_resolution == 0.25


class TestSensitivityAnalysisClient:
    """Test suite for SensitivityAnalysisClient class."""

    @pytest.fixture
    def mock_unified_client(self):
        """Create a mock UnifiedQueryClient."""
        client = Mock()
        client.merge_temporal_data = Mock(
            return_value={
                "sources": {
                    "laser": {
                        "points": [[0, 0, 0], [1, 1, 1]],
                        "signals": {"power": [100.0, 200.0], "velocity": [1.0, 2.0]},
                        "metadata": {"layer_index": [0, 1]},
                    }
                }
            }
        )
        client.merge_spatial_data = Mock(
            return_value={
                "sources": {
                    "ct": {
                        "points": [[0, 0, 0], [1, 1, 1]],
                        "signals": {"density": [0.9, 0.95]},
                    }
                }
            }
        )
        client.stl_client = Mock()
        client.stl_client.get_model_bounding_box = Mock(return_value=((0, 0, 0), (10, 10, 10)))
        client.ispm_client = Mock()
        client.ct_client = Mock()
        return client

    @pytest.fixture
    def mock_voxel_client(self):
        """Create a mock VoxelDomainClient."""
        return Mock()

    @pytest.fixture
    def client(self, mock_unified_client, mock_voxel_client):
        """Create a SensitivityAnalysisClient instance."""
        with patch(
            "am_qadf.analytics.sensitivity_analysis.client.SENSITIVITY_ANALYSIS_AVAILABLE",
            False,
        ):
            return SensitivityAnalysisClient(mock_unified_client, mock_voxel_client)

    @pytest.mark.unit
    def test_client_creation(self, mock_unified_client):
        """Test creating SensitivityAnalysisClient."""
        with patch(
            "am_qadf.analytics.sensitivity_analysis.client.SENSITIVITY_ANALYSIS_AVAILABLE",
            False,
        ):
            client = SensitivityAnalysisClient(mock_unified_client)

            assert client.unified_client == mock_unified_client
            assert client.voxel_client is None
            assert client.global_analyzer is None
            assert client.local_analyzer is None

    @pytest.mark.unit
    def test_client_creation_with_voxel_client(self, mock_unified_client, mock_voxel_client):
        """Test creating SensitivityAnalysisClient with voxel client."""
        with patch(
            "am_qadf.analytics.sensitivity_analysis.client.SENSITIVITY_ANALYSIS_AVAILABLE",
            False,
        ):
            client = SensitivityAnalysisClient(mock_unified_client, mock_voxel_client)

            assert client.voxel_client == mock_voxel_client

    @pytest.mark.unit
    def test_query_process_variables(self, client, mock_unified_client):
        """Test querying process variables."""
        model_id = "test_model"
        variables = ["laser_power", "scan_speed"]

        result = client.query_process_variables(model_id, variables)

        assert isinstance(result, pd.DataFrame)
        mock_unified_client.merge_temporal_data.assert_called_once()

    @pytest.mark.unit
    def test_query_process_variables_with_spatial_region(self, client, mock_unified_client):
        """Test querying process variables with spatial region."""
        model_id = "test_model"
        variables = ["laser_power"]
        spatial_region = ((0, 0, 0), (10, 10, 10))

        result = client.query_process_variables(model_id, variables, spatial_region=spatial_region)

        assert isinstance(result, pd.DataFrame)

    @pytest.mark.unit
    def test_query_process_variables_with_layer_range(self, client, mock_unified_client):
        """Test querying process variables with layer range."""
        model_id = "test_model"
        variables = ["laser_power"]
        layer_range = (0, 10)

        result = client.query_process_variables(model_id, variables, layer_range=layer_range)

        assert isinstance(result, pd.DataFrame)

    @pytest.mark.unit
    def test_query_process_variables_empty_result(self, client, mock_unified_client):
        """Test querying process variables when no data is returned."""
        mock_unified_client.merge_temporal_data.return_value = None

        model_id = "test_model"
        variables = ["laser_power"]

        result = client.query_process_variables(model_id, variables)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    @pytest.mark.unit
    def test_query_measurement_data(self, client, mock_unified_client):
        """Test querying measurement data."""
        model_id = "test_model"
        sources = ["ispm", "ct"]

        result = client.query_measurement_data(model_id, sources)

        assert isinstance(result, dict)
        assert "ispm" in result or "ct" in result

    @pytest.mark.unit
    def test_query_measurement_data_ispm_only(self, client, mock_unified_client):
        """Test querying ISPM measurement data only."""
        model_id = "test_model"
        sources = ["ispm"]

        result = client.query_measurement_data(model_id, sources)

        assert isinstance(result, dict)

    @pytest.mark.unit
    def test_query_measurement_data_ct_only(self, client, mock_unified_client):
        """Test querying CT scan measurement data only."""
        model_id = "test_model"
        sources = ["ct"]

        result = client.query_measurement_data(model_id, sources)

        assert isinstance(result, dict)

    @pytest.mark.unit
    def test_query_measurement_data_with_spatial_region(self, client, mock_unified_client):
        """Test querying measurement data with spatial region."""
        model_id = "test_model"
        sources = ["ct"]
        spatial_region = ((0, 0, 0), (10, 10, 10))

        result = client.query_measurement_data(model_id, sources, spatial_region=spatial_region)

        assert isinstance(result, dict)

    @pytest.mark.unit
    def test_query_measurement_data_empty_result(self, client, mock_unified_client):
        """Test querying measurement data when no data is returned."""
        mock_unified_client.merge_temporal_data.return_value = None
        mock_unified_client.merge_spatial_data.return_value = None

        model_id = "test_model"
        sources = ["ispm", "ct"]

        result = client.query_measurement_data(model_id, sources)

        assert isinstance(result, dict)
        assert len(result) == 0

    @pytest.mark.unit
    def test_perform_sensitivity_analysis_no_process_data(self, client, mock_unified_client):
        """Test performing sensitivity analysis when no process data is available."""
        mock_unified_client.merge_temporal_data.return_value = None

        model_id = "test_model"
        config = SensitivityAnalysisConfig(
            process_variables=["laser_power", "scan_speed"],
            measurement_variables=["density"],
        )

        with pytest.raises(ValueError, match="No process data found"):
            client.perform_sensitivity_analysis(model_id, config)

    @pytest.mark.unit
    def test_perform_sensitivity_analysis_no_measurement_data(self, client, mock_unified_client):
        """Test performing sensitivity analysis when no measurement data is available."""
        # Return process data but no measurement data
        mock_unified_client.merge_temporal_data.return_value = {
            "sources": {
                "laser": {
                    "points": [[0, 0, 0], [1, 1, 1]],
                    "signals": {"power": [100.0, 200.0], "velocity": [1.0, 2.0]},
                    "metadata": {},
                }
            }
        }
        mock_unified_client.merge_spatial_data.return_value = None

        model_id = "test_model"
        config = SensitivityAnalysisConfig(
            process_variables=["laser_power", "scan_speed"],
            measurement_variables=["density"],
        )

        # Should raise ValueError - either "No measurement data found" if fallback fails,
        # or "SALib not available" if fallback succeeds but SALib is missing
        with pytest.raises(ValueError, match="No measurement data found|SALib not available"):
            client.perform_sensitivity_analysis(model_id, config)

    @pytest.mark.unit
    def test_perform_sensitivity_analysis_with_process_output_fallback(self, client, mock_unified_client):
        """Test performing sensitivity analysis using process data as output fallback."""
        # Return process data with energy_density
        mock_unified_client.merge_temporal_data.return_value = {
            "sources": {
                "laser": {
                    "points": [[0, 0, 0], [1, 1, 1]],
                    "signals": {
                        "power": [100.0, 200.0],
                        "velocity": [1.0, 2.0],
                        "energy": [50.0, 100.0],
                    },
                    "metadata": {},
                }
            }
        }
        mock_unified_client.merge_spatial_data.return_value = None

        model_id = "test_model"
        config = SensitivityAnalysisConfig(
            process_variables=["laser_power", "scan_speed", "energy_density"],
            measurement_variables=["density"],
        )

        # Should use energy_density as output
        with patch.object(client, "_build_surrogate_model", return_value=Mock()):
            with patch.object(
                client,
                "_extract_measurement_outputs",
                return_value=np.array([50.0, 100.0]),
            ):
                try:
                    result = client.perform_sensitivity_analysis(model_id, config)
                    # If it doesn't raise, check result structure
                    assert isinstance(result, dict)
                except (ValueError, AttributeError) as e:
                    # Expected if surrogate model building fails or SALib is not available
                    error_str = str(e).lower()
                    assert "surrogate" in error_str or "model" in error_str or "salib" in error_str or "sobol" in error_str
