"""
Unit tests for VoxelDomainClient.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch, call
from am_qadf.voxel_domain.voxel_domain_client import VoxelDomainClient

try:
    from am_qadf.voxelization.voxel_grid import VoxelGrid
    from am_qadf.voxelization.adaptive_resolution import AdaptiveResolutionGrid

    VOXELIZATION_AVAILABLE = True
except ImportError:
    VOXELIZATION_AVAILABLE = False


class MockUnifiedQueryClient:
    """Mock UnifiedQueryClient for testing."""

    def __init__(self):
        self.stl_client = Mock()
        self.hatching_client = Mock()
        self.laser_client = Mock()
        self.ct_client = Mock()
        self.ispm_client = Mock()

    def get_all_data(self, model_id):
        """Return mock data."""
        return {"stl_model": {"bounding_box": {"min": [0.0, 0.0, 0.0], "max": [10.0, 10.0, 10.0]}}}


class MockSTLClient:
    """Mock STL client for testing."""

    def get_model_bounding_box(self, model_id):
        """Return mock bounding box."""
        return ([0.0, 0.0, 0.0], [10.0, 10.0, 10.0])

    def get_model(self, model_id):
        """Return mock STL model data."""
        return {"bounding_box": {"min": [0.0, 0.0, 0.0], "max": [10.0, 10.0, 10.0]}}


class MockQueryResult:
    """Mock QueryResult for testing."""

    def __init__(self, points=None, signals=None):
        self.points = points or [[0, 0, 0], [1, 1, 1]]
        self.signals = signals or {"power": [10.0, 20.0]}


@pytest.mark.skipif(not VOXELIZATION_AVAILABLE, reason="Voxelization module not available")
class TestVoxelDomainClient:
    """Test cases for VoxelDomainClient."""

    @pytest.fixture
    def unified_client(self):
        """Create a mock unified query client."""
        client = MockUnifiedQueryClient()
        client.stl_client = MockSTLClient()
        return client

    @pytest.fixture
    def client(self, unified_client):
        """Create a VoxelDomainClient instance."""
        return VoxelDomainClient(
            unified_query_client=unified_client,
            base_resolution=0.5,
            adaptive=False,
            target_coordinate_system="build_platform",
        )

    @pytest.mark.unit
    def test_initialization(self, unified_client):
        """Test client initialization."""
        client = VoxelDomainClient(unified_query_client=unified_client, base_resolution=0.5, adaptive=True)
        assert client.unified_client == unified_client
        assert client.base_resolution == 0.5
        assert client.adaptive == True
        assert client.target_coordinate_system == "build_platform"

    @pytest.mark.unit
    def test_initialization_defaults(self, unified_client):
        """Test client initialization with defaults."""
        client = VoxelDomainClient(unified_query_client=unified_client)
        assert client.base_resolution == 0.5
        assert client.adaptive == True
        assert client.target_coordinate_system == "build_platform"

    @pytest.mark.unit
    def test_create_voxel_grid_basic(self, client):
        """Test basic voxel grid creation."""
        grid = client.create_voxel_grid(
            model_id="test_model",
            resolution=1.0,
            bbox_min=(0, 0, 0),
            bbox_max=(10, 10, 10),
        )
        assert isinstance(grid, VoxelGrid)
        assert grid.resolution == 1.0
        assert np.array_equal(grid.bbox_min, [0, 0, 0])
        assert np.array_equal(grid.bbox_max, [10, 10, 10])

    @pytest.mark.unit
    def test_create_voxel_grid_with_stl_bbox(self, client):
        """Test voxel grid creation using STL bounding box."""
        grid = client.create_voxel_grid(model_id="test_model")
        assert isinstance(grid, VoxelGrid)
        assert np.array_equal(grid.bbox_min, [0.0, 0.0, 0.0])
        assert np.array_equal(grid.bbox_max, [10.0, 10.0, 10.0])

    @pytest.mark.unit
    def test_create_voxel_grid_adaptive(self, unified_client):
        """Test adaptive resolution grid creation."""
        client = VoxelDomainClient(unified_query_client=unified_client, adaptive=True)
        grid = client.create_voxel_grid(model_id="test_model", bbox_min=(0, 0, 0), bbox_max=(10, 10, 10))
        assert isinstance(grid, AdaptiveResolutionGrid)

    @pytest.mark.unit
    def test_create_voxel_grid_override_adaptive(self, client):
        """Test overriding adaptive setting."""
        grid = client.create_voxel_grid(
            model_id="test_model",
            adaptive=True,
            bbox_min=(0, 0, 0),
            bbox_max=(10, 10, 10),
        )
        assert isinstance(grid, AdaptiveResolutionGrid)

    @pytest.mark.unit
    def test_create_voxel_grid_uses_base_resolution(self, client):
        """Test that base resolution is used when not specified."""
        grid = client.create_voxel_grid(model_id="test_model", bbox_min=(0, 0, 0), bbox_max=(10, 10, 10))
        assert grid.resolution == client.base_resolution

    @pytest.mark.unit
    def test_create_voxel_grid_stl_bbox_fallback(self, unified_client):
        """Test STL bounding box fallback when get_model_bounding_box fails."""
        unified_client.stl_client.get_model_bounding_box = Mock(side_effect=Exception("Error"))
        unified_client.stl_client.get_model = Mock(
            return_value={"bounding_box": {"min": [0.0, 0.0, 0.0], "max": [10.0, 10.0, 10.0]}}
        )
        client = VoxelDomainClient(
            unified_query_client=unified_client,
            adaptive=False,  # Ensure we get a VoxelGrid, not AdaptiveResolutionGrid
        )

        grid = client.create_voxel_grid(model_id="test_model")
        assert isinstance(grid, VoxelGrid)
        # Should fall back to get_model
        assert unified_client.stl_client.get_model.called

    @pytest.mark.unit
    def test_create_voxel_grid_no_stl_client(self, unified_client):
        """Test grid creation when STL client is not available."""
        unified_client.stl_client = None
        unified_client.get_all_data = Mock(return_value={})  # Return empty dict
        client = VoxelDomainClient(unified_query_client=unified_client)

        with pytest.raises(ValueError, match="Could not retrieve bounding box"):
            client.create_voxel_grid(model_id="test_model")

    @pytest.mark.unit
    def test_map_signals_to_voxels_sequential(self, client):
        """Test sequential signal mapping."""
        grid = client.create_voxel_grid(model_id="test_model", bbox_min=(0, 0, 0), bbox_max=(10, 10, 10))

        with patch.object(client, "_map_source_data") as mock_map:
            result = client.map_signals_to_voxels(
                model_id="test_model",
                voxel_grid=grid,
                sources=["laser"],
                use_parallel_sources=False,
            )
            assert result == grid
            assert mock_map.called

    @pytest.mark.unit
    def test_map_signals_to_voxels_parallel(self, client):
        """Test parallel signal mapping."""
        grid = client.create_voxel_grid(model_id="test_model", bbox_min=(0, 0, 0), bbox_max=(10, 10, 10))

        with patch.object(client, "_map_source_data") as mock_map:
            result = client.map_signals_to_voxels(
                model_id="test_model",
                voxel_grid=grid,
                sources=["laser", "hatching"],
                use_parallel_sources=True,
            )
            assert result == grid
            # Should be called for each source
            assert mock_map.call_count == 2

    @pytest.mark.unit
    def test_map_signals_to_voxels_all_sources(self, client):
        """Test mapping signals from all sources."""
        grid = client.create_voxel_grid(model_id="test_model", bbox_min=(0, 0, 0), bbox_max=(10, 10, 10))

        with patch.object(client, "_map_source_data") as mock_map:
            client.map_signals_to_voxels(
                model_id="test_model",
                voxel_grid=grid,
                sources=None,  # Should default to all sources
            )
            # Should be called for all 4 sources
            assert mock_map.call_count == 4

    @pytest.mark.unit
    def test_map_signals_to_voxels_finalize(self, client):
        """Test that adaptive grid is finalized after mapping."""
        grid = client.create_voxel_grid(
            model_id="test_model",
            adaptive=True,
            bbox_min=(0, 0, 0),
            bbox_max=(10, 10, 10),
        )
        grid.finalize = Mock()

        with patch.object(client, "_map_source_data"):
            client.map_signals_to_voxels(model_id="test_model", voxel_grid=grid, sources=["laser"])
            grid.finalize.assert_called_once()

    @pytest.mark.unit
    def test_map_hatching_data(self, client):
        """Test hatching data mapping."""
        grid = client.create_voxel_grid(model_id="test_model", bbox_min=(0, 0, 0), bbox_max=(10, 10, 10))

        # Mock hatching layers
        client.unified_client.hatching_client.get_layers = Mock(
            return_value=[
                {
                    "hatches": [
                        {
                            "points": [[0, 0, 0], [1, 1, 1]],
                            "laser_power": 100.0,
                            "scan_speed": 1000.0,
                        }
                    ]
                }
            ]
        )

        with patch("am_qadf.voxel_domain.voxel_domain_client.interpolate_hatching_paths") as mock_interp:
            client._map_hatching_data(
                model_id="test_model",
                voxel_grid=grid,
                spatial_query=None,
                temporal_query=None,
                interpolation_method="nearest",
            )
            mock_interp.assert_called_once()

    @pytest.mark.unit
    def test_map_hatching_data_no_client(self, client):
        """Test hatching mapping when client is not available."""
        client.unified_client.hatching_client = None
        grid = client.create_voxel_grid(model_id="test_model", bbox_min=(0, 0, 0), bbox_max=(10, 10, 10))

        # Should not raise error, just log warning
        client._map_hatching_data(
            model_id="test_model",
            voxel_grid=grid,
            spatial_query=None,
            temporal_query=None,
        )

    @pytest.mark.unit
    def test_map_hatching_data_no_layers(self, client):
        """Test hatching mapping when no layers found."""
        client.unified_client.hatching_client.get_layers = Mock(return_value=[])
        grid = client.create_voxel_grid(model_id="test_model", bbox_min=(0, 0, 0), bbox_max=(10, 10, 10))

        # Should not raise error
        client._map_hatching_data(
            model_id="test_model",
            voxel_grid=grid,
            spatial_query=None,
            temporal_query=None,
        )

    @pytest.mark.unit
    def test_map_laser_data(self, client):
        """Test laser data mapping."""
        grid = client.create_voxel_grid(model_id="test_model", bbox_min=(0, 0, 0), bbox_max=(10, 10, 10))

        # Mock laser query result
        client.unified_client.laser_client.query = Mock(return_value=MockQueryResult())

        with patch("am_qadf.voxel_domain.voxel_domain_client.interpolate_to_voxels") as mock_interp:
            client._map_laser_data(
                model_id="test_model",
                voxel_grid=grid,
                spatial_query=None,
                temporal_query=None,
                interpolation_method="nearest",
            )
            mock_interp.assert_called_once()

    @pytest.mark.unit
    def test_map_laser_data_no_client(self, client):
        """Test laser mapping when client is not available."""
        client.unified_client.laser_client = None
        grid = client.create_voxel_grid(model_id="test_model", bbox_min=(0, 0, 0), bbox_max=(10, 10, 10))

        # Should not raise error
        client._map_laser_data(
            model_id="test_model",
            voxel_grid=grid,
            spatial_query=None,
            temporal_query=None,
        )

    @pytest.mark.unit
    def test_map_laser_data_no_data(self, client):
        """Test laser mapping when no data found."""
        client.unified_client.laser_client.query = Mock(return_value=MockQueryResult(points=[], signals={}))
        grid = client.create_voxel_grid(model_id="test_model", bbox_min=(0, 0, 0), bbox_max=(10, 10, 10))

        # Should not raise error
        client._map_laser_data(
            model_id="test_model",
            voxel_grid=grid,
            spatial_query=None,
            temporal_query=None,
        )

    @pytest.mark.unit
    def test_map_ct_data(self, client):
        """Test CT data mapping."""
        grid = client.create_voxel_grid(model_id="test_model", bbox_min=(0, 0, 0), bbox_max=(10, 10, 10))

        # Mock CT query result
        client.unified_client.ct_client.query = Mock(return_value=MockQueryResult())

        with patch("am_qadf.voxel_domain.voxel_domain_client.interpolate_to_voxels") as mock_interp:
            client._map_ct_data(
                model_id="test_model",
                voxel_grid=grid,
                spatial_query=None,
                interpolation_method="nearest",
            )
            mock_interp.assert_called_once()

    @pytest.mark.unit
    def test_map_ct_data_no_client(self, client):
        """Test CT mapping when client is not available."""
        client.unified_client.ct_client = None
        grid = client.create_voxel_grid(model_id="test_model", bbox_min=(0, 0, 0), bbox_max=(10, 10, 10))

        # Should not raise error
        client._map_ct_data(model_id="test_model", voxel_grid=grid, spatial_query=None)

    @pytest.mark.unit
    def test_map_ispm_data(self, client):
        """Test ISPM data mapping."""
        grid = client.create_voxel_grid(model_id="test_model", bbox_min=(0, 0, 0), bbox_max=(10, 10, 10))

        # Mock ISPM query result
        client.unified_client.ispm_client.query = Mock(return_value=MockQueryResult())

        with patch("am_qadf.voxel_domain.voxel_domain_client.interpolate_to_voxels") as mock_interp:
            client._map_ispm_data(
                model_id="test_model",
                voxel_grid=grid,
                spatial_query=None,
                temporal_query=None,
                interpolation_method="nearest",
            )
            mock_interp.assert_called_once()

    @pytest.mark.unit
    def test_map_ispm_data_no_client(self, client):
        """Test ISPM mapping when client is not available."""
        client.unified_client.ispm_client = None
        grid = client.create_voxel_grid(model_id="test_model", bbox_min=(0, 0, 0), bbox_max=(10, 10, 10))

        # Should not raise error
        client._map_ispm_data(
            model_id="test_model",
            voxel_grid=grid,
            spatial_query=None,
            temporal_query=None,
        )

    @pytest.mark.unit
    def test_map_source_data_unknown_source(self, client):
        """Test mapping unknown source."""
        grid = client.create_voxel_grid(model_id="test_model", bbox_min=(0, 0, 0), bbox_max=(10, 10, 10))

        # Should log warning but not raise error
        client._map_source_data(
            source="unknown_source",
            model_id="test_model",
            voxel_grid=grid,
            spatial_query=None,
            temporal_query=None,
            interpolation_method="nearest",
            use_parallel_interpolation=False,
            use_spark=False,
            spark_session=None,
            max_workers=None,
        )

    @pytest.mark.unit
    def test_get_voxel_statistics(self, client):
        """Test getting voxel statistics."""
        grid = client.create_voxel_grid(model_id="test_model", bbox_min=(0, 0, 0), bbox_max=(10, 10, 10))

        # Mock get_statistics method
        grid.get_statistics = Mock(return_value={"test": "stats"})

        stats = client.get_voxel_statistics(grid)
        assert stats == {"test": "stats"}
        grid.get_statistics.assert_called_once()

    @pytest.mark.unit
    def test_get_voxel_statistics_fallback(self, client):
        """Test getting statistics when grid doesn't have get_statistics method."""
        grid = client.create_voxel_grid(model_id="test_model", bbox_min=(0, 0, 0), bbox_max=(10, 10, 10))

        # Create a mock grid without get_statistics
        class MockGrid:
            def __init__(self):
                self.available_signals = {"power", "velocity"}
                self.resolution = 1.0
                self.bbox_min = np.array([0, 0, 0])
                self.bbox_max = np.array([10, 10, 10])

        mock_grid = MockGrid()

        stats = client.get_voxel_statistics(mock_grid)
        assert "available_signals" in stats
        assert "resolution" in stats
        assert "bbox_min" in stats
        assert "bbox_max" in stats

    @pytest.mark.unit
    def test_map_signals_error_handling(self, client):
        """Test error handling in signal mapping."""
        grid = client.create_voxel_grid(model_id="test_model", bbox_min=(0, 0, 0), bbox_max=(10, 10, 10))

        # Mock _map_source_data to raise error
        client._map_source_data = Mock(side_effect=Exception("Test error"))

        # Should handle error gracefully
        result = client.map_signals_to_voxels(model_id="test_model", voxel_grid=grid, sources=["laser"])
        assert result == grid
