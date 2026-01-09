"""
Integration tests for signal mapping pipeline.

Tests the complete workflow: query → transform → voxelize → interpolate → store
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from am_qadf.voxel_domain.voxel_domain_client import VoxelDomainClient
from am_qadf.query.unified_query_client import UnifiedQueryClient


class MockUnifiedQueryClient:
    """Mock unified query client for integration testing."""

    def __init__(self):
        self.stl_client = Mock()
        self.hatching_client = Mock()
        self.laser_client = Mock()
        self.ct_client = Mock()
        self.ispm_client = Mock()

        # Setup STL client
        self.stl_client.get_model_bounding_box = Mock(return_value=([0.0, 0.0, 0.0], [10.0, 10.0, 10.0]))

        # Setup hatching client
        self.hatching_client.get_layers = Mock(
            return_value=[
                {
                    "hatches": [
                        {
                            "points": [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
                            "laser_power": 200.0,
                            "scan_speed": 1000.0,
                        }
                    ]
                }
            ]
        )

        # Setup laser client
        class MockQueryResult:
            def __init__(self):
                self.points = [[0, 0, 0], [1, 1, 1]]
                self.signals = {
                    "laser_power": [250.0, 250.0],
                    "laser_speed": [150.0, 150.0],
                }

        self.laser_client.query = Mock(return_value=MockQueryResult())


@pytest.mark.integration
class TestSignalMappingPipeline:
    """Integration tests for signal mapping pipeline."""

    @pytest.fixture
    def unified_client(self):
        """Create mock unified query client."""
        return MockUnifiedQueryClient()

    @pytest.fixture
    def voxel_domain_client(self, unified_client):
        """Create voxel domain client."""
        return VoxelDomainClient(unified_query_client=unified_client, base_resolution=1.0, adaptive=False)

    @pytest.mark.integration
    def test_signal_mapping_pipeline_complete(self, voxel_domain_client):
        """Test complete signal mapping pipeline."""
        # Step 1: Create voxel grid
        grid = voxel_domain_client.create_voxel_grid(
            model_id="test_model",
            resolution=1.0,
            bbox_min=(0, 0, 0),
            bbox_max=(10, 10, 10),
        )
        assert grid is not None

        # Step 2: Map signals from multiple sources
        result = voxel_domain_client.map_signals_to_voxels(
            model_id="test_model",
            voxel_grid=grid,
            sources=["hatching", "laser"],
            interpolation_method="nearest",
        )

        # Step 3: Verify signals are mapped
        assert result is not None
        assert hasattr(result, "available_signals")

    @pytest.mark.integration
    def test_signal_mapping_with_stl_bbox(self, voxel_domain_client):
        """Test signal mapping using STL bounding box."""
        # Create grid using STL bbox
        grid = voxel_domain_client.create_voxel_grid(model_id="test_model")
        assert grid is not None
        assert np.array_equal(grid.bbox_min, [0.0, 0.0, 0.0])
        assert np.array_equal(grid.bbox_max, [10.0, 10.0, 10.0])

        # Map signals
        result = voxel_domain_client.map_signals_to_voxels(model_id="test_model", voxel_grid=grid, sources=["hatching"])
        assert result is not None

    @pytest.mark.integration
    def test_signal_mapping_all_sources(self, voxel_domain_client):
        """Test mapping signals from all available sources."""
        grid = voxel_domain_client.create_voxel_grid(model_id="test_model", bbox_min=(0, 0, 0), bbox_max=(10, 10, 10))

        result = voxel_domain_client.map_signals_to_voxels(model_id="test_model", voxel_grid=grid, sources=None)  # All sources
        assert result is not None

    @pytest.mark.integration
    def test_signal_mapping_parallel_sources(self, voxel_domain_client):
        """Test parallel source processing."""
        grid = voxel_domain_client.create_voxel_grid(model_id="test_model", bbox_min=(0, 0, 0), bbox_max=(10, 10, 10))

        result = voxel_domain_client.map_signals_to_voxels(
            model_id="test_model",
            voxel_grid=grid,
            sources=["hatching", "laser"],
            use_parallel_sources=True,
        )
        assert result is not None

    @pytest.mark.integration
    def test_signal_mapping_different_interpolation_methods(self, voxel_domain_client):
        """Test signal mapping with different interpolation methods."""
        grid = voxel_domain_client.create_voxel_grid(model_id="test_model", bbox_min=(0, 0, 0), bbox_max=(10, 10, 10))

        methods = ["nearest", "linear", "idw"]
        for method in methods:
            result = voxel_domain_client.map_signals_to_voxels(
                model_id="test_model",
                voxel_grid=grid,
                sources=["laser"],
                interpolation_method=method,
            )
            assert result is not None

    @pytest.mark.integration
    def test_signal_mapping_adaptive_resolution(self, unified_client):
        """Test signal mapping with adaptive resolution grid."""
        client = VoxelDomainClient(unified_query_client=unified_client, adaptive=True)

        grid = client.create_voxel_grid(model_id="test_model", bbox_min=(0, 0, 0), bbox_max=(10, 10, 10))

        result = client.map_signals_to_voxels(model_id="test_model", voxel_grid=grid, sources=["hatching"])
        assert result is not None
        assert hasattr(result, "finalize")  # Adaptive grids have finalize method

    @pytest.mark.integration
    def test_signal_mapping_with_spatial_query(self, voxel_domain_client):
        """Test signal mapping with spatial query filter."""
        grid = voxel_domain_client.create_voxel_grid(model_id="test_model", bbox_min=(0, 0, 0), bbox_max=(10, 10, 10))

        # Create spatial query
        class MockSpatialQuery:
            def __init__(self):
                self.bbox_min = [0, 0, 0]
                self.bbox_max = [5, 5, 5]

        result = voxel_domain_client.map_signals_to_voxels(
            model_id="test_model",
            voxel_grid=grid,
            sources=["laser"],
            spatial_query=MockSpatialQuery(),
        )
        assert result is not None

    @pytest.mark.integration
    def test_signal_mapping_with_temporal_query(self, voxel_domain_client):
        """Test signal mapping with temporal query filter."""
        grid = voxel_domain_client.create_voxel_grid(model_id="test_model", bbox_min=(0, 0, 0), bbox_max=(10, 10, 10))

        # Create temporal query
        class MockTemporalQuery:
            def __init__(self):
                self.layer_range = (0, 10)

        result = voxel_domain_client.map_signals_to_voxels(
            model_id="test_model",
            voxel_grid=grid,
            sources=["hatching"],
            temporal_query=MockTemporalQuery(),
        )
        assert result is not None

    @pytest.mark.integration
    def test_signal_mapping_error_recovery(self, voxel_domain_client):
        """Test error recovery in signal mapping pipeline."""
        # Make one source fail
        voxel_domain_client.unified_client.hatching_client.get_layers = Mock(side_effect=Exception("Hatching error"))

        grid = voxel_domain_client.create_voxel_grid(model_id="test_model", bbox_min=(0, 0, 0), bbox_max=(10, 10, 10))

        # Should handle error gracefully and continue with other sources
        result = voxel_domain_client.map_signals_to_voxels(
            model_id="test_model", voxel_grid=grid, sources=["hatching", "laser"]
        )
        assert result is not None  # Should still succeed with laser source

    @pytest.mark.integration
    def test_signal_mapping_statistics(self, voxel_domain_client):
        """Test getting statistics after signal mapping."""
        grid = voxel_domain_client.create_voxel_grid(model_id="test_model", bbox_min=(0, 0, 0), bbox_max=(10, 10, 10))

        result = voxel_domain_client.map_signals_to_voxels(model_id="test_model", voxel_grid=grid, sources=["hatching"])

        stats = voxel_domain_client.get_voxel_statistics(result)
        assert stats is not None
        assert "available_signals" in stats or "resolution" in stats
