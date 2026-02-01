"""
End-to-end tests for multi-source data fusion workflow.

Tests the complete workflow of fusing data from multiple sources
(hatching, laser, CT, ISPM) into a unified voxel domain representation.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

try:
    from am_qadf.query.unified_query_client import UnifiedQueryClient
    from am_qadf.voxel_domain.voxel_domain_client import VoxelDomainClient
    from am_qadf.voxelization.uniform_resolution import VoxelGrid
    from am_qadf.fusion.voxel_fusion import VoxelFusion
    from am_qadf.fusion.fusion_methods import (
        WeightedAverageFusion,
        MedianFusion,
        QualityBasedFusion,
    )
    from am_qadf.fusion.data_fusion import DataFusion, FusionStrategy

    FUSION_AVAILABLE = True
except ImportError:
    FUSION_AVAILABLE = False


@pytest.mark.skipif(not FUSION_AVAILABLE, reason="Fusion components not available")
@pytest.mark.e2e
@pytest.mark.slow
class TestMultiSourceFusion:
    """End-to-end tests for multi-source fusion workflow."""

    @pytest.fixture
    def mock_unified_query_client(self):
        """Create mock unified query client with multiple data sources."""
        client = Mock(spec=UnifiedQueryClient)

        # Mock STL client
        client.stl_client = Mock()
        client.stl_client.get_model_bounding_box.return_value = (
            (-50.0, -50.0, -50.0),
            (50.0, 50.0, 50.0),
        )

        np.random.seed(42)

        # Mock hatching client (minimal data for fast tests)
        client.hatching_client = Mock()
        layers = []
        for layer_idx in range(2):  # Reduced from 10 to 2
            hatches = []
            for hatch_idx in range(3):  # Reduced from 20 to 3
                n_points = 10  # Reduced from 100 to 10
                points = np.random.rand(n_points, 3) * 100.0 - 50.0
                hatches.append(
                    {
                        "points": points.tolist(),
                        "laser_power": float(200.0 + np.random.rand() * 50.0),
                        "scan_speed": float(1000.0 + np.random.rand() * 200.0),
                        "energy_density": float(50.0 + np.random.rand() * 20.0),
                        "hatch_spacing": 0.1,
                        "overlap_percentage": float(50.0 + np.random.rand() * 10.0),
                    }
                )
            layers.append(
                {
                    "layer_index": layer_idx,
                    "hatches": hatches,
                    "z_position": layer_idx * 0.03,
                }
            )
        client.hatching_client.get_layers.return_value = layers

        # Mock laser client (minimal data for fast tests)
        client.laser_client = Mock()
        laser_points = np.random.rand(50, 3) * 100.0 - 50.0  # Reduced from 2000 to 50
        laser_result = Mock()
        laser_result.points = laser_points
        laser_result.signals = {
            "laser_power": np.random.rand(50) * 300.0,
            "scan_speed": np.random.rand(50) * 2000.0,
            "energy_density": np.random.rand(50) * 100.0,
        }
        client.laser_client.query.return_value = laser_result

        # Mock CT client (minimal data for fast tests)
        client.ct_client = Mock()
        ct_points = np.random.rand(30, 3) * 100.0 - 50.0  # Reduced from 1000 to 30
        ct_result = Mock()
        ct_result.points = ct_points
        ct_result.signals = {
            "density": np.random.rand(30) * 1.0 + 4.0,
            "porosity": np.random.rand(30) * 0.1,
        }
        client.ct_client.query.return_value = ct_result

        # Mock ISPM client (minimal data for fast tests)
        client.ispm_client = Mock()
        ispm_points = np.random.rand(50, 3) * 100.0 - 50.0  # Reduced from 5000 to 50
        ispm_result = Mock()
        ispm_result.points = ispm_points
        ispm_result.signals = {
            "temperature": np.random.rand(50) * 1000.0,
            "pressure": np.random.rand(50) * 1.0,
        }
        client.ispm_client.query.return_value = ispm_result

        return client

    @pytest.fixture
    def mock_mongo_client(self):
        """Create mock MongoDB client."""
        from tests.fixtures.mocks import MockMongoClient

        return MockMongoClient()

    @pytest.mark.e2e
    def test_multi_source_fusion_workflow(self, mock_unified_query_client, mock_mongo_client):
        """Test complete multi-source fusion workflow."""
        # Step 1: Create voxel domain client
        voxel_client = VoxelDomainClient(
            unified_query_client=mock_unified_query_client,
            base_resolution=5.0,  # Larger resolution for faster tests
        )

        bbox_min, bbox_max = mock_unified_query_client.stl_client.get_model_bounding_box()

        # Step 2: Create separate grids for each source
        grids = {}
        sources = ["hatching", "laser", "ct", "ispm"]

        for source in sources:
            grid = VoxelGrid(bbox_min=bbox_min, bbox_max=bbox_max, resolution=5.0)  # Larger resolution for faster tests
            mapped_grid = voxel_client.map_signals_to_voxels(
                model_id="test_model_001",
                voxel_grid=grid,
                sources=[source],
                interpolation_method="nearest",
            )
            grids[source] = mapped_grid

        # Step 3: Fuse grids using VoxelFusion (no need to extract signals; fuse_grids takes grids)

        fusion_client = VoxelFusion()

        # Ensure grids are finalized
        for grid in grids.values():
            grid.finalize()

        if len(grids) >= 2:
            grid_list = list(grids.values())
            fused_result = fusion_client.fuse_grids(grids=grid_list, strategy="weighted_average")

            # Assertions
            assert fused_result is not None

    @pytest.mark.e2e
    def test_fusion_with_quality_scores(self, mock_unified_query_client, mock_mongo_client):
        """Test fusion workflow with quality-based weighting."""
        # Step 1: Create grids with different quality sources
        voxel_client = VoxelDomainClient(
            unified_query_client=mock_unified_query_client,
            base_resolution=5.0,  # Larger resolution for faster tests
        )

        bbox_min, bbox_max = mock_unified_query_client.stl_client.get_model_bounding_box()

        # Create grids
        laser_grid = VoxelGrid(bbox_min=bbox_min, bbox_max=bbox_max, resolution=1.0)
        laser_grid = voxel_client.map_signals_to_voxels(
            model_id="test_model_001",
            voxel_grid=laser_grid,
            sources=["laser"],
            interpolation_method="nearest",
        )

        ct_grid = VoxelGrid(bbox_min=bbox_min, bbox_max=bbox_max, resolution=1.0)
        ct_grid = voxel_client.map_signals_to_voxels(
            model_id="test_model_001",
            voxel_grid=ct_grid,
            sources=["ct"],
            interpolation_method="nearest",
        )

        # Step 2: Fuse with weighted average (quality via weights)
        laser_grid.finalize()
        ct_grid.finalize()

        fusion_client = VoxelFusion()
        # Use weights to reflect quality: laser 0.9, ct 0.7 (normalized)
        weights = [0.9 / 1.6, 0.7 / 1.6]
        fused_result = fusion_client.fuse_grids(
            grids=[laser_grid, ct_grid],
            strategy="weighted_average",
            weights=weights,
        )

        assert fused_result is not None

    @pytest.mark.e2e
    def test_fusion_with_different_methods(self, mock_unified_query_client, mock_mongo_client):
        """Test fusion with different fusion methods."""
        # Step 1: Create grids
        voxel_client = VoxelDomainClient(
            unified_query_client=mock_unified_query_client,
            base_resolution=5.0,  # Larger resolution for faster tests
        )

        bbox_min, bbox_max = mock_unified_query_client.stl_client.get_model_bounding_box()

        grid1 = VoxelGrid(bbox_min=bbox_min, bbox_max=bbox_max, resolution=1.0)
        grid1 = voxel_client.map_signals_to_voxels(
            model_id="test_model_001",
            voxel_grid=grid1,
            sources=["laser"],
            interpolation_method="nearest",
        )

        grid2 = VoxelGrid(bbox_min=bbox_min, bbox_max=bbox_max, resolution=1.0)
        grid2 = voxel_client.map_signals_to_voxels(
            model_id="test_model_001",
            voxel_grid=grid2,
            sources=["ispm"],
            interpolation_method="nearest",
        )

        # Step 2: Test different fusion methods (fuse_grids with strategy)
        grid1.finalize()
        grid2.finalize()

        strategies = ["weighted_average", "median"]
        for strategy in strategies:
            fusion_client = VoxelFusion()
            fused_result = fusion_client.fuse_grids(grids=[grid1, grid2], strategy=strategy)
            assert fused_result is not None

    @pytest.mark.e2e
    def test_fusion_with_data_fusion_engine(self, mock_unified_query_client, mock_mongo_client):
        """Test fusion using DataFusion engine directly."""
        # Step 1: Create grids
        voxel_client = VoxelDomainClient(
            unified_query_client=mock_unified_query_client,
            base_resolution=5.0,  # Larger resolution for faster tests
        )

        bbox_min, bbox_max = mock_unified_query_client.stl_client.get_model_bounding_box()

        grid = VoxelGrid(bbox_min=bbox_min, bbox_max=bbox_max, resolution=1.0)
        grid = voxel_client.map_signals_to_voxels(
            model_id="test_model_001",
            voxel_grid=grid,
            sources=["laser", "ct", "ispm"],
            interpolation_method="nearest",
        )

        # Step 2: Extract all signals as arrays; ensure grid is finalized
        grid.finalize()

        signals = {}
        for signal_name in grid.available_signals:
            signal_array = grid.get_signal_array(signal_name, default=0.0)
            signals[signal_name] = signal_array.flatten()

        # Step 3: Use DataFusion engine (from am_qadf.fusion.data_fusion)
        if len(signals) >= 2:
            signal_shapes = [arr.shape for arr in signals.values()]
            if len(set(signal_shapes)) == 1:
                fusion_engine = DataFusion(default_strategy=FusionStrategy.WEIGHTED_AVERAGE)
                for signal_name in signals.keys():
                    fusion_engine.register_source_quality(signal_name, 0.8)
                fused = fusion_engine.fuse_signals(signals=signals, strategy=FusionStrategy.WEIGHTED_AVERAGE)
                assert fused is not None
                assert isinstance(fused, np.ndarray)
                assert len(fused) > 0
                assert fused.shape == signal_shapes[0]
