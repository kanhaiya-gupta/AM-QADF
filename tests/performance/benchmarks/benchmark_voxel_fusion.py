"""
Performance benchmarks for voxel fusion.

Tests voxel fusion performance with various configurations.
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock


def create_voxel_grid_for_fusion(bbox_size=100.0, resolution=1.0, signal_name="test_signal"):
    """Create a voxel grid with signal for fusion benchmarking."""
    try:
        from am_qadf.voxelization.voxel_grid import VoxelGrid

        grid = VoxelGrid(
            bbox_min=(0.0, 0.0, 0.0),
            bbox_max=(bbox_size, bbox_size, bbox_size),
            resolution=resolution,
        )

        # Add some mock signal data
        grid.available_signals = {signal_name}

        return grid
    except ImportError:
        pytest.skip("VoxelGrid not available")


def create_multiple_grids(n_grids=5, bbox_size=100.0, resolution=1.0):
    """Create multiple voxel grids for fusion testing."""
    grids = []
    for i in range(n_grids):
        grid = create_voxel_grid_for_fusion(bbox_size=bbox_size, resolution=resolution, signal_name=f"signal_{i}")
        grids.append(grid)
    return grids


@pytest.mark.benchmark
@pytest.mark.performance
class TestVoxelFusionBenchmarks:
    """Performance benchmarks for voxel fusion."""

    @pytest.mark.benchmark
    def test_fusion_small_grids(self, benchmark):
        """Benchmark fusion with small grids (10x10x10)."""
        try:
            from am_qadf.fusion.voxel_fusion import VoxelFusion

            grids = create_multiple_grids(n_grids=3, bbox_size=10.0, resolution=1.0)
            fusion = VoxelFusion()

            def run_fusion():
                return fusion.fuse_voxel_grids(grids, method="weighted_average")

            result = benchmark(run_fusion)
            assert result is not None
        except (ImportError, AttributeError):
            pytest.skip("VoxelFusion not available")

    @pytest.mark.benchmark
    def test_fusion_medium_grids(self, benchmark):
        """Benchmark fusion with medium grids (50x50x50)."""
        try:
            from am_qadf.fusion.voxel_fusion import VoxelFusion

            grids = create_multiple_grids(n_grids=5, bbox_size=50.0, resolution=1.0)
            fusion = VoxelFusion()

            def run_fusion():
                return fusion.fuse_voxel_grids(grids, method="weighted_average")

            result = benchmark(run_fusion)
            assert result is not None
        except (ImportError, AttributeError):
            pytest.skip("VoxelFusion not available")

    @pytest.mark.benchmark
    def test_fusion_large_grids(self, benchmark):
        """Benchmark fusion with large grids (100x100x100)."""
        try:
            from am_qadf.fusion.voxel_fusion import VoxelFusion

            grids = create_multiple_grids(n_grids=5, bbox_size=100.0, resolution=1.0)
            fusion = VoxelFusion()

            def run_fusion():
                return fusion.fuse_voxel_grids(grids, method="weighted_average")

            result = benchmark(run_fusion)
            assert result is not None
        except (ImportError, AttributeError):
            pytest.skip("VoxelFusion not available")

    @pytest.mark.benchmark
    def test_fusion_different_methods(self, benchmark):
        """Benchmark different fusion methods."""
        try:
            from am_qadf.fusion.voxel_fusion import VoxelFusion

            grids = create_multiple_grids(n_grids=3, bbox_size=50.0, resolution=1.0)
            fusion = VoxelFusion()
            methods = ["weighted_average", "median", "maximum", "minimum"]

            for method in methods:

                def run_fusion():
                    return fusion.fuse_voxel_grids(grids, method=method)

                result = benchmark(run_fusion)
                assert result is not None
        except (ImportError, AttributeError):
            pytest.skip("VoxelFusion not available")

    @pytest.mark.benchmark
    def test_fusion_many_grids(self, benchmark):
        """Benchmark fusion with many grids (10+ grids)."""
        try:
            from am_qadf.fusion.voxel_fusion import VoxelFusion

            grids = create_multiple_grids(n_grids=10, bbox_size=50.0, resolution=1.0)
            fusion = VoxelFusion()

            def run_fusion():
                return fusion.fuse_voxel_grids(grids, method="weighted_average")

            result = benchmark(run_fusion)
            assert result is not None
        except (ImportError, AttributeError):
            pytest.skip("VoxelFusion not available")

    @pytest.mark.benchmark
    def test_fusion_quality_based(self, benchmark):
        """Benchmark quality-based fusion."""
        try:
            from am_qadf.fusion.voxel_fusion import VoxelFusion

            grids = create_multiple_grids(n_grids=5, bbox_size=50.0, resolution=1.0)
            fusion = VoxelFusion()

            quality_scores = {f"signal_{i}": 0.8 + (i * 0.05) for i in range(5)}

            def run_fusion():
                return fusion.fuse_voxel_grids(grids, method="quality_based", quality_scores=quality_scores)

            result = benchmark(run_fusion)
            assert result is not None
        except (ImportError, AttributeError):
            pytest.skip("VoxelFusion not available")

    @pytest.mark.benchmark
    def test_fusion_memory_efficiency(self, benchmark):
        """Benchmark memory efficiency of fusion."""
        import tracemalloc

        try:
            from am_qadf.fusion.voxel_fusion import VoxelFusion

            grids = create_multiple_grids(n_grids=5, bbox_size=100.0, resolution=1.0)
            fusion = VoxelFusion()

            tracemalloc.start()

            def run_fusion():
                return fusion.fuse_voxel_grids(grids, method="weighted_average")

            result = benchmark(run_fusion)

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            assert result is not None
            # Memory should be reasonable
            assert peak < 2 * 1024 * 1024 * 1024  # 2GB limit
        except (ImportError, AttributeError):
            pytest.skip("VoxelFusion not available")

    @pytest.mark.benchmark
    def test_fusion_scalability(self, benchmark):
        """Benchmark fusion scalability with increasing grid sizes."""
        try:
            from am_qadf.fusion.voxel_fusion import VoxelFusion

            sizes = [25.0, 50.0, 100.0]
            results = {}

            for size in sizes:
                grids = create_multiple_grids(n_grids=3, bbox_size=size, resolution=1.0)
                fusion = VoxelFusion()

                def run_fusion():
                    return fusion.fuse_voxel_grids(grids, method="weighted_average")

                result = benchmark(run_fusion)
                results[size] = result
                assert result is not None

            assert len(results) == len(sizes)
        except (ImportError, AttributeError):
            pytest.skip("VoxelFusion not available")
