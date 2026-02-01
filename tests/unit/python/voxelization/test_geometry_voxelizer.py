"""
Unit tests for am_qadf.voxelization.geometry_voxelizer.

Tests geometry_voxelizer.py: get_stl_bounding_box, create_voxel_grid_from_stl_and_hatching,
export_to_paraview, calculate_adaptive_resolution.
Some tests require am_qadf_native and C++ STL voxelizer; they are skipped when unavailable.
"""

import pytest

try:
    from am_qadf.voxelization import geometry_voxelizer
    GEOMETRY_VOXELIZER_AVAILABLE = True
except ImportError:
    GEOMETRY_VOXELIZER_AVAILABLE = False


@pytest.mark.skipif(not GEOMETRY_VOXELIZER_AVAILABLE, reason="geometry_voxelizer not importable")
class TestGetStlBoundingBox:
    """Tests for get_stl_bounding_box."""

    def test_nonexistent_stl_raises_file_not_found(self):
        from am_qadf.voxelization.geometry_voxelizer import get_stl_bounding_box
        with pytest.raises(FileNotFoundError, match="STL file not found"):
            get_stl_bounding_box("/nonexistent/path/to/file.stl")


@pytest.mark.skipif(not GEOMETRY_VOXELIZER_AVAILABLE, reason="geometry_voxelizer not importable")
class TestCreateVoxelGridFromStlAndHatching:
    """Tests for create_voxel_grid_from_stl_and_hatching."""

    def test_nonexistent_stl_raises(self):
        from am_qadf.voxelization.geometry_voxelizer import create_voxel_grid_from_stl_and_hatching
        # May raise ImportError (no C++), NotImplementedError (no HatchingVoxelizer), or FileNotFoundError (STL missing)
        with pytest.raises((FileNotFoundError, ImportError, NotImplementedError)):
            create_voxel_grid_from_stl_and_hatching(
                stl_path="/nonexistent/file.stl",
                hatching_result=None,
                resolution=0.1,
            )

    def test_resolution_zero_raises_when_stl_exists(self, tmp_path):
        """When STL exists but resolution is invalid, ValueError is raised (after STL check)."""
        from am_qadf.voxelization.geometry_voxelizer import create_voxel_grid_from_stl_and_hatching
        # Create a minimal empty file so path exists (C++ may still fail to parse; we only care about resolution check order)
        stl = tmp_path / "dummy.stl"
        stl.write_bytes(b"")
        try:
            create_voxel_grid_from_stl_and_hatching(
                stl_path=str(stl),
                hatching_result=None,
                resolution=0.0,
            )
        except (ValueError, ImportError, NotImplementedError) as e:
            if isinstance(e, ValueError):
                assert "Resolution" in str(e) or "resolution" in str(e).lower()
            # ImportError/NotImplementedError: C++ or Hatching not available, skip assertion
            return
        # If we get here without raising, C++ accepted resolution=0 (unexpected)
        pytest.fail("Expected ValueError for resolution=0 or ImportError/NotImplementedError")


@pytest.mark.skipif(not GEOMETRY_VOXELIZER_AVAILABLE, reason="geometry_voxelizer not importable")
class TestCalculateAdaptiveResolution:
    """Tests for calculate_adaptive_resolution."""

    def test_nonexistent_stl_raises(self):
        from am_qadf.voxelization.geometry_voxelizer import calculate_adaptive_resolution
        with pytest.raises((FileNotFoundError, ImportError)):
            calculate_adaptive_resolution("/nonexistent/file.stl")


@pytest.mark.skipif(not GEOMETRY_VOXELIZER_AVAILABLE, reason="geometry_voxelizer not importable")
class TestExportToParaview:
    """Tests for export_to_paraview."""

    def test_nonexistent_stl_raises(self):
        from am_qadf.voxelization.geometry_voxelizer import export_to_paraview
        with pytest.raises((FileNotFoundError, ImportError, NotImplementedError)):
            export_to_paraview(
                stl_path="/nonexistent/file.stl",
                hatching_result=None,
                output_path="/tmp/out.vdb",
            )
