"""
Unit tests for VoxelRenderer.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from am_qadf.visualization.voxel_renderer import VoxelRenderer

try:
    import pyvista as pv

    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False


class MockVoxelGrid:
    """Mock VoxelGrid for testing."""

    def __init__(self):
        self.dims = np.array([10, 10, 10])
        self.resolution = 1.0
        self.bbox_min = np.array([0.0, 0.0, 0.0])
        self.bbox_max = np.array([10.0, 10.0, 10.0])
        self.available_signals = {"power", "velocity", "energy"}

    def get_signal_array(self, signal_name, default=0.0):
        """Return a mock signal array."""
        return np.random.rand(*self.dims) * 100.0


@pytest.mark.skipif(not PYVISTA_AVAILABLE, reason="PyVista not installed")
class TestVoxelRenderer:
    """Test cases for VoxelRenderer."""

    @pytest.fixture
    def voxel_grid(self):
        """Create a mock voxel grid."""
        return MockVoxelGrid()

    @pytest.fixture
    def renderer(self, voxel_grid):
        """Create a VoxelRenderer instance."""
        return VoxelRenderer(voxel_grid=voxel_grid)

    @pytest.mark.unit
    def test_initialization_with_grid(self, voxel_grid):
        """Test renderer initialization with voxel grid."""
        renderer = VoxelRenderer(voxel_grid=voxel_grid)
        assert renderer.voxel_grid == voxel_grid
        assert renderer._pyvista_grid is None

    @pytest.mark.unit
    def test_initialization_without_grid(self):
        """Test renderer initialization without voxel grid."""
        renderer = VoxelRenderer()
        assert renderer.voxel_grid is None
        assert renderer._pyvista_grid is None

    @pytest.mark.unit
    def test_initialization_without_pyvista(self):
        """Test that initialization fails without PyVista."""
        with patch("am_qadf.visualization.voxel_renderer.PYVISTA_AVAILABLE", False):
            with pytest.raises(ImportError, match="PyVista is required"):
                VoxelRenderer()

    @pytest.mark.unit
    def test_set_voxel_grid(self, renderer, voxel_grid):
        """Test setting voxel grid."""
        new_grid = MockVoxelGrid()
        renderer.set_voxel_grid(new_grid)
        assert renderer.voxel_grid == new_grid
        assert renderer._pyvista_grid is None  # Should reset cached grid

    @pytest.mark.unit
    def test_create_pyvista_grid(self, renderer, voxel_grid):
        """Test PyVista grid creation."""
        grid = renderer._create_pyvista_grid()
        assert grid is not None
        assert grid.dimensions == tuple(voxel_grid.dims)
        assert grid.spacing == (voxel_grid.resolution,) * 3
        assert grid.origin == tuple(voxel_grid.bbox_min)

    @pytest.mark.unit
    def test_create_pyvista_grid_without_voxel_grid(self):
        """Test PyVista grid creation without voxel grid."""
        renderer = VoxelRenderer()
        grid = renderer._create_pyvista_grid()
        assert grid is None

    @pytest.mark.unit
    def test_create_pyvista_grid_caching(self, renderer):
        """Test that PyVista grid is cached."""
        grid1 = renderer._create_pyvista_grid()
        grid2 = renderer._create_pyvista_grid()
        assert grid1 is grid2  # Same object (cached)

    @pytest.mark.unit
    def test_render_3d_basic(self, renderer):
        """Test basic 3D rendering."""
        plotter = renderer.render_3d(signal_name="power", colormap="plasma", threshold=0.1, auto_show=False)
        assert plotter is not None
        assert isinstance(plotter, pv.Plotter)

    @pytest.mark.unit
    def test_render_3d_without_grid(self):
        """Test 3D rendering without voxel grid raises error."""
        renderer = VoxelRenderer()
        with pytest.raises(ValueError, match="No voxel grid set"):
            renderer.render_3d()

    @pytest.mark.unit
    def test_render_3d_with_title(self, renderer):
        """Test 3D rendering with title."""
        plotter = renderer.render_3d(signal_name="power", title="Test Title", auto_show=False)
        assert plotter is not None

    @pytest.mark.unit
    def test_render_3d_with_custom_colormap(self, renderer):
        """Test 3D rendering with custom colormap."""
        plotter = renderer.render_3d(signal_name="power", colormap="viridis", auto_show=False)
        assert plotter is not None

    @pytest.mark.unit
    def test_render_3d_with_opacity(self, renderer):
        """Test 3D rendering with custom opacity."""
        plotter = renderer.render_3d(signal_name="power", opacity=0.5, auto_show=False)
        assert plotter is not None

    @pytest.mark.unit
    def test_render_3d_without_scalar_bar(self, renderer):
        """Test 3D rendering without scalar bar."""
        plotter = renderer.render_3d(signal_name="power", show_scalar_bar=False, auto_show=False)
        assert plotter is not None

    @pytest.mark.unit
    def test_render_3d_large_grid_warning(self, renderer):
        """Test that large grids trigger warning."""
        # Create a large grid
        large_grid = MockVoxelGrid()
        large_grid.dims = np.array([100, 100, 100])  # 1M voxels
        renderer.set_voxel_grid(large_grid)

        with pytest.warns(UserWarning):
            plotter = renderer.render_3d(signal_name="power", threshold=0.1, auto_show=False)
        assert plotter is not None

    @pytest.mark.unit
    def test_render_3d_empty_mesh(self, renderer):
        """Test 3D rendering with empty mesh (all values below threshold)."""
        # Create grid with all zeros
        empty_grid = MockVoxelGrid()
        empty_grid.get_signal_array = lambda name, default: np.zeros((10, 10, 10))
        renderer.set_voxel_grid(empty_grid)

        plotter = renderer.render_3d(signal_name="power", threshold=1.0, auto_show=False)  # High threshold
        assert plotter is not None

    @pytest.mark.unit
    def test_render_slice_x(self, renderer):
        """Test X-axis slice rendering."""
        plotter = renderer.render_slice(signal_name="power", axis="x", position=5.0, auto_show=False)
        assert plotter is not None

    @pytest.mark.unit
    def test_render_slice_y(self, renderer):
        """Test Y-axis slice rendering."""
        plotter = renderer.render_slice(signal_name="power", axis="y", position=5.0, auto_show=False)
        assert plotter is not None

    @pytest.mark.unit
    def test_render_slice_z(self, renderer):
        """Test Z-axis slice rendering."""
        plotter = renderer.render_slice(signal_name="power", axis="z", position=5.0, auto_show=False)
        assert plotter is not None

    @pytest.mark.unit
    def test_render_slice_default_position(self, renderer):
        """Test slice rendering with default position (center)."""
        plotter = renderer.render_slice(
            signal_name="power",
            axis="z",
            position=None,  # Should use center
            auto_show=False,
        )
        assert plotter is not None

    @pytest.mark.unit
    def test_render_slice_without_grid(self):
        """Test slice rendering without voxel grid raises error."""
        renderer = VoxelRenderer()
        with pytest.raises(ValueError, match="No voxel grid set"):
            renderer.render_slice()

    @pytest.mark.unit
    def test_render_slice_invalid_axis(self, renderer):
        """Test slice rendering with invalid axis."""
        # Should default to 'z' or handle gracefully
        plotter = renderer.render_slice(signal_name="power", axis="invalid", auto_show=False)
        # Should still work (defaults to 'z')
        assert plotter is not None

    @pytest.mark.unit
    def test_render_isosurface(self, renderer):
        """Test isosurface rendering."""
        plotter = renderer.render_isosurface(signal_name="power", isovalue=50.0, auto_show=False)
        assert plotter is not None

    @pytest.mark.unit
    def test_render_isosurface_default_value(self, renderer):
        """Test isosurface rendering with default isovalue (mean)."""
        plotter = renderer.render_isosurface(signal_name="power", isovalue=None, auto_show=False)  # Should use mean
        assert plotter is not None

    @pytest.mark.unit
    def test_render_isosurface_without_grid(self):
        """Test isosurface rendering without voxel grid raises error."""
        renderer = VoxelRenderer()
        with pytest.raises(ValueError, match="No voxel grid set"):
            renderer.render_isosurface()

    @pytest.mark.unit
    def test_render_multi_slice(self, renderer):
        """Test multi-slice rendering (3 slices + 3D view)."""
        plotter = renderer.render_multi_slice(signal_name="power", auto_show=False)
        assert plotter is not None
        assert plotter.shape == (2, 2)  # 2x2 subplot layout

    @pytest.mark.unit
    def test_render_multi_slice_without_grid(self):
        """Test multi-slice rendering without voxel grid raises error."""
        renderer = VoxelRenderer()
        with pytest.raises(ValueError, match="No voxel grid set"):
            renderer.render_multi_slice()

    @pytest.mark.unit
    def test_render_3d_error_handling(self, renderer):
        """Test error handling in render_3d."""
        # Mock get_signal_array to raise an error
        renderer.voxel_grid.get_signal_array = Mock(side_effect=Exception("Test error"))

        # Should return a plotter with error message
        plotter = renderer.render_3d(signal_name="power", auto_show=False)
        assert plotter is not None

    @pytest.mark.unit
    def test_render_3d_auto_show(self, renderer):
        """Test render_3d with auto_show=True."""
        with patch.object(pv.Plotter, "show") as mock_show:
            plotter = renderer.render_3d(signal_name="power", auto_show=True)
            assert plotter is not None
            mock_show.assert_called_once()

    @pytest.mark.unit
    def test_render_slice_auto_show(self, renderer):
        """Test render_slice with auto_show=True."""
        with patch.object(pv.Plotter, "show") as mock_show:
            plotter = renderer.render_slice(signal_name="power", axis="z", auto_show=True)
            assert plotter is not None
            mock_show.assert_called_once()

    @pytest.mark.unit
    def test_render_with_different_signals(self, renderer):
        """Test rendering with different signal names."""
        signals = ["power", "velocity", "energy"]
        for signal in signals:
            plotter = renderer.render_3d(signal_name=signal, auto_show=False)
            assert plotter is not None

    @pytest.mark.unit
    def test_render_with_missing_signal(self, renderer):
        """Test rendering with missing signal (should use default)."""
        plotter = renderer.render_3d(signal_name="nonexistent_signal", auto_show=False)
        assert plotter is not None
