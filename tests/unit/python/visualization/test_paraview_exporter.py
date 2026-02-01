"""
Unit tests for ParaView exporter (C++ VDBWriter wrapper).

Tests export_voxel_grid_to_paraview and export_multiple_grids_to_paraview.
Requires am_qadf_native.io.VDBWriter and a VoxelGrid with _get_or_create_grid(signal).get_grid().
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from am_qadf.visualization.paraview_exporter import (
    export_voxel_grid_to_paraview,
    export_multiple_grids_to_paraview,
)


class MockUniformGrid:
    """Mock UniformVoxelGrid that returns a C++ FloatGrid or mock."""

    def __init__(self, grid=None):
        self._grid = grid  # FloatGrid or mock

    def get_grid(self):
        return self._grid


def _make_mock_voxel_grid(available_signals=None, float_grid=None):
    """VoxelGrid-like object with _get_or_create_grid(signal).get_grid()."""
    available_signals = available_signals or {"temperature"}
    uniform = MockUniformGrid(float_grid)
    mock_vg = Mock()
    mock_vg.available_signals = set(available_signals)
    mock_vg._get_or_create_grid = Mock(return_value=uniform)
    return mock_vg


@pytest.mark.unit
def test_export_voxel_grid_to_paraview_with_real_grid():
    """Export with real VoxelGrid when am_qadf_native is available."""
    pytest.importorskip("am_qadf_native", reason="VDBWriter C++ bindings required")
    try:
        from am_qadf.voxelization.uniform_resolution import VoxelGrid
    except ImportError:
        pytest.skip("VoxelGrid (uniform_resolution) required for export test")
    grid = VoxelGrid(bbox_min=(0, 0, 0), bbox_max=(10, 10, 10), resolution=1.0)
    # Ensure at least one signal exists
    grid._get_or_create_grid("temperature")
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "out.vdb"
        result = export_voxel_grid_to_paraview(grid, str(out), signal_names=["temperature"])
        assert result.endswith(".vdb")
        assert Path(result).exists()


@pytest.mark.unit
def test_export_voxel_grid_to_paraview_adds_vdb_extension():
    """Export adds .vdb if path has no extension."""
    pytest.importorskip("am_qadf_native", reason="VDBWriter required")
    try:
        from am_qadf.voxelization.uniform_resolution import VoxelGrid
    except ImportError:
        pytest.skip("VoxelGrid required")
    grid = VoxelGrid(bbox_min=(0, 0, 0), bbox_max=(10, 10, 10), resolution=1.0)
    grid._get_or_create_grid("temperature")
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "out"  # no extension
        result = export_voxel_grid_to_paraview(grid, str(out))
        assert result.endswith(".vdb")
        assert Path(result).exists()


@pytest.mark.unit
def test_export_voxel_grid_to_paraview_no_signals_raises():
    """Export with no signals raises ValueError."""
    mock_grid = Mock()
    mock_grid.available_signals = set()
    with tempfile.NamedTemporaryFile(suffix=".vdb", delete=False) as f:
        path = f.name
    try:
        with pytest.raises(ValueError, match="No signals to export"):
            export_voxel_grid_to_paraview(mock_grid, path)
    finally:
        Path(path).unlink(missing_ok=True)


@pytest.mark.unit
def test_export_multiple_grids_to_paraview_no_grids_raises():
    """Export multiple with no valid grids raises ValueError."""
    mock_grid = Mock()
    mock_grid.available_signals = set()
    with tempfile.NamedTemporaryFile(suffix=".vdb", delete=False) as f:
        path = f.name
    try:
        with pytest.raises(ValueError, match="No grids to export"):
            export_multiple_grids_to_paraview({"g1": mock_grid}, path)
    finally:
        Path(path).unlink(missing_ok=True)
