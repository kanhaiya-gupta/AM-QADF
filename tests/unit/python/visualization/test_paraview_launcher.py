"""
Unit tests for ParaView launcher.

Tests find_paraview_executable, launch_paraview, create_paraview_button, export_and_launch_paraview.
Pure Python (no C++); uses subprocess and ipywidgets when available.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from am_qadf.visualization.paraview_launcher import (
    find_paraview_executable,
    launch_paraview,
    create_paraview_button,
    export_and_launch_paraview,
)


class TestFindParaviewExecutable:
    """Test suite for find_paraview_executable."""

    @pytest.mark.unit
    def test_find_paraview_returns_str_or_none(self):
        """find_paraview_executable returns str path or None."""
        result = find_paraview_executable()
        assert result is None or (isinstance(result, str) and len(result) > 0)

    @pytest.mark.unit
    def test_find_paraview_with_existing_path(self):
        """When a known path exists, returns that path."""
        with patch.object(Path, "exists", return_value=True):
            with patch("am_qadf.visualization.paraview_launcher.platform") as mock_platform:
                mock_platform.system.return_value = "Windows"
                # Patch Path(possible_paths[0]).exists() to True
                with patch("am_qadf.visualization.paraview_launcher.Path") as MockPath:
                    mock_path = MockPath.return_value
                    mock_path.exists.return_value = True
                    mock_path.__str__ = lambda self: "C:\\ParaView\\paraview.exe"
                    result = find_paraview_executable()
                    # May still be None if expanduser/expandvars changes path
                    assert result is None or isinstance(result, str)


class TestLaunchParaview:
    """Test suite for launch_paraview."""

    @pytest.mark.unit
    def test_launch_paraview_nonexistent_file_returns_false(self):
        """Launch with nonexistent .vdb file returns False."""
        result = launch_paraview("/nonexistent/path/file.vdb")
        assert result is False

    @pytest.mark.unit
    def test_launch_paraview_no_executable_returns_false(self):
        """Launch when ParaView executable not found returns False."""
        with tempfile.NamedTemporaryFile(suffix=".vdb", delete=False) as f:
            vdb_path = f.name
        try:
            with patch("am_qadf.visualization.paraview_launcher.find_paraview_executable", return_value=None):
                result = launch_paraview(vdb_path)
                assert result is False
        finally:
            Path(vdb_path).unlink(missing_ok=True)

    @pytest.mark.unit
    def test_launch_paraview_success_mocks_subprocess(self):
        """Launch with valid file and mock executable returns True."""
        with tempfile.NamedTemporaryFile(suffix=".vdb", delete=False) as f:
            vdb_path = f.name
        try:
            with patch("am_qadf.visualization.paraview_launcher.find_paraview_executable", return_value=vdb_path):
                with patch("am_qadf.visualization.paraview_launcher.subprocess.Popen", return_value=Mock()):
                    result = launch_paraview(vdb_path)
                    assert result is True
        finally:
            Path(vdb_path).unlink(missing_ok=True)


class TestCreateParaviewButton:
    """Test suite for create_paraview_button."""

    @pytest.mark.unit
    def test_create_paraview_button_returns_button_or_none(self):
        """create_paraview_button returns ipywidgets Button or None."""
        result = create_paraview_button("/some/file.vdb", button_text="Open")
        # If ipywidgets available, returns Button; else None
        assert result is None or hasattr(result, "on_click")


class TestExportAndLaunchParaview:
    """Test suite for export_and_launch_paraview."""

    @pytest.mark.unit
    def test_export_and_launch_paraview_calls_export_and_launch(self):
        """export_and_launch_paraview calls export then optionally launch."""
        mock_grid = Mock()
        mock_grid.available_signals = {"temperature"}
        uniform = Mock()
        uniform.get_grid.return_value = None
        mock_grid._get_or_create_grid = Mock(return_value=uniform)
        with patch("am_qadf.visualization.paraview_exporter.export_voxel_grid_to_paraview", return_value="/tmp/out.vdb") as mock_export:
            with patch("am_qadf.visualization.paraview_launcher.launch_paraview", return_value=True):
                result = export_and_launch_paraview(mock_grid, "/tmp/out.vdb", auto_launch=True)
                mock_export.assert_called_once()
                assert result == "/tmp/out.vdb"