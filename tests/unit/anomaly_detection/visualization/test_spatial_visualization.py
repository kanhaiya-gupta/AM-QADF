"""
Unit tests for spatial anomaly visualization.

Tests for SpatialAnomalyVisualizer class.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from am_qadf.anomaly_detection.visualization.spatial_visualization import (
    SpatialAnomalyVisualizer,
)


class TestSpatialAnomalyVisualizer:
    """Test suite for SpatialAnomalyVisualizer class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample voxel data."""
        np.random.seed(42)
        n_samples = 100
        voxel_coords = np.random.randn(n_samples, 3) * 10 + 50
        anomaly_scores = np.random.rand(n_samples)
        anomaly_labels = anomaly_scores > 0.7
        return voxel_coords, anomaly_scores, anomaly_labels

    @pytest.mark.unit
    def test_visualizer_creation(self):
        """Test creating SpatialAnomalyVisualizer."""
        visualizer = SpatialAnomalyVisualizer()

        assert visualizer.voxel_coordinates is None
        assert visualizer.anomaly_scores is None
        assert visualizer.anomaly_labels is None

    @pytest.mark.unit
    def test_set_data(self, sample_data):
        """Test setting data for visualization."""
        voxel_coords, anomaly_scores, anomaly_labels = sample_data
        visualizer = SpatialAnomalyVisualizer()
        visualizer.set_data(voxel_coords, anomaly_scores, anomaly_labels)

        assert visualizer.voxel_coordinates is not None
        assert visualizer.anomaly_scores is not None
        assert visualizer.anomaly_labels is not None
        assert np.array_equal(visualizer.voxel_coordinates, voxel_coords)
        assert np.array_equal(visualizer.anomaly_scores, anomaly_scores)

    @pytest.mark.unit
    def test_set_data_without_labels(self, sample_data):
        """Test setting data without labels."""
        voxel_coords, anomaly_scores, _ = sample_data
        visualizer = SpatialAnomalyVisualizer()
        visualizer.set_data(voxel_coords, anomaly_scores)

        assert visualizer.voxel_coordinates is not None
        assert visualizer.anomaly_scores is not None
        assert visualizer.anomaly_labels is None

    @pytest.mark.unit
    def test_create_voxel_grid(self, sample_data):
        """Test creating voxel grid."""
        voxel_coords, anomaly_scores, _ = sample_data
        visualizer = SpatialAnomalyVisualizer()
        visualizer.set_data(voxel_coords, anomaly_scores)

        grid = visualizer.create_voxel_grid(resolution=1.0)

        assert grid is not None
        assert isinstance(grid, np.ndarray)
        assert len(grid.shape) == 3

    @pytest.mark.unit
    def test_create_voxel_grid_no_data(self):
        """Test creating voxel grid without data."""
        visualizer = SpatialAnomalyVisualizer()
        grid = visualizer.create_voxel_grid()

        assert grid is None

    @pytest.mark.unit
    @patch("matplotlib.pyplot.show")
    def test_plot_3d_voxel_map_matplotlib(self, mock_show, sample_data):
        """Test plotting 3D voxel map with matplotlib."""
        voxel_coords, anomaly_scores, _ = sample_data
        visualizer = SpatialAnomalyVisualizer()
        visualizer.set_data(voxel_coords, anomaly_scores)

        with patch(
            "am_qadf.anomaly_detection.visualization.spatial_visualization.PYVISTA_AVAILABLE",
            False,
        ):
            fig = visualizer.plot_3d_voxel_map(show=False)

        assert fig is not None
        # Should not call show when show=False
        mock_show.assert_not_called()

    @pytest.mark.unit
    def test_plot_3d_voxel_map_no_data(self):
        """Test plotting 3D voxel map without data."""
        visualizer = SpatialAnomalyVisualizer()
        fig = visualizer.plot_3d_voxel_map(show=False)

        assert fig is None

    @pytest.mark.unit
    @patch("matplotlib.pyplot.show")
    def test_plot_3d_voxel_map_with_threshold(self, mock_show, sample_data):
        """Test plotting 3D voxel map with threshold."""
        voxel_coords, anomaly_scores, _ = sample_data
        visualizer = SpatialAnomalyVisualizer()
        visualizer.set_data(voxel_coords, anomaly_scores)

        with patch(
            "am_qadf.anomaly_detection.visualization.spatial_visualization.PYVISTA_AVAILABLE",
            False,
        ):
            fig = visualizer.plot_3d_voxel_map(threshold=0.5, show=False)

        assert fig is not None

    @pytest.mark.unit
    @patch("matplotlib.pyplot.show")
    def test_plot_slice_view(self, mock_show, sample_data):
        """Test plotting slice view."""
        voxel_coords, anomaly_scores, _ = sample_data
        visualizer = SpatialAnomalyVisualizer()
        visualizer.set_data(voxel_coords, anomaly_scores)

        fig = visualizer.plot_slice_view(axis="z", show=False)

        assert fig is not None
        mock_show.assert_not_called()

    @pytest.mark.unit
    @patch("matplotlib.pyplot.show")
    def test_plot_slice_view_different_axes(self, mock_show, sample_data):
        """Test plotting slice view for different axes."""
        voxel_coords, anomaly_scores, _ = sample_data
        visualizer = SpatialAnomalyVisualizer()
        visualizer.set_data(voxel_coords, anomaly_scores)

        for axis in ["x", "y", "z"]:
            fig = visualizer.plot_slice_view(axis=axis, show=False)
            assert fig is not None

    @pytest.mark.unit
    def test_plot_slice_view_invalid_axis(self, sample_data):
        """Test plotting slice view with invalid axis."""
        voxel_coords, anomaly_scores, _ = sample_data
        visualizer = SpatialAnomalyVisualizer()
        visualizer.set_data(voxel_coords, anomaly_scores)

        with pytest.raises(ValueError, match="Axis must be"):
            visualizer.plot_slice_view(axis="invalid", show=False)

    @pytest.mark.unit
    def test_plot_slice_view_no_data(self):
        """Test plotting slice view without data."""
        visualizer = SpatialAnomalyVisualizer()
        fig = visualizer.plot_slice_view(show=False)

        assert fig is None

    @pytest.mark.unit
    @patch("matplotlib.pyplot.show")
    def test_plot_anomaly_heatmap(self, mock_show, sample_data):
        """Test plotting anomaly heatmap."""
        voxel_coords, anomaly_scores, _ = sample_data
        visualizer = SpatialAnomalyVisualizer()
        visualizer.set_data(voxel_coords, anomaly_scores)

        fig = visualizer.plot_anomaly_heatmap(show=False)

        assert fig is not None
        mock_show.assert_not_called()

    @pytest.mark.unit
    def test_plot_anomaly_heatmap_no_data(self):
        """Test plotting heatmap without data."""
        visualizer = SpatialAnomalyVisualizer()
        fig = visualizer.plot_anomaly_heatmap(show=False)

        assert fig is None

    @pytest.mark.unit
    @patch("matplotlib.pyplot.show")
    def test_plot_anomaly_heatmap_custom_resolution(self, mock_show, sample_data):
        """Test plotting heatmap with custom resolution."""
        voxel_coords, anomaly_scores, _ = sample_data
        visualizer = SpatialAnomalyVisualizer()
        visualizer.set_data(voxel_coords, anomaly_scores)

        fig = visualizer.plot_anomaly_heatmap(resolution=0.5, show=False)

        assert fig is not None
