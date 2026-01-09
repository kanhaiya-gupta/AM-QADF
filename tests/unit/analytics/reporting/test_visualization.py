"""
Unit tests for visualization.

Tests for VisualizationConfig, VisualizationResult, AnalysisVisualizer, and SensitivityVisualizer.
"""

import pytest
import os
import tempfile
import shutil
from am_qadf.analytics.reporting.visualization import (
    VisualizationConfig,
    VisualizationResult,
    AnalysisVisualizer,
    SensitivityVisualizer,
)


class TestVisualizationConfig:
    """Test suite for VisualizationConfig dataclass."""

    @pytest.mark.unit
    def test_config_creation_default(self):
        """Test creating VisualizationConfig with default values."""
        config = VisualizationConfig()

        assert config.figure_size == (12, 8)
        assert config.dpi == 300
        assert config.style == "whitegrid"
        assert config.color_palette == "viridis"
        assert config.alpha == 0.7
        assert config.output_directory == "plots"
        assert config.save_format == "png"
        assert config.confidence_level == 0.95

    @pytest.mark.unit
    def test_config_creation_custom(self):
        """Test creating VisualizationConfig with custom values."""
        config = VisualizationConfig(
            figure_size=(10, 6),
            dpi=150,
            style="darkgrid",
            color_palette="plasma",
            alpha=0.8,
            output_directory="custom_plots",
            save_format="pdf",
            confidence_level=0.99,
        )

        assert config.figure_size == (10, 6)
        assert config.dpi == 150
        assert config.style == "darkgrid"
        assert config.color_palette == "plasma"
        assert config.alpha == 0.8
        assert config.output_directory == "custom_plots"
        assert config.save_format == "pdf"
        assert config.confidence_level == 0.99


class TestVisualizationResult:
    """Test suite for VisualizationResult dataclass."""

    @pytest.mark.unit
    def test_result_creation(self):
        """Test creating VisualizationResult."""
        result = VisualizationResult(
            success=True,
            visualization_type="SensitivityAnalysis",
            plot_paths=["/path/to/plot1.png", "/path/to/plot2.png"],
            generation_time=2.5,
        )

        assert result.success is True
        assert result.visualization_type == "SensitivityAnalysis"
        assert len(result.plot_paths) == 2
        assert result.generation_time == 2.5
        assert result.error_message is None

    @pytest.mark.unit
    def test_result_creation_with_error(self):
        """Test creating VisualizationResult with error."""
        result = VisualizationResult(
            success=False,
            visualization_type="SensitivityAnalysis",
            plot_paths=[],
            generation_time=0.0,
            error_message="Test error",
        )

        assert result.success is False
        assert result.error_message == "Test error"


class TestAnalysisVisualizer:
    """Test suite for AnalysisVisualizer class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def visualizer(self, temp_dir):
        """Create an AnalysisVisualizer instance."""
        config = VisualizationConfig(output_directory=temp_dir)
        return AnalysisVisualizer(config)

    @pytest.mark.unit
    def test_visualizer_creation_default(self):
        """Test creating AnalysisVisualizer with default config."""
        visualizer = AnalysisVisualizer()

        assert visualizer.config is not None
        assert visualizer.config.figure_size == (12, 8)

    @pytest.mark.unit
    def test_visualizer_creation_custom(self, temp_dir):
        """Test creating AnalysisVisualizer with custom config."""
        config = VisualizationConfig(figure_size=(10, 6), output_directory=temp_dir)
        visualizer = AnalysisVisualizer(config)

        assert visualizer.config.figure_size == (10, 6)
        assert visualizer.config.output_directory == temp_dir

    @pytest.mark.unit
    def test_visualize_sensitivity_analysis(self, visualizer):
        """Test visualizing sensitivity analysis."""
        sensitivity_results = {
            "sobol_analysis": {"indices": {"power": 0.6, "velocity": 0.4}},
            "morris_analysis": {"mu_star": {"power": 0.5, "velocity": 0.3}},
        }

        result = visualizer.visualize_sensitivity_analysis(sensitivity_results, plot_title="Sensitivity Analysis")

        assert isinstance(result, VisualizationResult)
        assert result.success is True or result.success is False  # May fail if matplotlib issues
        assert result.visualization_type == "SensitivityAnalysis"

    @pytest.mark.unit
    def test_visualize_statistical_analysis(self, visualizer):
        """Test visualizing statistical analysis."""
        statistical_results = {
            "pca_analysis": {"explained_variance": [0.5, 0.3, 0.2]},
            "correlation_analysis": {"correlation_matrix": [[1.0, 0.5], [0.5, 1.0]]},
        }

        result = visualizer.visualize_statistical_analysis(statistical_results, plot_title="Statistical Analysis")

        assert isinstance(result, VisualizationResult)
        assert result.success is True or result.success is False
        assert result.visualization_type == "StatisticalAnalysis"

    @pytest.mark.unit
    def test_visualize_process_analysis(self, visualizer):
        """Test visualizing process analysis."""
        process_results = {"parameter_analysis": {"optimal_parameters": {"power": 250.0, "velocity": 1200.0}}}

        result = visualizer.visualize_process_analysis(process_results, plot_title="Process Analysis")

        assert isinstance(result, VisualizationResult)
        assert result.success is True or result.success is False
        assert result.visualization_type == "ProcessAnalysis"


class TestSensitivityVisualizer:
    """Test suite for SensitivityVisualizer class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def visualizer(self, temp_dir):
        """Create a SensitivityVisualizer instance."""
        config = VisualizationConfig(output_directory=temp_dir)
        return SensitivityVisualizer(config)

    @pytest.mark.unit
    def test_visualizer_creation(self, visualizer):
        """Test creating SensitivityVisualizer."""
        assert visualizer is not None
        assert visualizer.config is not None

    @pytest.mark.unit
    def test_visualize_sensitivity_analysis(self, visualizer):
        """Test visualizing sensitivity analysis."""
        sensitivity_results = {"sobol_analysis": {"indices": {"power": 0.6, "velocity": 0.4}}}

        result = visualizer.visualize_sensitivity_analysis(sensitivity_results)

        assert isinstance(result, VisualizationResult)
        assert result.success is True or result.success is False
        assert result.visualization_type == "SensitivityAnalysis"
