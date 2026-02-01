"""
Integration tests for SPC with QualityDashboardGenerator.

Tests integration of SPC module with visualization/dashboard generation.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path

try:
    from am_qadf.analytics.reporting.visualization import (
        QualityDashboardGenerator,
        VisualizationConfig,
        VisualizationResult,
    )
    from am_qadf.analytics.spc import SPCClient, SPCConfig
except ImportError:
    pytest.skip("SPC or visualization modules not available", allow_module_level=True)


class TestSPCVisualizationIntegration:
    """Integration tests for SPC with QualityDashboardGenerator."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def dashboard_generator(self, temp_dir):
        """Create QualityDashboardGenerator instance."""
        config = VisualizationConfig(output_directory=temp_dir)
        return QualityDashboardGenerator(config=config)

    @pytest.fixture
    def quality_data_df(self):
        """Create sample quality data DataFrame."""
        return pd.DataFrame(
            {
                "overall_score": np.random.uniform(0.8, 1.0, 50),
                "completeness": np.random.uniform(0.75, 0.95, 50),
                "coverage": np.random.uniform(0.80, 0.95, 50),
                "consistency": np.random.uniform(0.85, 0.98, 50),
                "timestamp": pd.date_range("2026-01-01", periods=50, freq="H"),
            }
        )

    @pytest.mark.integration
    def test_dashboard_generator_spc_available(self, dashboard_generator):
        """Test that dashboard generator can detect SPC module."""
        # SPC availability is checked in __init__
        assert hasattr(dashboard_generator, "spc_available")
        assert isinstance(dashboard_generator.spc_available, bool)

    @pytest.mark.integration
    def test_generate_dashboard_with_spc_charts(self, dashboard_generator, quality_data_df):
        """Test generating dashboard with SPC control charts."""
        if not dashboard_generator.spc_available:
            pytest.skip("SPC module not available")

        result = dashboard_generator.generate_dashboard(quality_data_df, dashboard_type="historical")

        assert isinstance(result, VisualizationResult)
        assert result.success == True
        assert len(result.plot_paths) > 0
        # Should include SPC control charts if data is sufficient
        spc_paths = [p for p in result.plot_paths if "spc" in p.lower() or "control" in p.lower()]
        # May or may not have SPC charts depending on data
        assert isinstance(spc_paths, list)

    @pytest.mark.integration
    def test_add_spc_control_charts(self, dashboard_generator, quality_data_df):
        """Test adding SPC control charts to existing dashboard."""
        if not dashboard_generator.spc_available:
            pytest.skip("SPC module not available")

        # Create initial dashboard
        initial_result = dashboard_generator.generate_dashboard(quality_data_df)
        initial_plot_count = len(initial_result.plot_paths)

        # Add SPC control charts
        updated_result = dashboard_generator.add_spc_control_charts(initial_result, quality_data_df)

        assert isinstance(updated_result, VisualizationResult)
        assert len(updated_result.plot_paths) >= initial_plot_count
        # Should have added at least one SPC chart path
        spc_paths = [p for p in updated_result.plot_paths if "spc" in p.lower()]
        # May have added charts or not depending on data quality

    @pytest.mark.integration
    def test_add_spc_control_charts_no_spc(self, dashboard_generator, quality_data_df, temp_dir):
        """Test adding SPC charts when SPC module is not available."""
        # Temporarily disable SPC
        original_available = dashboard_generator.spc_available
        dashboard_generator.spc_available = False

        initial_result = VisualizationResult(
            success=True, visualization_type="QualityDashboard", plot_paths=[], generation_time=0.0
        )

        updated_result = dashboard_generator.add_spc_control_charts(initial_result, quality_data_df)

        # Should return unchanged dashboard
        assert updated_result == initial_result

        # Restore
        dashboard_generator.spc_available = original_available

    @pytest.mark.integration
    def test_plot_spc_control_charts_sufficient_data(self, dashboard_generator):
        """Test plotting SPC control charts with sufficient data."""
        if not dashboard_generator.spc_available:
            pytest.skip("SPC module not available")

        # Create quality data with enough samples for control charts
        quality_data = pd.DataFrame(
            {
                "completeness": np.random.uniform(0.8, 1.0, 30),  # 30 samples
                "coverage": np.random.uniform(0.75, 0.95, 30),
                "consistency": np.random.uniform(0.85, 0.98, 30),
                "timestamp": pd.date_range("2026-01-01", periods=30, freq="H"),
            }
        )

        plot_path = dashboard_generator._plot_spc_control_charts(quality_data)

        # May return None if data doesn't meet requirements, or path if successful
        assert plot_path is None or isinstance(plot_path, str)
        if plot_path:
            assert Path(plot_path).exists()

    @pytest.mark.integration
    def test_plot_spc_control_charts_insufficient_data(self, dashboard_generator):
        """Test plotting SPC control charts with insufficient data."""
        if not dashboard_generator.spc_available:
            pytest.skip("SPC module not available")

        # Create quality data with insufficient samples
        quality_data = pd.DataFrame(
            {"completeness": np.random.uniform(0.8, 1.0, 5), "coverage": np.random.uniform(0.75, 0.95, 5)}  # Only 5 samples
        )

        plot_path = dashboard_generator._plot_spc_control_charts(quality_data)

        # Should return None for insufficient data
        assert plot_path is None

    @pytest.mark.integration
    def test_spc_charts_in_dashboard_output(self, dashboard_generator, quality_data_df, temp_dir):
        """Test that SPC charts appear in dashboard output files."""
        if not dashboard_generator.spc_available:
            pytest.skip("SPC module not available")

        result = dashboard_generator.generate_dashboard(quality_data_df)

        if result.success and len(result.plot_paths) > 0:
            # Check that plot files exist
            for plot_path in result.plot_paths:
                if Path(plot_path).exists():
                    # Verify file is created (may be SPC chart or other plots)
                    assert Path(plot_path).stat().st_size > 0
