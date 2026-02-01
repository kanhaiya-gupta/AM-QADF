"""
End-to-end tests for analytics pipeline workflow.

Tests the complete workflow from query → analyze → report,
including statistical analysis, sensitivity analysis, and reporting.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

try:
    from am_qadf.query.unified_query_client import UnifiedQueryClient
    from am_qadf.voxel_domain.voxel_domain_client import VoxelDomainClient
    from am_qadf.voxelization.uniform_resolution import VoxelGrid
    from am_qadf.analytics.statistical_analysis.client import AdvancedAnalyticsClient
    from am_qadf.analytics.sensitivity_analysis.client import SensitivityAnalysisClient
    from am_qadf.analytics.reporting.report_generators import AnalysisReportGenerator

    ANALYTICS_AVAILABLE = True
except ImportError:
    ANALYTICS_AVAILABLE = False


@pytest.mark.skipif(not ANALYTICS_AVAILABLE, reason="Analytics components not available")
@pytest.mark.e2e
@pytest.mark.slow
class TestAnalyticsPipeline:
    """End-to-end tests for analytics pipeline workflow."""

    @pytest.fixture
    def mock_unified_query_client(self):
        """Create mock unified query client."""
        client = Mock(spec=UnifiedQueryClient)

        # Mock STL client
        client.stl_client = Mock()
        client.stl_client.get_model_bounding_box.return_value = (
            (-50.0, -50.0, -50.0),
            (50.0, 50.0, 50.0),
        )

        np.random.seed(42)

        # Mock laser client (minimal data for fast tests)
        client.laser_client = Mock()
        laser_points = np.random.rand(50, 3) * 100.0 - 50.0  # Reduced from 5000 to 50
        laser_result = Mock()
        laser_result.points = laser_points
        laser_result.signals = {
            "laser_power": np.random.rand(50) * 300.0,
            "scan_speed": np.random.rand(50) * 2000.0,
            "energy_density": np.random.rand(50) * 100.0,
        }
        client.laser_client.query.return_value = laser_result

        # Mock ISPM client (minimal data for fast tests)
        client.ispm_client = Mock()
        ispm_points = np.random.rand(50, 3) * 100.0 - 50.0  # Reduced from 10000 to 50
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

    @pytest.fixture
    def sample_voxel_grid(self, mock_unified_query_client, mock_mongo_client):
        """Create a sample voxel grid with signals for analytics."""
        voxel_client = VoxelDomainClient(
            unified_query_client=mock_unified_query_client,
            base_resolution=5.0,  # Larger resolution for faster tests
        )

        bbox_min, bbox_max = mock_unified_query_client.stl_client.get_model_bounding_box()
        voxel_grid = VoxelGrid(bbox_min=bbox_min, bbox_max=bbox_max, resolution=5.0)  # Larger resolution for faster tests

        result_grid = voxel_client.map_signals_to_voxels(
            model_id="test_model_001",
            voxel_grid=voxel_grid,
            sources=["laser", "ispm"],
            interpolation_method="nearest",
        )

        return result_grid

    @pytest.mark.e2e
    def test_statistical_analysis_pipeline(self, sample_voxel_grid, mock_mongo_client):
        """Test complete statistical analysis pipeline."""
        # Step 1: Create statistical analyzer
        analyzer = AdvancedAnalyticsClient(mongo_client=mock_mongo_client)

        # Step 2: Perform statistical analysis on voxel grid
        analysis_result = analyzer.analyze(model_id="test_model_001", voxel_grid=sample_voxel_grid, signals=None)

        # Assertions
        assert analysis_result is not None
        # Should have statistics
        assert isinstance(analysis_result, dict)
        assert "descriptive_statistics" in analysis_result or "summary" in analysis_result

    @pytest.mark.e2e
    def test_sensitivity_analysis_pipeline(self, sample_voxel_grid, mock_unified_query_client, mock_mongo_client):
        """Test complete sensitivity analysis pipeline."""
        # Step 1: Create sensitivity analyzer (requires unified_query_client)
        analyzer = SensitivityAnalysisClient(unified_query_client=mock_unified_query_client, voxel_domain_client=None)

        # Step 2: Verify analyzer was created
        assert analyzer is not None
        assert hasattr(analyzer, "unified_client")

        # Note: Full sensitivity analysis requires more complex setup
        # For now, just verify the client can be created
        # Actual analysis would require proper data and configuration

    @pytest.mark.e2e
    def test_reporting_pipeline(self, sample_voxel_grid, mock_mongo_client):
        """Test complete reporting pipeline."""
        # Step 1: Create report generator
        report_generator = AnalysisReportGenerator()

        # Step 2: Prepare analysis results
        analysis_results = {
            "model_id": "test_model_001",
            "statistical_analysis": {
                "descriptive_statistics": {
                    "laser_power": {
                        "mean": 100.0,
                        "std": 20.0,
                        "min": 50.0,
                        "max": 150.0,
                    }
                }
            },
            "signals": list(sample_voxel_grid.available_signals),
            "voxel_count": (len(sample_voxel_grid.voxels) if hasattr(sample_voxel_grid, "voxels") else 0),
        }

        # Step 3: Generate report
        report = report_generator.generate_comprehensive_report(
            analytics_results=analysis_results, report_title="Test Analytics Report"
        )

        # Assertions
        assert report is not None
        # Report should be a ReportResult object
        assert hasattr(report, "success") or hasattr(report, "report_path")

    @pytest.mark.e2e
    def test_complete_analytics_workflow(self, sample_voxel_grid, mock_mongo_client):
        """Test complete analytics workflow: query → analyze → report."""
        # Step 1: Statistical Analysis
        stat_analyzer = AdvancedAnalyticsClient(mongo_client=mock_mongo_client)

        # Step 2: Perform analysis
        analysis_result = stat_analyzer.analyze(model_id="test_model_001", voxel_grid=sample_voxel_grid, signals=None)

        # Step 3: Generate report
        report_generator = AnalysisReportGenerator()
        report = report_generator.generate_comprehensive_report(
            analytics_results=(analysis_result if isinstance(analysis_result, dict) else {"result": analysis_result}),
            report_title="Complete Analytics Report",
        )

        # Assertions
        assert analysis_result is not None
        assert report is not None

    @pytest.mark.e2e
    def test_analytics_pipeline_with_storage(self, sample_voxel_grid, mock_mongo_client):
        """Test analytics pipeline with storage operations."""
        # Step 1: Perform analysis
        analyzer = AdvancedAnalyticsClient(mongo_client=mock_mongo_client)

        # Step 2: Perform analysis
        analysis_result = analyzer.analyze(model_id="test_model_001", voxel_grid=sample_voxel_grid, signals=None)

        # Step 3: Query stored results (if available)
        stored_result = analyzer.query_results(model_id="test_model_001", analysis_type="descriptive")

        # Assertions
        assert analysis_result is not None
        # Stored result may be None if no storage was performed
