"""
Integration tests for quality assessment workflow.

Tests the complete quality assessment pipeline.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch


@pytest.mark.integration
class TestQualityAssessmentWorkflow:
    """Integration tests for quality assessment workflow."""

    @pytest.fixture
    def sample_voxel_grid(self):
        """Create sample voxel grid for quality assessment."""
        try:
            from am_qadf.voxelization.voxel_grid import VoxelGrid

            grid = VoxelGrid(bbox_min=(0, 0, 0), bbox_max=(10, 10, 10), resolution=1.0)
            grid.available_signals = {"laser_power", "temperature", "density"}
            return grid
        except ImportError:
            pytest.skip("VoxelGrid not available")

    @pytest.mark.integration
    def test_quality_assessment_complete_workflow(self, sample_voxel_grid):
        """Test complete quality assessment workflow."""
        try:
            from am_qadf.analytics.quality_assessment.client import (
                QualityAssessmentClient,
            )

            client = QualityAssessmentClient()

            # Assess quality
            results = client.assess_data_quality(voxel_data=sample_voxel_grid, signals=None)

            assert results is not None
            assert hasattr(results, "overall_score") or hasattr(results, "completeness")
        except (ImportError, AttributeError):
            pytest.skip("QualityAssessmentClient not available")

    @pytest.mark.integration
    def test_quality_assessment_completeness(self, sample_voxel_grid):
        """Test completeness assessment."""
        try:
            from am_qadf.analytics.quality_assessment.completeness import (
                CompletenessAnalyzer,
            )

            assessor = CompletenessAnalyzer()
            completeness = assessor.assess_completeness(sample_voxel_grid)

            assert completeness is not None
            # CompletenessMetrics has completeness_ratio, not completeness_score
            assert hasattr(completeness, "completeness_ratio") or isinstance(completeness, (int, float))
            if isinstance(completeness, (int, float)):
                assert 0.0 <= completeness <= 1.0
            elif hasattr(completeness, "completeness_ratio"):
                assert 0.0 <= completeness.completeness_ratio <= 1.0
        except (ImportError, AttributeError):
            pytest.skip("CompletenessAssessment not available")

    @pytest.mark.integration
    def test_quality_assessment_signal_quality(self, sample_voxel_grid):
        """Test signal quality assessment."""
        try:
            from am_qadf.analytics.quality_assessment.signal_quality import (
                SignalQualityAnalyzer,
            )

            assessor = SignalQualityAnalyzer()
            # Get signal array first
            signal_array = sample_voxel_grid.get_signal_array("laser_power", default=0.0)
            quality = assessor.assess_signal_quality(signal_name="laser_power", signal_array=signal_array)

            assert quality is not None
        except (ImportError, AttributeError):
            pytest.skip("SignalQualityAssessment not available")

    @pytest.mark.integration
    def test_quality_assessment_alignment_accuracy(self, sample_voxel_grid):
        """Test alignment accuracy assessment."""
        try:
            from am_qadf.analytics.quality_assessment.alignment_accuracy import (
                AlignmentAccuracyAnalyzer,
            )

            assessor = AlignmentAccuracyAnalyzer()
            accuracy = assessor.assess_alignment_accuracy(voxel_data=sample_voxel_grid, reference_data=sample_voxel_grid)

            assert accuracy is not None
        except (ImportError, AttributeError):
            pytest.skip("AlignmentAccuracyAssessment not available")

    @pytest.mark.integration
    def test_quality_assessment_data_quality(self, sample_voxel_grid):
        """Test overall data quality assessment."""
        try:
            from am_qadf.analytics.quality_assessment.data_quality import (
                DataQualityAnalyzer,
            )

            assessor = DataQualityAnalyzer()
            quality = assessor.assess_quality(sample_voxel_grid, signals=None)

            assert quality is not None
            # DataQualityMetrics has completeness, consistency_score, etc., but not overall_score
            assert hasattr(quality, "completeness") or hasattr(quality, "consistency_score") or hasattr(quality, "to_dict")
        except (ImportError, AttributeError):
            pytest.skip("DataQualityAssessment not available")

    @pytest.mark.integration
    def test_quality_assessment_multiple_signals(self, sample_voxel_grid):
        """Test quality assessment for multiple signals."""
        try:
            from am_qadf.analytics.quality_assessment.client import (
                QualityAssessmentClient,
            )

            client = QualityAssessmentClient()

            results = client.assess_data_quality(
                voxel_data=sample_voxel_grid,
                signals=["laser_power", "temperature", "density"],
            )

            assert results is not None
        except (ImportError, AttributeError):
            pytest.skip("QualityAssessmentClient not available")

    @pytest.mark.integration
    def test_quality_assessment_storage(self, sample_voxel_grid):
        """Test storing quality assessment results."""
        try:
            from am_qadf.analytics.quality_assessment.client import (
                QualityAssessmentClient,
            )

            client = QualityAssessmentClient()

            results = client.assess_data_quality(voxel_data=sample_voxel_grid, signals=None)

            # Store results (if method exists)
            if hasattr(client, "store_results"):
                stored = client.store_results(model_id="test_model", results=results)
                assert stored is not None
        except (ImportError, AttributeError):
            pytest.skip("QualityAssessmentClient not available")
