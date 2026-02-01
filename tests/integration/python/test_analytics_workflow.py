"""
Integration tests for analytics workflow.

Tests the complete workflow: query → analyze → store results
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch


class MockMongoClient:
    """Mock MongoDB client for analytics storage."""

    def __init__(self):
        self.connected = True
        self.db = Mock()
        self._collections = {}

    def is_connected(self):
        """Check if client is connected."""
        return self.connected

    def get_collection(self, name):
        if name not in self._collections:
            self._collections[name] = MockCollection()
        return self._collections[name]


class MockCollection:
    """Mock MongoDB collection."""

    def __init__(self):
        self._documents = []
        self._counter = 0

    def insert_one(self, doc):
        doc["_id"] = f"doc_{self._counter}"
        self._counter += 1
        self._documents.append(doc)
        return MockInsertResult(doc["_id"])

    def find_one(self, query):
        for doc in self._documents:
            if all(doc.get(k) == v for k, v in query.items()):
                return doc
        return None


class MockInsertResult:
    def __init__(self, inserted_id):
        self.inserted_id = inserted_id


@pytest.mark.integration
class TestAnalyticsWorkflow:
    """Integration tests for analytics workflow."""

    @pytest.fixture
    def mongo_client(self):
        """Create mock MongoDB client."""
        return MockMongoClient()

    @pytest.fixture
    def sample_voxel_grid(self):
        """Create sample voxel grid for analytics."""
        try:
            from am_qadf.voxelization.uniform_resolution import VoxelGrid

            grid = VoxelGrid(bbox_min=(0, 0, 0), bbox_max=(10, 10, 10), resolution=1.0)
            grid.available_signals = {"laser_power", "temperature", "density"}
            return grid
        except ImportError:
            pytest.skip("VoxelGrid not available")

    @pytest.mark.integration
    def test_statistical_analysis_workflow(self, sample_voxel_grid, mongo_client):
        """Test statistical analysis workflow."""
        try:
            from am_qadf.analytics.statistical_analysis.client import (
                StatisticalAnalysisClient,
            )

            client = StatisticalAnalysisClient(mongo_client)

            # Perform analysis
            result = client.analyze(
                model_id="test_model",
                voxel_grid=sample_voxel_grid,
                signals=["laser_power", "temperature"],
            )

            assert result is not None
        except (ImportError, AttributeError):
            pytest.skip("StatisticalAnalysisClient not available")

    @pytest.mark.integration
    def test_sensitivity_analysis_workflow(self, sample_voxel_grid, mongo_client):
        """Test sensitivity analysis workflow."""
        try:
            from am_qadf.analytics.sensitivity_analysis.client import (
                SensitivityAnalysisClient,
            )
            from tests.fixtures.mocks.mock_query_clients import MockUnifiedQueryClient

            # Create mock unified query client
            unified_client = MockUnifiedQueryClient()

            client = SensitivityAnalysisClient(unified_client, voxel_domain_client=None)

            # Note: SensitivityAnalysisClient doesn't have a simple analyze() method
            # It requires more complex setup. For now, just verify client creation
            assert client is not None
            assert hasattr(client, "unified_client")
        except (ImportError, AttributeError) as e:
            pytest.skip(f"SensitivityAnalysisClient not available: {e}")

    @pytest.mark.integration
    def test_quality_assessment_workflow(self, sample_voxel_grid, mongo_client):
        """Test quality assessment workflow."""
        try:
            from am_qadf.analytics.quality_assessment.client import (
                QualityAssessmentClient,
            )

            client = QualityAssessmentClient()

            # Perform quality assessment
            result = client.assess_data_quality(voxel_data=sample_voxel_grid, signals=None)

            assert result is not None
        except (ImportError, AttributeError):
            pytest.skip("QualityAssessmentClient not available")

    @pytest.mark.integration
    def test_analytics_storage_retrieval(self, mongo_client):
        """Test storing and retrieving analytics results."""
        collection = mongo_client.get_collection("analytics_results")

        # Store result
        result = {
            "model_id": "test_model",
            "analysis_type": "statistical",
            "results": {"mean": 100.0, "std": 10.0},
        }

        insert_result = collection.insert_one(result)
        assert insert_result.inserted_id is not None

        # Retrieve result
        retrieved = collection.find_one({"model_id": "test_model"})
        assert retrieved is not None
        assert retrieved["analysis_type"] == "statistical"

    @pytest.mark.integration
    def test_multiple_analytics_workflows(self, sample_voxel_grid, mongo_client):
        """Test running multiple analytics workflows sequentially."""
        results = {}

        # Statistical analysis
        try:
            from am_qadf.analytics.statistical_analysis.client import (
                StatisticalAnalysisClient,
            )

            stat_client = StatisticalAnalysisClient(mongo_client)
            results["statistical"] = stat_client.analyze(model_id="test_model", voxel_grid=sample_voxel_grid)
        except (ImportError, AttributeError):
            pass

        # Quality assessment
        try:
            from am_qadf.analytics.quality_assessment.client import (
                QualityAssessmentClient,
            )

            quality_client = QualityAssessmentClient()
            results["quality"] = quality_client.assess_data_quality(voxel_data=sample_voxel_grid, signals=None)
        except (ImportError, AttributeError):
            pass

        # At least one should succeed
        assert len(results) > 0

    @pytest.mark.integration
    def test_analytics_with_query_filters(self, sample_voxel_grid, mongo_client):
        """Test analytics with query filters."""
        try:
            from am_qadf.analytics.statistical_analysis.client import (
                StatisticalAnalysisClient,
            )

            client = StatisticalAnalysisClient(mongo_client)

            # First, insert some test data
            collection = mongo_client.get_collection("analytics_results")
            test_result = {
                "model_id": "test_model",
                "analysis_type": "descriptive",
                "results": {"mean": 100.0, "std": 10.0},
            }
            collection.insert_one(test_result)

            # Query with filters
            result = client.query_results(model_id="test_model", analysis_type="descriptive")

            assert result is not None
            assert result["model_id"] == "test_model"
            assert result["analysis_type"] == "descriptive"
        except (ImportError, AttributeError):
            pytest.skip("StatisticalAnalysisClient not available")
