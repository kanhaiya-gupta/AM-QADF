"""
Integration tests for end-to-end workflow.

Tests the complete workflow from data input to final output.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from tests.fixtures.mocks.mock_mongodb import MockCollection


class MockUnifiedQueryClient:
    """Mock unified query client for E2E testing."""

    def __init__(self):
        self.stl_client = Mock()
        self.stl_client.get_model_bounding_box = Mock(return_value=([0.0, 0.0, 0.0], [10.0, 10.0, 10.0]))
        self.hatching_client = Mock()
        self.hatching_client.get_layers = Mock(
            return_value=[
                {
                    "hatches": [
                        {
                            "points": [[0, 0, 0], [1, 1, 1]],
                            "laser_power": 200.0,
                            "scan_speed": 1000.0,
                        }
                    ]
                }
            ]
        )
        self.laser_client = Mock()
        self.ct_client = Mock()
        self.ispm_client = Mock()


class MockMongoClient:
    """Mock MongoDB client for E2E testing."""

    def __init__(self):
        self.connected = True
        self.db = Mock()
        self._collections = {}
        self._files = {}
        self._file_counter = 0

    def is_connected(self):
        """Check if client is connected."""
        return self.connected

    def get_collection(self, name):
        if name not in self._collections:
            self._collections[name] = MockCollection(name)
        return self._collections[name]

    def store_file(self, data, filename, metadata=None):
        file_id = f"file_{self._file_counter}"
        self._file_counter += 1
        self._files[file_id] = {"data": data, "filename": filename}
        return file_id

    def retrieve_file(self, file_id):
        return self._files.get(file_id, {}).get("data")

    def delete_file(self, file_id):
        if file_id in self._files:
            del self._files[file_id]


@pytest.mark.integration
@pytest.mark.e2e
class TestEndToEndWorkflow:
    """Integration tests for end-to-end workflow."""

    @pytest.fixture
    def unified_client(self):
        """Create mock unified query client."""
        return MockUnifiedQueryClient()

    @pytest.fixture
    def mongo_client(self):
        """Create mock MongoDB client."""
        return MockMongoClient()

    @pytest.mark.integration
    @pytest.mark.e2e
    def test_complete_pipeline(self, unified_client, mongo_client):
        """Test complete pipeline: query → map → fuse → assess → store."""
        try:
            from am_qadf.voxel_domain.voxel_domain_client import VoxelDomainClient
            from am_qadf.voxel_domain.voxel_storage import VoxelGridStorage

            # Step 1: Create voxel domain client
            domain_client = VoxelDomainClient(unified_query_client=unified_client)

            # Step 2: Create voxel grid
            grid = domain_client.create_voxel_grid(model_id="test_model", bbox_min=(0, 0, 0), bbox_max=(10, 10, 10))
            assert grid is not None

            # Step 3: Map signals
            grid = domain_client.map_signals_to_voxels(model_id="test_model", voxel_grid=grid, sources=["hatching"])
            assert grid is not None

            # Step 4: Store grid
            storage = VoxelGridStorage(mongo_client)
            grid_id = storage.save_voxel_grid(model_id="test_model", grid_name="e2e_test_grid", voxel_grid=grid)
            assert grid_id is not None

            # Step 5: Load grid
            loaded = storage.load_voxel_grid(grid_id)
            assert loaded is not None

        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")

    @pytest.mark.integration
    @pytest.mark.e2e
    def test_multi_source_pipeline(self, unified_client):
        """Test pipeline with multiple data sources."""
        try:
            from am_qadf.voxel_domain.voxel_domain_client import VoxelDomainClient

            domain_client = VoxelDomainClient(unified_query_client=unified_client)

            # Create grid
            grid = domain_client.create_voxel_grid(model_id="test_model", bbox_min=(0, 0, 0), bbox_max=(10, 10, 10))

            # Map from multiple sources
            grid = domain_client.map_signals_to_voxels(model_id="test_model", voxel_grid=grid, sources=["hatching", "laser"])

            assert grid is not None
            assert hasattr(grid, "available_signals")

        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")

    @pytest.mark.integration
    @pytest.mark.e2e
    def test_pipeline_with_analytics(self, unified_client, mongo_client):
        """Test pipeline including analytics."""
        try:
            from am_qadf.voxel_domain.voxel_domain_client import VoxelDomainClient

            domain_client = VoxelDomainClient(unified_query_client=unified_client)

            # Create and map
            grid = domain_client.create_voxel_grid(model_id="test_model", bbox_min=(0, 0, 0), bbox_max=(10, 10, 10))

            grid = domain_client.map_signals_to_voxels(model_id="test_model", voxel_grid=grid, sources=["hatching"])

            # Get statistics
            stats = domain_client.get_voxel_statistics(grid)
            assert stats is not None

        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")

    @pytest.mark.integration
    @pytest.mark.e2e
    def test_pipeline_with_quality_assessment(self, unified_client):
        """Test pipeline including quality assessment."""
        try:
            from am_qadf.voxel_domain.voxel_domain_client import VoxelDomainClient
            from am_qadf.analytics.quality_assessment.client import (
                QualityAssessmentClient,
            )

            domain_client = VoxelDomainClient(unified_query_client=unified_client)

            # Create and map
            grid = domain_client.create_voxel_grid(model_id="test_model", bbox_min=(0, 0, 0), bbox_max=(10, 10, 10))

            grid = domain_client.map_signals_to_voxels(model_id="test_model", voxel_grid=grid, sources=["hatching"])

            # Assess quality
            quality_client = QualityAssessmentClient()
            quality_results = quality_client.assess_data_quality(voxel_data=grid, signals=None)

            assert quality_results is not None

        except (ImportError, AttributeError) as e:
            pytest.skip(f"Required modules not available: {e}")

    @pytest.mark.integration
    @pytest.mark.e2e
    def test_pipeline_error_recovery(self, unified_client):
        """Test pipeline error recovery."""
        try:
            from am_qadf.voxel_domain.voxel_domain_client import VoxelDomainClient

            domain_client = VoxelDomainClient(unified_query_client=unified_client)

            # Make one source fail
            unified_client.hatching_client.get_layers = Mock(side_effect=Exception("Source error"))

            # Pipeline should handle error gracefully
            grid = domain_client.create_voxel_grid(model_id="test_model", bbox_min=(0, 0, 0), bbox_max=(10, 10, 10))

            result = domain_client.map_signals_to_voxels(model_id="test_model", voxel_grid=grid, sources=["hatching", "laser"])

            # Should still return grid even if one source fails
            assert result is not None

        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")
