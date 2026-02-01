"""
Integration tests for Streaming module components.

Tests integration of streaming components working together.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
import time

try:
    from am_qadf.streaming import (
        StreamingClient,
        StreamingConfig,
        IncrementalProcessor,
        BufferManager,
        StreamProcessor,
        StreamStorage,
    )
except ImportError:
    pytest.skip("Streaming module not available", allow_module_level=True)


class TestStreamingIntegration:
    """Integration tests for streaming components."""

    @pytest.fixture
    def streaming_config(self):
        """Create streaming configuration."""
        return StreamingConfig(
            buffer_size=100,
            processing_batch_size=10,
        )

    @pytest.fixture
    def streaming_client(self, streaming_config):
        """Create streaming client."""
        return StreamingClient(config=streaming_config)

    @pytest.fixture
    def buffer_manager(self):
        """Create buffer manager."""
        return BufferManager(window_size=10, buffer_size=50)

    @pytest.fixture
    def incremental_processor(self):
        """Create incremental processor with mock voxel grid."""
        mock_voxel_grid = MagicMock()
        mock_voxel_grid.add_point = MagicMock()
        return IncrementalProcessor(voxel_grid=mock_voxel_grid)

    @pytest.mark.integration
    def test_streaming_client_with_buffer_manager(self, streaming_client, buffer_manager):
        """Test streaming client integrated with buffer manager."""
        # Process data through buffer manager
        for i in range(15):
            data = np.array([float(i)])
            buffer_manager.add_data(data, datetime.now())

        # Get window from buffer
        window = buffer_manager.get_window()

        # Process window through streaming client
        window_list = window.tolist() if hasattr(window, "tolist") else [{"value": float(x)} for x in window.flatten()]
        result = streaming_client.process_stream_batch(window_list)

        assert isinstance(result, type(streaming_client.process_stream_batch([{"value": 1.0}])))
        assert result.processed_count > 0

    @pytest.mark.integration
    def test_streaming_client_with_incremental_processor(self, streaming_client, incremental_processor):
        """Test streaming client integrated with incremental processor."""

        # Register incremental processor
        def voxel_processor(data_batch):
            coordinates = np.array([[10.0, 20.0, 30.0]] * len(data_batch))
            new_data = np.array([item.get("value", 0.0) for item in data_batch])
            incremental_processor.update_voxel_grid(new_data, coordinates)
            return {"updated": True, "points": len(data_batch)}

        streaming_client.register_processor("voxel_processor", voxel_processor)

        # Process batch
        data_batch = [
            {"value": 1.0, "timestamp": datetime.now()},
            {"value": 2.0, "timestamp": datetime.now()},
            {"value": 3.0, "timestamp": datetime.now()},
        ]

        result = streaming_client.process_stream_batch(data_batch)

        assert result.voxel_updates is not None
        assert result.voxel_updates["updated"] is True
        assert incremental_processor.voxel_grid.add_point.call_count == 3

    @pytest.mark.integration
    def test_buffer_manager_with_incremental_processor(self, buffer_manager, incremental_processor):
        """Test buffer manager integrated with incremental processor."""
        # Add data to buffer
        coordinates_list = []
        for i in range(10):
            coordinates_list.append([10.0 + i, 20.0 + i, 30.0 + i])
            buffer_manager.add_data(np.array([float(i)]), datetime.now(), metadata={"coordinates": coordinates_list[-1]})

        # Get window and process with incremental processor
        window, timestamps, metadata_list = buffer_manager.flush_buffer()

        # Update voxel grid
        coordinates = np.array([m["coordinates"] for m in metadata_list])
        new_data = window.flatten()[: len(coordinates)]

        incremental_processor.update_voxel_grid(new_data, coordinates)

        stats = incremental_processor.get_statistics()
        assert stats["total_points_processed"] == 10

    @pytest.mark.integration
    def test_stream_processor_pipeline(self, streaming_config):
        """Test stream processor with processing pipeline."""
        processor = StreamProcessor(config=streaming_config)

        # Create pipeline stages
        def stage1(data):
            # Multiply by 2
            return [x * 2 for x in data if isinstance(x, (int, float))] or data

        def stage2(data):
            # Add 10
            return [x + 10 for x in data if isinstance(x, (int, float))] or data

        def stage3(data):
            # Filter > 20
            return [x for x in data if isinstance(x, (int, float)) and x > 20] or data

        stages = [stage1, stage2, stage3]
        pipeline = processor.create_processing_pipeline(stages)

        # Process data
        data = [5, 10, 15]
        result = processor.process_with_pipeline(data, pipeline)

        # Expected: [5*2+10=20, 10*2+10=30, 15*2+10=40] -> filter > 20 -> [30, 40]
        assert len(result) == 2
        assert 30 in result
        assert 40 in result

    @pytest.mark.integration
    def test_stream_processor_with_checkpoints(self, streaming_config):
        """Test stream processor with quality checkpoints."""
        processor = StreamProcessor(config=streaming_config)

        # Add checkpoints
        processor.add_quality_checkpoint("non_empty", lambda x: len(x) > 0)
        processor.add_quality_checkpoint("has_positive", lambda x: any(v > 0 for v in x if isinstance(v, (int, float))))

        # Process data
        pipeline = processor.create_processing_pipeline([lambda x: x])
        data = [1, 2, 3]
        result = processor.process_with_pipeline(data, pipeline)

        # Validate checkpoints
        checkpoint_results = processor.validate_all_checkpoints(data)

        assert checkpoint_results["non_empty"] is True
        assert checkpoint_results["has_positive"] is True

    @pytest.mark.integration
    def test_complete_streaming_workflow(self, streaming_client, buffer_manager, incremental_processor):
        """Test complete streaming workflow with all components."""

        # Register processor
        def complete_processor(data_batch):
            # Extract coordinates and values
            values = []
            coordinates = []
            for item in data_batch:
                if isinstance(item, dict):
                    values.append(item.get("value", 0.0))
                    coords = item.get("coordinates", [10.0, 20.0, 30.0])
                    coordinates.append(coords)

            if values and coordinates:
                incremental_processor.update_voxel_grid(np.array(values), np.array(coordinates))

            return {"processed": True, "points": len(values), "regions": incremental_processor.get_updated_regions()}

        streaming_client.register_processor("complete_processor", complete_processor)

        # Simulate streaming data
        for i in range(20):
            data = {"value": float(i), "coordinates": [10.0 + i, 20.0 + i, 30.0 + i], "timestamp": datetime.now()}
            buffer_manager.add_data(np.array([data["value"]]), data["timestamp"], metadata=data)

            # Process in batches
            if (i + 1) % 5 == 0:
                window, timestamps, metadata_list = buffer_manager.get_sliding_window(5)

                # Convert to batch format
                batch = []
                for idx, metadata in enumerate(metadata_list[-5:]):
                    batch.append(
                        {
                            "value": window.flatten()[idx] if len(window.flatten()) > idx else 0.0,
                            "coordinates": metadata.get("coordinates", [0.0, 0.0, 0.0]),
                            "timestamp": timestamps[idx],
                        }
                    )

                # Process through streaming client
                result = streaming_client.process_stream_batch(batch)
                assert result.processed_count == 5

        # Check statistics
        stats = streaming_client.get_stream_statistics()
        assert stats["messages_processed"] == 20
        assert stats["batches_processed"] == 4

        # Check incremental processor
        inc_stats = incremental_processor.get_statistics()
        assert inc_stats["total_points_processed"] == 20

    @pytest.mark.integration
    def test_stream_storage_integration(self):
        """Test stream storage integration (mocked)."""
        mock_redis = MagicMock()
        mock_redis.ping.return_value = True
        mock_redis.setex = MagicMock()
        mock_redis.get.return_value = None

        mock_mongo = MagicMock()
        mock_collection = MagicMock()
        mock_collection.insert_many.return_value = MagicMock(inserted_ids=["id1", "id2"])
        mock_db = MagicMock()
        mock_db.get_collection.return_value = mock_collection
        mock_mongo.get_database.return_value = mock_db

        storage = StreamStorage(redis_client=mock_redis, mongo_client=mock_mongo)

        # Cache data
        storage.cache_recent_data("test_key", {"value": 123}, ttl_seconds=3600)
        mock_redis.setex.assert_called_once()

        # Store batch
        batch_data = [
            {"value": 1.0, "timestamp": datetime.now()},
            {"value": 2.0, "timestamp": datetime.now()},
        ]
        inserted = storage.store_batch(batch_data)
        assert inserted == 2

    @pytest.mark.integration
    def test_parallel_processing(self, streaming_config):
        """Test parallel processing with stream processor."""
        processor = StreamProcessor(config=streaming_config)
        processor.enable_parallel_processing(num_workers=2)

        # Create simple pipeline
        pipeline = processor.create_processing_pipeline([lambda x: [v * 2 for v in x]])

        # Process multiple batches in parallel
        batches = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]

        results = processor.process_parallel(batches, pipeline)

        assert len(results) == 3
        assert results[0] == [2, 4, 6]
        assert results[1] == [8, 10, 12]
        assert results[2] == [14, 16, 18]

        # Cleanup
        processor.disable_parallel_processing()
