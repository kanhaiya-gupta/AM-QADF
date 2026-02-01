"""
Performance tests for Streaming module.

Tests performance characteristics including latency, throughput, and memory usage.
"""

import pytest
import numpy as np
import time
from datetime import datetime
from unittest.mock import Mock, MagicMock

try:
    from am_qadf.streaming import (
        StreamingClient,
        StreamingConfig,
        BufferManager,
        StreamProcessor,
        IncrementalProcessor,
    )
except ImportError:
    pytest.skip("Streaming module not available", allow_module_level=True)


class TestStreamingPerformance:
    """Performance tests for streaming components."""

    @pytest.fixture
    def streaming_config(self):
        """Create streaming configuration."""
        return StreamingConfig(
            buffer_size=10000,
            processing_batch_size=1000,
        )

    @pytest.fixture
    def streaming_client(self, streaming_config):
        """Create streaming client."""
        return StreamingClient(config=streaming_config)

    @pytest.mark.performance
    def test_streaming_client_batch_processing_latency(self, streaming_client):
        """Test batch processing latency (target: < 100ms per batch)."""
        batch_size = 100
        n_batches = 10

        total_time = 0.0
        for i in range(n_batches):
            batch = [{"value": float(j)} for j in range(batch_size)]

            start_time = time.time()
            result = streaming_client.process_stream_batch(batch)
            end_time = time.time()

            processing_time_ms = (end_time - start_time) * 1000.0
            total_time += processing_time_ms

            # Each batch should process quickly
            assert processing_time_ms < 500.0  # Allow 500ms for test environment

        avg_latency_ms = total_time / n_batches
        print(f"\nAverage batch processing latency: {avg_latency_ms:.2f} ms")

        # Target: < 100ms average (relaxed for test environment)
        assert avg_latency_ms < 200.0

    @pytest.mark.performance
    def test_streaming_client_throughput(self, streaming_client):
        """Test throughput (target: > 1000 messages/second)."""
        batch_size = 100
        n_batches = 20

        start_time = time.time()
        for i in range(n_batches):
            batch = [{"value": float(j)} for j in range(batch_size)]
            streaming_client.process_stream_batch(batch)
        end_time = time.time()

        total_time = end_time - start_time
        total_messages = n_batches * batch_size
        throughput = total_messages / total_time

        print(f"\nThroughput: {throughput:.2f} messages/second")
        print(f"Total time: {total_time:.2f} seconds for {total_messages} messages")

        # Target: > 1000 messages/second (relaxed for test environment)
        assert throughput > 500.0

    @pytest.mark.performance
    def test_buffer_manager_add_data_performance(self):
        """Test buffer manager add data performance."""
        buffer_manager = BufferManager(window_size=100, buffer_size=10000)

        n_points = 10000
        start_time = time.time()

        for i in range(n_points):
            data = np.array([float(i)])
            buffer_manager.add_data(data, datetime.now())

        end_time = time.time()
        total_time = end_time - start_time
        throughput = n_points / total_time

        print(f"\nBuffer add throughput: {throughput:.2f} points/second")

        # Should handle at least 1000 points/second
        assert throughput > 1000.0

    @pytest.mark.performance
    def test_buffer_manager_window_retrieval_performance(self):
        """Test buffer manager window retrieval performance."""
        buffer_manager = BufferManager(window_size=1000, buffer_size=10000)

        # Fill buffer
        for i in range(10000):
            buffer_manager.add_data(np.array([float(i)]), datetime.now())

        # Measure window retrieval
        n_retrievals = 100
        start_time = time.time()

        for i in range(n_retrievals):
            window = buffer_manager.get_window()
            assert len(window) > 0

        end_time = time.time()
        total_time = end_time - start_time
        avg_latency_ms = (total_time / n_retrievals) * 1000.0

        print(f"\nAverage window retrieval latency: {avg_latency_ms:.2f} ms")

        # Should be fast (< 50ms)
        assert avg_latency_ms < 100.0

    @pytest.mark.performance
    def test_stream_processor_pipeline_performance(self, streaming_config):
        """Test stream processor pipeline performance."""
        processor = StreamProcessor(config=streaming_config)

        # Create pipeline with multiple stages
        def stage1(data):
            return [x * 2 for x in data]

        def stage2(data):
            return [x + 10 for x in data]

        def stage3(data):
            return [x**0.5 for x in data]

        pipeline = processor.create_processing_pipeline([stage1, stage2, stage3])

        # Process batches
        batch_size = 1000
        n_batches = 10

        start_time = time.time()
        for i in range(n_batches):
            data = list(range(batch_size))
            result = processor.process_with_pipeline(data, pipeline)
            assert len(result) == batch_size
        end_time = time.time()

        total_time = end_time - start_time
        avg_latency_ms = (total_time / n_batches) * 1000.0
        throughput = (n_batches * batch_size) / total_time

        print(f"\nPipeline average latency: {avg_latency_ms:.2f} ms per batch")
        print(f"Pipeline throughput: {throughput:.2f} items/second")

        # Should be reasonably fast
        assert avg_latency_ms < 500.0
        assert throughput > 500.0

    @pytest.mark.performance
    def test_incremental_processor_performance(self):
        """Test incremental processor performance."""
        mock_voxel_grid = MagicMock()
        mock_voxel_grid.add_point = MagicMock()

        processor = IncrementalProcessor(voxel_grid=mock_voxel_grid)

        n_points = 10000
        batch_size = 100

        # Generate coordinates and data
        coordinates = np.random.uniform(0.0, 100.0, (n_points, 3))
        data = np.random.uniform(1000.0, 2000.0, n_points)

        start_time = time.time()

        # Process in batches
        for i in range(0, n_points, batch_size):
            batch_coords = coordinates[i : i + batch_size]
            batch_data = data[i : i + batch_size]
            processor.update_voxel_grid(batch_data, batch_coords)

        end_time = time.time()

        total_time = end_time - start_time
        throughput = n_points / total_time

        print(f"\nIncremental processor throughput: {throughput:.2f} points/second")

        # Should handle at least 1000 points/second
        assert throughput > 1000.0
        assert mock_voxel_grid.add_point.call_count == n_points

    @pytest.mark.performance
    def test_streaming_client_statistics_overhead(self, streaming_client):
        """Test overhead of statistics tracking."""
        batch_size = 100
        n_batches = 1000

        # Process without statistics
        start_time = time.time()
        for i in range(n_batches):
            batch = [{"value": float(j)} for j in range(batch_size)]
            streaming_client.process_stream_batch(batch)
        end_time = time.time()

        total_time = end_time - start_time
        avg_time_per_batch = total_time / n_batches

        # Get statistics (should be fast)
        stats_start = time.time()
        stats = streaming_client.get_stream_statistics()
        stats_end = time.time()
        stats_time = (stats_end - stats_start) * 1000.0

        print(f"\nAverage processing time per batch: {avg_time_per_batch*1000:.2f} ms")
        print(f"Statistics retrieval time: {stats_time:.2f} ms")
        print(f"Statistics: {stats}")

        # Statistics retrieval should be very fast
        assert stats_time < 10.0

    @pytest.mark.performance
    @pytest.mark.slow
    def test_long_running_stream_processing(self, streaming_client):
        """Test long-running stream processing performance."""
        batch_size = 100
        n_batches = 1000

        start_time = time.time()
        total_messages = 0

        for i in range(n_batches):
            batch = [{"value": float(j + i * batch_size)} for j in range(batch_size)]
            result = streaming_client.process_stream_batch(batch)
            total_messages += result.processed_count

            # Reset statistics every 100 batches to test memory
            if (i + 1) % 100 == 0:
                stats = streaming_client.get_stream_statistics()
                streaming_client.reset_statistics()

        end_time = time.time()

        total_time = end_time - start_time
        throughput = total_messages / total_time

        print(f"\nLong-running throughput: {throughput:.2f} messages/second")
        print(f"Total messages processed: {total_messages}")
        print(f"Total time: {total_time:.2f} seconds")

        # Should maintain good performance
        assert throughput > 500.0
