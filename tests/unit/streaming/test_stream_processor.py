"""
Unit tests for StreamProcessor.

Tests for low-latency stream processing pipeline.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
from datetime import datetime

from am_qadf.streaming.stream_processor import StreamProcessor
from am_qadf.streaming.streaming_client import StreamingConfig


class TestStreamProcessor:
    """Test suite for StreamProcessor class."""

    @pytest.fixture
    def config(self):
        """Create a StreamingConfig instance."""
        return StreamingConfig()

    @pytest.fixture
    def processor(self, config):
        """Create a StreamProcessor instance."""
        return StreamProcessor(config=config)

    @pytest.mark.unit
    def test_processor_creation(self, processor):
        """Test creating StreamProcessor."""
        assert processor is not None
        assert processor.config is not None
        assert len(processor._pipeline_stages) == 0
        assert len(processor._checkpoints) == 0
        assert processor._enable_parallel is False

    @pytest.mark.unit
    def test_create_processing_pipeline(self, processor):
        """Test creating processing pipeline."""

        def stage1(data):
            return data * 2

        def stage2(data):
            return data + 1

        stages = [stage1, stage2]
        pipeline = processor.create_processing_pipeline(stages)

        assert pipeline is not None
        assert callable(pipeline)
        assert len(processor._pipeline_stages) == 2

        # Test pipeline
        result = pipeline(5)
        assert result == 11  # (5 * 2) + 1

    @pytest.mark.unit
    def test_create_processing_pipeline_empty(self, processor):
        """Test creating pipeline with no stages."""
        with pytest.raises(ValueError, match="Pipeline must have at least one stage"):
            processor.create_processing_pipeline([])

    @pytest.mark.unit
    def test_process_with_pipeline(self, processor):
        """Test processing data with pipeline."""

        def stage(data):
            return [x * 2 for x in data]

        pipeline = processor.create_processing_pipeline([stage])

        data = [1, 2, 3]
        result = processor.process_with_pipeline(data, pipeline)

        assert result == [2, 4, 6]

        # Check statistics
        stats = processor.get_statistics()
        assert stats["batches_processed"] == 1
        assert stats["average_latency_ms"] > 0

    @pytest.mark.unit
    def test_process_with_pipeline_no_stages(self, processor):
        """Test processing with no pipeline stages."""
        data = [1, 2, 3]
        result = processor.process_with_pipeline(data)

        # Should return data unchanged and log warning
        assert result == data

    @pytest.mark.unit
    def test_add_quality_checkpoint(self, processor):
        """Test adding quality checkpoint."""

        def validator(data):
            return len(data) > 0

        processor.add_quality_checkpoint("non_empty", validator)

        assert "non_empty" in processor._checkpoints

    @pytest.mark.unit
    def test_validate_checkpoint(self, processor):
        """Test validating checkpoint."""

        def validator(data):
            return len(data) > 0

        processor.add_quality_checkpoint("non_empty", validator)

        # Valid data
        assert processor.validate_checkpoint("non_empty", [1, 2, 3]) is True

        # Invalid data
        assert processor.validate_checkpoint("non_empty", []) is False

    @pytest.mark.unit
    def test_validate_all_checkpoints(self, processor):
        """Test validating all checkpoints."""
        processor.add_quality_checkpoint("non_empty", lambda x: len(x) > 0)
        processor.add_quality_checkpoint(
            "has_positive", lambda x: any(v > 0 for v in x) if isinstance(x, (list, np.ndarray)) else x > 0
        )

        results = processor.validate_all_checkpoints([1, 2, 3])

        assert "non_empty" in results
        assert "has_positive" in results
        assert results["non_empty"] is True
        assert results["has_positive"] is True

    @pytest.mark.unit
    def test_enable_parallel_processing(self, processor):
        """Test enabling parallel processing."""
        processor.enable_parallel_processing(num_workers=4)

        assert processor._enable_parallel is True
        assert processor._num_workers == 4
        assert processor._executor is not None

        # Cleanup
        processor.disable_parallel_processing()

    @pytest.mark.unit
    def test_disable_parallel_processing(self, processor):
        """Test disabling parallel processing."""
        processor.enable_parallel_processing(num_workers=2)
        assert processor._enable_parallel is True

        processor.disable_parallel_processing()

        assert processor._enable_parallel is False
        assert processor._executor is None

    @pytest.mark.unit
    def test_measure_latency(self, processor):
        """Test measuring latency."""
        start_time = datetime.now()
        end_time = datetime.now()

        latency = processor.measure_latency(start_time, end_time)

        assert latency >= 0

    @pytest.mark.unit
    def test_get_statistics(self, processor):
        """Test getting statistics."""
        # Process some data
        pipeline = processor.create_processing_pipeline([lambda x: x])
        processor.process_with_pipeline([1, 2, 3], pipeline)

        stats = processor.get_statistics()

        assert stats["batches_processed"] == 1
        assert stats["average_latency_ms"] > 0
        assert stats["parallel_enabled"] is False
        assert stats["num_stages"] == 1
        assert stats["num_checkpoints"] == 0

    @pytest.mark.unit
    def test_reset_statistics(self, processor):
        """Test resetting statistics."""
        # Process some data
        pipeline = processor.create_processing_pipeline([lambda x: x])
        processor.process_with_pipeline([1, 2, 3], pipeline)

        stats_before = processor.get_statistics()
        assert stats_before["batches_processed"] == 1

        processor.reset_statistics()

        stats_after = processor.get_statistics()
        assert stats_after["batches_processed"] == 0
        assert stats_after["total_processing_time_ms"] == 0.0

    @pytest.mark.unit
    def test_add_stage(self, processor):
        """Test adding processing stage."""

        def stage(data):
            return data * 2

        processor.add_stage(stage)

        assert len(processor._pipeline_stages) == 1
        assert processor._pipeline_stages[0] == stage

    @pytest.mark.unit
    def test_add_stage_at_position(self, processor):
        """Test adding stage at specific position."""

        def stage1(data):
            return data

        def stage2(data):
            return data

        processor.add_stage(stage1)
        processor.add_stage(stage2, position=0)

        assert processor._pipeline_stages[0] == stage2
        assert processor._pipeline_stages[1] == stage1

    @pytest.mark.unit
    def test_remove_stage(self, processor):
        """Test removing processing stage."""

        def stage(data):
            return data

        processor.add_stage(stage)
        assert len(processor._pipeline_stages) == 1

        processor.remove_stage(0)

        assert len(processor._pipeline_stages) == 0

    @pytest.mark.unit
    def test_clear_pipeline(self, processor):
        """Test clearing pipeline."""
        processor.add_stage(lambda x: x)
        processor.add_stage(lambda x: x * 2)

        assert len(processor._pipeline_stages) == 2

        processor.clear_pipeline()

        assert len(processor._pipeline_stages) == 0

    @pytest.mark.unit
    def test_remove_quality_checkpoint(self, processor):
        """Test removing quality checkpoint."""
        processor.add_quality_checkpoint("test", lambda x: True)
        assert "test" in processor._checkpoints

        processor.remove_quality_checkpoint("test")

        assert "test" not in processor._checkpoints

    @pytest.mark.unit
    def test_process_with_pipeline_error_handling(self, processor):
        """Test error handling in pipeline processing."""

        def failing_stage(data):
            raise ValueError("Stage failed")

        pipeline = processor.create_processing_pipeline([failing_stage])

        with pytest.raises(ValueError):
            processor.process_with_pipeline([1, 2, 3], pipeline)

        stats = processor.get_statistics()
        # Error is incremented once in the stage handler and once in the outer exception handler
        assert stats["errors"] == 2
