"""
Stream Processor

Low-latency stream processing pipeline.
Provides pipeline creation, parallel processing, quality checkpoints, and latency measurement.
"""

from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
from datetime import datetime
import threading
import time
from functools import wraps
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

from .streaming_client import StreamingConfig


class StreamProcessor:
    """
    Low-latency stream processing pipeline.

    Provides:
    - Processing pipeline creation from stages
    - Parallel processing for multiple streams
    - Quality checkpoints
    - Latency measurement
    - Performance monitoring
    """

    def __init__(self, config: StreamingConfig):
        """
        Initialize stream processor.

        Args:
            config: StreamingConfig with processing settings
        """
        self.config = config

        # Processing pipeline stages
        self._pipeline_stages: List[Callable] = []

        # Quality checkpoints
        self._checkpoints: Dict[str, Callable] = {}

        # Parallel processing
        self._executor: Optional[ThreadPoolExecutor] = None
        self._num_workers = 1
        self._enable_parallel = False

        # Statistics
        self._stats = {
            "batches_processed": 0,
            "total_processing_time_ms": 0.0,
            "average_latency_ms": 0.0,
            "min_latency_ms": float("inf"),
            "max_latency_ms": 0.0,
            "checkpoints_passed": 0,
            "checkpoints_failed": 0,
            "errors": 0,
        }

        self._lock = threading.Lock()

        logger.info("StreamProcessor initialized")

    def create_processing_pipeline(self, stages: List[Callable]) -> Callable:
        """
        Create processing pipeline from stages.

        Args:
            stages: List of callable processing stages

        Returns:
            Pipeline function that processes data through all stages
        """
        if not stages:
            raise ValueError("Pipeline must have at least one stage")

        self._pipeline_stages = stages

        def pipeline(data: Any) -> Any:
            """Pipeline function that processes data through stages."""
            result = data
            for i, stage in enumerate(self._pipeline_stages):
                try:
                    result = stage(result)
                except Exception as e:
                    logger.error(f"Error in pipeline stage {i}: {e}")
                    with self._lock:
                        self._stats["errors"] += 1
                    raise
            return result

        logger.info(f"Created processing pipeline with {len(stages)} stages")
        return pipeline

    def process_with_pipeline(self, data: Any, pipeline: Optional[Callable] = None) -> Any:
        """
        Process data through pipeline.

        Args:
            data: Data to process
            pipeline: Optional pipeline function (uses default if None)

        Returns:
            Processed data
        """
        start_time = time.time()

        try:
            # Use provided pipeline or default
            if pipeline is None:
                if not self._pipeline_stages:
                    logger.warning("No pipeline stages defined, returning data unchanged")
                    return data

                # Create temporary pipeline
                pipeline = self.create_processing_pipeline(self._pipeline_stages)

            # Process data
            result = pipeline(data)

            # Measure latency
            processing_time_ms = (time.time() - start_time) * 1000.0

            # Update statistics
            with self._lock:
                self._stats["batches_processed"] += 1
                self._stats["total_processing_time_ms"] += processing_time_ms

                batches = self._stats["batches_processed"]
                self._stats["average_latency_ms"] = self._stats["total_processing_time_ms"] / batches
                self._stats["min_latency_ms"] = min(self._stats["min_latency_ms"], processing_time_ms)
                self._stats["max_latency_ms"] = max(self._stats["max_latency_ms"], processing_time_ms)

            return result

        except Exception as e:
            logger.error(f"Error processing data with pipeline: {e}")
            with self._lock:
                self._stats["errors"] += 1
            raise

    def enable_parallel_processing(self, num_workers: int = 4) -> None:
        """
        Enable parallel processing for multiple streams.

        Args:
            num_workers: Number of worker threads
        """
        if num_workers < 1:
            raise ValueError(f"num_workers must be >= 1, got {num_workers}")

        self._num_workers = num_workers
        self._enable_parallel = True

        # Create thread pool executor
        if self._executor:
            self._executor.shutdown(wait=False)

        self._executor = ThreadPoolExecutor(max_workers=num_workers)

        logger.info(f"Enabled parallel processing with {num_workers} workers")

    def disable_parallel_processing(self) -> None:
        """Disable parallel processing."""
        self._enable_parallel = False

        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None

        logger.info("Disabled parallel processing")

    def process_parallel(self, data_batches: List[Any], pipeline: Optional[Callable] = None) -> List[Any]:
        """
        Process multiple data batches in parallel.

        Args:
            data_batches: List of data batches to process
            pipeline: Optional pipeline function

        Returns:
            List of processed results
        """
        if not self._enable_parallel or not self._executor:
            # Fall back to sequential processing
            return [self.process_with_pipeline(batch, pipeline) for batch in data_batches]

        # Process in parallel
        futures = [self._executor.submit(self.process_with_pipeline, batch, pipeline) for batch in data_batches]

        results = []
        for future in futures:
            try:
                result = future.result(timeout=30.0)  # 30 second timeout
                results.append(result)
            except Exception as e:
                logger.error(f"Error in parallel processing: {e}")
                with self._lock:
                    self._stats["errors"] += 1
                results.append(None)

        return results

    def add_quality_checkpoint(self, checkpoint_name: str, validator: Callable) -> None:
        """
        Add quality checkpoint to pipeline.

        Args:
            checkpoint_name: Name of checkpoint
            validator: Callable that validates data and returns True/False
        """
        self._checkpoints[checkpoint_name] = validator
        logger.info(f"Added quality checkpoint: {checkpoint_name}")

    def remove_quality_checkpoint(self, checkpoint_name: str) -> None:
        """
        Remove quality checkpoint.

        Args:
            checkpoint_name: Name of checkpoint to remove
        """
        if checkpoint_name in self._checkpoints:
            del self._checkpoints[checkpoint_name]
            logger.info(f"Removed quality checkpoint: {checkpoint_name}")
        else:
            logger.warning(f"Checkpoint {checkpoint_name} not found")

    def validate_checkpoint(self, checkpoint_name: str, data: Any) -> bool:
        """
        Validate data against a specific checkpoint.

        Args:
            checkpoint_name: Name of checkpoint
            data: Data to validate

        Returns:
            True if validation passes, False otherwise
        """
        if checkpoint_name not in self._checkpoints:
            logger.warning(f"Checkpoint {checkpoint_name} not found")
            return False

        try:
            validator = self._checkpoints[checkpoint_name]
            result = validator(data)

            with self._lock:
                if result:
                    self._stats["checkpoints_passed"] += 1
                else:
                    self._stats["checkpoints_failed"] += 1

            return bool(result)

        except Exception as e:
            logger.error(f"Error validating checkpoint {checkpoint_name}: {e}")
            with self._lock:
                self._stats["checkpoints_failed"] += 1
            return False

    def validate_all_checkpoints(self, data: Any) -> Dict[str, bool]:
        """
        Validate data against all checkpoints.

        Args:
            data: Data to validate

        Returns:
            Dictionary mapping checkpoint names to validation results
        """
        results = {}
        for checkpoint_name in self._checkpoints:
            results[checkpoint_name] = self.validate_checkpoint(checkpoint_name, data)
        return results

    def measure_latency(self, start_time: datetime, end_time: datetime) -> float:
        """
        Measure processing latency.

        Args:
            start_time: Start time
            end_time: End time

        Returns:
            Latency in milliseconds
        """
        if isinstance(start_time, datetime) and isinstance(end_time, datetime):
            delta = end_time - start_time
            latency_ms = delta.total_seconds() * 1000.0
        else:
            # Assume timestamps
            latency_ms = (end_time - start_time) * 1000.0

        return latency_ms

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get processing statistics.

        Returns:
            Dictionary with statistics
        """
        with self._lock:
            stats = self._stats.copy()

            # Calculate throughput
            if stats["total_processing_time_ms"] > 0:
                stats["throughput_batches_per_sec"] = stats["batches_processed"] / (stats["total_processing_time_ms"] / 1000.0)
            else:
                stats["throughput_batches_per_sec"] = 0.0

            # Checkpoint pass rate
            total_checkpoints = stats["checkpoints_passed"] + stats["checkpoints_failed"]
            if total_checkpoints > 0:
                stats["checkpoint_pass_rate"] = stats["checkpoints_passed"] / total_checkpoints
            else:
                stats["checkpoint_pass_rate"] = 0.0

            # Error rate
            total_operations = stats["batches_processed"] + stats["errors"]
            if total_operations > 0:
                stats["error_rate"] = stats["errors"] / total_operations
            else:
                stats["error_rate"] = 0.0

            stats["parallel_enabled"] = self._enable_parallel
            stats["num_workers"] = self._num_workers
            stats["num_stages"] = len(self._pipeline_stages)
            stats["num_checkpoints"] = len(self._checkpoints)

            return stats

    def reset_statistics(self) -> None:
        """Reset processing statistics."""
        with self._lock:
            self._stats = {
                "batches_processed": 0,
                "total_processing_time_ms": 0.0,
                "average_latency_ms": 0.0,
                "min_latency_ms": float("inf"),
                "max_latency_ms": 0.0,
                "checkpoints_passed": 0,
                "checkpoints_failed": 0,
                "errors": 0,
            }
        logger.info("Statistics reset")

    def add_stage(self, stage: Callable, position: Optional[int] = None) -> None:
        """
        Add processing stage to pipeline.

        Args:
            stage: Processing stage callable
            position: Optional position to insert (None = append)
        """
        if position is None:
            self._pipeline_stages.append(stage)
        else:
            self._pipeline_stages.insert(position, stage)
        logger.info(
            f"Added processing stage at position {position if position is not None else len(self._pipeline_stages) - 1}"
        )

    def remove_stage(self, position: int) -> None:
        """
        Remove processing stage from pipeline.

        Args:
            position: Position of stage to remove
        """
        if 0 <= position < len(self._pipeline_stages):
            removed = self._pipeline_stages.pop(position)
            logger.info(f"Removed processing stage at position {position}")
        else:
            raise IndexError(f"Invalid stage position: {position}")

    def clear_pipeline(self) -> None:
        """Clear all pipeline stages."""
        self._pipeline_stages.clear()
        logger.info("Pipeline cleared")

    def __del__(self):
        """Cleanup on deletion."""
        if self._executor:
            self._executor.shutdown(wait=False)
