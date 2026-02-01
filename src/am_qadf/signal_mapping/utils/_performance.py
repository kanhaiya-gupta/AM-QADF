"""
Performance monitoring utilities for interpolation methods.
"""

import time
import logging
from typing import Any

logger = logging.getLogger(__name__)


def performance_monitor(func):
    """Decorator to monitor interpolation performance (minimal overhead, no numpy data access)."""

    def wrapper(*args, **kwargs):
        start_time = time.time()

        result = func(*args, **kwargs)

        end_time = time.time()
        processing_time = end_time - start_time

        # Get point count efficiently (O(1) - just shape access, no data copy)
        num_points = 0
        try:
            import numpy as np
            # Check first arg (points array) - isinstance is fast, no data access
            if args and hasattr(args[0], 'shape'):
                # Direct shape access (O(1), no data copy)
                num_points = args[0].shape[0] if len(args[0].shape) > 0 else 0
            elif "points" in kwargs and hasattr(kwargs["points"], 'shape'):
                num_points = kwargs["points"].shape[0] if len(kwargs["points"].shape) > 0 else 0
        except (AttributeError, IndexError):
            # Fallback: not a numpy array or no shape
            pass

        if num_points > 0:
            throughput = num_points / processing_time if processing_time > 0 else 0
            logger.info(
                f"{func.__name__}: Processed {num_points:,} points in {processing_time:.2f}s "
                f"({throughput:,.0f} points/sec)"
            )

        return result

    return wrapper
