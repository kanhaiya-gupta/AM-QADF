"""
Performance monitoring utilities for interpolation methods.
"""

import time
import logging
from typing import Any

logger = logging.getLogger(__name__)


def performance_monitor(func):
    """Decorator to monitor interpolation performance."""

    def wrapper(*args, **kwargs):
        start_time = time.time()

        result = func(*args, **kwargs)

        end_time = time.time()
        processing_time = end_time - start_time

        # Try to get point count from arguments
        import numpy as np

        points = None
        if args and isinstance(args[0], np.ndarray):
            points = args[0]
        elif "points" in kwargs:
            points = kwargs["points"]

        if points is not None and len(points) > 0:
            throughput = len(points) / processing_time if processing_time > 0 else 0
            logger.info(
                f"{func.__name__}: Processed {len(points):,} points in {processing_time:.2f}s "
                f"({throughput:,.0f} points/sec)"
            )

        return result

    return wrapper
