"""
Cache Infrastructure Module

Provides caching infrastructure including Redis support.
"""

from .redis_config import RedisConfig
from .redis_client import RedisClient

# Framework-wide cache access
try:
    from .framework_cache import (
        get_framework_cache_service,
        get_cached_query_result,
        cache_query_result,
        clear_framework_cache,
    )
    __all__ = [
        "RedisConfig",
        "RedisClient",
        "get_framework_cache_service",
        "get_cached_query_result",
        "cache_query_result",
        "clear_framework_cache",
    ]
except ImportError:
    # If framework_cache can't be imported (e.g., missing dependencies), just export Redis components
    __all__ = [
        "RedisConfig",
        "RedisClient",
    ]
