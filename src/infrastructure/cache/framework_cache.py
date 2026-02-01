"""
Framework-Wide Cache Access

Provides shared cache access for all framework modules (voxelization, signal mapping, plotting, etc.)
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Global cache service instance
_framework_cache_service = None


def get_framework_cache_service():
    """
    Get or create framework-wide cache service instance.
    
    This allows any module (voxelization, signal mapping, plotting) to access
    the same Redis cache that the data query module uses.
    
    Returns:
        CacheService instance (RedisCacheService if Redis available, otherwise in-memory)
    """
    global _framework_cache_service
    
    if _framework_cache_service is None:
        try:
            # Try to get Redis cache service (same as data query module uses)
            from src.infrastructure.cache.redis_client import RedisClient
            from src.infrastructure.cache.redis_config import RedisConfig
            
            config = RedisConfig.from_env()
            redis_client = RedisClient(config)
            
            if redis_client and redis_client.is_connected():
                # Use Redis cache
                from client.modules.data_layer.data_query.services.redis_cache_service import RedisCacheService
                _framework_cache_service = RedisCacheService(
                    redis_client=redis_client,
                    default_ttl_seconds=3600
                )
                logger.info("Framework cache service initialized with Redis")
            else:
                # Fallback to in-memory cache
                from client.modules.data_layer.data_query.services.cache_service import CacheService
                _framework_cache_service = CacheService(max_size=100, default_ttl_seconds=3600)
                logger.info("Framework cache service initialized with in-memory cache (Redis not available)")
        except Exception as e:
            logger.warning(f"Failed to initialize framework cache service: {e}. Using in-memory fallback.")
            try:
                from client.modules.data_layer.data_query.services.cache_service import CacheService
                _framework_cache_service = CacheService(max_size=100, default_ttl_seconds=3600)
            except Exception as e2:
                logger.error(f"Failed to initialize even in-memory cache: {e2}")
                _framework_cache_service = None
    
    return _framework_cache_service


def get_cached_query_result(query_request: Dict[str, Any]) -> Optional[Any]:
    """
    Get cached query result for a query request.
    
    This can be used by any framework module to check if query results
    are already cached before executing expensive queries.
    
    Args:
        query_request: Query request dictionary (same format as data query module)
                      Can include SpatialQuery/TemporalQuery objects - will be serialized
        
    Returns:
        Cached result if found, None otherwise
    """
    cache_service = get_framework_cache_service()
    if cache_service is None:
        return None
    
    try:
        # Serialize query request (convert objects to dicts)
        serialized_request = _serialize_query_request(query_request)
        return cache_service.get(serialized_request)
    except Exception as e:
        logger.warning(f"Failed to get cached query result: {e}")
        return None


def _serialize_query_request(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Serialize query request to make it cacheable.
    Converts SpatialQuery/TemporalQuery objects to dictionaries.
    """
    serialized = {}
    for key, value in request.items():
        if value is None:
            serialized[key] = None
        elif hasattr(value, '__dict__'):
            # Convert objects to dict
            try:
                serialized[key] = value.__dict__
            except:
                # Fallback: try to convert to dict
                try:
                    serialized[key] = dict(value)
                except:
                    serialized[key] = str(value)
        elif isinstance(value, dict):
            # Recursively serialize nested dicts
            serialized[key] = _serialize_query_request(value)
        elif isinstance(value, list):
            # Serialize list items
            serialized[key] = [
                item.__dict__ if hasattr(item, '__dict__') else item
                for item in value
            ]
        else:
            serialized[key] = value
    return serialized


def cache_query_result(query_request: Dict[str, Any], result: Any, ttl_seconds: Optional[int] = None):
    """
    Cache a query result for framework-wide access.
    
    Args:
        query_request: Query request dictionary (can include objects - will be serialized)
        result: Query result to cache
        ttl_seconds: Optional TTL in seconds (uses default if None)
    """
    cache_service = get_framework_cache_service()
    if cache_service is None:
        return
    
    try:
        # Serialize query request before caching
        serialized_request = _serialize_query_request(query_request)
        cache_service.set(serialized_request, result, ttl_seconds=ttl_seconds)
    except Exception as e:
        logger.warning(f"Failed to cache query result: {e}")


def clear_framework_cache(pattern: Optional[str] = None):
    """
    Clear framework cache entries.
    
    Args:
        pattern: Optional pattern to match (if None, clears all cache)
    """
    cache_service = get_framework_cache_service()
    if cache_service is None:
        return
    
    try:
        cache_service.invalidate(pattern)
    except Exception as e:
        logger.warning(f"Failed to clear framework cache: {e}")
