"""
Redis Client Wrapper
Provides connection pooling and error handling for Redis operations.
"""
import logging
import json
import pickle
from typing import Any, Optional, Dict
from redis import Redis
from redis.connection import ConnectionPool
from redis.exceptions import RedisError, ConnectionError, TimeoutError, ResponseError, AuthenticationError

from .redis_config import RedisConfig

logger = logging.getLogger(__name__)


class RedisClient:
    """
    Redis client wrapper with connection pooling and error handling.
    """
    
    def __init__(self, config: RedisConfig):
        """
        Initialize Redis client.
        
        Args:
            config: Redis configuration
        """
        self.config = config
        self._pool: Optional[ConnectionPool] = None
        self._client: Optional[Redis] = None
        self._connected = False
        
        self._connect()
    
    def _connect(self):
        """Establish Redis connection with connection pooling."""
        try:
            # Only use password if it's not None and not empty
            password = self.config.password if self.config.password and self.config.password.strip() else None
            
            # Create connection pool
            self._pool = ConnectionPool(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                password=password,  # None if no password configured
                max_connections=self.config.max_connections,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                retry_on_timeout=self.config.retry_on_timeout,
                decode_responses=self.config.decode_responses
            )
            
            # Create Redis client
            self._client = Redis(connection_pool=self._pool)
            
            # Test connection
            try:
                self._client.ping()
            except (AuthenticationError, ResponseError) as auth_err:
                # If ping fails with auth error and we have a password, try without password
                error_msg = str(auth_err).lower()
                if password and ('auth' in error_msg or 'password' in error_msg):
                    logger.warning(f"Redis authentication failed during ping. Retrying without password...")
                    # Close current connection
                    if self._client:
                        try:
                            self._client.close()
                        except:
                            pass
                    # Recreate without password
                    self._pool = ConnectionPool(
                        host=self.config.host,
                        port=self.config.port,
                        db=self.config.db,
                        password=None,
                        max_connections=self.config.max_connections,
                        socket_timeout=self.config.socket_timeout,
                        socket_connect_timeout=self.config.socket_connect_timeout,
                        retry_on_timeout=self.config.retry_on_timeout,
                        decode_responses=self.config.decode_responses
                    )
                    self._client = Redis(connection_pool=self._pool)
                    self._client.ping()  # Try ping again
            
            # Configure max memory and eviction policy
            self._configure_memory_policy()
            
            self._connected = True
            auth_status = "with password" if password else "without password"
            logger.info(f"Redis connected to {self.config.host}:{self.config.port} ({auth_status})")
            
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self._connected = False
            raise
        except (AuthenticationError, ResponseError) as e:
            # Check if it's an authentication error and password was provided
            error_msg = str(e).lower()
            if password and ('auth' in error_msg or 'password' in error_msg):
                logger.warning(f"Redis authentication failed (Redis doesn't require password). Trying without password...")
                # Try again without password
                try:
                    self._pool = ConnectionPool(
                        host=self.config.host,
                        port=self.config.port,
                        db=self.config.db,
                        password=None,  # Try without password
                        max_connections=self.config.max_connections,
                        socket_timeout=self.config.socket_timeout,
                        socket_connect_timeout=self.config.socket_connect_timeout,
                        retry_on_timeout=self.config.retry_on_timeout,
                        decode_responses=self.config.decode_responses
                    )
                    self._client = Redis(connection_pool=self._pool)
                    self._client.ping()
                    self._configure_memory_policy()
                    self._connected = True
                    logger.info(f"Redis connected to {self.config.host}:{self.config.port} (without password - Redis doesn't require auth)")
                except Exception as retry_error:
                    logger.error(f"Failed to connect to Redis even without password: {retry_error}")
                    self._connected = False
                    raise
            else:
                logger.error(f"Redis authentication error: {e}")
                self._connected = False
                raise
        except Exception as e:
            # Check if it's an authentication error (might be wrapped)
            error_msg = str(e).lower()
            if password and ('auth' in error_msg and 'password' in error_msg):
                logger.warning(f"Redis authentication failed. Trying without password...")
                # Try again without password
                try:
                    self._pool = ConnectionPool(
                        host=self.config.host,
                        port=self.config.port,
                        db=self.config.db,
                        password=None,  # Try without password
                        max_connections=self.config.max_connections,
                        socket_timeout=self.config.socket_timeout,
                        socket_connect_timeout=self.config.socket_connect_timeout,
                        retry_on_timeout=self.config.retry_on_timeout,
                        decode_responses=self.config.decode_responses
                    )
                    self._client = Redis(connection_pool=self._pool)
                    self._client.ping()
                    self._configure_memory_policy()
                    self._connected = True
                    logger.info(f"Redis connected to {self.config.host}:{self.config.port} (without password - Redis doesn't require auth)")
                except Exception as retry_error:
                    logger.error(f"Failed to connect to Redis even without password: {retry_error}")
                    self._connected = False
                    raise
            else:
                logger.error(f"Unexpected error connecting to Redis: {e}")
                self._connected = False
                raise
    
    def _configure_memory_policy(self):
        """Configure Redis memory eviction policy."""
        try:
            # Set max memory (in bytes)
            max_memory_bytes = self.config.max_cache_size_mb * 1024 * 1024
            self._client.config_set('maxmemory', str(max_memory_bytes))
            
            # Set eviction policy: allkeys-lru (evict least recently used keys)
            self._client.config_set('maxmemory-policy', 'allkeys-lru')
            
            logger.info(f"Redis memory policy configured: {self.config.max_cache_size_mb}MB, allkeys-lru")
        except Exception as e:
            logger.warning(f"Failed to configure Redis memory policy: {e}")
    
    @property
    def client(self) -> Redis:
        """Get Redis client instance."""
        if not self._connected or self._client is None:
            raise RuntimeError("Redis client not connected")
        return self._client
    
    def is_connected(self) -> bool:
        """Check if Redis is connected."""
        if not self._connected or self._client is None:
            return False
        
        try:
            self._client.ping()
            return True
        except Exception:
            self._connected = False
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from Redis cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        if not self.is_connected():
            return None
        
        try:
            value = self._client.get(key)
            if value is None:
                return None
            
            # Deserialize value based on decode_responses setting
            if self.config.decode_responses:
                # Value is already a string, try JSON first
                try:
                    return json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    # Fallback to pickle (stored as base64)
                    try:
                        import base64
                        return pickle.loads(base64.b64decode(value))
                    except:
                        # Last resort: return as-is (might be plain string)
                        return value
            else:
                # Value is bytes
                try:
                    return json.loads(value.decode('utf-8'))
                except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
                    # Fallback to pickle
                    try:
                        return pickle.loads(value)
                    except:
                        # Last resort: return decoded string
                        return value.decode('utf-8', errors='ignore')
                
        except RedisError as e:
            logger.warning(f"Redis get error for key {key}: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """
        Set value in Redis cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time to live in seconds (uses default if None)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected():
            return False
        
        try:
            ttl = ttl_seconds if ttl_seconds is not None else self.config.default_ttl_seconds
            
            # Serialize value
            try:
                serialized = json.dumps(value, default=str)
                # If decode_responses=True, store as string
                # If decode_responses=False, encode to bytes
                if not self.config.decode_responses:
                    serialized = serialized.encode('utf-8')
            except (TypeError, ValueError):
                # Fallback to pickle for complex objects
                serialized = pickle.dumps(value)
                # If decode_responses=True, encode pickle as base64 string
                if self.config.decode_responses:
                    import base64
                    serialized = base64.b64encode(serialized).decode('utf-8')
                # If decode_responses=False, keep as bytes
            
            # Set with TTL
            self._client.setex(key, ttl, serialized)
            return True
            
        except RedisError as e:
            logger.warning(f"Redis set error for key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """
        Delete key from Redis cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted, False otherwise
        """
        if not self.is_connected():
            return False
        
        try:
            return bool(self._client.delete(key))
        except RedisError as e:
            logger.warning(f"Redis delete error for key {key}: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in Redis."""
        if not self.is_connected():
            return False
        
        try:
            return bool(self._client.exists(key))
        except RedisError:
            return False
    
    def clear_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching pattern.
        
        Args:
            pattern: Key pattern (e.g., "query:*")
            
        Returns:
            Number of keys deleted
        """
        if not self.is_connected():
            return 0
        
        try:
            keys = self._client.keys(pattern)
            if keys:
                return self._client.delete(*keys)
            return 0
        except RedisError as e:
            logger.warning(f"Redis clear_pattern error for pattern {pattern}: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Redis cache statistics."""
        if not self.is_connected():
            return {"status": "disconnected"}
        
        try:
            info = self._client.info('memory')
            return {
                "status": "connected",
                "used_memory_mb": round(info.get('used_memory', 0) / 1024 / 1024, 2),
                "max_memory_mb": self.config.max_cache_size_mb,
                "keys": self._client.dbsize(),
                "eviction_policy": "allkeys-lru"
            }
        except Exception as e:
            logger.error(f"Failed to get Redis stats: {e}")
            return {"status": "error", "error": str(e)}
    
    def close(self):
        """Close Redis connection."""
        if self._client:
            try:
                self._client.close()
            except Exception as e:
                logger.error(f"Error closing Redis connection: {e}")
            finally:
                self._client = None
                self._pool = None
                self._connected = False
