"""
Redis Configuration
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class RedisConfig:
    """Redis connection configuration."""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    max_connections: int = 50
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    retry_on_timeout: bool = True
    decode_responses: bool = True
    
    # Cache-specific settings
    default_ttl_seconds: int = 3600
    max_cache_size_mb: int = 2048  # Max memory for cache (2GB)
    
    @classmethod
    def from_env(cls) -> 'RedisConfig':
        """Load configuration from environment variables."""
        import os
        # Get password, but treat empty string as None
        password = os.getenv('REDIS_PASSWORD')
        if password and password.strip():
            password = password.strip()
        else:
            password = None
        
        return cls(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', '6379')),
            db=int(os.getenv('REDIS_DB', '0')),
            password=password,
            max_connections=int(os.getenv('REDIS_MAX_CONNECTIONS', '50')),
            socket_timeout=int(os.getenv('REDIS_SOCKET_TIMEOUT', '5')),
            socket_connect_timeout=int(os.getenv('REDIS_SOCKET_CONNECT_TIMEOUT', '5')),
            retry_on_timeout=os.getenv('REDIS_RETRY_ON_TIMEOUT', 'true').lower() == 'true',
            decode_responses=os.getenv('REDIS_DECODE_RESPONSES', 'true').lower() == 'true',
            default_ttl_seconds=int(os.getenv('REDIS_DEFAULT_TTL', '3600')),
            max_cache_size_mb=int(os.getenv('REDIS_MAX_CACHE_SIZE_MB', '512'))
        )
