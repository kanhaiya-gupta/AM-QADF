"""
Unit tests for Redis client.
"""
import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
src_dir = project_root / "src"
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from src.infrastructure.cache.redis_client import RedisClient
from src.infrastructure.cache.redis_config import RedisConfig


@pytest.fixture
def redis_config():
    """Create Redis config for testing."""
    return RedisConfig(
        host="localhost",
        port=6379,
        db=1,  # Use db=1 for testing to avoid conflicts
        default_ttl_seconds=60,
        max_cache_size_mb=128
    )


@pytest.fixture
def redis_client(redis_config):
    """Create Redis client for testing (requires Redis server running)."""
    try:
        client = RedisClient(redis_config)
        if client.is_connected():
            # Clean up test keys before test
            client.clear_pattern("test:*")
            yield client
            # Clean up test keys after test
            client.clear_pattern("test:*")
            client.close()
        else:
            pytest.skip("Redis server not available")
    except Exception as e:
        pytest.skip(f"Redis server not available: {e}")


@pytest.mark.skipif(
    not RedisClient(RedisConfig(host="localhost", port=6379, db=1)).is_connected(),
    reason="Redis server not available"
)
class TestRedisClient:
    """Test Redis client functionality."""
    
    def test_connection(self, redis_client):
        """Test Redis connection."""
        assert redis_client.is_connected()
    
    def test_set_get_string(self, redis_client):
        """Test setting and getting string values."""
        key = "test:string"
        value = "test_value"
        
        success = redis_client.set(key, value, ttl_seconds=60)
        assert success
        
        result = redis_client.get(key)
        assert result == value
    
    def test_set_get_dict(self, redis_client):
        """Test setting and getting dictionary values."""
        key = "test:dict"
        value = {"key1": "value1", "key2": 123, "key3": [1, 2, 3]}
        
        success = redis_client.set(key, value, ttl_seconds=60)
        assert success
        
        result = redis_client.get(key)
        assert result == value
    
    def test_set_get_none(self, redis_client):
        """Test getting non-existent key returns None."""
        result = redis_client.get("test:nonexistent")
        assert result is None
    
    def test_delete(self, redis_client):
        """Test deleting keys."""
        key = "test:delete"
        value = "to_delete"
        
        redis_client.set(key, value, ttl_seconds=60)
        assert redis_client.exists(key)
        
        deleted = redis_client.delete(key)
        assert deleted
        
        assert not redis_client.exists(key)
        assert redis_client.get(key) is None
    
    def test_exists(self, redis_client):
        """Test key existence check."""
        key = "test:exists"
        
        assert not redis_client.exists(key)
        
        redis_client.set(key, "value", ttl_seconds=60)
        assert redis_client.exists(key)
    
    def test_clear_pattern(self, redis_client):
        """Test clearing keys by pattern."""
        # Set multiple test keys
        redis_client.set("test:pattern:1", "value1", ttl_seconds=60)
        redis_client.set("test:pattern:2", "value2", ttl_seconds=60)
        redis_client.set("test:other", "value3", ttl_seconds=60)
        
        # Clear pattern
        deleted = redis_client.clear_pattern("test:pattern:*")
        assert deleted >= 2
        
        # Verify pattern keys are deleted
        assert redis_client.get("test:pattern:1") is None
        assert redis_client.get("test:pattern:2") is None
        
        # Verify other key still exists
        assert redis_client.get("test:other") == "value3"
    
    def test_get_stats(self, redis_client):
        """Test getting Redis statistics."""
        stats = redis_client.get_stats()
        
        assert "status" in stats
        assert stats["status"] == "connected"
        assert "used_memory_mb" in stats
        assert "max_memory_mb" in stats
        assert "keys" in stats
    
    def test_ttl_expiration(self, redis_client):
        """Test TTL expiration (if supported by test environment)."""
        key = "test:ttl"
        value = "expires_soon"
        
        redis_client.set(key, value, ttl_seconds=1)
        assert redis_client.get(key) == value
        
        # Note: Actual expiration test would require waiting, which is slow
        # This is tested in integration tests instead


@pytest.mark.skipif(
    not RedisClient(RedisConfig(host="localhost", port=6379, db=1)).is_connected(),
    reason="Redis server not available"
)
class TestRedisConfig:
    """Test Redis configuration."""
    
    def test_from_env(self, monkeypatch):
        """Test loading configuration from environment variables."""
        monkeypatch.setenv("REDIS_HOST", "test_host")
        monkeypatch.setenv("REDIS_PORT", "6380")
        monkeypatch.setenv("REDIS_DB", "2")
        monkeypatch.setenv("REDIS_PASSWORD", "test_password")
        monkeypatch.setenv("REDIS_DEFAULT_TTL", "7200")
        monkeypatch.setenv("REDIS_MAX_CACHE_SIZE_MB", "1024")
        
        config = RedisConfig.from_env()
        
        assert config.host == "test_host"
        assert config.port == 6380
        assert config.db == 2
        assert config.password == "test_password"
        assert config.default_ttl_seconds == 7200
        assert config.max_cache_size_mb == 1024
    
    def test_default_config(self):
        """Test default configuration values."""
        config = RedisConfig()
        
        assert config.host == "localhost"
        assert config.port == 6379
        assert config.db == 0
        assert config.password is None
        assert config.default_ttl_seconds == 3600
        assert config.max_cache_size_mb == 512
