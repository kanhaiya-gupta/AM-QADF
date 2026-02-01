# Infrastructure Layer

Infrastructure layer for database connection management in AM-QADF framework.

## Overview

This layer provides centralized database connection management, keeping the framework (`src/am_qadf`) database-agnostic. It handles:

- Environment variable loading from `.env` files
- Database connection configuration parsing
- Connection pooling and management
- Health checks for Docker environments
- MongoDB, Cassandra, Neo4j, Elasticsearch support
- Redis caching for query results

## Structure

```
AM-QADF/
├── src/
│   └── infrastructure/         # Infrastructure layer (database connections)
│       ├── config/
│       │   ├── env_loader.py          # Load .env files
│       │   └── database_config.py     # Parse DB configs from env vars
│       │
│       ├── database/
│       │   ├── connection_manager.py  # Centralized connection management
│       │   ├── mongodb_client.py      # MongoDB client wrapper
│       │   ├── connection_pool.py      # Connection pooling
│       │   └── health_check.py         # Health checks
│       │
│       └── cache/
│           ├── redis_config.py        # Redis configuration
│           └── redis_client.py        # Redis client wrapper
│
└── docs/
    └── Infrastructure/          # Documentation (this directory)
```

## Usage

### Basic Usage

```python
from src.infrastructure.database import get_connection_manager

# Get connection manager (loads from development.env by default)
manager = get_connection_manager(env_name="development")

# Get MongoDB client
mongodb_client = manager.get_mongodb_client()

# Use MongoDB client
collection = mongodb_client.get_collection("hatching_layers")
documents = collection.find({"model_id": "my_model"})
```

### In Docker Environment

The infrastructure automatically detects Docker environment variables:

```python
# In Docker, environment variables are set by docker-compose
# No need to load .env file manually
from src.infrastructure.database import get_connection_manager

# Automatically uses environment variables
manager = get_connection_manager(env_name="production")
mongodb_client = manager.get_mongodb_client()
```

### With AM-QADF Query Clients

```python
from src.infrastructure.database import get_connection_manager
from am_qadf.query import UnifiedQueryClient

# Get MongoDB connection
manager = get_connection_manager()
mongodb_client = manager.get_mongodb_client()

# Pass to query clients (framework stays database-agnostic)
query_client = UnifiedQueryClient(mongo_client=mongodb_client)

# Use query client
result = query_client.query(
    model_id="my_model",
    sources=['hatching', 'laser', 'ct']
)
```

### Health Checks

```python
from src.infrastructure.database import check_all_connections

# Check all database connections
health_status = check_all_connections(env_name="development")

# health_status = {
#     'mongodb': {'status': 'healthy', 'server_version': '7.0.0', ...},
#     'cassandra': {'status': 'not_configured'},
#     ...
# }
```

## Environment Variables

The infrastructure reads from `.env` files or Docker environment variables:

### MongoDB
- `MONGODB_URL` - Full MongoDB connection string
- `MONGO_DATABASE` - Database name
- `MONGO_ROOT_USERNAME` - Username
- `MONGO_ROOT_PASSWORD` - Password
- `MONGODB_HOST` - Host (if not using URL)
- `MONGODB_PORT` - Port (default: 27017)

### Redis (Optional - for Caching)
- `REDIS_HOST` - Redis host (default: localhost)
- `REDIS_PORT` - Redis port (default: 6379)
- `REDIS_DB` - Redis database number (default: 0)
- `REDIS_PASSWORD` - Redis password (optional)
- `REDIS_DEFAULT_TTL` - Default cache TTL in seconds (default: 3600)
- `REDIS_MAX_CACHE_SIZE_MB` - Maximum Redis memory for cache (default: 512)

### Docker Integration

In `docker-compose.yml`, environment variables are automatically passed:

```yaml
services:
  api:
    environment:
      MONGODB_URL: mongodb://admin:password@mongodb:27017/pbf_data_lake
      MONGO_DATABASE: pbf_data_lake
```

## Benefits

1. **Separation of Concerns**: Framework code stays database-agnostic
2. **Docker-Ready**: Works seamlessly with Docker Compose
3. **Centralized Management**: Single point for connection management
4. **Health Checks**: Built-in health checks for monitoring
5. **Connection Pooling**: Efficient connection reuse
6. **Easy Testing**: Can mock connections for testing

## Best Practices

1. **Use Connection Manager**: Always use `get_connection_manager()` instead of creating connections directly
2. **Environment-Aware**: Use appropriate `env_name` for different environments
3. **Health Checks**: Regularly check connection health in long-running services
4. **Close Connections**: Properly close connections when shutting down
5. **Error Handling**: Always handle connection errors gracefully
6. **Redis Caching**: Use Redis for persistent caching (see [Redis Caching](redis-caching.md))

## Related Documentation

- [Infrastructure Setup](SETUP.md) - Setup and installation guide
- [Build System](build/README.md) - Build system documentation
- [Third-Party Dependencies](third-party/README.md) - Third-party library documentation
- [Redis Caching](redis-caching.md) - Redis caching infrastructure
- [Redis Troubleshooting](redis-troubleshooting.md) - Redis troubleshooting guide

## Utility Scripts

Utility scripts for managing infrastructure services are available in `generation/scripts/`:

### MongoDB
- `start_mongodb.py` - Start MongoDB container
- `stop_mongodb.py` - Stop MongoDB container
- `check_mongodb.py` - Check MongoDB connection and status

### Redis
- `start_redis.py` - Start Redis container
- `stop_redis.py` - Stop Redis container
- `check_redis.py` - Check Redis connection and cache status

