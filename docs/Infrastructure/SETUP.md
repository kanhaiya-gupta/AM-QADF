# Infrastructure Layer Setup

## Dependencies

The infrastructure layer requires the following Python packages:

### Required Packages

Core infrastructure dependencies are in the project root `requirements.txt` (python-dotenv, pymongo). Optional database drivers (Cassandra, Neo4j, Elasticsearch) can be added if needed.

## Installation

```bash
pip install -r requirements.txt
```

Or install only infrastructure-related packages:

```bash
pip install python-dotenv pymongo
```

## Docker Integration

The infrastructure layer automatically works with Docker Compose:

1. **Environment Variables**: Set in `docker-compose.yml` or `.env` files
2. **Service Discovery**: Uses service names (e.g., `mongodb:27017`)
3. **Health Checks**: Built-in health checks for Docker healthcheck endpoints

### Environment File Loading

Docker Compose automatically loads `.env` files from the same directory as the compose file. To use a different env file:

```bash
# From project root, specify env file explicitly
docker-compose -f docker/docker-compose.dev.yml --env-file development.env up -d

# Or from docker directory
cd docker
docker-compose -f docker-compose.dev.yml --env-file ../development.env up -d
```

### Example Docker Compose Integration

```yaml
services:
  api:
    environment:
      MONGODB_URL: mongodb://admin:password@mongodb:27017/pbf_data_lake
      MONGO_DATABASE: pbf_data_lake
    depends_on:
      mongodb:
        condition: service_healthy
```

## Usage in Code

```python
from src.infrastructure.database import get_connection_manager

# Automatically loads from environment variables
manager = get_connection_manager(env_name="development")
mongodb_client = manager.get_mongodb_client()
```

## Environment Variables

The infrastructure reads from:
1. `.env` files (development.env, production.env)
2. Docker environment variables (set by docker-compose)
3. System environment variables

Priority: System env > Docker env > .env file

