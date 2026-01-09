# Infrastructure Layer Setup

## Dependencies

The infrastructure layer requires the following Python packages:

### Required Packages

Add these to your `requirements/requirements_core.txt` or create `requirements/requirements_infrastructure.txt`:

```
python-dotenv>=1.0.0    # For loading .env files
pymongo>=4.6.0          # For MongoDB connections
```

### Optional Packages (for other databases)

```
cassandra-driver>=3.28.0    # For Cassandra (if needed)
neo4j>=5.15.0               # For Neo4j (if needed)
elasticsearch>=8.11.0       # For Elasticsearch (if needed)
```

## Installation

```bash
pip install python-dotenv pymongo
```

Or if using requirements file:

```bash
pip install -r requirements/requirements_core.txt
```

## Docker Integration

The infrastructure layer automatically works with Docker Compose:

1. **Environment Variables**: Set in `docker-compose.yml` or `.env` files
2. **Service Discovery**: Uses service names (e.g., `mongodb:27017`)
3. **Health Checks**: Built-in health checks for Docker healthcheck endpoints

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

