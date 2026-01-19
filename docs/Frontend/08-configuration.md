# Configuration Guide

## Overview

The AM-QADF Frontend Client can be configured through environment variables and configuration files.

## Environment Variables

### MongoDB Configuration

```env
# MongoDB Connection
MONGODB_CONNECTION_STRING=mongodb://localhost:27017
MONGODB_DATABASE=am_qadf_data
MONGODB_USERNAME=admin
MONGODB_PASSWORD=password

# MongoDB Options
MONGODB_MAX_POOL_SIZE=100
MONGODB_MIN_POOL_SIZE=10
```

### FastAPI Configuration

```env
# Server Configuration
API_HOST=0.0.0.0
API_PORT=8000
RELOAD=true  # Set to false in production

# CORS Configuration
CORS_ORIGINS=*  # In production, specify allowed origins
CORS_ALLOW_CREDENTIALS=true
```

### Authentication Configuration

```env
# JWT Configuration
JWT_SECRET_KEY=your-secret-key-here
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24
JWT_REFRESH_EXPIRATION_DAYS=7

# Authentication Settings
AUTH_ENABLED=true
SESSION_TIMEOUT_MINUTES=30
```

### Feature Flags

```env
# Feature Toggles
ENABLE_3D_VIZ=true
ENABLE_REALTIME=false
ENABLE_ANALYTICS=true
ENABLE_STREAMING=false
ENABLE_MONITORING=true
```

### Logging Configuration

```env
# Logging
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
DEBUG=false
LOG_FILE=logs/app.log
LOG_MAX_SIZE=10MB
LOG_BACKUP_COUNT=5
```

## Configuration Files

### Application Configuration

The FastAPI application is configured in `client/app.py`:

```python
app = FastAPI(
    title="AM-QADF Web Client",
    description="Web frontend for AM-QADF Framework",
    version="1.0.0",
)
```

### Static Files Configuration

Static files are mounted in `client/app.py`:

```python
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
```

### Templates Configuration

Templates are configured in `client/app.py`:

```python
templates_dir = client_dir / "templates"
templates = Jinja2Templates(directory=str(templates_dir))
```

## Module-Specific Configuration

### Data Query Configuration

```env
# Query Settings
QUERY_CACHE_ENABLED=true
QUERY_CACHE_TTL_SECONDS=300
QUERY_MAX_RESULTS=10000
QUERY_TIMEOUT_SECONDS=30
```

### Visualization Configuration

```env
# Visualization Settings
VIZ_MAX_POINTS=1000000
VIZ_RESOLUTION=1024
VIZ_ENABLE_3D=true
VIZ_ENABLE_2D=true
```

### Analytics Configuration

```env
# Analytics Settings
ANALYTICS_CACHE_ENABLED=true
ANALYTICS_MAX_SAMPLES=100000
ANALYTICS_TIMEOUT_SECONDS=60
```

## Production Configuration

### Security Settings

```env
# Security
SECRET_KEY=your-production-secret-key
ALLOWED_HOSTS=yourdomain.com,www.yourdomain.com
CORS_ORIGINS=https://yourdomain.com
DEBUG=false
```

### Performance Settings

```env
# Performance
WORKERS=4
MAX_CONNECTIONS=1000
KEEP_ALIVE_TIMEOUT=5
```

### Database Connection Pooling

```env
# Connection Pooling
MONGODB_MAX_POOL_SIZE=200
MONGODB_MIN_POOL_SIZE=20
MONGODB_MAX_IDLE_TIME_MS=45000
```

## Environment-Specific Configuration

### Development

Create `development.env`:

```env
DEBUG=true
LOG_LEVEL=DEBUG
RELOAD=true
CORS_ORIGINS=*
```

### Production

Create `production.env`:

```env
DEBUG=false
LOG_LEVEL=INFO
RELOAD=false
CORS_ORIGINS=https://yourdomain.com
WORKERS=4
```

## Loading Configuration

Configuration is loaded from environment variables. Use `python-dotenv` to load from `.env` file:

```python
from dotenv import load_dotenv
load_dotenv()
```

## Configuration Validation

Configuration values are validated on startup. Invalid configurations will cause the application to fail with clear error messages.

## Best Practices

1. **Never commit `.env` files** - Add to `.gitignore`
2. **Use strong secrets** - Generate random secret keys
3. **Limit CORS origins** - Don't use `*` in production
4. **Set appropriate timeouts** - Prevent long-running requests
5. **Configure logging** - Set appropriate log levels
6. **Use connection pooling** - Optimize database connections

## Related Documentation

- [Installation](03-installation.md) - Installation guide
- [Architecture](02-architecture.md) - System architecture
- [Troubleshooting](10-troubleshooting.md) - Common issues

---

**Next**: [Performance](09-performance.md) | [Troubleshooting](10-troubleshooting.md)
