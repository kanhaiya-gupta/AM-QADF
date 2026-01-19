# Performance Guide

## Overview

This guide covers performance optimization strategies for the AM-QADF Frontend Client.

## Performance Metrics

### Key Metrics

- **Response Time**: API endpoint response times
- **Throughput**: Requests per second
- **Concurrent Users**: Number of simultaneous users
- **Resource Usage**: CPU, memory, disk usage

## Optimization Strategies

### 1. Static File Optimization

#### Enable Compression

Configure your web server (nginx, Apache) to compress static files:

```nginx
# nginx configuration
gzip on;
gzip_types text/css application/javascript application/json;
gzip_min_length 1000;
```

#### Browser Caching

Set appropriate cache headers for static files:

```python
from fastapi.responses import Response

@app.get("/static/{file_path:path}")
async def static_file(file_path: str):
    response = FileResponse(f"static/{file_path}")
    response.headers["Cache-Control"] = "public, max-age=31536000"
    return response
```

### 2. Database Optimization

#### Connection Pooling

Configure MongoDB connection pooling:

```python
from pymongo import MongoClient

client = MongoClient(
    connection_string,
    maxPoolSize=200,
    minPoolSize=20,
    maxIdleTimeMS=45000
)
```

#### Query Optimization

- Use indexes on frequently queried fields
- Limit result sets with pagination
- Use projection to return only needed fields

### 3. Template Optimization

#### Template Caching

Jinja2 templates are cached by default. Ensure template caching is enabled:

```python
templates = Jinja2Templates(
    directory=str(templates_dir),
    auto_reload=False  # Disable in production
)
```

#### Minimize Template Complexity

- Keep templates simple
- Avoid complex logic in templates
- Use template includes for reusable components

### 4. API Response Optimization

#### Response Compression

Enable response compression in FastAPI:

```python
from fastapi.middleware.gzip import GZipMiddleware

app.add_middleware(GZipMiddleware, minimum_size=1000)
```

#### Pagination

Implement pagination for large result sets:

```python
@app.get("/api/data-query/query")
async def query_data(
    page: int = 1,
    page_size: int = 100
):
    skip = (page - 1) * page_size
    results = await query_service.query(skip=skip, limit=page_size)
    return {
        "data": results,
        "page": page,
        "page_size": page_size,
        "total": total_count
    }
```

### 5. Caching Strategies

#### Response Caching

Cache frequently accessed data:

```python
from functools import lru_cache
from cachetools import TTLCache

cache = TTLCache(maxsize=1000, ttl=300)

@app.get("/api/data-query/models")
async def get_models():
    cache_key = "models_list"
    if cache_key in cache:
        return cache[cache_key]
    
    models = await model_service.get_all()
    cache[cache_key] = models
    return models
```

#### Browser Caching

Set appropriate cache headers for API responses:

```python
response.headers["Cache-Control"] = "public, max-age=300"
```

### 6. Frontend Optimization

#### JavaScript Optimization

- Minify JavaScript files
- Use code splitting for large modules
- Lazy load non-critical JavaScript

#### CSS Optimization

- Minify CSS files
- Remove unused CSS
- Use CSS variables for theming

#### Image Optimization

- Compress images
- Use appropriate image formats (WebP, AVIF)
- Implement lazy loading for images

### 7. Server Configuration

#### Worker Processes

Configure multiple worker processes for production:

```bash
uvicorn client.app:app --workers 4 --host 0.0.0.0 --port 8000
```

#### Resource Limits

Set appropriate resource limits:

```env
WORKERS=4
MAX_CONNECTIONS=1000
KEEP_ALIVE_TIMEOUT=5
```

## Monitoring Performance

### Application Metrics

Monitor key metrics:

- Request count and response times
- Error rates
- Database query performance
- Memory and CPU usage

### Logging

Enable performance logging:

```python
import time
import logging

logger = logging.getLogger(__name__)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"{request.method} {request.url.path} - {process_time:.3f}s")
    return response
```

## Performance Testing

### Load Testing

Use tools like `locust` or `wrk` for load testing:

```bash
# Install locust
pip install locust

# Run load test
locust -f load_test.py --host http://localhost:8000
```

### Benchmarking

Benchmark critical endpoints:

```python
import time

def benchmark_endpoint(func):
    async def wrapper(*args, **kwargs):
        start = time.time()
        result = await func(*args, **kwargs)
        duration = time.time() - start
        logger.info(f"{func.__name__} took {duration:.3f}s")
        return result
    return wrapper
```

## Best Practices

1. **Monitor regularly** - Track performance metrics
2. **Optimize bottlenecks** - Focus on slow endpoints
3. **Use caching** - Cache frequently accessed data
4. **Limit data transfer** - Use pagination and compression
5. **Optimize queries** - Use indexes and projections
6. **Minimize dependencies** - Reduce external API calls
7. **Use async/await** - Leverage async I/O

## Related Documentation

- [Architecture](02-architecture.md) - System architecture
- [Configuration](08-configuration.md) - Configuration guide
- [Troubleshooting](10-troubleshooting.md) - Common issues

---

**Next**: [Troubleshooting](10-troubleshooting.md)
