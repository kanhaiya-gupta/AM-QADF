# Frontend API Reference

This directory contains complete API reference documentation for the AM-QADF Frontend Client.

## API Overview

The frontend client provides REST API endpoints organized by architectural layers. All API endpoints are prefixed with `/api`.

## API Endpoints by Layer

### Core Layer
- **[Authentication API](authentication-api.md)** - User authentication and authorization endpoints

### Data Layer
- **[Data Query API](data-query-api.md)** - Data querying endpoints
- **[Voxelization API](voxelization-api.md)** - Voxel grid operations endpoints

### Processing Layer
- **[Signal Mapping API](signal-mapping-api.md)** - Signal mapping endpoints
- **[Synchronization API](synchronization-api.md)** - Alignment endpoints
- **[Correction API](correction-api.md)** - Correction endpoints
- **[Fusion API](fusion-api.md)** - Data fusion endpoints
- **[Processing API](processing-api.md)** - Processing pipeline endpoints

### Application Layer
- **[Analytics API](analytics-api.md)** - Analytics endpoints (statistical, sensitivity, SPC)
- **[Quality API](quality-api.md)** - Quality assessment endpoints
- **[Validation API](validation-api.md)** - Validation endpoints
- **[Visualization API](visualization-api.md)** - Visualization endpoints
- **[Anomaly Detection API](anomaly-detection-api.md)** - Anomaly detection endpoints

### System Layer
- **[Monitoring API](monitoring-api.md)** - Monitoring endpoints
- **[Streaming API](streaming-api.md)** - Streaming endpoints
- **[Workflow API](workflow-api.md)** - Workflow management endpoints

## API Base URL

All API endpoints are served from:
```
http://localhost:8000/api
```

## Response Format

### Success Response

```json
{
  "status": "success",
  "data": { ... },
  "message": "Operation completed successfully"
}
```

### Error Response

```json
{
  "status": "error",
  "error": {
    "code": "ERROR_CODE",
    "message": "Error message",
    "details": { ... }
  }
}
```

## Authentication

Most API endpoints require authentication. Include JWT token in request headers:

```
Authorization: Bearer <jwt_token>
```

## Common Endpoints

### Health Check

```http
GET /health
```

Returns server health status.

**Response:**
```json
{
  "status": "healthy",
  "service": "am-qadf-client"
}
```

### API Documentation

```http
GET /docs
```

FastAPI automatic API documentation (Swagger UI).

```http
GET /redoc
```

Alternative API documentation (ReDoc).

## Request/Response Examples

### Example: Query Data

**Request:**
```http
POST /api/data-query/query
Content-Type: application/json
Authorization: Bearer <token>

{
  "model_id": "my_model",
  "sources": ["hatching", "laser"],
  "spatial_bbox": {
    "min": [-50, -50, -50],
    "max": [50, 50, 50]
  }
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "query_id": "query_123",
    "points": [...],
    "signals": {...},
    "metadata": {...}
  }
}
```

## Rate Limiting

API endpoints may be rate-limited. Check response headers:
- `X-RateLimit-Limit`: Maximum requests per window
- `X-RateLimit-Remaining`: Remaining requests
- `X-RateLimit-Reset`: Reset time

## Error Codes

Common error codes:

| Code | Description |
|------|-------------|
| `AUTH_REQUIRED` | Authentication required |
| `AUTH_INVALID` | Invalid authentication token |
| `VALIDATION_ERROR` | Request validation failed |
| `NOT_FOUND` | Resource not found |
| `INTERNAL_ERROR` | Internal server error |

## Related Documentation

- [Modules](../05-modules/README.md) - Module documentation
- [Architecture](../02-architecture.md) - System architecture
- [Quick Start](../04-quick-start.md) - Getting started guide

---

**Parent**: [Frontend Documentation](../README.md)
