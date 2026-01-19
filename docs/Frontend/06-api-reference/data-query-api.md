# Data Query API Reference

## Overview

API endpoints for querying multi-source data from the AM-QADF data warehouse.

## Endpoints

### GET /data-query/

Serve the data query page.

**Response**: HTML page

---

### POST /api/data-query/query

Execute a data query.

**Request Body**:
```json
{
  "model_id": "string",
  "sources": ["hatching", "laser"],
  "spatial_bbox": {
    "min": [-50, -50, -50],
    "max": [50, 50, 50]
  },
  "temporal_range": {
    "start": 0,
    "end": 1000
  },
  "signal_types": ["temperature", "power"]
}
```

**Response**:
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

---

### GET /api/data-query/models

List all available models.

**Response**:
```json
{
  "status": "success",
  "data": [
    {
      "model_id": "model_1",
      "name": "Model 1",
      "sources": ["hatching", "laser"]
    }
  ]
}
```

---

### GET /api/data-query/models/{model_id}

Get model details.

**Response**:
```json
{
  "status": "success",
  "data": {
    "model_id": "model_1",
    "name": "Model 1",
    "bounding_box": {
      "min": [-50, -50, -50],
      "max": [50, 50, 50]
    },
    "available_signals": ["temperature", "power"]
  }
}
```

---

### GET /api/data-query/query/{query_id}

Get query status or results.

**Response**:
```json
{
  "status": "success",
  "data": {
    "query_id": "query_123",
    "status": "completed",
    "results": {...}
  }
}
```

---

### GET /api/data-query/history

Get query history.

**Query Parameters**:
- `page`: Page number (default: 1)
- `page_size`: Items per page (default: 20)

**Response**:
```json
{
  "status": "success",
  "data": {
    "queries": [...],
    "page": 1,
    "page_size": 20,
    "total": 100
  }
}
```

---

### POST /api/data-query/save

Save a query for reuse.

**Request Body**:
```json
{
  "name": "My Query",
  "query": {
    "model_id": "model_1",
    "sources": ["hatching"]
  }
}
```

**Response**:
```json
{
  "status": "success",
  "data": {
    "saved_query_id": "saved_123"
  }
}
```

---

### GET /api/data-query/saved

List saved queries.

**Response**:
```json
{
  "status": "success",
  "data": [
    {
      "saved_query_id": "saved_123",
      "name": "My Query",
      "query": {...}
    }
  ]
}
```

## Error Responses

```json
{
  "status": "error",
  "error": {
    "code": "ERROR_CODE",
    "message": "Error message",
    "details": {...}
  }
}
```

## Related Documentation

- [Data Query Module](../05-modules/data-query.md)
- [API Reference](README.md)

---

**Parent**: [API Reference](README.md)
