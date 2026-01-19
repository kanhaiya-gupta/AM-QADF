# Data Query Module

## Overview

The Data Query module provides interfaces for querying multi-source data from the AM-QADF data warehouse.

## Features

- **Multi-Source Querying**: Query data from hatching, laser, CT, ISPM, and other sources
- **Spatial Filtering**: Filter by bounding box and spatial regions
- **Temporal Filtering**: Filter by time ranges and layer ranges
- **Signal Type Selection**: Select specific signal types to retrieve
- **Query History**: Track and manage query history
- **Saved Queries**: Save and reuse common queries
- **Async Queries**: Support for long-running queries with status tracking
- **Pagination**: Efficient handling of large result sets

## Components

### Routes (`routes.py`)

- `GET /data-query/` - Data query page
- `POST /api/data-query/query` - Execute query
- `GET /api/data-query/models` - List available models
- `GET /api/data-query/models/{model_id}` - Get model details
- `GET /api/data-query/query/{query_id}` - Get query status/results
- `GET /api/data-query/history` - Get query history
- `POST /api/data-query/save` - Save a query
- `GET /api/data-query/saved` - List saved queries

### Services

- **QueryService**: Core query execution
- **ModelService**: Model management and metadata
- **ValidationService**: Query validation
- **CacheService**: Query result caching
- **PaginationService**: Result pagination
- **AsyncQueryService**: Asynchronous query execution
- **QueryHistoryService**: Query history management
- **SavedQueriesService**: Saved query management

### Templates

- `templates/data_query/index.html` - Main data query page

### Static Files

- `static/css/modules/data_query/` - Module-specific styles
- `static/js/data_query/` - Module-specific JavaScript

## Usage

### Basic Query

```javascript
// Execute a query
const queryRequest = {
  model_id: "my_model",
  sources: ["hatching", "laser"],
  spatial_bbox: {
    min: [-50, -50, -50],
    max: [50, 50, 50]
  }
};

const response = await API.post('/data-query/query', queryRequest);
console.log(response.data);
```

### Get Available Models

```javascript
// List all models
const models = await API.get('/data-query/models');
console.log(models.data);
```

### Saved Queries

```javascript
// Save a query
const saveRequest = {
  name: "My Query",
  query: queryRequest
};

await API.post('/data-query/save', saveRequest);

// Get saved queries
const saved = await API.get('/data-query/saved');
console.log(saved.data);
```

## Related Documentation

- [Data Query API Reference](../06-api-reference/data-query-api.md)
- [Architecture](../02-architecture.md)

---

**Parent**: [Modules](README.md)
