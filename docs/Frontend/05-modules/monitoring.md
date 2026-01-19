# Monitoring Module

## Overview

The Monitoring module provides interfaces for system and process monitoring.

## Features

- **System Monitoring**: Monitor system health and resources
- **Process Monitoring**: Monitor processing operations
- **Alerts**: Configure and manage alerts
- **Health Checks**: System health status

## Components

### Routes (`routes.py`)

- `GET /monitoring/` - Monitoring dashboard
- `GET /api/monitoring/health` - System health status
- `GET /api/monitoring/metrics` - System metrics
- `GET /api/monitoring/alerts` - List alerts

### Services

- **MonitoringService**: Core monitoring operations
- **ConnectionService**: AM-QADF framework connection

### Templates

- `templates/monitoring/index.html` - Main monitoring page

## Usage

```javascript
// Get system health
const health = await API.get('/monitoring/health');
console.log(health.data);

// Get metrics
const metrics = await API.get('/monitoring/metrics');
console.log(metrics.data);
```

## Related Documentation

- [Monitoring API Reference](../06-api-reference/monitoring-api.md)
- [Architecture](../02-architecture.md)

---

**Parent**: [Modules](README.md)
