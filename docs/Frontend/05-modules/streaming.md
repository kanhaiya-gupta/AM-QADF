# Streaming Module

## Overview

The Streaming module provides interfaces for real-time data streaming and processing.

## Features

- **Real-Time Streaming**: Stream data from Kafka
- **Incremental Processing**: Process streaming data incrementally
- **Buffer Management**: Manage streaming buffers
- **Stream Monitoring**: Monitor stream status

## Components

### Routes (`routes.py`)

- `GET /streaming/` - Streaming page
- `POST /api/streaming/start` - Start streaming
- `POST /api/streaming/stop` - Stop streaming
- `GET /api/streaming/status` - Get stream status

### Services

- **StreamingService**: Core streaming operations
- **ConnectionService**: AM-QADF framework connection

### Templates

- `templates/streaming/index.html` - Main streaming page

## Usage

```javascript
// Start streaming
const streamRequest = {
  model_id: "my_model",
  sources: ["hatching", "laser"],
  buffer_size: 1000
};

const response = await API.post('/streaming/start', streamRequest);
const streamId = response.data.stream_id;

// Get stream status
const status = await API.get(`/streaming/status/${streamId}`);
console.log(status.data);
```

## Related Documentation

- [Streaming API Reference](../06-api-reference/streaming-api.md)
- [Architecture](../02-architecture.md)

---

**Parent**: [Modules](README.md)
