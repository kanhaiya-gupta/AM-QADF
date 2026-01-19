# Workflow Module

## Overview

The Workflow module provides interfaces for managing and executing processing workflows.

## Features

- **Workflow Definition**: Define processing workflows
- **Workflow Execution**: Execute workflows
- **Workflow History**: Track workflow execution history
- **Workflow Templates**: Reusable workflow templates

## Components

### Routes (`routes.py`)

- `GET /workflow/` - Workflow page
- `POST /api/workflow/create` - Create workflow
- `POST /api/workflow/execute` - Execute workflow
- `GET /api/workflow/history` - Get workflow history

### Services

- **WorkflowService**: Core workflow operations
- **ConnectionService**: AM-QADF framework connection

### Templates

- `templates/workflow/index.html` - Main workflow page

## Usage

```javascript
const workflowRequest = {
  name: "My Workflow",
  steps: [
    { type: "query", config: {...} },
    { type: "voxelize", config: {...} },
    { type: "map_signals", config: {...} }
  ]
};

const response = await API.post('/workflow/create', workflowRequest);
const workflowId = response.data.workflow_id;

// Execute workflow
await API.post(`/workflow/execute/${workflowId}`, {});
```

## Related Documentation

- [Workflow API Reference](../06-api-reference/workflow-api.md)
- [Architecture](../02-architecture.md)

---

**Parent**: [Modules](README.md)
