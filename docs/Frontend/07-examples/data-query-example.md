# Data Query Example

## Overview

This example demonstrates how to query data from the AM-QADF data warehouse.

## Step 1: Get Available Models

```javascript
// List all available models
const modelsResponse = await API.get('/data-query/models');
const models = modelsResponse.data;

console.log('Available models:', models);
```

## Step 2: Get Model Details

```javascript
// Get details for a specific model
const modelId = models[0].model_id;
const modelDetails = await API.get(`/data-query/models/${modelId}`);

console.log('Model details:', modelDetails.data);
console.log('Available signals:', modelDetails.data.available_signals);
```

## Step 3: Execute Query

```javascript
// Execute a query
const queryRequest = {
  model_id: modelId,
  sources: ["hatching", "laser"],
  spatial_bbox: {
    min: [-50, -50, -50],
    max: [50, 50, 50]
  },
  temporal_range: {
    start: 0,
    end: 1000
  },
  signal_types: ["temperature", "power"]
};

const queryResponse = await API.post('/data-query/query', queryRequest);
console.log('Query ID:', queryResponse.data.query_id);
console.log('Results:', queryResponse.data);
```

## Step 4: Save Query

```javascript
// Save the query for later use
const saveRequest = {
  name: "My Standard Query",
  query: queryRequest
};

const saveResponse = await API.post('/data-query/save', saveRequest);
console.log('Saved query ID:', saveResponse.data.saved_query_id);
```

## Step 5: Retrieve Saved Query

```javascript
// Get all saved queries
const savedQueries = await API.get('/data-query/saved');
console.log('Saved queries:', savedQueries.data);
```

## Complete Example

```html
<!DOCTYPE html>
<html>
<head>
    <title>Data Query Example</title>
</head>
<body>
    <h1>Data Query Example</h1>
    <button onclick="executeQuery()">Execute Query</button>
    <div id="results"></div>

    <script>
        async function executeQuery() {
            try {
                // Get models
                const models = await API.get('/data-query/models');
                const modelId = models.data[0].model_id;

                // Execute query
                const queryRequest = {
                    model_id: modelId,
                    sources: ["hatching", "laser"],
                    spatial_bbox: {
                        min: [-50, -50, -50],
                        max: [50, 50, 50]
                    }
                };

                const response = await API.post('/data-query/query', queryRequest);
                document.getElementById('results').innerHTML = 
                    JSON.stringify(response.data, null, 2);
            } catch (error) {
                console.error('Query failed:', error);
            }
        }
    </script>
</body>
</html>
```

## Related Documentation

- [Data Query Module](../05-modules/data-query.md)
- [Data Query API](../06-api-reference/data-query-api.md)

---

**Parent**: [Examples](README.md)
