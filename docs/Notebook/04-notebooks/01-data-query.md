# Notebook 01: Data Query and Access

**File**: `01_Data_Query_and_Access.ipynb`  
**Category**: Introduction and Fundamentals  
**Duration**: 45-60 minutes

## Purpose

This notebook teaches you how to query and access multi-source data from the AM-QADF data warehouse. You'll learn to use the UnifiedQueryClient to retrieve data from various sources with interactive query interfaces.

## Learning Objectives

By the end of this notebook, you will:

- ✅ Query data from multiple sources (MongoDB, PostgreSQL, etc.)
- ✅ Apply spatial, temporal, and parameter filters
- ✅ Understand QueryResult structure
- ✅ Export query results
- ✅ Use interactive query interfaces

## Topics Covered

### Data Sources

- **MongoDB**: Unstructured data (ISPM, CT scans, process logs)
- **PostgreSQL**: Structured data (process parameters, build metadata)
- **Multi-Source**: Querying across multiple sources
- **UnifiedQueryClient**: Unified interface for all sources

### Query Types

- **Spatial Queries**: Filter by spatial bounds
- **Temporal Queries**: Filter by time ranges
- **Parameter Queries**: Filter by process parameters
- **Combined Queries**: Multiple filters together

### QueryResult Structure

- **Data**: Retrieved data arrays
- **Metadata**: Source information, timestamps
- **Statistics**: Data statistics and summaries
- **Quality**: Data quality metrics

## Interactive Widgets

### Top Panel

- **Model Selection**: Dropdown to select build/model
- **Query Button**: Execute query
- **Clear Button**: Clear results

### Left Panel

- **Data Source Selection**: Checkboxes for data sources
- **Spatial Filters**: Bounding box controls
- **Temporal Filters**: Time range selectors
- **Parameter Filters**: Parameter value filters

### Center Panel

- **Results Display**: Tabbed view with:
  - **Table View**: Data table
  - **Statistics View**: Statistical summaries
  - **Visualization View**: Data visualizations
  - **Export View**: Export options

### Right Panel

- **Quick Actions**: Common query operations
- **Status Display**: Query status
- **Summary**: Query summary information

### Bottom Panel

- **Progress Bar**: Query progress
- **Error Display**: Error messages

## Usage

### Step 1: Select Data Sources

1. Check data sources you want to query
2. Select model/build from dropdown
3. Configure filters as needed

### Step 2: Execute Query

1. Click "Execute Query" button
2. Wait for query to complete
3. View results in center panel

### Step 3: Explore Results

1. Switch between table/statistics/visualization tabs
2. Review data in table view
3. Check statistics in statistics view
4. Visualize data in visualization view

### Step 4: Export Results

1. Go to export view
2. Select export format
3. Click export button

## Example Workflow

1. **Select Sources**: Check ISPM and CT scan sources
2. **Set Filters**: Set spatial bounds and time range
3. **Execute**: Click "Execute Query"
4. **Review**: Check table view for data
5. **Analyze**: View statistics and visualizations
6. **Export**: Export results for further analysis

## Key Takeaways

1. **Multi-Source Access**: Query data from multiple sources simultaneously
2. **Flexible Filtering**: Apply spatial, temporal, and parameter filters
3. **Unified Interface**: UnifiedQueryClient provides consistent interface
4. **Rich Results**: QueryResult includes data, metadata, and statistics
5. **Export Capabilities**: Export results in various formats

## Related Notebooks

- **Previous**: [00: Introduction to AM-QADF](00-introduction.md)
- **Next**: [02: Voxel Grid Creation](02-voxel-grid.md)
- **Related**: [17: Complete Workflow Example](17-complete-workflow.md)

## Related Documentation

- **[Query Module](../../AM_QADF/05-modules/query.md)** - Query module details
- **[Query API](../../AM_QADF/06-api-reference/query-api.md)** - Query API reference

---

**Last Updated**: 2024

