# Notebook 03: Signal Mapping Fundamentals

**File**: `03_Signal_Mapping_Fundamentals.ipynb`  
**Category**: Core Processing  
**Duration**: 60-90 minutes

## Purpose

This notebook teaches you how to map point cloud signals to voxel grids using various interpolation methods. You'll learn about different interpolation techniques, their parameters, and how to compare methods interactively.

## Learning Objectives

By the end of this notebook, you will:

- ✅ Understand interpolation methods (Nearest Neighbor, Linear, IDW, Gaussian KDE)
- ✅ Map signals to voxel grids
- ✅ Configure interpolation parameters
- ✅ Compare interpolation methods
- ✅ Visualize mapping results

## Topics Covered

### Interpolation Methods

- **Nearest Neighbor**: Fast, simple interpolation
- **Linear Interpolation**: Smooth linear interpolation
- **Inverse Distance Weighting (IDW)**: Distance-weighted interpolation
- **Gaussian KDE**: Kernel density estimation

### Signal Mapping Process

- **Point Cloud Data**: Input point cloud with signals
- **Voxel Grid**: Target voxel grid
- **Interpolation**: Mapping signals to voxels
- **Result**: Voxel grid with mapped signals

### Method Comparison

- **Accuracy**: Comparison of mapping accuracy
- **Performance**: Comparison of execution time
- **Visualization**: Side-by-side comparison

## Interactive Widgets

### Top Panel

- **Method Selector**: Dropdown to select interpolation method
- **Generate Sample Data**: Button to generate sample data
- **Map Signals**: Button to execute mapping
- **Compare Methods**: Button to compare all methods

### Left Panel

- **Method Parameters**: Accordion with method-specific parameters
  - **Nearest Neighbor**: No parameters
  - **Linear**: k_neighbors parameter
  - **IDW**: Power parameter, search radius
  - **Gaussian KDE**: Bandwidth parameter
- **Grid Resolution**: Resolution settings
- **Signal Selector**: Select signals to map
- **Data Info**: Information about input data

### Center Panel

- **Visualization Modes**: Radio buttons for views
  - **2D Slice**: Interactive 2D slice viewer
  - **3D View**: 3D volume visualization
  - **Comparison**: Side-by-side method comparison

### Right Panel

- **Mapping Results**: Mapping statistics and metrics
- **Performance Metrics**: Execution time, accuracy
- **Method Comparison**: Comparison results

### Bottom Panel

- **Status Display**: Current operation status
- **Progress Bar**: Mapping progress
- **Error Display**: Error messages

## Usage

### Step 1: Generate or Load Data

1. Click "Generate Sample Data" or load real data
2. Review data information in left panel
3. Select signals to map

### Step 2: Configure Method

1. Select interpolation method
2. Adjust method parameters
3. Set grid resolution

### Step 3: Execute Mapping

1. Click "Map Signals" button
2. Wait for mapping to complete
3. View results in center panel

### Step 4: Compare Methods

1. Click "Compare Methods" button
2. View side-by-side comparison
3. Review performance metrics

## Example Workflow

1. **Generate Data**: Create sample point cloud with temperature signal
2. **Select Method**: Choose "Gaussian KDE"
3. **Set Parameters**: Configure bandwidth parameter
4. **Map**: Execute signal mapping
5. **Visualize**: View 2D slice to verify mapping
6. **Compare**: Compare with other methods

## Key Takeaways

1. **Method Selection**: Choose appropriate method for your data
2. **Parameter Tuning**: Adjust parameters for optimal results
3. **Performance Trade-offs**: Balance accuracy and speed
4. **Visualization**: Use visualizations to verify mapping quality
5. **Comparison**: Compare methods to find best fit

## Related Notebooks

- **Previous**: [02: Voxel Grid Creation](02-voxel-grid.md)
- **Next**: [04: Temporal and Spatial Alignment](04-alignment.md)
- **Related**: [17: Complete Workflow Example](17-complete-workflow.md)

## Related Documentation

- **[Signal Mapping Module](../../AM_QADF/05-modules/signal-mapping.md)** - ⭐ CRITICAL MODULE
- **[Signal Mapping API](../../AM_QADF/06-api-reference/signal-mapping-api.md)** - Signal mapping API

---

**Last Updated**: 2024

