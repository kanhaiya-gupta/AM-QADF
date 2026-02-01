# Notebook 04: Temporal and Spatial Alignment

**File**: `04_Temporal_and_Spatial_Alignment.ipynb`  
**Category**: Core Processing  
**Duration**: 60-90 minutes

## Purpose

This notebook teaches you how to align multi-source data temporally and spatially. You'll learn temporal alignment techniques, spatial transformations, and coordinate system alignment.

## Learning Objectives

By the end of this notebook, you will:

- ✅ Align data temporally using layer-based and time-based methods
- ✅ Transform data spatially (translation, rotation, scaling)
- ✅ Align coordinate systems
- ✅ Validate alignment results
- ✅ Visualize before/after alignment

## Topics Covered

### Temporal Alignment

- **Layer-Based Alignment**: Align by build layers
- **Time-Based Alignment**: Align by timestamps
- **Temporal Interpolation**: Interpolate between time points
- **Temporal Validation**: Validate temporal alignment

### Spatial Alignment

- **Bbox-corner correspondence**: Transformation is computed from 8 bounding-box corners per source (24 permutations × 56 triplets); no point-to-point matching. See [SPATIAL_ALIGNMENT_DESIGN.md](../../Infrastructure/SPATIAL_ALIGNMENT_DESIGN.md).
- **Translation, rotation, scaling**: Combined into a 4×4 similarity transform (Kabsch + Umeyama).
- **Transformation Matrix**: 4×4; apply to points to bring them into reference (hatching) frame.
- **Python API**: `UnifiedQueryClient.query_and_transform_points(...)`.

### Coordinate System Alignment

- **Source Systems**: Different coordinate systems
- **Target System**: Unified coordinate system
- **Transformation**: Convert between systems
- **Validation**: Verify alignment accuracy

## Interactive Widgets

### Top Panel

- **Alignment Mode**: Radio buttons (Temporal/Spatial/Both)
- **Data Source Selection**: Checkboxes for data sources
- **Execute Alignment**: Button to execute alignment
- **Reset**: Button to reset alignment

### Left Panel

- **Temporal Alignment Configuration**:
  - **Alignment Method**: Layer-based or time-based
  - **Time Range**: Time range selector
  - **Interpolation**: Temporal interpolation options
- **Spatial Alignment Configuration**:
  - **Transformation Type**: Translation, rotation, scaling
  - **Transformation Parameters**: Dynamic controls based on type
  - **Coordinate System**: Source and target systems

### Center Panel

- **Visualization Modes**: Radio buttons
  - **Before**: Data before alignment
  - **After**: Data after alignment
  - **Overlay**: Overlay comparison
  - **Difference**: Difference visualization

### Right Panel

- **Alignment Metrics**: Alignment quality metrics
- **Transformation Matrix**: Display transformation matrix
- **Error Statistics**: Alignment error statistics
- **Validation Status**: Alignment validation results
- **Export Options**: Export aligned data

### Bottom Panel

- **Status Display**: Current operation status
- **Progress Bar**: Alignment progress
- **Error Display**: Error messages

## Usage

### Step 1: Select Data Sources

1. Check data sources to align
2. Select alignment mode (Temporal/Spatial/Both)
3. Review data information

### Step 2: Configure Alignment

1. For temporal: Set alignment method and time range
2. For spatial: Configure transformation parameters
3. Set coordinate system settings

### Step 3: Execute Alignment

1. Click "Execute Alignment" button
2. Wait for alignment to complete
3. Review alignment metrics

### Step 4: Validate and Visualize

1. Check validation status
2. View before/after visualizations
3. Review error statistics
4. Export aligned data if needed

## Example Workflow

1. **Select Sources**: Choose ISPM and CT scan data
2. **Temporal Alignment**: Align by build layers
3. **Spatial Alignment**: Apply rotation and translation
4. **Execute**: Run alignment
5. **Validate**: Check alignment metrics
6. **Visualize**: Compare before/after views

## Key Takeaways

1. **Temporal Alignment**: Align data by time or layers
2. **Spatial Transformation**: Apply translation, rotation, scaling
3. **Coordinate Systems**: Align different coordinate systems
4. **Validation**: Always validate alignment results
5. **Visualization**: Use visualizations to verify alignment

## Related Notebooks

- **Previous**: [03: Signal Mapping Fundamentals](03-signal-mapping.md)
- **Next**: [05: Data Correction and Processing](05-correction.md)
- **Related**: [06: Multi-Source Data Fusion](06-fusion.md)

## Related Documentation

- **[Synchronization Module](../../AM_QADF/05-modules/synchronization.md)** - Synchronization details
- **[Synchronization API](../../AM_QADF/06-api-reference/synchronization-api.md)** - Synchronization API

---

**Last Updated**: 2024

