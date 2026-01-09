# Visualization Module API Reference

## Overview

The Visualization module provides 3D visualization capabilities for voxel domain data.

## VoxelRenderer

3D rendering engine for voxel grids.

```python
from am_qadf.visualization import VoxelRenderer

renderer = VoxelRenderer()
```

### Methods

#### `render_3d(voxel_grid: VoxelGrid, signal_name: str, colormap: str = 'viridis', opacity: float = 1.0, show: bool = True) -> Any`

Render 3D voxel grid.

**Parameters**:
- `voxel_grid` (VoxelGrid): Voxel grid to render
- `signal_name` (str): Signal name to visualize
- `colormap` (str): Colormap name (default: 'viridis')
- `opacity` (float): Opacity (0-1, default: 1.0)
- `show` (bool): Whether to display immediately (default: True)

**Returns**: Plotter object (PyVista)

**Example**:
```python
plotter = renderer.render_3d(
    voxel_grid=grid,
    signal_name='power',
    colormap='hot',
    opacity=0.8
)
```

#### `render_slice(voxel_grid: VoxelGrid, signal_name: str, axis: str = 'z', index: int = 0, colormap: str = 'viridis') -> Any`

Render 2D slice of voxel grid.

**Parameters**:
- `voxel_grid` (VoxelGrid): Voxel grid
- `signal_name` (str): Signal name
- `axis` (str): Slice axis ('x', 'y', 'z')
- `index` (int): Slice index
- `colormap` (str): Colormap name

**Returns**: Matplotlib figure

---

## MultiResolutionViewer

Viewer for multi-resolution grids.

```python
from am_qadf.visualization import MultiResolutionViewer

viewer = MultiResolutionViewer()
```

### Methods

#### `view_level(grid: MultiResolutionGrid, level: int, signal_name: str) -> Any`

View specific resolution level.

**Parameters**:
- `grid` (MultiResolutionGrid): Multi-resolution grid
- `level` (int): Resolution level
- `signal_name` (str): Signal name

**Returns**: Plotter object

---

## NotebookWidgets

Interactive widgets for Jupyter notebooks.

```python
from am_qadf.visualization import NotebookWidget

widget = NotebookWidget(voxel_grid: VoxelGrid)
```

### Methods

#### `create_interactive_viewer(signal_name: str) -> Any`

Create interactive viewer widget.

**Parameters**:
- `signal_name` (str): Signal name

**Returns**: ipywidgets widget

**Example**:
```python
widget = NotebookWidget(grid)
viewer = widget.create_interactive_viewer('power')
display(viewer)  # In Jupyter notebook
```

---

## Utility Functions

### `plot_signal_distribution(signal_array: np.ndarray, signal_name: str = 'signal') -> Any`

Plot signal distribution histogram.

**Parameters**:
- `signal_array` (np.ndarray): Signal array
- `signal_name` (str): Signal name for labeling

**Returns**: Matplotlib figure

### `plot_signal_over_time(signal_array: np.ndarray, time_array: np.ndarray, signal_name: str = 'signal') -> Any`

Plot signal over time.

**Parameters**:
- `signal_array` (np.ndarray): Signal array
- `time_array` (np.ndarray): Time array
- `signal_name` (str): Signal name

**Returns**: Matplotlib figure

---

## Related

- [Visualization Module Documentation](../05-modules/visualization.md) - Module overview
- [All API References](README.md) - Other API references

---

**Parent**: [API Reference](README.md)

