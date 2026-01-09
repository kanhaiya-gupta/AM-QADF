"""
Visualization Example

Demonstrates 3D visualization capabilities:
1. Render 3D voxel grid
2. Render 2D slices
3. Interactive visualization (if in Jupyter)
4. Multi-resolution viewing
"""

import numpy as np
from am_qadf.voxelization import VoxelGrid
from am_qadf.visualization import VoxelRenderer


def create_sample_voxel_grid_for_visualization():
    """Create a sample voxel grid with interesting signal patterns."""
    grid = VoxelGrid(
        bbox_min=(0, 0, 0),
        bbox_max=(10, 10, 10),
        resolution=0.5,  # Smaller voxels for better visualization
        aggregation='mean'
    )
    
    dims = grid.dims
    
    # Create signal with spatial patterns
    power_signal = np.zeros(dims)
    
    # Create gradient pattern
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                # Gradient from low to high
                power_signal[i, j, k] = 100 + (i / dims[0]) * 200
                
                # Add some spatial variation
                distance_from_center = np.sqrt(
                    (i - dims[0]/2)**2 + (j - dims[1]/2)**2 + (k - dims[2]/2)**2
                )
                power_signal[i, j, k] += 50 * np.sin(distance_from_center / 2)
    
    # Add signal to grid
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                x, y, z = grid._voxel_to_world(i, j, k)
                grid.add_point(x, y, z, signals={'power': power_signal[i, j, k]})
    
    grid.finalize()
    return grid


def main():
    """Visualization example."""
    print("=" * 60)
    print("AM-QADF Visualization Example")
    print("=" * 60)
    
    # Step 1: Create sample voxel grid
    print("\n1. Creating sample voxel grid for visualization...")
    voxel_grid = create_sample_voxel_grid_for_visualization()
    print(f"   Grid dimensions: {voxel_grid.dims}")
    print(f"   Grid size: {voxel_grid.size} mm")
    
    # Get signal statistics
    power_array = voxel_grid.get_signal_array('power', default=0.0)
    print(f"   Power signal range: [{np.min(power_array):.2f}, {np.max(power_array):.2f}]")
    
    # Step 2: Initialize renderer
    print("\n2. Initializing 3D renderer...")
    renderer = VoxelRenderer()
    
    # Step 3: Render 3D visualization
    print("\n3. Rendering 3D voxel grid...")
    print("   (This will open an interactive 3D window if PyVista is available)")
    
    try:
        plotter = renderer.render_3d(
            voxel_grid=voxel_grid,
            signal_name='power',
            colormap='viridis',
            opacity=0.8,
            show=True  # Set to False if running in headless mode
        )
        print("   ✅ 3D visualization rendered successfully!")
        print("   Close the window to continue...")
        
    except Exception as e:
        print(f"   ⚠️  3D rendering not available: {e}")
        print("      (Requires PyVista. Install with: pip install pyvista)")
    
    # Step 4: Render 2D slices
    print("\n4. Rendering 2D slices...")
    try:
        # Render slices along different axes
        for axis in ['x', 'y', 'z']:
            slice_index = voxel_grid.dims[{'x': 0, 'y': 1, 'z': 2}[axis]] // 2
            
            print(f"\n   {axis.upper()}-axis slice at index {slice_index}:")
            fig = renderer.render_slice(
                voxel_grid=voxel_grid,
                signal_name='power',
                axis=axis,
                index=slice_index,
                colormap='hot'
            )
            
            if fig:
                print(f"      ✅ Slice rendered")
                # In Jupyter, this would display automatically
                # In script mode, you might want to save: fig.savefig(f'slice_{axis}_{slice_index}.png')
            else:
                print(f"      ⚠️  Slice rendering returned None")
                
    except Exception as e:
        print(f"   ⚠️  Slice rendering error: {e}")
        print("      (Requires Matplotlib)")
    
    # Step 5: Signal distribution plot
    print("\n5. Plotting signal distribution...")
    try:
        from am_qadf.visualization import plot_signal_distribution
        
        fig = plot_signal_distribution(
            signal_array=power_array[power_array != 0],
            signal_name='power'
        )
        print("   ✅ Distribution plot created")
        # In script mode: fig.savefig('power_distribution.png')
        
    except Exception as e:
        print(f"   ⚠️  Distribution plotting error: {e}")
    
    # Step 6: Interactive widgets (Jupyter only)
    print("\n6. Interactive widgets (Jupyter notebooks only)...")
    print("   To use interactive widgets in Jupyter:")
    print("   ```python")
    print("   from am_qadf.visualization import NotebookWidget")
    print("   widget = NotebookWidget(voxel_grid)")
    print("   viewer = widget.create_interactive_viewer('power')")
    print("   display(viewer)")
    print("   ```")
    
    print("\n" + "=" * 60)
    print("Visualization Example Summary")
    print("=" * 60)
    print("\nVisualization Features:")
    print("  ✅ 3D voxel grid rendering (PyVista)")
    print("  ✅ 2D slice visualization (Matplotlib)")
    print("  ✅ Signal distribution plots")
    print("  ✅ Interactive widgets (Jupyter)")
    print("\nNote: Some features require additional dependencies:")
    print("  - PyVista for 3D rendering: pip install pyvista")
    print("  - Matplotlib for 2D plots: pip install matplotlib")
    print("  - ipywidgets for interactive widgets: pip install ipywidgets")
    
    print("\n" + "=" * 60)
    print("Visualization example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

