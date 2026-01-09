"""
Signal Mapping Example

Demonstrates different interpolation methods for mapping point cloud signals
to voxel grids:
1. Nearest neighbor interpolation
2. Linear interpolation
3. IDW interpolation
4. KDE interpolation
5. RBF interpolation
"""

import numpy as np
from am_qadf.voxelization import VoxelGrid
from am_qadf.signal_mapping.methods import (
    NearestNeighborInterpolation,
    LinearInterpolation,
    IDWInterpolation,
    GaussianKDEInterpolation,
    RBFInterpolation,
)


def generate_sample_data(n_points=1000):
    """Generate sample point cloud data."""
    # Generate random points in a 10x10x10 mm cube
    points = np.random.rand(n_points, 3) * 10.0
    
    # Generate sample signals
    power = 100 + 50 * np.sin(points[:, 0] / 2) + np.random.randn(n_points) * 5
    temperature = 1000 + 200 * np.cos(points[:, 1] / 2) + np.random.randn(n_points) * 10
    
    signals = {
        'power': power,
        'temperature': temperature
    }
    
    return points, signals


def main():
    """Signal mapping example."""
    print("=" * 60)
    print("AM-QADF Signal Mapping Example")
    print("=" * 60)
    
    # Generate sample data
    print("\n1. Generating sample point cloud data...")
    points, signals = generate_sample_data(n_points=1000)
    print(f"   Generated {len(points)} points")
    print(f"   Signals: {list(signals.keys())}")
    
    # Create voxel grid
    print("\n2. Creating voxel grid...")
    voxel_grid = VoxelGrid(
        bbox_min=(0, 0, 0),
        bbox_max=(10, 10, 10),
        resolution=0.5,  # 0.5mm voxels
        aggregation='mean'
    )
    print(f"   Grid dimensions: {voxel_grid.dims}")
    
    # Test different interpolation methods
    methods = [
        ('Nearest Neighbor', NearestNeighborInterpolation()),
        ('Linear', LinearInterpolation(k_neighbors=8)),
        ('IDW', IDWInterpolation(power=2.0, k_neighbors=8)),
        ('Gaussian KDE', GaussianKDEInterpolation(bandwidth=1.0)),
        ('RBF (Gaussian)', RBFInterpolation(kernel='gaussian', epsilon=1.0, smoothing=0.0))
    ]
    
    # Note: RBF has O(N³) complexity, so use with smaller datasets or Spark backend
    print("\n   Note: RBF interpolation has O(N³) complexity.")
    print("         For large datasets, consider using Spark backend or alternative methods.")
    
    print("\n3. Testing interpolation methods...")
    results = {}
    
    for method_name, interpolator in methods:
        print(f"\n   {method_name} Interpolation:")
        try:
            result_grid = interpolator.interpolate(
                points=points,
                signals=signals,
                voxel_grid=voxel_grid
            )
            
            # Get interpolated signal
            power_array = result_grid.get_signal_array('power', default=0.0)
            filled_voxels = np.sum(power_array != 0)
            total_voxels = np.prod(power_array.shape)
            
            print(f"      Filled voxels: {filled_voxels}/{total_voxels} ({100*filled_voxels/total_voxels:.1f}%)")
            print(f"      Power range: [{np.min(power_array[power_array != 0]):.2f}, "
                  f"{np.max(power_array):.2f}]")
            
            results[method_name] = result_grid
            
        except Exception as e:
            print(f"      ❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("Signal mapping example completed!")
    print("=" * 60)
    print("\nNote: Compare the results from different methods to choose")
    print("      the best interpolation method for your use case.")


if __name__ == "__main__":
    main()

