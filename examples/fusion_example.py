"""
Data Fusion Example

Demonstrates multi-source data fusion:
1. Create multiple signal sources
2. Fuse signals using different strategies
3. Compare fusion results
"""

import numpy as np
from am_qadf.voxelization import VoxelGrid
from am_qadf.fusion import VoxelFusion
from am_qadf.synchronization.data_fusion import FusionStrategy


def create_sample_voxel_grid(signal_name, base_value, noise_level=0.1):
    """Create a sample voxel grid with a signal."""
    grid = VoxelGrid(
        bbox_min=(0, 0, 0),
        bbox_max=(10, 10, 10),
        resolution=1.0,
        aggregation='mean'
    )
    
    # Generate signal values for each voxel
    dims = grid.dims
    signal_array = np.ones(dims) * base_value
    signal_array += np.random.randn(*dims) * noise_level * base_value
    
    # Add signal to grid
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                x, y, z = grid._voxel_to_world(i, j, k)
                grid.add_point(x, y, z, signals={signal_name: signal_array[i, j, k]})
    
    grid.finalize()
    return grid


def main():
    """Data fusion example."""
    print("=" * 60)
    print("AM-QADF Data Fusion Example")
    print("=" * 60)
    
    # Step 1: Create multiple signal sources
    print("\n1. Creating multiple signal sources...")
    
    # Source 1: Hatching power (higher value, lower noise)
    hatching_grid = create_sample_voxel_grid('hatching_power', base_value=200.0, noise_level=0.05)
    print("   Created hatching power signal (mean=200, low noise)")
    
    # Source 2: Laser power (lower value, higher noise)
    laser_grid = create_sample_voxel_grid('laser_power', base_value=180.0, noise_level=0.15)
    print("   Created laser power signal (mean=180, higher noise)")
    
    # Source 3: CT scan density
    ct_grid = create_sample_voxel_grid('ct_density', base_value=150.0, noise_level=0.10)
    print("   Created CT scan density signal (mean=150, medium noise)")
    
    # Step 2: Initialize fusion engine
    print("\n2. Initializing fusion engine...")
    fusion = VoxelFusion(
        default_strategy=FusionStrategy.WEIGHTED_AVERAGE,
        use_quality_scores=True
    )
    
    # Step 3: Fuse signals with different strategies
    print("\n3. Fusing signals with different strategies...")
    
    strategies = [
        ('Weighted Average', FusionStrategy.WEIGHTED_AVERAGE),
        ('Median', FusionStrategy.MEDIAN),
        ('Average', FusionStrategy.AVERAGE),
        ('Max', FusionStrategy.MAX),
        ('Min', FusionStrategy.MIN)
    ]
    
    # Use hatching grid as base (it has the best quality)
    base_grid = hatching_grid
    
    # Add other signals to base grid
    for signal_name in ['laser_power', 'ct_density']:
        if signal_name == 'laser_power':
            source_grid = laser_grid
        else:
            source_grid = ct_grid
        
        signal_array = source_grid.get_signal_array(signal_name, default=0.0)
        for i in range(base_grid.dims[0]):
            for j in range(base_grid.dims[1]):
                for k in range(base_grid.dims[2]):
                    x, y, z = base_grid._voxel_to_world(i, j, k)
                    value = signal_array[i, j, k]
                    if value != 0:
                        base_grid.add_point(x, y, z, signals={signal_name: value})
    
    base_grid.finalize()
    
    # Define quality scores (higher = better)
    quality_scores = {
        'hatching_power': 0.9,  # High quality
        'laser_power': 0.7,     # Medium quality
        'ct_density': 0.8       # Good quality
    }
    
    results = {}
    for strategy_name, strategy in strategies:
        print(f"\n   {strategy_name}:")
        try:
            fused = fusion.fuse_voxel_signals(
                voxel_data=base_grid,
                signals=['hatching_power', 'laser_power', 'ct_density'],
                fusion_strategy=strategy,
                quality_scores=quality_scores,
                output_signal_name='fused_power'
            )
            
            # Calculate statistics
            mean_val = np.mean(fused)
            std_val = np.std(fused)
            print(f"      Mean: {mean_val:.2f}")
            print(f"      Std: {std_val:.2f}")
            print(f"      Range: [{np.min(fused):.2f}, {np.max(fused):.2f}]")
            
            results[strategy_name] = fused
            
        except Exception as e:
            print(f"      ❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("Data fusion example completed!")
    print("=" * 60)
    print("\nNote: Compare fusion results to choose the best strategy")
    print("      for your specific use case.")


if __name__ == "__main__":
    main()

