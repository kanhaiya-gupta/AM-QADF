"""
Analytics Example

Demonstrates analytics capabilities:
1. Statistical analysis (descriptive statistics, correlation)
2. Sensitivity analysis (Sobol, Morris)
3. Process analysis
4. Virtual experiments
"""

import numpy as np
from am_qadf.voxelization import VoxelGrid
from am_qadf.analytics import AnalyticsClient
from am_qadf.query.mongodb_client import MongoDBClient


def create_sample_voxel_grid_with_multiple_signals():
    """Create a sample voxel grid with multiple correlated signals."""
    grid = VoxelGrid(
        bbox_min=(0, 0, 0),
        bbox_max=(10, 10, 10),
        resolution=1.0,
        aggregation='mean'
    )
    
    dims = grid.dims
    
    # Create correlated signals
    # Signal 1: Power (base signal)
    power = 100 + 50 * np.sin(np.arange(dims[0])[:, None, None] / 2)
    power = np.tile(power, (1, dims[1], dims[2]))
    power += np.random.randn(*dims) * 5
    
    # Signal 2: Temperature (correlated with power)
    temperature = 1000 + power * 2 + np.random.randn(*dims) * 10
    
    # Signal 3: Velocity (inversely correlated with power)
    velocity = 100 - power * 0.3 + np.random.randn(*dims) * 2
    
    # Add signals to grid
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                x, y, z = grid._voxel_to_world(i, j, k)
                grid.add_point(x, y, z, signals={
                    'power': power[i, j, k],
                    'temperature': temperature[i, j, k],
                    'velocity': velocity[i, j, k]
                })
    
    grid.finalize()
    return grid


def main():
    """Analytics example."""
    print("=" * 60)
    print("AM-QADF Analytics Example")
    print("=" * 60)
    
    # Step 1: Create sample voxel grid
    print("\n1. Creating sample voxel grid with multiple signals...")
    voxel_grid = create_sample_voxel_grid_with_multiple_signals()
    print(f"   Grid dimensions: {voxel_grid.dims}")
    print(f"   Available signals: {voxel_grid.available_signals}")
    
    # Step 2: Initialize analytics client
    print("\n2. Initializing analytics client...")
    try:
        mongo_client = MongoDBClient(
            connection_string="mongodb://localhost:27017",
            database_name="am_qadf"
        )
        analytics_client = AnalyticsClient(mongo_client=mongo_client)
    except Exception as e:
        print(f"   ⚠️  MongoDB not available, using client without storage: {e}")
        analytics_client = AnalyticsClient(mongo_client=None)
    
    # Step 3: Statistical Analysis
    print("\n3. Performing statistical analysis...")
    try:
        stats_client = analytics_client.get_statistical_analysis_client()
        
        # Get signal arrays
        power_array = voxel_grid.get_signal_array('power', default=0.0)
        temp_array = voxel_grid.get_signal_array('temperature', default=0.0)
        vel_array = voxel_grid.get_signal_array('velocity', default=0.0)
        
        # Flatten arrays (remove zeros for analysis)
        power_flat = power_array[power_array != 0]
        temp_flat = temp_array[temp_array != 0]
        vel_flat = vel_array[vel_array != 0]
        
        # Descriptive statistics
        print("\n   Descriptive Statistics:")
        for signal_name, signal_data in [('Power', power_flat), ('Temperature', temp_flat), ('Velocity', vel_flat)]:
            stats = stats_client.compute_descriptive_statistics(signal_data)
            print(f"\n      {signal_name}:")
            print(f"         Mean: {stats.get('mean', 0):.2f}")
            print(f"         Std: {stats.get('std', 0):.2f}")
            print(f"         Min: {stats.get('min', 0):.2f}")
            print(f"         Max: {stats.get('max', 0):.2f}")
            print(f"         Median: {stats.get('median', 0):.2f}")
        
        # Correlation analysis
        print("\n   Correlation Analysis:")
        # Align arrays by removing zeros
        min_len = min(len(power_flat), len(temp_flat), len(vel_flat))
        power_aligned = power_flat[:min_len]
        temp_aligned = temp_flat[:min_len]
        vel_aligned = vel_flat[:min_len]
        
        corr_power_temp = stats_client.compute_correlation(power_aligned, temp_aligned)
        corr_power_vel = stats_client.compute_correlation(power_aligned, vel_aligned)
        corr_temp_vel = stats_client.compute_correlation(temp_aligned, vel_aligned)
        
        print(f"      Power vs Temperature: {corr_power_temp:.3f}")
        print(f"      Power vs Velocity: {corr_power_vel:.3f}")
        print(f"      Temperature vs Velocity: {corr_temp_vel:.3f}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Step 4: Sensitivity Analysis (if SALib is available)
    print("\n4. Performing sensitivity analysis...")
    try:
        sensitivity_client = analytics_client.get_sensitivity_analysis_client()
        
        # Prepare parameter arrays (using signal arrays as parameters)
        parameters = {
            'power': power_flat[:100],  # Use subset for faster computation
            'velocity': vel_flat[:100]
        }
        output = temp_flat[:100]  # Temperature as output
        
        # Sobol analysis
        print("\n   Sobol Sensitivity Analysis:")
        try:
            sobol_results = sensitivity_client.sobol_analysis(
                parameters=parameters,
                output=output,
                n_samples=100  # Reduced for example
            )
            for param, sensitivity in sobol_results.items():
                print(f"      {param}: {sensitivity:.3f}")
        except Exception as e:
            print(f"      ⚠️  Sobol analysis not available: {e}")
        
    except Exception as e:
        print(f"   ⚠️  Sensitivity analysis not available: {e}")
    
    # Step 5: Process Analysis
    print("\n5. Performing process analysis...")
    try:
        process_client = analytics_client.get_process_analysis_client()
        
        # Analyze parameter effects
        parameters_dict = {
            'power': power_flat[:100],
            'velocity': vel_flat[:100]
        }
        
        effects = process_client.analyze_parameter_effects(
            parameters=parameters_dict,
            output=temp_flat[:100]
        )
        
        print("\n   Parameter Effects on Temperature:")
        for param, effect in effects.items():
            print(f"      {param}: {effect:.3f}")
        
    except Exception as e:
        print(f"   ⚠️  Process analysis error: {e}")
    
    print("\n" + "=" * 60)
    print("Analytics example completed!")
    print("=" * 60)
    print("\nNote: Some analytics features require additional dependencies")
    print("      (e.g., SALib for sensitivity analysis).")


if __name__ == "__main__":
    main()

