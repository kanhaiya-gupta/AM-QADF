"""
Quality Assessment Example

Demonstrates quality assessment capabilities:
1. Assess data completeness
2. Assess signal quality (SNR, uncertainty)
3. Assess alignment accuracy
4. Generate quality report
"""

import numpy as np
from am_qadf.voxelization import VoxelGrid
from am_qadf.quality import QualityAssessmentClient
from am_qadf.query.mongodb_client import MongoDBClient


def create_sample_voxel_grid_with_quality_issues():
    """Create a sample voxel grid with various quality issues."""
    grid = VoxelGrid(
        bbox_min=(0, 0, 0),
        bbox_max=(10, 10, 10),
        resolution=1.0,
        aggregation='mean'
    )
    
    dims = grid.dims
    
    # Create signal with some missing regions (completeness issue)
    power_signal = np.ones(dims) * 200.0
    
    # Add some missing regions (gaps)
    power_signal[2:4, 2:4, :] = 0  # Missing region
    
    # Add noise (signal quality issue)
    noise = np.random.randn(*dims) * 10.0
    power_signal += noise
    
    # Add some outliers (alignment/quality issue)
    power_signal[5, 5, 5] = 500.0  # Outlier
    
    # Add signal to grid
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                if power_signal[i, j, k] > 0:  # Skip missing regions
                    x, y, z = grid._voxel_to_world(i, j, k)
                    grid.add_point(x, y, z, signals={'power': power_signal[i, j, k]})
    
    grid.finalize()
    return grid


def main():
    """Quality assessment example."""
    print("=" * 60)
    print("AM-QADF Quality Assessment Example")
    print("=" * 60)
    
    # Step 1: Create sample voxel grid with quality issues
    print("\n1. Creating sample voxel grid with quality issues...")
    voxel_grid = create_sample_voxel_grid_with_quality_issues()
    print(f"   Grid dimensions: {voxel_grid.dims}")
    print(f"   Filled voxels: {voxel_grid.get_filled_voxel_count()}/{voxel_grid.get_voxel_count()}")
    
    # Step 2: Initialize quality assessment client
    print("\n2. Initializing quality assessment client...")
    try:
        mongo_client = MongoDBClient(
            connection_string="mongodb://localhost:27017",
            database_name="am_qadf"
        )
        quality_client = QualityAssessmentClient(mongo_client=mongo_client)
    except Exception as e:
        print(f"   ⚠️  MongoDB not available, using client without storage: {e}")
        quality_client = QualityAssessmentClient(mongo_client=None)
    
    # Step 3: Assess data completeness
    print("\n3. Assessing data completeness...")
    try:
        completeness_metrics = quality_client.assess_data_quality(
            voxel_data=voxel_grid,
            signals=['power']
        )
        print(f"   Overall quality: {completeness_metrics.overall_quality:.3f}")
        print(f"   Completeness: {completeness_metrics.completeness:.3f}")
        print(f"   Coverage: {completeness_metrics.coverage:.3f}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Step 4: Assess signal quality
    print("\n4. Assessing signal quality...")
    try:
        power_array = voxel_grid.get_signal_array('power', default=0.0)
        
        signal_metrics = quality_client.assess_signal_quality(
            signal_name='power',
            signal_array=power_array,
            measurement_uncertainty=0.05,  # 5% uncertainty
            store_maps=False
        )
        
        print(f"   Signal-to-Noise Ratio (SNR): {signal_metrics.snr:.2f} dB")
        print(f"   Uncertainty: {signal_metrics.uncertainty:.3f}")
        print(f"   Signal completeness: {signal_metrics.completeness:.3f}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Step 5: Assess all signals
    print("\n5. Assessing all signals...")
    try:
        all_metrics = quality_client.assess_all_signals(
            voxel_data=voxel_grid,
            signals=['power'],
            store_maps=False
        )
        
        for signal_name, metrics in all_metrics.items():
            print(f"\n   Signal: {signal_name}")
            print(f"      SNR: {metrics.snr:.2f} dB")
            print(f"      Uncertainty: {metrics.uncertainty:.3f}")
            print(f"      Completeness: {metrics.completeness:.3f}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Step 6: Quality summary
    print("\n" + "=" * 60)
    print("Quality Assessment Summary")
    print("=" * 60)
    print("\nQuality Issues Detected:")
    print("  - Missing regions (completeness)")
    print("  - Noise in signal (SNR)")
    print("  - Outliers (data quality)")
    print("\nRecommendations:")
    print("  - Fill missing regions using interpolation")
    print("  - Apply noise reduction filters")
    print("  - Remove or correct outliers")
    
    print("\n" + "=" * 60)
    print("Quality assessment example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

