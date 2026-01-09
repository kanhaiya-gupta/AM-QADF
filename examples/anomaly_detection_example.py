"""
Anomaly Detection Example

Demonstrates anomaly detection capabilities:
1. Statistical detectors (Z-score, IQR)
2. Clustering detectors (DBSCAN, Isolation Forest)
3. ML-based detectors (Autoencoder)
4. Compare detection results
"""

import numpy as np
from am_qadf.voxelization import VoxelGrid
from am_qadf.anomaly_detection import AnomalyDetectionClient
from am_qadf.query.mongodb_client import MongoDBClient


def create_sample_voxel_grid_with_anomalies():
    """Create a sample voxel grid with injected anomalies."""
    grid = VoxelGrid(
        bbox_min=(0, 0, 0),
        bbox_max=(10, 10, 10),
        resolution=1.0,
        aggregation='mean'
    )
    
    dims = grid.dims
    
    # Create normal signal with some anomalies
    power_signal = np.ones(dims) * 200.0
    power_signal += np.random.randn(*dims) * 10.0  # Normal noise
    
    # Inject anomalies
    # 1. High-value outliers
    power_signal[2, 2, 2] = 500.0  # Extreme high
    power_signal[5, 5, 5] = 450.0  # High
    
    # 2. Low-value outliers
    power_signal[8, 8, 8] = 50.0   # Extreme low
    power_signal[1, 1, 1] = 80.0   # Low
    
    # 3. Spatial cluster of anomalies
    power_signal[7:9, 7:9, 7:9] = 300.0  # Cluster
    
    # Add signal to grid
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                x, y, z = grid._voxel_to_world(i, j, k)
                grid.add_point(x, y, z, signals={'power': power_signal[i, j, k]})
    
    grid.finalize()
    return grid


def main():
    """Anomaly detection example."""
    print("=" * 60)
    print("AM-QADF Anomaly Detection Example")
    print("=" * 60)
    
    # Step 1: Create sample voxel grid with anomalies
    print("\n1. Creating sample voxel grid with injected anomalies...")
    voxel_grid = create_sample_voxel_grid_with_anomalies()
    print(f"   Grid dimensions: {voxel_grid.dims}")
    print(f"   Total voxels: {voxel_grid.get_voxel_count()}")
    
    # Step 2: Initialize anomaly detection client
    print("\n2. Initializing anomaly detection client...")
    try:
        mongo_client = MongoDBClient(
            connection_string="mongodb://localhost:27017",
            database_name="am_qadf"
        )
        anomaly_client = AnomalyDetectionClient(mongo_client=mongo_client)
    except Exception as e:
        print(f"   ⚠️  MongoDB not available, using client without storage: {e}")
        anomaly_client = AnomalyDetectionClient(mongo_client=None)
    
    # Step 3: Test different detectors
    print("\n3. Testing different anomaly detectors...")
    
    detectors = [
        ('Z-Score', 'zscore', {'threshold': 3.0}),
        ('IQR', 'iqr', {'factor': 1.5}),
        ('Isolation Forest', 'isolation_forest', {'contamination': 0.1}),
        ('DBSCAN', 'dbscan', {'eps': 0.5, 'min_samples': 5}),
    ]
    
    results = {}
    
    for detector_name, detector_type, params in detectors:
        print(f"\n   {detector_name} Detector:")
        try:
            result = anomaly_client.detect_anomalies(
                voxel_data=voxel_grid,
                signal_name='power',
                detector_type=detector_type,
                **params
            )
            
            # Calculate statistics
            n_anomalies = np.sum(result.anomaly_mask)
            total_voxels = len(result.anomaly_mask)
            anomaly_rate = n_anomalies / total_voxels if total_voxels > 0 else 0
            
            print(f"      Detected anomalies: {n_anomalies}/{total_voxels} ({100*anomaly_rate:.1f}%)")
            print(f"      Anomaly score range: [{np.min(result.anomaly_scores):.2f}, "
                  f"{np.max(result.anomaly_scores):.2f}]")
            
            # Show some anomaly locations
            anomaly_indices = np.where(result.anomaly_mask)[0]
            if len(anomaly_indices) > 0):
                print(f"      Sample anomaly locations: {anomaly_indices[:5]}")
            
            results[detector_name] = result
            
        except Exception as e:
            print(f"      ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Step 4: Compare detector results
    print("\n4. Comparing detector results...")
    if len(results) > 1:
        detector_names = list(results.keys())
        print("\n   Detector Comparison:")
        print(f"   {'Detector':<20} {'Anomalies':<15} {'Rate':<10}")
        print("   " + "-" * 45)
        
        for name, result in results.items():
            n_anomalies = np.sum(result.anomaly_mask)
            total = len(result.anomaly_mask)
            rate = n_anomalies / total if total > 0 else 0
            print(f"   {name:<20} {n_anomalies:<15} {100*rate:>6.1f}%")
    
    # Step 5: ML-based detector (if available)
    print("\n5. Testing ML-based detector (Autoencoder)...")
    try:
        result = anomaly_client.detect_anomalies(
            voxel_data=voxel_grid,
            signal_name='power',
            detector_type='autoencoder',
            threshold=0.1
        )
        
        n_anomalies = np.sum(result.anomaly_mask)
        total_voxels = len(result.anomaly_mask)
        print(f"   Detected anomalies: {n_anomalies}/{total_voxels}")
        print(f"   Anomaly rate: {100*n_anomalies/total_voxels:.1f}%")
        
    except Exception as e:
        print(f"   ⚠️  Autoencoder detector not available: {e}")
        print("      (Requires TensorFlow/Keras)")
    
    print("\n" + "=" * 60)
    print("Anomaly Detection Summary")
    print("=" * 60)
    print("\nAnomaly Types Detected:")
    print("  - High-value outliers (500, 450)")
    print("  - Low-value outliers (50, 80)")
    print("  - Spatial clusters (region 7:9, 7:9, 7:9)")
    print("\nRecommendations:")
    print("  - Use multiple detectors for robust detection")
    print("  - Tune detector parameters based on your data")
    print("  - Validate detected anomalies with domain knowledge")
    
    print("\n" + "=" * 60)
    print("Anomaly detection example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

