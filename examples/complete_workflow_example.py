"""
Complete Workflow Example

Demonstrates a complete end-to-end workflow:
1. Query multi-source data
2. Create voxel grid
3. Map signals to voxels
4. Synchronize and correct data
5. Process signals
6. Fuse signals
7. Assess quality
8. Perform analytics
9. Detect anomalies
10. Visualize results
"""

import numpy as np
from am_qadf.query import UnifiedQueryClient
from am_qadf.voxel_domain import VoxelDomainClient
from am_qadf.query.mongodb_client import MongoDBClient
from am_qadf.quality import QualityAssessmentClient
from am_qadf.analytics import AnalyticsClient
from am_qadf.anomaly_detection import AnomalyDetectionClient
from am_qadf.visualization import VoxelRenderer


def main():
    """Complete workflow example."""
    print("=" * 70)
    print("AM-QADF Complete Workflow Example")
    print("=" * 70)
    
    # Configuration
    model_id = "example_model_id"  # Replace with actual model ID
    resolution = 1.0  # mm
    
    try:
        # ====================================================================
        # STEP 1: Initialize Clients
        # ====================================================================
        print("\n" + "=" * 70)
        print("STEP 1: Initializing Clients")
        print("=" * 70)
        
        mongo_client = MongoDBClient(
            connection_string="mongodb://localhost:27017",
            database_name="am_qadf"
        )
        
        query_client = UnifiedQueryClient(mongo_client=mongo_client)
        voxel_client = VoxelDomainClient(
            unified_query_client=query_client,
            base_resolution=resolution,
            adaptive=False,
            mongo_client=mongo_client
        )
        
        quality_client = QualityAssessmentClient(mongo_client=mongo_client)
        analytics_client = AnalyticsClient(mongo_client=mongo_client)
        anomaly_client = AnomalyDetectionClient(mongo_client=mongo_client)
        
        print("✅ All clients initialized")
        
        # ====================================================================
        # STEP 2: Query Data
        # ====================================================================
        print("\n" + "=" * 70)
        print("STEP 2: Querying Multi-Source Data")
        print("=" * 70)
        
        all_data = query_client.get_all_data(model_id)
        print(f"✅ Retrieved data from {len([k for k, v in all_data.items() if v is not None])} sources")
        print(f"   Available sources: {[k for k, v in all_data.items() if v is not None]}")
        
        # ====================================================================
        # STEP 3: Create Voxel Grid
        # ====================================================================
        print("\n" + "=" * 70)
        print("STEP 3: Creating Voxel Grid")
        print("=" * 70)
        
        voxel_grid = voxel_client.create_voxel_grid(
            model_id=model_id,
            resolution=resolution
        )
        print(f"✅ Voxel grid created")
        print(f"   Dimensions: {voxel_grid.dims}")
        print(f"   Size: {voxel_grid.size} mm")
        print(f"   Resolution: {resolution} mm")
        
        # ====================================================================
        # STEP 4: Map Signals to Voxels
        # ====================================================================
        print("\n" + "=" * 70)
        print("STEP 4: Mapping Signals to Voxels")
        print("=" * 70)
        
        voxel_grid = voxel_client.map_signals_to_voxels(
            model_id=model_id,
            voxel_grid=voxel_grid,
            sources=['hatching', 'laser', 'ct', 'ispm'],
            method='linear',  # Use linear interpolation for smooth results
            n_workers=4
        )
        print(f"✅ Signals mapped to voxels")
        print(f"   Available signals: {voxel_grid.available_signals}")
        
        # ====================================================================
        # STEP 5: Quality Assessment
        # ====================================================================
        print("\n" + "=" * 70)
        print("STEP 5: Assessing Data Quality")
        print("=" * 70)
        
        quality_metrics = quality_client.assess_data_quality(
            voxel_data=voxel_grid,
            signals=list(voxel_grid.available_signals)
        )
        print(f"✅ Quality assessment completed")
        print(f"   Overall quality: {quality_metrics.overall_quality:.3f}")
        print(f"   Completeness: {quality_metrics.completeness:.3f}")
        print(f"   Coverage: {quality_metrics.coverage:.3f}")
        
        # ====================================================================
        # STEP 6: Fuse Signals
        # ====================================================================
        print("\n" + "=" * 70)
        print("STEP 6: Fusing Multi-Source Signals")
        print("=" * 70)
        
        # Get quality scores for fusion
        signal_quality = quality_client.assess_all_signals(
            voxel_data=voxel_grid,
            signals=list(voxel_grid.available_signals),
            store_maps=False
        )
        
        # Create quality scores dictionary (normalize to 0-1)
        quality_scores = {}
        for signal_name, metrics in signal_quality.items():
            # Use SNR as quality indicator (normalized)
            quality_scores[signal_name] = min(1.0, max(0.0, metrics.snr / 50.0))
        
        # Fuse signals
        voxel_grid = voxel_client.fuse_signals(
            voxel_grid=voxel_grid,
            signals=list(voxel_grid.available_signals),
            quality_scores=quality_scores,
            output_signal_name='fused_power'
        )
        print(f"✅ Signals fused")
        print(f"   Fused signal 'fused_power' added to grid")
        
        # ====================================================================
        # STEP 7: Statistical Analysis
        # ====================================================================
        print("\n" + "=" * 70)
        print("STEP 7: Performing Statistical Analysis")
        print("=" * 70)
        
        stats_client = analytics_client.get_statistical_analysis_client()
        
        fused_array = voxel_grid.get_signal_array('fused_power', default=0.0)
        fused_flat = fused_array[fused_array != 0]
        
        stats = stats_client.compute_descriptive_statistics(fused_flat)
        print(f"✅ Statistical analysis completed")
        print(f"   Mean: {stats.get('mean', 0):.2f}")
        print(f"   Std: {stats.get('std', 0):.2f}")
        print(f"   Min: {stats.get('min', 0):.2f}")
        print(f"   Max: {stats.get('max', 0):.2f}")
        print(f"   Median: {stats.get('median', 0):.2f}")
        
        # ====================================================================
        # STEP 8: Anomaly Detection
        # ====================================================================
        print("\n" + "=" * 70)
        print("STEP 8: Detecting Anomalies")
        print("=" * 70)
        
        anomaly_result = anomaly_client.detect_anomalies(
            voxel_data=voxel_grid,
            signal_name='fused_power',
            detector_type='isolation_forest',
            contamination=0.1
        )
        
        n_anomalies = np.sum(anomaly_result.anomaly_mask)
        total_voxels = len(anomaly_result.anomaly_mask)
        print(f"✅ Anomaly detection completed")
        print(f"   Detected anomalies: {n_anomalies}/{total_voxels} ({100*n_anomalies/total_voxels:.1f}%)")
        
        # ====================================================================
        # STEP 9: Save Results
        # ====================================================================
        print("\n" + "=" * 70)
        print("STEP 9: Saving Voxel Grid")
        print("=" * 70)
        
        grid_id = voxel_client.save_voxel_grid(
            voxel_grid=voxel_grid,
            model_id=model_id
        )
        print(f"✅ Voxel grid saved")
        print(f"   Grid ID: {grid_id}")
        
        # ====================================================================
        # STEP 10: Visualization
        # ====================================================================
        print("\n" + "=" * 70)
        print("STEP 10: Visualizing Results")
        print("=" * 70)
        
        renderer = VoxelRenderer()
        try:
            plotter = renderer.render_3d(
                voxel_grid=voxel_grid,
                signal_name='fused_power',
                colormap='hot',
                opacity=0.8,
                show=True
            )
            print("✅ 3D visualization rendered")
            print("   Close the window to finish...")
        except Exception as e:
            print(f"⚠️  3D visualization not available: {e}")
            print("   (Requires PyVista)")
        
        # ====================================================================
        # Summary
        # ====================================================================
        print("\n" + "=" * 70)
        print("WORKFLOW COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\nSummary:")
        print(f"  ✅ Queried data from {len([k for k, v in all_data.items() if v is not None])} sources")
        print(f"  ✅ Created voxel grid ({voxel_grid.dims})")
        print(f"  ✅ Mapped {len(voxel_grid.available_signals)} signals")
        print(f"  ✅ Assessed quality (score: {quality_metrics.overall_quality:.3f})")
        print(f"  ✅ Fused signals with quality weighting")
        print(f"  ✅ Performed statistical analysis")
        print(f"  ✅ Detected {n_anomalies} anomalies")
        print(f"  ✅ Saved grid (ID: {grid_id})")
        print(f"  ✅ Visualized results")
        
    except Exception as e:
        print("\n" + "=" * 70)
        print("ERROR IN WORKFLOW")
        print("=" * 70)
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("  1. Ensure MongoDB is running")
        print("  2. Verify model_id exists in database")
        print("  3. Check that data sources are available")
        print("  4. Verify all dependencies are installed")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

