"""
Script to generate all test fixture files.

Run this script to create pre-computed test fixtures for testing.
This script generates:
- Voxel grid fixtures (small, medium, large)
- Point cloud fixtures (hatching paths, laser points, CT points)
- Signal fixtures (sample signals)
- Validation fixtures (ground truth data, MPM comparison data, test datasets)
- Monitoring fixtures (alert_fixtures.json, health_fixtures.json)
- SPC fixtures (spc_fixtures.npz, capability_fixtures.json)
- Streaming fixtures (kafka_fixtures.json)
- OpenVDB fixtures (test_uniform_grid.vdb, test_multi_res_grid.vdb, test_adaptive_grid.vdb)
- Ensures mocks/ directory exists (mocks are code, not generated)

Usage:
    python scripts/generate_test_fixtures.py
"""

import sys
from pathlib import Path
import pickle
import json
import numpy as np

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    from am_qadf.voxelization.uniform_resolution import VoxelGrid
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure AM-QADF is properly installed (am_qadf_native with OpenVDB bindings).")
    sys.exit(1)


# ============================================================================
# Voxel Grid Generation
# ============================================================================

def create_small_voxel_grid():
    """Create small voxel grid (10x10x10)."""
    print("Creating small voxel grid (10x10x10)...")
    
    grid = VoxelGrid(
        bbox_min=(0.0, 0.0, 0.0),
        bbox_max=(10.0, 10.0, 10.0),
        resolution=1.0
    )
    
    # Add sample data
    np.random.seed(42)
    n_points = 100
    points = np.random.rand(n_points, 3) * 10.0
    
    for point in points:
        grid.add_point(
            point[0], point[1], point[2],
            signals={
                'laser_power': float(np.random.rand() * 300.0),
                'temperature': float(np.random.rand() * 1000.0)
            }
        )
    
    # Finalize to aggregate signals
    grid.finalize()
    
    return grid


def create_medium_voxel_grid():
    """Create medium voxel grid (50x50x50)."""
    print("Creating medium voxel grid (50x50x50)...")
    
    grid = VoxelGrid(
        bbox_min=(0.0, 0.0, 0.0),
        bbox_max=(50.0, 50.0, 50.0),
        resolution=1.0
    )
    
    # Add sample data
    np.random.seed(42)
    n_points = 1000
    points = np.random.rand(n_points, 3) * 50.0
    
    for point in points:
        grid.add_point(
            point[0], point[1], point[2],
            signals={
                'laser_power': float(np.random.rand() * 300.0),
                'temperature': float(np.random.rand() * 1000.0),
                'density': float(np.random.rand() * 1.0)
            }
        )
    
    # Finalize to aggregate signals
    grid.finalize()
    
    return grid


def create_large_voxel_grid():
    """Create large voxel grid (100x100x100)."""
    print("Creating large voxel grid (100x100x100)...")
    
    grid = VoxelGrid(
        bbox_min=(0.0, 0.0, 0.0),
        bbox_max=(100.0, 100.0, 100.0),
        resolution=1.0
    )
    
    # Add sample data (sparse to keep file size reasonable)
    np.random.seed(42)
    n_points = 5000
    points = np.random.rand(n_points, 3) * 100.0
    
    for point in points:
        grid.add_point(
            point[0], point[1], point[2],
            signals={
                'laser_power': float(np.random.rand() * 300.0),
                'temperature': float(np.random.rand() * 1000.0),
                'density': float(np.random.rand() * 1.0),
                'velocity': float(np.random.rand() * 100.0)
            }
        )
    
    # Finalize to aggregate signals
    grid.finalize()
    
    return grid


# ============================================================================
# Point Cloud Generation
# ============================================================================

def generate_hatching_paths():
    """Generate hatching paths fixture."""
    print("Generating hatching paths fixture...")
    np.random.seed(42)
    
    layers = []
    for layer_idx in range(5):
        hatches = []
        for hatch_idx in range(10):
            # Generate hatch path points
            n_points = 50
            x_start = np.random.rand() * 100.0
            y_start = np.random.rand() * 100.0
            z = layer_idx * 0.03  # 30 micron layer height
            
            points = []
            for i in range(n_points):
                x = x_start + (i / n_points) * 10.0  # 10mm hatch length
                y = y_start + np.random.rand() * 0.1  # Small variation
                points.append([float(x), float(y), float(z)])
            
            hatch = {
                'start_point': points[0],
                'end_point': points[-1],
                'points': points,
                'laser_power': float(200.0 + np.random.rand() * 50.0),
                'scan_speed': float(1000.0 + np.random.rand() * 200.0),
                'energy_density': float(50.0 + np.random.rand() * 20.0),
                'laser_beam_width': 0.1,
                'hatch_spacing': 0.15,  # 0.15mm spacing allows gaps to be visible at 0.1mm voxel resolution
                'overlap_percentage': float(50.0 + np.random.rand() * 10.0),
                'hatch_type': 'line',
                'scan_order': hatch_idx
            }
            hatches.append(hatch)
        
        layer = {
            'model_id': 'test_model_001',
            'layer_index': layer_idx,
            'layer_height': 0.03,
            'z_position': z,
            'hatches': hatches,
            'contours': [],
            'processing_time': '2024-01-01T00:00:00Z'
        }
        layers.append(layer)
    
    return layers


def generate_laser_points(n_points=1000):
    """Generate laser points fixture.
    
    Args:
        n_points: Number of points to generate (default: 1000)
    """
    print(f"Generating laser points fixture ({n_points:,} points)...")
    np.random.seed(42)
    
    points = []
    for i in range(n_points):
        point = {
            'model_id': 'test_model_001',
            'layer_index': int(np.random.randint(0, 5)),
            'point_id': f'point_{i}',
            'spatial_coordinates': [
                float(np.random.rand() * 100.0),
                float(np.random.rand() * 100.0),
                float(np.random.rand() * 5.0 * 0.03)  # 5 layers
            ],
            'laser_power': float(200.0 + np.random.rand() * 50.0),
            'scan_speed': float(1000.0 + np.random.rand() * 200.0),
            'hatch_spacing': 0.1,
            'energy_density': float(50.0 + np.random.rand() * 20.0),
            'exposure_time': float(0.1 + np.random.rand() * 0.05),
            'timestamp': '2024-01-01T00:00:00Z',
            'region_type': np.random.choice(['contour', 'hatch', 'support']).item()
        }
        points.append(point)
    
    return points


def generate_large_laser_points():
    """Generate large laser points fixture (10K points) for signal mapping tests."""
    return generate_laser_points(n_points=10000)


def generate_xlarge_laser_points():
    """Generate extra-large laser points fixture (100K points) for performance tests."""
    return generate_laser_points(n_points=100000)


def generate_ct_points(n_points=500):
    """Generate CT scan points fixture.
    
    Args:
        n_points: Number of points to generate (default: 500)
    """
    print(f"Generating CT scan points fixture ({n_points:,} points)...")
    np.random.seed(42)
    
    # Generate CT scan data structure
    ct_data = {
        'model_id': 'test_model_001',
        'scan_id': 'scan_001',
        'scan_timestamp': '2024-01-01T00:00:00Z',
        'voxel_grid': {
            'dimensions': [100, 100, 100],
            'spacing': [0.1, 0.1, 0.1],
            'origin': [0.0, 0.0, 0.0]
        },
        'points': []
    }
    
    # Generate sample points (sparse representation)
    for i in range(n_points):
        point = {
            'x': float(np.random.rand() * 100.0),
            'y': float(np.random.rand() * 100.0),
            'z': float(np.random.rand() * 10.0),
            'density': float(4.0 + np.random.rand() * 0.5),  # Ti6Al4V density range
            'porosity': float(np.random.rand() * 0.05),  # 0-5% porosity
            'defect': np.random.rand() < 0.1  # 10% chance of defect
        }
        ct_data['points'].append(point)
    
    # Add defect locations
    ct_data['defect_locations'] = [
        [p['x'], p['y'], p['z']] 
        for p in ct_data['points'] 
        if p.get('defect', False)
    ]
    
    return ct_data


def generate_large_ct_points():
    """Generate large CT scan points fixture (5K points) for signal mapping tests."""
    return generate_ct_points(n_points=5000)


# ============================================================================
# Signal Generation
# ============================================================================

def generate_sample_signals():
    """Generate sample signals fixture."""
    print("Generating sample signals fixture...")
    np.random.seed(42)
    
    n_points = 1000
    
    signals = {
        'laser_power': np.random.rand(n_points) * 300.0,  # 0-300 W
        'scan_speed': np.random.rand(n_points) * 2000.0,  # 0-2000 mm/s
        'temperature': np.random.rand(n_points) * 1000.0,  # 0-1000 °C
        'density': np.random.rand(n_points) * 1.0 + 4.0,  # 4.0-5.0 g/cm³
        'porosity': np.random.rand(n_points) * 0.1,  # 0-10%
        'velocity': np.random.rand(n_points) * 100.0,  # 0-100 mm/s
        'energy_density': np.random.rand(n_points) * 100.0,  # 0-100 J/mm²
        'exposure_time': np.random.rand(n_points) * 0.2,  # 0-0.2 s
    }
    
    return signals


# ============================================================================
# Validation Fixture Generation
# ============================================================================

def generate_ground_truth_fixtures():
    """Generate ground truth fixtures for validation testing."""
    print("Generating ground truth fixtures...")
    np.random.seed(42)
    
    fixtures = {}
    
    # Ground truth signal (medium size)
    print("  - Ground truth signal (50x50x10)...")
    signal_3d = np.zeros((50, 50, 10))
    z_coords = np.arange(10)[:, np.newaxis, np.newaxis]
    y_coords = np.arange(50)[np.newaxis, :, np.newaxis]
    x_coords = np.arange(50)[np.newaxis, np.newaxis, :]
    signal_3d = 100 + 10 * np.sin(x_coords * 0.1) + 10 * np.cos(y_coords * 0.1) + 5 * np.sin(z_coords * 0.2)
    fixtures['ground_truth_signal'] = signal_3d.astype(np.float32)
    
    # Ground truth coordinates (1000 points)
    print("  - Ground truth coordinates (1000 points)...")
    coords = np.random.rand(1000, 3) * 10.0
    fixtures['ground_truth_coordinates'] = coords.astype(np.float32)
    
    # Ground truth quality metrics
    print("  - Ground truth quality metrics...")
    fixtures['ground_truth_quality_metrics'] = {
        'overall_quality_score': 0.9,
        'data_quality_score': 0.85,
        'signal_quality_score': 0.92,
        'alignment_score': 0.88,
        'completeness_score': 0.95,
        'completeness': 0.90,
        'snr': 25.5,
        'alignment_accuracy': 0.95,
    }
    
    return fixtures


def generate_mpm_comparison_fixtures():
    """Generate MPM comparison fixtures for validation testing."""
    print("Generating MPM comparison fixtures...")
    np.random.seed(42)
    
    fixtures = {}
    
    # Framework quality metrics
    framework_metrics = {
        'completeness': 0.9,
        'snr': 25.5,
        'alignment_accuracy': 0.95,
        'overall_quality_score': 0.9,
    }
    
    # MPM metrics (slightly different, with known correlation ~0.9)
    print("  - MPM quality metrics...")
    fixtures['mpm_quality_metrics'] = {
        'completeness': 0.88,
        'snr': 24.8,
        'alignment_accuracy': 0.94,
        'overall_quality_score': 0.88,
    }
    
    # Framework output array
    print("  - Framework output array (100x100)...")
    framework_array = np.random.rand(100, 100) * 100.0
    fixtures['framework_array'] = framework_array.astype(np.float32)
    
    # MPM output array (correlated with framework)
    print("  - MPM output array (100x100, correlated)...")
    mpm_array = framework_array * 0.9 + np.random.rand(100, 100) * 10.0
    fixtures['mpm_array'] = mpm_array.astype(np.float32)
    
    # Framework metrics dict
    fixtures['framework_metrics'] = framework_metrics
    
    return fixtures


def generate_validation_test_datasets():
    """Generate validation test datasets (small, medium, large)."""
    print("Generating validation test datasets...")
    np.random.seed(42)
    
    datasets = {}
    
    sizes = {
        'small': {'signal_shape': (20, 20, 5), 'n_points': 100},
        'medium': {'signal_shape': (50, 50, 10), 'n_points': 1000},
        'large': {'signal_shape': (100, 100, 20), 'n_points': 10000},
    }
    
    for size_name, config in sizes.items():
        print(f"  - {size_name.capitalize()} dataset ({config['signal_shape']}, {config['n_points']} points)...")
        dataset = {
            'framework_signal': (np.random.rand(*config['signal_shape']) * 100.0).astype(np.float32),
            'ground_truth_signal': (np.random.rand(*config['signal_shape']) * 100.0).astype(np.float32),
            'framework_coords': (np.random.rand(config['n_points'], 3) * 10.0).astype(np.float32),
            'ground_truth_coords': (np.random.rand(config['n_points'], 3) * 10.0).astype(np.float32),
            'framework_metrics': {
                'completeness': 0.9,
                'snr': 25.5,
                'alignment_accuracy': 0.95,
            },
            'mpm_metrics': {
                'completeness': 0.88,
                'snr': 24.8,
                'alignment_accuracy': 0.94,
            },
        }
        datasets[size_name] = dataset
    
    return datasets


# ============================================================================
# Monitoring Fixtures (alert, health)
# ============================================================================

def generate_monitoring_fixtures():
    """Generate monitoring test data (alerts, health) as JSON-serializable dicts."""
    from datetime import datetime, timedelta
    
    np.random.seed(42)
    alerts = []
    for i in range(20):
        alerts.append({
            "alert_id": f"alert_{i}",
            "alert_type": np.random.choice(["quality_threshold", "temperature", "power", "defect"]),
            "severity": np.random.choice(["low", "medium", "high", "critical"]),
            "message": f"Test alert {i} from MonitoringFixture",
            "timestamp": (datetime.now() - timedelta(minutes=i * 5)).isoformat(),
            "source": "TestSource",
            "metadata": {"metric_value": float(np.random.uniform(80, 200)), "threshold": 150.0},
            "acknowledged": i % 3 == 0,
        })
    
    health = {
        "cpu_percent": 45.0,
        "memory_percent": 62.0,
        "disk_percent": 55.0,
        "timestamp": datetime.now().isoformat(),
        "status": "healthy",
    }
    return {"alerts": alerts, "health": health}


# ============================================================================
# SPC Fixtures (baseline, capability, control chart)
# ============================================================================

def generate_spc_fixtures():
    """Generate SPC test data (baseline, capability, control chart)."""
    np.random.seed(42)
    
    baseline_stable = np.random.normal(10.0, 1.0, 200)
    baseline_unstable = np.concatenate([
        np.random.normal(10.0, 1.0, 50),
        np.random.normal(11.0, 1.0, 50),
        np.random.normal(9.5, 1.0, 50),
        np.random.normal(10.5, 1.0, 50),
    ])
    
    capability = {
        "cp": 1.33,
        "cpk": 1.2,
        "mean": 10.0,
        "std": 0.5,
        "usl": 12.0,
        "lsl": 8.0,
        "n_samples": 100,
    }
    
    control_chart = np.random.normal(10.0, 1.0, 100).tolist()
    
    return {
        "baseline_stable": baseline_stable,
        "baseline_unstable": baseline_unstable,
        "capability": capability,
        "control_chart": control_chart,
    }


# ============================================================================
# Streaming (Kafka) Fixtures
# ============================================================================

def generate_streaming_fixtures():
    """Generate streaming/Kafka test messages as JSON-serializable list."""
    from datetime import datetime
    
    np.random.seed(42)
    messages = []
    for i in range(30):
        messages.append({
            "topic": "am_qadf_monitoring",
            "key": f"key_{i}",
            "partition": 0,
            "offset": i,
            "timestamp": (datetime.now()).isoformat(),
            "value": {
                "sensor_id": f"sensor_{i % 5}",
                "temperature": float(np.random.normal(1000.0, 50.0)),
                "power": float(np.random.normal(200.0, 10.0)),
                "velocity": float(np.random.normal(100.0, 5.0)),
                "x": float(np.random.uniform(0.0, 100.0)),
                "y": float(np.random.uniform(0.0, 100.0)),
                "z": float(np.random.uniform(0.0, 10.0)),
                "timestamp": datetime.now().isoformat(),
            },
        })
    return messages


# ============================================================================
# OpenVDB Fixtures (.vdb files for C++/ParaView tests)
# ============================================================================

def generate_openvdb_fixtures(voxel_grid_small, voxel_grid_medium, openvdb_dir):
    """Export voxel grids to .vdb files if am_qadf_native is available."""
    try:
        from am_qadf.visualization.paraview_exporter import export_voxel_grid_to_paraview
    except ImportError:
        print("   ⚠ ParaView exporter not available (am_qadf_native); skipping .vdb generation.")
        (openvdb_dir / "README.txt").write_text(
            "OpenVDB fixtures require am_qadf_native with OpenVDB bindings.\n"
            "Run: python scripts/generate_test_fixtures.py (with native build).\n"
            "Or create test_uniform_grid.vdb, test_multi_res_grid.vdb, test_adaptive_grid.vdb manually.\n"
        )
        return
    
    openvdb_dir.mkdir(parents=True, exist_ok=True)
    
    # test_uniform_grid.vdb (from small grid)
    path_uniform = openvdb_dir / "test_uniform_grid.vdb"
    export_voxel_grid_to_paraview(voxel_grid_small, str(path_uniform))
    print(f"   ✓ {path_uniform.name}")
    
    # test_multi_res_grid.vdb (from medium grid; same format, different size)
    path_multi = openvdb_dir / "test_multi_res_grid.vdb"
    export_voxel_grid_to_paraview(voxel_grid_medium, str(path_multi))
    print(f"   ✓ {path_multi.name}")
    
    # test_adaptive_grid.vdb: try AdaptiveResolutionGrid if available, else export small again with different name
    try:
        from am_qadf.voxelization.adaptive_resolution import AdaptiveResolutionGrid
        adaptive = AdaptiveResolutionGrid(
            bbox_min=(0.0, 0.0, 0.0),
            bbox_max=(10.0, 10.0, 10.0),
            base_resolution=1.0,
        )
        np.random.seed(42)
        for _ in range(50):
            x, y, z = np.random.rand(3) * 10.0
            adaptive.add_point(x, y, z, signals={"temperature": float(np.random.rand() * 1000.0)})
        adaptive.finalize()
        path_adaptive = openvdb_dir / "test_adaptive_grid.vdb"
        export_voxel_grid_to_paraview(adaptive, str(path_adaptive))
        print(f"   ✓ {path_adaptive.name}")
    except Exception as e:
        path_adaptive = openvdb_dir / "test_adaptive_grid.vdb"
        export_voxel_grid_to_paraview(voxel_grid_small, str(path_adaptive))
        print(f"   ✓ {path_adaptive.name} (uniform fallback; adaptive not available: {e})")


# ============================================================================
# Main Generation Function
# ============================================================================

def main():
    """Generate all test fixtures."""
    # Define fixture directories
    tests_dir = project_root / 'tests'
    fixtures_dir = tests_dir / 'fixtures'
    voxel_dir = fixtures_dir / 'voxel_data'
    point_cloud_dir = fixtures_dir / 'point_clouds'
    signals_dir = fixtures_dir / 'signals'
    validation_dir = fixtures_dir / 'validation'
    mocks_dir = fixtures_dir / 'mocks'
    monitoring_dir = fixtures_dir / 'monitoring'
    spc_dir = fixtures_dir / 'spc'
    streaming_dir = fixtures_dir / 'streaming'
    openvdb_dir = fixtures_dir / 'openvdb'
    
    # Create all directories (mocks is code-only; we just ensure it exists)
    for d in (voxel_dir, point_cloud_dir, signals_dir, validation_dir,
              mocks_dir, monitoring_dir, spc_dir, streaming_dir, openvdb_dir):
        d.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Generating Test Fixtures")
    print("=" * 60)
    
    # ========================================================================
    # Generate Voxel Grids
    # ========================================================================
    print("\n" + "=" * 60)
    print("Voxel Grid Fixtures")
    print("=" * 60)
    
    # Voxel grids use C++ OpenVDB (not picklable). We save metadata only; grids are
    # generated on load via tests/fixtures/voxel_data (generate_*_voxel_grid).
    print("\n1. Small Voxel Grid (10x10x10)")
    small_grid = create_small_voxel_grid()
    small_meta = voxel_dir / 'small_voxel_grid.meta.json'
    with open(small_meta, 'w') as f:
        json.dump({
            "bbox_min": list(small_grid.bbox_min),
            "bbox_max": list(small_grid.bbox_max),
            "resolution": float(small_grid.resolution),
            "signals": sorted(small_grid.available_signals),
        }, f, indent=2)
    print(f"   ✓ Metadata saved to {small_meta} (grid generated on load)")
    print(f"   Dimensions: {small_grid.dims}")
    print(f"   Resolution: {small_grid.resolution} mm")
    print(f"   Available signals: {small_grid.available_signals}")

    print("\n2. Medium Voxel Grid (50x50x50)")
    medium_grid = create_medium_voxel_grid()
    medium_meta = voxel_dir / 'medium_voxel_grid.meta.json'
    with open(medium_meta, 'w') as f:
        json.dump({
            "bbox_min": list(medium_grid.bbox_min),
            "bbox_max": list(medium_grid.bbox_max),
            "resolution": float(medium_grid.resolution),
            "signals": sorted(medium_grid.available_signals),
        }, f, indent=2)
    print(f"   ✓ Metadata saved to {medium_meta} (grid generated on load)")
    print(f"   Dimensions: {medium_grid.dims}")
    print(f"   Resolution: {medium_grid.resolution} mm")
    print(f"   Available signals: {medium_grid.available_signals}")

    print("\n3. Large Voxel Grid (100x100x100)")
    large_grid = create_large_voxel_grid()
    large_meta = voxel_dir / 'large_voxel_grid.meta.json'
    with open(large_meta, 'w') as f:
        json.dump({
            "bbox_min": list(large_grid.bbox_min),
            "bbox_max": list(large_grid.bbox_max),
            "resolution": float(large_grid.resolution),
            "signals": sorted(large_grid.available_signals),
        }, f, indent=2)
    print(f"   ✓ Metadata saved to {large_meta} (grid generated on load)")
    print(f"   Dimensions: {large_grid.dims}")
    print(f"   Resolution: {large_grid.resolution} mm")
    print(f"   Available signals: {large_grid.available_signals}")
    
    # ========================================================================
    # Generate Point Clouds
    # ========================================================================
    print("\n" + "=" * 60)
    print("Point Cloud Fixtures")
    print("=" * 60)
    
    # Hatching paths
    print("\n1. Hatching Paths")
    hatching_paths = generate_hatching_paths()
    hatching_path = point_cloud_dir / 'hatching_paths.json'
    with open(hatching_path, 'w') as f:
        json.dump(hatching_paths, f, indent=2)
    print(f"   ✓ Saved to {hatching_path}")
    print(f"   Layers: {len(hatching_paths)}")
    print(f"   Total hatches: {sum(len(layer['hatches']) for layer in hatching_paths)}")
    
    # Laser points
    print("\n2. Laser Points")
    laser_points = generate_laser_points()
    laser_path = point_cloud_dir / 'laser_points.json'
    with open(laser_path, 'w') as f:
        json.dump(laser_points, f, indent=2)
    print(f"   ✓ Saved to {laser_path}")
    print(f"   Points: {len(laser_points)}")
    
    # Large laser points (for signal mapping tests)
    print("\n3. Large Laser Points (10K)")
    large_laser_points = generate_large_laser_points()
    large_laser_path = point_cloud_dir / 'laser_points_large.json'
    with open(large_laser_path, 'w') as f:
        json.dump(large_laser_points, f, indent=2)
    print(f"   ✓ Saved to {large_laser_path}")
    print(f"   Points: {len(large_laser_points)}")
    
    # CT points
    print("\n4. CT Scan Points")
    ct_points = generate_ct_points()
    ct_path = point_cloud_dir / 'ct_points.json'
    with open(ct_path, 'w') as f:
        json.dump(ct_points, f, indent=2)
    print(f"   ✓ Saved to {ct_path}")
    print(f"   Points: {len(ct_points['points'])}")
    print(f"   Defect locations: {len(ct_points['defect_locations'])}")
    
    # Large CT points (for signal mapping tests)
    print("\n5. Large CT Scan Points (5K)")
    large_ct_points = generate_large_ct_points()
    large_ct_path = point_cloud_dir / 'ct_points_large.json'
    with open(large_ct_path, 'w') as f:
        json.dump(large_ct_points, f, indent=2)
    print(f"   ✓ Saved to {large_ct_path}")
    print(f"   Points: {len(large_ct_points['points'])}")
    print(f"   Defect locations: {len(large_ct_points['defect_locations'])}")
    
    # ========================================================================
    # Generate Signals
    # ========================================================================
    print("\n" + "=" * 60)
    print("Signal Fixtures")
    print("=" * 60)
    
    # Sample signals
    print("\n1. Sample Signals")
    signals = generate_sample_signals()
    signals_path = signals_dir / 'sample_signals.npz'
    np.savez(signals_path, **signals)
    print(f"   ✓ Saved to {signals_path}")
    print(f"   Signals: {list(signals.keys())}")
    print(f"   Points per signal: {len(signals['laser_power'])}")
    
    # ========================================================================
    # Generate Validation Fixtures
    # ========================================================================
    print("\n" + "=" * 60)
    print("Validation Fixtures")
    print("=" * 60)
    
    # Ground truth fixtures
    print("\n1. Ground Truth Fixtures")
    ground_truth = generate_ground_truth_fixtures()
    ground_truth_path = validation_dir / 'ground_truth_data.pkl'
    with open(ground_truth_path, 'wb') as f:
        pickle.dump(ground_truth, f)
    print(f"   ✓ Saved to {ground_truth_path}")
    print(f"   Keys: {list(ground_truth.keys())}")
    print(f"   Signal shape: {ground_truth['ground_truth_signal'].shape}")
    print(f"   Coordinates shape: {ground_truth['ground_truth_coordinates'].shape}")
    
    # MPM comparison fixtures
    print("\n2. MPM Comparison Fixtures")
    mpm_fixtures = generate_mpm_comparison_fixtures()
    mpm_path = validation_dir / 'mpm_comparison_data.pkl'
    with open(mpm_path, 'wb') as f:
        pickle.dump(mpm_fixtures, f)
    print(f"   ✓ Saved to {mpm_path}")
    print(f"   Keys: {list(mpm_fixtures.keys())}")
    print(f"   Framework array shape: {mpm_fixtures['framework_array'].shape}")
    print(f"   MPM array shape: {mpm_fixtures['mpm_array'].shape}")
    
    # Validation test datasets
    print("\n3. Validation Test Datasets")
    validation_datasets = generate_validation_test_datasets()
    datasets_path = validation_dir / 'validation_test_datasets.pkl'
    with open(datasets_path, 'wb') as f:
        pickle.dump(validation_datasets, f)
    print(f"   ✓ Saved to {datasets_path}")
    print(f"   Sizes: {list(validation_datasets.keys())}")
    for size_name, dataset in validation_datasets.items():
        print(f"   - {size_name}: signal {dataset['framework_signal'].shape}, "
              f"coords {dataset['framework_coords'].shape}")
    
    # ========================================================================
    # Generate Monitoring Fixtures
    # ========================================================================
    print("\n" + "=" * 60)
    print("Monitoring Fixtures")
    print("=" * 60)
    monitoring_data = generate_monitoring_fixtures()
    alert_path = monitoring_dir / 'alert_fixtures.json'
    with open(alert_path, 'w') as f:
        json.dump(monitoring_data["alerts"], f, indent=2)
    print(f"   ✓ {alert_path.name}: {len(monitoring_data['alerts'])} alerts")
    health_path = monitoring_dir / 'health_fixtures.json'
    with open(health_path, 'w') as f:
        json.dump(monitoring_data["health"], f, indent=2)
    print(f"   ✓ {health_path.name}")
    
    # ========================================================================
    # Generate SPC Fixtures
    # ========================================================================
    print("\n" + "=" * 60)
    print("SPC Fixtures")
    print("=" * 60)
    spc_data = generate_spc_fixtures()
    spc_path = spc_dir / 'spc_fixtures.npz'
    np.savez(
        spc_path,
        baseline_stable=spc_data["baseline_stable"],
        baseline_unstable=spc_data["baseline_unstable"],
        control_chart=np.array(spc_data["control_chart"]),
    )
    print(f"   ✓ {spc_path.name}: baseline_stable, baseline_unstable, control_chart")
    capability_path = spc_dir / 'capability_fixtures.json'
    with open(capability_path, 'w') as f:
        json.dump(spc_data["capability"], f, indent=2)
    print(f"   ✓ {capability_path.name}")
    
    # ========================================================================
    # Generate Streaming (Kafka) Fixtures
    # ========================================================================
    print("\n" + "=" * 60)
    print("Streaming Fixtures")
    print("=" * 60)
    kafka_messages = generate_streaming_fixtures()
    kafka_path = streaming_dir / 'kafka_fixtures.json'
    with open(kafka_path, 'w') as f:
        json.dump(kafka_messages, f, indent=2)
    print(f"   ✓ {kafka_path.name}: {len(kafka_messages)} messages")
    
    # ========================================================================
    # Generate OpenVDB Fixtures (.vdb for C++/ParaView)
    # ========================================================================
    print("\n" + "=" * 60)
    print("OpenVDB Fixtures")
    print("=" * 60)
    generate_openvdb_fixtures(small_grid, medium_grid, openvdb_dir)
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 60)
    print("All fixtures generated successfully!")
    print("=" * 60)
    print(f"\nFixture locations:")
    print(f"  - Voxel grids: {voxel_dir}")
    print(f"  - Point clouds: {point_cloud_dir}")
    print(f"  - Signals: {signals_dir}")
    print(f"  - Validation: {validation_dir}")
    print(f"  - Mocks: {mocks_dir} (code only)")
    print(f"  - Monitoring: {monitoring_dir}")
    print(f"  - SPC: {spc_dir}")
    print(f"  - Streaming: {streaming_dir}")
    print(f"  - OpenVDB: {openvdb_dir}")


if __name__ == '__main__':
    main()

