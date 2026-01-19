#!/usr/bin/env python3
"""
Check Fused Data in MongoDB

This script comprehensively checks fused grids saved from Notebook 06 (Multi-Source Data Fusion).
It verifies that:
1. Fusion metadata is properly stored
2. Fused signals are stored in GridFS and can be loaded
3. All source grid references are present and accessible
4. Fusion metrics are saved
5. Signal arrays have correct shapes and contain valid data
6. Naming convention is correct
7. No signals are missing
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
src_dir = project_root / 'src'

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import os
import numpy as np
import gzip
import io
from bson import ObjectId
from gridfs import GridFS
from typing import Dict, Any, List, Optional

# Load environment variables
env_file = project_root / 'development.env'
if env_file.exists():
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                value = value.strip('"\'')
                os.environ[key] = value

from src.infrastructure.config import MongoDBConfig
from src.infrastructure.database import MongoDBClient
from am_qadf.voxel_domain import VoxelGridStorage
from am_qadf.voxel_domain.grid_naming import GridNaming


def check_gridfs_file(mongo_client, file_id, bucket_name='fs'):
    """Check if a GridFS file exists and get its metadata."""
    try:
        fs = GridFS(mongo_client.database, collection=bucket_name)
        grid_file = fs.get(ObjectId(file_id))
        
        metadata = grid_file.metadata or {}
        size_mb = grid_file.length / (1024 * 1024)
        
        return {
            'exists': True,
            'size_mb': size_mb,
            'filename': grid_file.filename,
            'metadata': metadata
        }
    except Exception as e:
        return {
            'exists': False,
            'error': str(e)
        }


def load_signal_array(mongo_client, file_id, bucket_name='fs'):
    """Load and verify a signal array from GridFS."""
    try:
        fs = GridFS(mongo_client.database, collection=bucket_name)
        grid_file = fs.get(ObjectId(file_id))
        
        # Read and decompress
        compressed_data = grid_file.read()
        decompressed = gzip.decompress(compressed_data)
        data = np.load(io.BytesIO(decompressed), allow_pickle=True)
        
        # Check if it's a dictionary-like object (npz format) or direct array (npy format)
        if isinstance(data, np.lib.npyio.NpzFile):
            # .npz format (sparse or multi-array)
            if 'format' in data.files and data['format'] == 'sparse':
                # Sparse format: reconstruct array
                indices = data['indices']
                values = data['values']
                dims = data['dims']
                default = data.get('default', 0.0)
                
                # Reconstruct dense array
                array = np.full(tuple(dims), default, dtype=np.float32)
                if len(indices) > 0:
                    array[indices[:, 0], indices[:, 1], indices[:, 2]] = values
                
                return {
                    'success': True,
                    'array': array,
                    'shape': tuple(dims),
                    'dtype': str(array.dtype),
                    'format': 'sparse',
                    'non_zero_count': len(values),
                    'total_voxels': np.prod(dims)
                }
            else:
                # Dense format in npz - get first array
                if len(data.files) > 0:
                    first_key = data.files[0]
                    array = data[first_key]
                    return {
                        'success': True,
                        'array': array,
                        'shape': array.shape,
                        'dtype': str(array.dtype),
                        'format': 'dense_npz',
                        'non_zero_count': np.count_nonzero(array),
                        'total_voxels': array.size
                    }
        elif isinstance(data, np.ndarray):
            # Direct numpy array (.npy format - dense)
            return {
                'success': True,
                'array': data,
                'shape': data.shape,
                'dtype': str(data.dtype),
                'format': 'dense_npy',
                'non_zero_count': np.count_nonzero(data),
                'total_voxels': data.size
            }
        
        return {
            'success': False,
            'error': f'Unknown data format: {type(data)}'
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def verify_naming_convention(grid_name: str) -> Dict[str, Any]:
    """Verify grid name follows the naming convention."""
    try:
        parsed = GridNaming.parse_grid_name(grid_name)
        return {
            'valid': True,
            'parsed': parsed
        }
    except Exception as e:
        return {
            'valid': False,
            'error': str(e)
        }


def check_source_grid(voxel_storage, source_grid_id: str) -> Dict[str, Any]:
    """Check if a source grid exists and is accessible."""
    try:
        grid_data = voxel_storage.load_voxel_grid(grid_id=source_grid_id)
        if grid_data:
            return {
                'exists': True,
                'grid_name': grid_data.get('grid_name', 'Unknown'),
                'available_signals': grid_data.get('available_signals', []),
                'metadata': grid_data.get('metadata', {})
            }
        else:
            return {
                'exists': False,
                'error': 'Grid not found'
            }
    except Exception as e:
        return {
            'exists': False,
            'error': str(e)
        }


def main():
    print("=" * 80)
    print("🔍 Comprehensive Fused Data Check")
    print("=" * 80)
    print()
    
    # Connect to MongoDB
    print("🔌 Connecting to MongoDB...")
    try:
        config = MongoDBConfig.from_env()
        if not config.username:
            config.username = os.getenv('MONGO_ROOT_USERNAME', 'admin')
        if not config.password:
            config.password = os.getenv('MONGO_ROOT_PASSWORD', 'password')
        
        mongo_client = MongoDBClient(config=config)
        if not mongo_client.is_connected():
            print("❌ Failed to connect to MongoDB")
            return
        
        print(f"✅ Connected to MongoDB: {config.database}")
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        return
    
    # Initialize VoxelGridStorage
    try:
        voxel_storage = VoxelGridStorage(mongo_client=mongo_client)
    except Exception as e:
        print(f"❌ Failed to initialize VoxelGridStorage: {e}")
        return
    
    # Get all grids
    collection = mongo_client.get_collection('voxel_grids')
    all_grids = list(collection.find({}).sort('created_at', -1).limit(1000))
    
    # Filter for fused grids
    fused_grids = []
    for grid in all_grids:
        metadata = grid.get('metadata', {})
        config_meta = metadata.get('configuration_metadata', {})
        if not config_meta:
            config_meta = metadata
        
        if config_meta.get('fusion_applied', False):
            fused_grids.append(grid)
    
    print()
    print("📊 Summary Statistics")
    print("-" * 80)
    print(f"Total Grids: {len(all_grids)}")
    print(f"Fused Grids: {len(fused_grids)}")
    print()
    
    if not fused_grids:
        print("⚠️ No fused grids found in MongoDB")
        return
    
    print("=" * 80)
    print("📋 Detailed Fused Grid Verification")
    print("=" * 80)
    print()
    
    # Track overall statistics
    total_issues = 0
    grids_with_issues = []
    
    for idx, grid in enumerate(fused_grids, 1):
        grid_id = str(grid.get('_id', ''))
        grid_name = grid.get('grid_name', 'Unknown')
        model_id = grid.get('model_id', 'Unknown')
        model_name = grid.get('model_name', 'Unknown')
        created_at = grid.get('created_at', 'Unknown')
        
        metadata = grid.get('metadata', {})
        config_meta = metadata.get('configuration_metadata', {})
        if not config_meta:
            config_meta = metadata
        
        print("-" * 80)
        print(f"📦 Grid {idx}/{len(fused_grids)}")
        print("-" * 80)
        print()
        print("📌 Basic Information:")
        print(f"  Grid ID: {grid_id}")
        print(f"  Grid Name: {grid_name}")
        print(f"  Model ID: {model_id}")
        print(f"  Model Name: {model_name}")
        print(f"  Created: {created_at}")
        print()
        
        # Verify naming convention
        print("🏷️  Naming Convention Check:")
        naming_check = verify_naming_convention(grid_name)
        if naming_check['valid']:
            parsed = naming_check['parsed']
            print(f"  ✅ Valid naming convention")
            print(f"     Source: {parsed.get('source', 'N/A')}")
            print(f"     Grid Type: {parsed.get('grid_type', 'N/A')}")
            print(f"     Resolution: {parsed.get('resolution', 'N/A')} ({GridNaming.parse_resolution(parsed.get('resolution', '100'))} mm)")
            print(f"     Stage: {parsed.get('stage', 'N/A')}")
            if 'fusion_strategy' in parsed:
                print(f"     Fusion Strategy: {parsed.get('fusion_strategy', 'N/A')}")
        else:
            print(f"  ❌ Invalid naming convention: {naming_check.get('error', 'Unknown error')}")
            total_issues += 1
        print()
        
        # Fusion metadata
        print("🔀 Fusion Details:")
        fusion_strategy = config_meta.get('fusion_strategy', 'Unknown')
        fusion_timestamp = config_meta.get('fusion_timestamp', 'Unknown')
        num_sources = config_meta.get('num_sources', 0)
        source_grids = config_meta.get('source_grids', [])
        source_names = config_meta.get('source_names', [])
        
        print(f"  Fusion Strategy: {fusion_strategy}")
        print(f"  Fusion Timestamp: {fusion_timestamp}")
        print(f"  Number of Sources: {num_sources}")
        print(f"  Source Grid IDs: {len(source_grids)} grid(s)")
        
        # Verify source grids
        source_grid_status = []
        for i, (grid_id_ref, source_name) in enumerate(zip(source_grids, source_names), 1):
            source_check = check_source_grid(voxel_storage, grid_id_ref)
            if source_check['exists']:
                print(f"    ✅ {i}. {source_name} (Grid ID: {grid_id_ref[:8]}...)")
                source_grid_status.append(True)
            else:
                print(f"    ❌ {i}. {source_name} (Grid ID: {grid_id_ref[:8]}...) - {source_check.get('error', 'Not found')}")
                source_grid_status.append(False)
                total_issues += 1
        
        if not all(source_grid_status):
            grids_with_issues.append(grid_id)
        
        print()
        
        # Strategy-specific parameters
        if fusion_strategy == 'weighted_average':
            weighted_avg = config_meta.get('weighted_average', {})
            if weighted_avg:
                print(f"  Weighted Average Parameters:")
                print(f"    Normalize Weights: {weighted_avg.get('normalize_weights', 'N/A')}")
                print(f"    Auto-weight by Quality: {weighted_avg.get('auto_weight_quality', 'N/A')}")
                weights = weighted_avg.get('weights', {})
                if weights:
                    print(f"    Weights:")
                    for source, weight in weights.items():
                        if weight is not None:
                            print(f"      {source}: {weight:.2f}")
        elif fusion_strategy == 'quality_based':
            quality_based = config_meta.get('quality_based', {})
            if quality_based:
                print(f"  Quality-Based Parameters:")
                print(f"    Quality Threshold: {quality_based.get('quality_threshold', 'N/A')}")
                print(f"    Quality Source: {quality_based.get('quality_source', 'N/A')}")
        print()
        
        # Fusion metrics
        fusion_metrics = config_meta.get('fusion_metrics', {})
        if fusion_metrics:
            print("📊 Fusion Metrics:")
            print(f"  Fusion Score: {fusion_metrics.get('fusion_score', 'N/A')}")
            print(f"  Coverage: {fusion_metrics.get('coverage', 'N/A')}")
            print(f"  Quality Score: {fusion_metrics.get('quality_score', 'N/A')}")
            print(f"  Consistency Score: {fusion_metrics.get('consistency_score', 'N/A')}")
            print()
        
        # Available signals
        available_signals = grid.get('available_signals', [])
        print(f"📊 Available Signals: {len(available_signals)} signal(s)")
        
        # Check for required "fused" signal
        expected_signals = ['fused']
        missing_signals = []
        for expected in expected_signals:
            if expected not in available_signals:
                missing_signals.append(expected)
                print(f"  ❌ Missing required signal: {expected}")
                total_issues += 1
        
        if available_signals:
            for signal in available_signals:
                status = "✅" if signal not in missing_signals else "❌"
                print(f"  {status} {signal}")
        print()
        
        # Signal storage verification and loading
        signal_references = grid.get('signal_references', {})
        print("💾 Signal Storage Verification:")
        if signal_references:
            total_size = 0
            valid_signals = 0
            loaded_signals = 0
            signal_shapes = {}
            
            for signal_name, file_id in signal_references.items():
                # Check GridFS file exists
                file_info = check_gridfs_file(mongo_client, file_id)
                if file_info['exists']:
                    total_size += file_info['size_mb']
                    valid_signals += 1
                    
                    # Try to load the signal array
                    print(f"  ✅ {signal_name}:")
                    print(f"     File ID: {file_id[:24]}...")
                    print(f"     Size: {file_info['size_mb']:.2f} MB")
                    
                    # Load and verify signal array
                    signal_data = load_signal_array(mongo_client, file_id)
                    if signal_data['success']:
                        loaded_signals += 1
                        signal_shapes[signal_name] = signal_data['shape']
                        
                        print(f"     ✅ Loaded successfully")
                        print(f"     Shape: {signal_data['shape']}")
                        print(f"     Dtype: {signal_data['dtype']}")
                        print(f"     Format: {signal_data['format']}")
                        print(f"     Non-zero values: {signal_data['non_zero_count']:,} / {signal_data['total_voxels']:,}")
                        
                        # Check for empty or invalid arrays
                        array = signal_data['array']
                        if array.size == 0:
                            print(f"     ⚠️ Warning: Empty array!")
                            total_issues += 1
                        elif np.all(np.isnan(array)):
                            print(f"     ⚠️ Warning: All values are NaN!")
                            total_issues += 1
                        elif np.all(array == 0):
                            print(f"     ⚠️ Warning: All values are zero!")
                        else:
                            print(f"     Statistics: min={np.nanmin(array):.2f}, max={np.nanmax(array):.2f}, mean={np.nanmean(array):.2f}, std={np.nanstd(array):.2f}")
                    else:
                        print(f"     ❌ Failed to load: {signal_data.get('error', 'Unknown error')}")
                        total_issues += 1
                else:
                    print(f"  ❌ {signal_name}: File not found ({file_info.get('error', 'Unknown error')})")
                    total_issues += 1
            
            print()
            print(f"  Storage Summary:")
            print(f"    Valid Files: {valid_signals}/{len(signal_references)}")
            print(f"    Loaded Signals: {loaded_signals}/{len(signal_references)}")
            print(f"    Total Storage: {total_size:.2f} MB")
            
            # Check if all signals have matching shapes
            if len(signal_shapes) > 1:
                shapes = list(signal_shapes.values())
                if len(set(shapes)) > 1:
                    print(f"    ⚠️ Warning: Signal shapes differ: {signal_shapes}")
                    total_issues += 1
                else:
                    print(f"    ✅ All signals have matching shape: {shapes[0]}")
        else:
            print("  ❌ No signal arrays stored in GridFS")
            total_issues += 1
        print()
        
        # Check metadata completeness
        print("📋 Metadata Completeness:")
        required_fields = ['source', 'grid_type', 'resolution', 'fusion_applied', 'fusion_strategy']
        missing_fields = []
        for field in required_fields:
            if field not in config_meta:
                missing_fields.append(field)
                print(f"  ❌ Missing: {field}")
                total_issues += 1
        
        if not missing_fields:
            print("  ✅ All required metadata fields present")
        print()
        
        # Description
        description = grid.get('description', '')
        if description:
            print(f"📝 Description: {description}")
            print()
    
    # Final summary
    print("=" * 80)
    print("📊 Final Summary")
    print("=" * 80)
    print(f"Total Fused Grids Checked: {len(fused_grids)}")
    print(f"Total Issues Found: {total_issues}")
    if grids_with_issues:
        print(f"Grids with Issues: {len(grids_with_issues)}")
        for grid_id in grids_with_issues:
            print(f"  - {grid_id[:8]}...")
    
    if total_issues == 0:
        print()
        print("✅ All checks passed! No issues found.")
    else:
        print()
        print(f"⚠️ Found {total_issues} issue(s) that need attention.")
    print("=" * 80)


if __name__ == '__main__':
    main()
