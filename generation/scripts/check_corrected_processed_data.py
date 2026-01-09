"""
Check Corrected and Processed Data in MongoDB

This script checks voxel grids that have been corrected or processed,
including correction/processing parameters, metadata, and GridFS storage.
"""

import sys
import os
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

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

# Import MongoDB client
try:
    from src.infrastructure.config import MongoDBConfig
    from src.infrastructure.database import MongoDBClient
    from am_qadf.voxel_domain import VoxelGridStorage
    
    def get_mongodb_config():
        """Get MongoDB config from environment."""
        return MongoDBConfig.from_env()
except Exception as e:
    print(f"❌ Error loading MongoDB client: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


def format_bbox(bbox: Dict[str, Any]) -> str:
    """Format bounding box for display."""
    if not bbox:
        return "N/A"
    
    if isinstance(bbox, dict):
        if 'min' in bbox and 'max' in bbox:
            min_vals = bbox['min']
            max_vals = bbox['max']
            return f"[{min_vals[0]:.2f}, {min_vals[1]:.2f}, {min_vals[2]:.2f}] to [{max_vals[0]:.2f}, {max_vals[1]:.2f}, {max_vals[2]:.2f}]"
        elif 'bbox_min' in bbox and 'bbox_max' in bbox:
            min_vals = bbox['bbox_min']
            max_vals = bbox['bbox_max']
            return f"[{min_vals[0]:.2f}, {min_vals[1]:.2f}, {min_vals[2]:.2f}] to [{max_vals[0]:.2f}, {max_vals[1]:.2f}, {max_vals[2]:.2f}]"
    
    return str(bbox)


def check_gridfs_file(mongo_client: MongoDBClient, file_id: str, file_type: str) -> Dict[str, Any]:
    """Check if a GridFS file exists and get its metadata."""
    try:
        from gridfs import GridFS
        from bson import ObjectId
        
        # Try default bucket first (where MongoDBClient actually stores files)
        buckets_to_try = ['fs']
        
        file_data = None
        found_bucket = None
        gridfs_metadata = {}
        for bucket in buckets_to_try:
            try:
                fs = GridFS(mongo_client.database, collection=bucket)
                
                # Try to get file by ObjectId
                if isinstance(file_id, str):
                    try:
                        file_id_obj = ObjectId(file_id)
                    except Exception:
                        continue
                    else:
                        try:
                            grid_file = fs.get(file_id_obj)
                            file_data = grid_file.read()
                            gridfs_metadata = grid_file.metadata if hasattr(grid_file, 'metadata') else {}
                            found_bucket = bucket
                            break
                        except Exception:
                            continue
            except Exception:
                continue
        
        if not file_data:
            return {'exists': False, 'valid': False, 'error': f'file not found in buckets: {buckets_to_try}'}
        
        if file_data:
            # Try to decompress and load to verify it's valid
            import gzip
            import io
            try:
                decompressed = gzip.decompress(file_data)
                data = np.load(io.BytesIO(decompressed), allow_pickle=True)
                
                # Handle different data formats
                shape_str = 'N/A'
                dtype_str = 'N/A'
                
                # Check if it's a sparse format (npz with 'format' key)
                if hasattr(data, 'files'):
                    if 'format' in data.files and data['format'] == 'sparse':
                        # Sparse format: extract dims from data
                        dims = data['dims'] if 'dims' in data.files else None
                        if dims is not None:
                            if isinstance(dims, np.ndarray):
                                dims = dims.tolist()
                            shape_str = f"{tuple(dims)} (sparse)"
                        if 'values' in data.files:
                            values = data['values']
                            if hasattr(values, 'dtype'):
                                dtype_str = str(values.dtype)
                            if hasattr(values, 'shape'):
                                num_values = len(values)
                                shape_str += f", {num_values} non-zero values"
                    else:
                        # Regular npz format - try to get shape from first array
                        if len(data.files) > 0:
                            first_key = data.files[0]
                            arr = data[first_key]
                            if hasattr(arr, 'shape'):
                                shape_str = str(arr.shape)
                            if hasattr(arr, 'dtype'):
                                dtype_str = str(arr.dtype)
                elif isinstance(data, np.ndarray):
                    # Direct numpy array
                    shape_str = str(data.shape)
                    dtype_str = str(data.dtype)
                elif isinstance(data, dict):
                    # Dictionary format - try to extract shape/dtype
                    if 'dims' in data:
                        dims = data['dims']
                        if isinstance(dims, np.ndarray):
                            dims = dims.tolist()
                        shape_str = f"{tuple(dims)} (sparse)" if isinstance(dims, (list, tuple)) else str(dims)
                    if 'values' in data:
                        values = data['values']
                        if hasattr(values, 'dtype'):
                            dtype_str = str(values.dtype)
                        if hasattr(values, 'shape'):
                            num_values = len(values)
                            if 'dims' in data:
                                shape_str += f", {num_values} non-zero values"
                            else:
                                shape_str = str(values.shape)
                
                result = {
                    'exists': True,
                    'valid': True,
                    'size_bytes': len(file_data),
                    'data_shape': shape_str,
                    'data_dtype': dtype_str,
                    'data_size_mb': len(file_data) / (1024 * 1024),
                    'gridfs_metadata': gridfs_metadata
                }
                if found_bucket:
                    result['bucket'] = found_bucket
                return result
            except Exception as e:
                return {
                    'exists': True,
                    'valid': False,
                    'size_bytes': len(file_data),
                    'error': str(e)
                }
        return {'exists': False, 'valid': False}
    except Exception as e:
        return {'exists': False, 'valid': False, 'error': str(e)}


def check_corrected_processed_data(
    model_id: Optional[str] = None,
    grid_id: Optional[str] = None,
    correction_only: bool = False,
    processing_only: bool = False,
    summary_only: bool = False,
    list_models: bool = False,
    verify_gridfs: bool = True
):
    """
    Check corrected and processed data stored in MongoDB.
    
    Args:
        model_id: Optional model ID to filter by
        grid_id: Optional grid ID to filter by
        correction_only: If True, only show corrected grids
        processing_only: If True, only show processed grids
        summary_only: If True, only show summary statistics
        list_models: If True, only list unique model IDs
        verify_gridfs: If True, verify GridFS files exist
    """
    print("=" * 80)
    print("🔍 Checking Corrected and Processed Data in MongoDB")
    print("=" * 80)
    
    # Connect to MongoDB
    print("\n🔌 Connecting to MongoDB...")
    config = get_mongodb_config()
    
    # Ensure credentials are set
    if not config.username:
        config.username = os.getenv('MONGO_ROOT_USERNAME', 'admin')
    if not config.password:
        config.password = os.getenv('MONGO_ROOT_PASSWORD', 'password')
    
    try:
        mongo_client = MongoDBClient(config=config)
        if not mongo_client.is_connected():
            print("❌ Failed to connect to MongoDB")
            return
        print(f"✅ Connected to MongoDB: {config.database}\n")
    except Exception as e:
        print(f"❌ Failed to connect to MongoDB: {e}")
        return
    
    # Initialize VoxelGridStorage
    try:
        voxel_storage = VoxelGridStorage(mongo_client=mongo_client)
    except Exception as e:
        print(f"❌ Failed to initialize VoxelGridStorage: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Get full documents including configuration_metadata
    try:
        collection = mongo_client.get_collection('voxel_grids')
        query = {}
        if model_id:
            query['model_id'] = model_id
        if grid_id:
            from bson import ObjectId
            try:
                query['_id'] = ObjectId(grid_id)
            except:
                print(f"⚠️ Invalid grid_id format: {grid_id}")
                return
        
        grid_docs = list(collection.find(query).sort('created_at', -1).limit(1000))
        
        # Convert to dict format and filter for corrected/processed grids
        grids = []
        for doc in grid_docs:
            metadata = doc.get('metadata', {})
            # Check both nested and flat structure for backward compatibility
            config_meta = metadata.get('configuration_metadata', {})
            if not config_meta:
                # Fallback: check if correction/processing flags are at top level of metadata
                config_meta = metadata
            
            is_corrected = config_meta.get('correction_applied', False)
            is_processed = config_meta.get('processing_applied', False)
            
            # Apply filters
            if correction_only and not is_corrected:
                continue
            if processing_only and not is_processed:
                continue
            if not correction_only and not processing_only and not is_corrected and not is_processed:
                continue  # Skip grids that are neither corrected nor processed
            
            grid = {
                'grid_id': str(doc['_id']),
                'model_id': doc.get('model_id'),
                'grid_name': doc.get('grid_name'),
                'model_name': doc.get('model_name'),
                'description': doc.get('description', ''),
                'tags': doc.get('tags', []),
                'metadata': metadata,
                'available_signals': doc.get('available_signals', []),
                'signal_references': doc.get('signal_references', {}),
                'voxel_data_reference': doc.get('voxel_data_reference'),
                'created_at': doc.get('created_at'),
                'updated_at': doc.get('updated_at'),
                'is_corrected': is_corrected,
                'is_processed': is_processed,
                'config_metadata': config_meta
            }
            grids.append(grid)
    except Exception as e:
        print(f"❌ Failed to list grids: {e}")
        import traceback
        traceback.print_exc()
        return
    
    if not grids:
        print("\n⚠️ No corrected or processed grids found in MongoDB")
        if correction_only:
            print("   (Filtered for corrected grids only)")
        if processing_only:
            print("   (Filtered for processed grids only)")
        return
    
    # Summary statistics
    print("\n📊 Summary Statistics")
    print("-" * 80)
    print(f"Total Grids: {len(grids)}")
    
    corrected_count = sum(1 for g in grids if g.get('is_corrected'))
    processed_count = sum(1 for g in grids if g.get('is_processed'))
    both_count = sum(1 for g in grids if g.get('is_corrected') and g.get('is_processed'))
    
    print(f"Corrected Grids: {corrected_count}")
    print(f"Processed Grids: {processed_count}")
    print(f"Both Corrected & Processed: {both_count}")
    
    unique_models = set(g.get('model_id') for g in grids)
    print(f"Unique Models: {len(unique_models)}")
    
    # Count by correction type
    correction_types = {}
    for grid in grids:
        if grid.get('is_corrected'):
            corr_type = grid.get('config_metadata', {}).get('correction_type', 'unknown')
            correction_types[corr_type] = correction_types.get(corr_type, 0) + 1
    
    if correction_types:
        print(f"\nCorrection Types:")
        for corr_type, count in sorted(correction_types.items()):
            print(f"  {corr_type}: {count}")
    
    # Count by processing methods
    processing_methods = {}
    for grid in grids:
        if grid.get('is_processed'):
            methods = grid.get('config_metadata', {}).get('processing_methods', [])
            for method in methods:
                processing_methods[method] = processing_methods.get(method, 0) + 1
    
    if processing_methods:
        print(f"\nProcessing Methods:")
        for method, count in sorted(processing_methods.items()):
            print(f"  {method}: {count}")
    
    if summary_only:
        return
    
    # List models only
    if list_models:
        print("\n📋 Unique Model IDs with Corrected/Processed Grids:")
        print("-" * 80)
        for mid in sorted(unique_models):
            count = sum(1 for g in grids if g.get('model_id') == mid)
            print(f"   {mid} ({count} grid(s))")
        return
    
    # Group grids by model
    grids_by_model_dict = {}
    for grid in grids:
        mid = grid.get('model_id', 'unknown')
        if mid not in grids_by_model_dict:
            grids_by_model_dict[mid] = []
        grids_by_model_dict[mid].append(grid)
    
    # Detailed information
    print("\n" + "=" * 80)
    print("📋 Detailed Correction and Processing Information")
    print("=" * 80)
    
    grid_num = 0
    for model_id_key, model_grids in sorted(grids_by_model_dict.items()):
        print(f"\n{'─' * 80}")
        print(f"Model: {model_id_key}")
        print(f"{'─' * 80}")
        
        for grid in sorted(model_grids, key=lambda g: g.get('created_at', datetime.min)):
            grid_num += 1
            print(f"\n📦 Grid {grid_num}/{len(grids)}")
            print("-" * 80)
            
            # Basic information
            print(f"📌 Basic Information:")
            print(f"  Grid ID: {grid.get('grid_id', 'N/A')}")
            print(f"  Grid Name: {grid.get('grid_name', 'N/A')}")
            print(f"  Model ID: {grid.get('model_id', 'N/A')}")
            
            model_name = grid.get('model_name', 'N/A')
            if model_name != 'N/A':
                print(f"  Model Name: {model_name}")
            
            created_at = grid.get('created_at')
            if created_at:
                print(f"  Created: {created_at}")
            
            # Correction/Processing status
            print(f"\n🔧 Correction & Processing Status:")
            if grid.get('is_corrected'):
                print(f"  ✅ Corrected: Yes")
            else:
                print(f"  ❌ Corrected: No")
            
            if grid.get('is_processed'):
                print(f"  ✅ Processed: Yes")
            else:
                print(f"  ❌ Processed: No")
            
            # Correction details
            if grid.get('is_corrected'):
                config_meta = grid.get('config_metadata', {})
                print(f"\n📐 Correction Details:")
                print(f"  Correction Type: {config_meta.get('correction_type', 'N/A')}")
                print(f"  Original Grid ID: {config_meta.get('original_grid_id', 'N/A')}")
                print(f"  Correction Timestamp: {config_meta.get('correction_timestamp', 'N/A')}")
                
                # Scaling parameters
                if 'scaling' in config_meta:
                    scaling = config_meta['scaling']
                    print(f"  Scaling:")
                    print(f"    X: {scaling.get('scale_x', 'N/A')}")
                    print(f"    Y: {scaling.get('scale_y', 'N/A')}")
                    print(f"    Z: {scaling.get('scale_z', 'N/A')}")
                    print(f"    Uniform: {scaling.get('uniform_scale', 'N/A')}")
                
                # Rotation parameters
                if 'rotation' in config_meta:
                    rotation = config_meta['rotation']
                    print(f"  Rotation:")
                    print(f"    X: {rotation.get('rot_x_deg', 'N/A')}°")
                    print(f"    Y: {rotation.get('rot_y_deg', 'N/A')}°")
                    print(f"    Z: {rotation.get('rot_z_deg', 'N/A')}°")
                    if 'rotation_center' in rotation:
                        center = rotation['rotation_center']
                        print(f"    Center: ({center.get('x', 'N/A')}, {center.get('y', 'N/A')}, {center.get('z', 'N/A')})")
                
                # Warping parameters
                if 'warping' in config_meta:
                    warping = config_meta['warping']
                    print(f"  Warping:")
                    print(f"    Type: {warping.get('warp_type', 'N/A')}")
                    print(f"    Degree: {warping.get('warp_degree', 'N/A')}")
                
                # Calibration
                if 'calibration' in config_meta:
                    calib = config_meta['calibration']
                    print(f"  Calibration:")
                    print(f"    ID: {calib.get('calibration_id', 'N/A')}")
                    print(f"    Used: {calib.get('calibration_used', 'N/A')}")
                
                # Correction metrics
                if 'correction_metrics' in config_meta:
                    metrics = config_meta['correction_metrics']
                    print(f"  Correction Metrics:")
                    print(f"    Mean Error: {metrics.get('mean_error', 'N/A')} mm")
                    print(f"    Max Error: {metrics.get('max_error', 'N/A')} mm")
                    print(f"    RMS Error: {metrics.get('rms_error', 'N/A')} mm")
                    print(f"    Score: {metrics.get('score', 'N/A')}")
                
                # Corrected bounding box
                if 'corrected_bbox' in config_meta:
                    bbox = config_meta['corrected_bbox']
                    print(f"  Corrected Bounding Box: {format_bbox(bbox)}")
            
            # Processing details
            if grid.get('is_processed'):
                config_meta = grid.get('config_metadata', {})
                print(f"\n⚙️ Processing Details:")
                print(f"  Processing Timestamp: {config_meta.get('processing_timestamp', 'N/A')}")
                
                # Processed signal(s)
                if 'processed_signal' in config_meta:
                    print(f"  Processed Signal: {config_meta.get('processed_signal', 'N/A')}")
                elif 'processed_signals' in config_meta:
                    signals = config_meta.get('processed_signals', [])
                    print(f"  Processed Signals ({config_meta.get('num_signals_processed', len(signals))}): {', '.join(signals)}")
                
                # Original grid ID
                if 'original_grid_id' in config_meta:
                    print(f"  Original Grid ID: {config_meta.get('original_grid_id', 'N/A')}")
                
                # Outlier detection
                if 'outlier_detection' in config_meta:
                    outlier = config_meta['outlier_detection']
                    if outlier.get('enabled', False):
                        print(f"  Outlier Detection:")
                        print(f"    Method: {outlier.get('method', 'N/A')}")
                        print(f"    Threshold: {outlier.get('threshold', 'N/A')}")
                
                # Smoothing
                if 'smoothing' in config_meta:
                    smooth = config_meta['smoothing']
                    print(f"  Smoothing:")
                    print(f"    Method: {smooth.get('method', 'N/A')}")
                    print(f"    Window Length: {smooth.get('window_length', 'N/A')}")
                    if 'poly_order' in smooth:
                        print(f"    Poly Order: {smooth.get('poly_order', 'N/A')}")
                
                # Noise reduction
                if 'noise_reduction' in config_meta:
                    noise = config_meta['noise_reduction']
                    print(f"  Noise Reduction:")
                    print(f"    Method: {noise.get('method', 'N/A')}")
                    print(f"    Kernel Size: {noise.get('kernel_size', 'N/A')}")
                
                # Processing metrics
                if 'processing_metrics' in config_meta:
                    metrics = config_meta['processing_metrics']
                    print(f"  Processing Metrics:")
                    print(f"    SNR Improvement: {metrics.get('snr_improvement', 'N/A')} dB")
                    print(f"    Noise Reduction: {metrics.get('noise_reduction', 'N/A')}")
                    print(f"    Quality Score: {metrics.get('quality_score', 'N/A')}")
            
            # Available signals
            available_signals = grid.get('available_signals', [])
            print(f"\n📊 Available Signals:")
            if available_signals:
                print(f"  {len(available_signals)} signal(s): {', '.join(sorted(available_signals))}")
            else:
                print(f"  No signals stored")
            
            # GridFS storage for signals
            if verify_gridfs:
                print(f"\n💾 Signal Storage (GridFS):")
                signal_references = grid.get('signal_references', {})
                
                if signal_references:
                    total_size_mb = 0
                    valid_signals = 0
                    invalid_signals = 0
                    
                    for signal_name, file_id in signal_references.items():
                        file_info = check_gridfs_file(mongo_client, file_id, f"signal_{signal_name}")
                        if file_info.get('exists') and file_info.get('valid'):
                            size_mb = file_info.get('data_size_mb', 0)
                            shape = file_info.get('data_shape', 'N/A')
                            dtype = file_info.get('data_dtype', 'N/A')
                            total_size_mb += size_mb
                            valid_signals += 1
                            print(f"    ✅ {signal_name}:")
                            print(f"       File ID: {file_id[:24]}...")
                            print(f"       Size: {size_mb:.2f} MB")
                            print(f"       Shape: {shape}")
                            print(f"       Dtype: {dtype}")
                        else:
                            invalid_signals += 1
                            error = file_info.get('error', 'file not found')
                            print(f"    ❌ {signal_name}: {file_id[:24]}... ({error})")
                    
                    print(f"\n  Storage Summary:")
                    print(f"    Valid Signals: {valid_signals}/{len(signal_references)}")
                    if invalid_signals > 0:
                        print(f"    Invalid/Missing: {invalid_signals}")
                    print(f"    Total Storage: {total_size_mb:.2f} MB")
                else:
                    print(f"  No signal arrays stored in GridFS")
            
            # Tags and description
            tags = grid.get('tags', [])
            if tags:
                print(f"\n🏷️ Tags: {', '.join(tags)}")
            
            description = grid.get('description', '')
            if description:
                print(f"\n📝 Description: {description}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Check corrected and processed data stored in MongoDB',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--model-id',
        type=str,
        help='Filter by model ID'
    )
    parser.add_argument(
        '--grid-id',
        type=str,
        help='Filter by grid ID'
    )
    parser.add_argument(
        '--correction-only',
        action='store_true',
        help='Only show corrected grids'
    )
    parser.add_argument(
        '--processing-only',
        action='store_true',
        help='Only show processed grids'
    )
    parser.add_argument(
        '--summary-only',
        action='store_true',
        help='Only show summary statistics'
    )
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='Only list unique model IDs with corrected/processed grids'
    )
    parser.add_argument(
        '--no-verify-gridfs',
        action='store_true',
        help='Skip GridFS file verification'
    )
    
    args = parser.parse_args()
    
    check_corrected_processed_data(
        model_id=args.model_id,
        grid_id=args.grid_id,
        correction_only=args.correction_only,
        processing_only=args.processing_only,
        summary_only=args.summary_only,
        list_models=args.list_models,
        verify_gridfs=not args.no_verify_gridfs
    )


if __name__ == '__main__':
    main()

