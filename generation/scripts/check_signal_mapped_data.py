"""
Check Signal Mapped Data in MongoDB

This script checks voxel grids that have signals mapped to them,
including signal types, GridFS storage, and mapping metadata.
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


def format_dimensions(dims: Any) -> str:
    """Format grid dimensions for display."""
    if dims is None:
        return "N/A"
    if isinstance(dims, (list, tuple)):
        if len(dims) == 3:
            return f"{dims[0]} × {dims[1]} × {dims[2]} = {dims[0] * dims[1] * dims[2]:,} voxels"
    return str(dims)


def check_gridfs_file(mongo_client: MongoDBClient, file_id: str, file_type: str, bucket_name: str = None) -> Dict[str, Any]:
    """Check if a GridFS file exists and get its metadata."""
    try:
        # VoxelGridStorage defines gridfs_bucket='voxel_grid_data' but actually uses MongoDBClient.store_file()
        # which stores in the default GridFS bucket ('fs'), not the custom bucket
        # So we need to check the default bucket first
        from gridfs import GridFS
        from bson import ObjectId
        
        # Try default bucket first (where MongoDBClient actually stores files)
        buckets_to_try = ['fs']
        if bucket_name and bucket_name != 'fs':
            buckets_to_try.append(bucket_name)
        
        file_data = None
        found_bucket = None
        gridfs_metadata = {}
        for bucket in buckets_to_try:
            try:
                fs = GridFS(mongo_client.database, collection=bucket)
                
                # Try to get file by ObjectId first
                if isinstance(file_id, str):
                    try:
                        file_id_obj = ObjectId(file_id)
                    except Exception:
                        # Try to find by filename if ObjectId conversion fails
                        grid_file = fs.find_one({'filename': file_id})
                        if grid_file:
                            file_data = grid_file.read()
                            # Get GridFS metadata
                            gridfs_metadata = grid_file.metadata if hasattr(grid_file, 'metadata') else {}
                            found_bucket = bucket
                            break
                    else:
                        try:
                            grid_file = fs.get(file_id_obj)
                            file_data = grid_file.read()
                            # Get GridFS metadata
                            gridfs_metadata = grid_file.metadata if hasattr(grid_file, 'metadata') else {}
                            found_bucket = bucket
                            break
                        except Exception:
                            continue
                else:
                    try:
                        grid_file = fs.get(file_id)
                        file_data = grid_file.read()
                        # Get GridFS metadata
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
                    # npz format (compressed numpy archive)
                    if 'format' in data.files:
                        # Sparse format: extract dims from data
                        format_val = data['format']
                        if format_val == 'sparse':
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


def check_signal_mapped_data(
    model_id: Optional[str] = None,
    grid_id: Optional[str] = None,
    summary_only: bool = False,
    list_models: bool = False,
    verify_gridfs: bool = True
):
    """
    Check signal mapped data stored in MongoDB.
    
    Args:
        model_id: Optional model ID to filter by
        grid_id: Optional grid ID to filter by
        summary_only: If True, only show summary statistics
        list_models: If True, only list unique model IDs
        verify_gridfs: If True, verify GridFS files exist
    """
    print("=" * 80)
    print("🔍 Checking Signal Mapped Data in MongoDB")
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
    
    # Get full documents including signal_references
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
        
        # Convert to dict format
        grids = []
        for doc in grid_docs:
            grid = {
                'grid_id': str(doc['_id']),
                'model_id': doc.get('model_id'),
                'grid_name': doc.get('grid_name'),
                'model_name': doc.get('model_name'),
                'description': doc.get('description', ''),
                'tags': doc.get('tags', []),
                'metadata': doc.get('metadata', {}),
                'available_signals': doc.get('available_signals', []),
                'signal_references': doc.get('signal_references', {}),
                'voxel_data_reference': doc.get('voxel_data_reference'),
                'created_at': doc.get('created_at'),
                'updated_at': doc.get('updated_at')
            }
            grids.append(grid)
    except Exception as e:
        print(f"❌ Failed to list grids: {e}")
        import traceback
        traceback.print_exc()
        return
    
    if not grids:
        print("\n⚠️ No voxel grids found in MongoDB")
        return
    
    # Filter for grids with signals (mapped grids)
    mapped_grids = [g for g in grids if g.get('available_signals') and len(g.get('available_signals', [])) > 0]
    
    if not mapped_grids:
        print("\n⚠️ No grids with mapped signals found in MongoDB")
        print(f"   (Found {len(grids)} total grid(s), but none have signals)")
        return
    
    # Summary statistics
    print("\n📊 Summary Statistics")
    print("-" * 80)
    print(f"Total Grids: {len(grids)}")
    print(f"Grids with Mapped Signals: {len(mapped_grids)}/{len(grids)}")
    
    unique_models = set(g.get('model_id') for g in mapped_grids)
    print(f"Unique Models with Mapped Signals: {len(unique_models)}")
    
    # Count signals by type
    all_signals = set()
    for grid in mapped_grids:
        all_signals.update(grid.get('available_signals', []))
    
    print(f"Unique Signal Types: {len(all_signals)}")
    if all_signals:
        print(f"  Signal Types: {', '.join(sorted(all_signals))}")
    
    # Count by mapping method
    mapping_methods = {}
    for grid in mapped_grids:
        metadata = grid.get('metadata', {})
        config_meta = metadata.get('configuration_metadata', {})
        method = config_meta.get('mapping_method', 'unknown')
        mapping_methods[method] = mapping_methods.get(method, 0) + 1
    
    if mapping_methods:
        print(f"\nGrids by Mapping Method:")
        for method, count in sorted(mapping_methods.items()):
            print(f"  {method}: {count}")
    
    # Count by model
    grids_by_model = {}
    for grid in mapped_grids:
        mid = grid.get('model_id', 'unknown')
        grids_by_model[mid] = grids_by_model.get(mid, 0) + 1
    
    if grids_by_model:
        print(f"\nMapped Grids per Model:")
        for mid, count in sorted(grids_by_model.items()):
            print(f"  {mid[:36]}...: {count} grid(s)")
    
    if summary_only:
        return
    
    # List models only
    if list_models:
        print("\n📋 Unique Model IDs with Mapped Signals:")
        print("-" * 80)
        for mid in sorted(unique_models):
            count = sum(1 for g in mapped_grids if g.get('model_id') == mid)
            print(f"   {mid} ({count} mapped grid(s))")
        return
    
    # Group grids by model
    grids_by_model_dict = {}
    for grid in mapped_grids:
        mid = grid.get('model_id', 'unknown')
        if mid not in grids_by_model_dict:
            grids_by_model_dict[mid] = []
        grids_by_model_dict[mid].append(grid)
    
    # Detailed information
    print("\n" + "=" * 80)
    print("📋 Detailed Signal Mapping Information")
    print("=" * 80)
    
    grid_num = 0
    for model_id_key, model_grids in sorted(grids_by_model_dict.items()):
        print(f"\n{'─' * 80}")
        print(f"Model: {model_id_key}")
        print(f"{'─' * 80}")
        
        for grid in sorted(model_grids, key=lambda g: g.get('created_at', datetime.min)):
            grid_num += 1
            print(f"\n📦 Mapped Grid {grid_num}/{len(mapped_grids)}")
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
                if isinstance(created_at, str):
                    print(f"  Created: {created_at}")
                else:
                    print(f"  Created: {created_at}")
            
            # Grid properties
            metadata = grid.get('metadata', {})
            print(f"\n📐 Grid Properties:")
            resolution = metadata.get('resolution')
            if resolution:
                print(f"  Resolution: {resolution} mm")
            
            bbox = metadata.get('bounding_box') or metadata.get('bbox')
            if bbox:
                print(f"  Bounding Box: {format_bbox(bbox)}")
            
            dimensions = metadata.get('dimensions') or metadata.get('grid_dimensions') or metadata.get('dims')
            if dimensions:
                print(f"  Dimensions: {format_dimensions(dimensions)}")
            
            grid_type = metadata.get('grid_type', 'uniform')
            print(f"  Grid Type: {grid_type}")
            
            # Mapping information
            config_meta = metadata.get('configuration_metadata', {})
            if config_meta:
                print(f"\n🗺️ Signal Mapping Configuration:")
                mapping_method = config_meta.get('mapping_method')
                if mapping_method:
                    print(f"  Mapping Method: {mapping_method}")
                
                mapped_signals = config_meta.get('mapped_signals', [])
                if mapped_signals:
                    print(f"  Mapped Signals: {', '.join(mapped_signals)}")
                
                grid_resolution = config_meta.get('grid_resolution')
                if grid_resolution:
                    print(f"  Grid Resolution: {grid_resolution} mm")
                
                # Method-specific parameters
                if mapping_method == 'linear':
                    if 'linear_k_neighbors' in config_meta:
                        print(f"  K Neighbors: {config_meta.get('linear_k_neighbors')}")
                    if 'linear_radius' in config_meta:
                        print(f"  Radius: {config_meta.get('linear_radius')} mm")
                elif mapping_method == 'idw':
                    if 'idw_power' in config_meta:
                        print(f"  Power: {config_meta.get('idw_power')}")
                    if 'idw_k_neighbors' in config_meta:
                        print(f"  K Neighbors: {config_meta.get('idw_k_neighbors')}")
                    if 'idw_radius' in config_meta:
                        print(f"  Radius: {config_meta.get('idw_radius')} mm")
                elif mapping_method == 'gaussian_kde':
                    if 'kde_bandwidth' in config_meta:
                        print(f"  Bandwidth: {config_meta.get('kde_bandwidth')}")
                    if 'kde_adaptive' in config_meta:
                        print(f"  Adaptive: {config_meta.get('kde_adaptive')}")
            
            # Available signals
            available_signals = grid.get('available_signals', [])
            print(f"\n📊 Mapped Signals:")
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
                            bucket = file_info.get('bucket', 'fs')
                            metadata = file_info.get('gridfs_metadata', {})
                            total_size_mb += size_mb
                            valid_signals += 1
                            print(f"    ✅ {signal_name}:")
                            print(f"       File ID: {file_id[:24]}...")
                            print(f"       Bucket: {bucket}")
                            if metadata:
                                stored_grid_id = metadata.get('grid_id', 'N/A')
                                stored_signal = metadata.get('signal_name', 'N/A')
                                data_type = metadata.get('data_type', 'N/A')
                                format_type = metadata.get('format', 'N/A')
                                print(f"       Grid ID (from metadata): {stored_grid_id}")
                                print(f"       Signal Name (from metadata): {stored_signal}")
                                print(f"       Data Type: {data_type}")
                                print(f"       Format: {format_type}")
                                # Verify grid_id matches
                                if stored_grid_id != 'N/A' and stored_grid_id != grid.get('grid_id'):
                                    print(f"       ⚠️ Warning: Grid ID mismatch! Expected {grid.get('grid_id')}, found {stored_grid_id}")
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
                
                # Voxel data reference
                voxel_data_ref = grid.get('voxel_data_reference')
                if voxel_data_ref:
                    file_info = check_gridfs_file(mongo_client, voxel_data_ref, "voxel_data")
                    if file_info.get('exists') and file_info.get('valid'):
                        size_mb = file_info.get('data_size_mb', 0)
                        shape = file_info.get('data_shape', 'N/A')
                        bucket = file_info.get('bucket', 'fs')
                        print(f"\n  ✅ Voxel Data Structure:")
                        print(f"     File ID: {voxel_data_ref[:24]}...")
                        print(f"     Bucket: {bucket}")
                        print(f"     Size: {size_mb:.2f} MB")
                        print(f"     Shape: {shape}")
                    else:
                        error = file_info.get('error', 'file not found')
                        print(f"\n  ❌ Voxel Data: {voxel_data_ref[:24]}... ({error})")
                else:
                    print(f"\n  Voxel Data Structure: None")
            
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
        description='Check signal mapped data stored in MongoDB',
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
        '--summary-only',
        action='store_true',
        help='Only show summary statistics'
    )
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='Only list unique model IDs with mapped signals'
    )
    parser.add_argument(
        '--no-verify-gridfs',
        action='store_true',
        help='Skip GridFS file verification'
    )
    
    args = parser.parse_args()
    
    check_signal_mapped_data(
        model_id=args.model_id,
        grid_id=args.grid_id,
        summary_only=args.summary_only,
        list_models=args.list_models,
        verify_gridfs=not args.no_verify_gridfs
    )


if __name__ == '__main__':
    main()

