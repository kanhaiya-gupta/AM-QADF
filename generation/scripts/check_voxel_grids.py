"""
Check Voxel Grids in MongoDB

This script checks voxel grids stored in MongoDB, including
grid properties, metadata, available signals, and GridFS storage.
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


def check_gridfs_file(mongo_client: MongoDBClient, file_id: str, file_type: str) -> Dict[str, Any]:
    """Check if a GridFS file exists and get its metadata."""
    try:
        file_data = mongo_client.get_file(file_id, bucket_name='voxel_grid_data')
        if file_data:
            # Try to decompress and load to verify it's valid
            import gzip
            import io
            try:
                decompressed = gzip.decompress(file_data)
                data = np.load(io.BytesIO(decompressed), allow_pickle=True)
                return {
                    'exists': True,
                    'valid': True,
                    'size_bytes': len(file_data),
                    'data_shape': data.shape if hasattr(data, 'shape') else 'N/A',
                    'data_dtype': str(data.dtype) if hasattr(data, 'dtype') else 'N/A'
                }
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


def check_voxel_grids(
    model_id: Optional[str] = None,
    grid_id: Optional[str] = None,
    summary_only: bool = False,
    list_models: bool = False,
    verify_gridfs: bool = True
):
    """
    Check voxel grids stored in MongoDB.
    
    Args:
        model_id: Optional model ID to filter by
        grid_id: Optional grid ID to filter by
        summary_only: If True, only show summary statistics
        list_models: If True, only list unique model IDs
        verify_gridfs: If True, verify GridFS files exist
    """
    print("=" * 80)
    print("🔍 Checking Voxel Grids in MongoDB")
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
    
    # List grids - we need to get full documents to access signal_references
    try:
        # Get full documents including signal_references and voxel_data_reference
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
        
        # Convert to dict format similar to list_grids
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
    
    # List models only
    if list_models:
        print("\n📋 Unique Model IDs with Voxel Grids:")
        print("-" * 80)
        model_ids = set(g.get('model_id') for g in grids)
        for mid in sorted(model_ids):
            count = sum(1 for g in grids if g.get('model_id') == mid)
            print(f"   {mid} ({count} grid(s))")
        return
    
    # Summary statistics
    print("\n📊 Summary Statistics")
    print("-" * 80)
    print(f"Total Grids: {len(grids)}")
    
    unique_models = set(g.get('model_id') for g in grids)
    print(f"Unique Models: {len(unique_models)}")
    
    grids_with_signals = sum(1 for g in grids if g.get('available_signals'))
    print(f"Grids with Signals: {grids_with_signals}/{len(grids)}")
    
    # Count by grid type
    grid_types = {}
    for grid in grids:
        metadata = grid.get('metadata', {})
        grid_type = metadata.get('grid_type', 'unknown')
        grid_types[grid_type] = grid_types.get(grid_type, 0) + 1
    
    if grid_types:
        print(f"\nGrids by Type:")
        for gtype, count in sorted(grid_types.items()):
            print(f"  {gtype}: {count}")
    
    # Count by model
    grids_by_model = {}
    for grid in grids:
        mid = grid.get('model_id', 'unknown')
        grids_by_model[mid] = grids_by_model.get(mid, 0) + 1
    
    if grids_by_model:
        print(f"\nGrids per Model:")
        for mid, count in sorted(grids_by_model.items()):
            print(f"  {mid[:36]}...: {count} grid(s)")
    
    if summary_only:
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
    print("📋 Detailed Grid Information")
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
            
            # Model name from metadata
            metadata = grid.get('metadata', {})
            model_name = metadata.get('model_name', 'N/A')
            if model_name != 'N/A':
                print(f"  Model Name: {model_name}")
            
            created_at = grid.get('created_at')
            if created_at:
                if isinstance(created_at, str):
                    print(f"  Created: {created_at}")
                else:
                    print(f"  Created: {created_at}")
            
            updated_at = grid.get('updated_at')
            if updated_at and updated_at != created_at:
                if isinstance(updated_at, str):
                    print(f"  Updated: {updated_at}")
                else:
                    print(f"  Updated: {updated_at}")
            
            # Grid properties
            print(f"\n📐 Grid Properties:")
            resolution = metadata.get('resolution')
            if resolution:
                print(f"  Resolution: {resolution} mm")
            
            bbox = metadata.get('bounding_box') or metadata.get('bbox')
            if bbox:
                print(f"  Bounding Box: {format_bbox(bbox)}")
            
            dimensions = metadata.get('dimensions') or metadata.get('grid_dimensions')
            if dimensions:
                print(f"  Dimensions: {format_dimensions(dimensions)}")
            
            grid_type = metadata.get('grid_type', 'uniform')
            print(f"  Grid Type: {grid_type}")
            
            coord_system = metadata.get('coordinate_system', 'build_platform')
            print(f"  Coordinate System: {coord_system}")
            
            # Additional grid properties from metadata
            bbox_min = metadata.get('bbox_min')
            bbox_max = metadata.get('bbox_max')
            if bbox_min and bbox_max:
                print(f"  BBox (min/max): [{bbox_min[0]:.2f}, {bbox_min[1]:.2f}, {bbox_min[2]:.2f}] to [{bbox_max[0]:.2f}, {bbox_max[1]:.2f}, {bbox_max[2]:.2f}]")
            
            dims = metadata.get('dims')
            if dims:
                print(f"  Dimensions (dims): {format_dimensions(dims)}")
            
            aggregation = metadata.get('aggregation')
            if aggregation:
                print(f"  Aggregation Method: {aggregation}")
            
            sparse_storage = metadata.get('sparse_storage')
            if sparse_storage is not None:
                print(f"  Sparse Storage: {sparse_storage}")
            
            compression = metadata.get('compression')
            if compression is not None:
                print(f"  Compression: {compression}")
            
            # Configuration metadata (user-selected settings)
            config_meta = metadata.get('configuration_metadata', {})
            if config_meta:
                print(f"\n⚙️ Configuration Metadata:")
                # Show all configuration fields
                if 'bbox_mode' in config_meta:
                    print(f"  BBox Source: {config_meta.get('bbox_mode', 'N/A')}")
                if 'alignment_id' in config_meta:
                    print(f"  Alignment ID: {config_meta.get('alignment_id', 'N/A')}")
                if 'created_from_alignment' in config_meta:
                    print(f"  Created from Alignment: {config_meta.get('created_from_alignment', False)}")
                if 'grid_type' in config_meta:
                    print(f"  Grid Type (config): {config_meta.get('grid_type', 'N/A')}")
                if 'resolution_mode' in config_meta:
                    print(f"  Resolution Mode: {config_meta.get('resolution_mode', 'N/A')}")
                if 'uniform_resolution' in config_meta:
                    print(f"  Uniform Resolution: {config_meta.get('uniform_resolution', 'N/A')} mm")
                if 'x_resolution' in config_meta or 'y_resolution' in config_meta or 'z_resolution' in config_meta:
                    res_str = []
                    if 'x_resolution' in config_meta:
                        res_str.append(f"X={config_meta['x_resolution']}")
                    if 'y_resolution' in config_meta:
                        res_str.append(f"Y={config_meta['y_resolution']}")
                    if 'z_resolution' in config_meta:
                        res_str.append(f"Z={config_meta['z_resolution']}")
                    print(f"  Per-Axis Resolution: {', '.join(res_str)} mm")
                if 'aggregation_method' in config_meta:
                    print(f"  Aggregation Method (config): {config_meta.get('aggregation_method', 'N/A')}")
                if 'adaptive_strategy' in config_meta:
                    print(f"  Adaptive Strategy: {config_meta.get('adaptive_strategy', 'N/A')}")
                if 'coordinate_system' in config_meta:
                    coord_sys = config_meta.get('coordinate_system')
                    if isinstance(coord_sys, dict):
                        print(f"  Coordinate System (config): {coord_sys.get('type', 'N/A')}")
                        if 'origin' in coord_sys:
                            print(f"    Origin: {coord_sys['origin']}")
                        if 'rotation' in coord_sys:
                            print(f"    Rotation: {coord_sys['rotation']}")
                    else:
                        print(f"  Coordinate System (config): {coord_sys}")
                
                # Show any other configuration fields not already displayed
                shown_keys = {
                    'bbox_mode', 'alignment_id', 'created_from_alignment', 'grid_type',
                    'resolution_mode', 'uniform_resolution', 'x_resolution', 'y_resolution',
                    'z_resolution', 'aggregation_method', 'adaptive_strategy', 'coordinate_system'
                }
                other_keys = set(config_meta.keys()) - shown_keys
                if other_keys:
                    print(f"  Other Settings:")
                    for key in sorted(other_keys):
                        value = config_meta[key]
                        if value is not None:
                            print(f"    {key}: {value}")
            
            # Full metadata dump (all other fields)
            print(f"\n📋 Full Metadata:")
            # Exclude already-displayed fields
            excluded_keys = {
                'resolution', 'bounding_box', 'bbox', 'dimensions', 'grid_dimensions',
                'grid_type', 'coordinate_system', 'configuration_metadata', 'model_name',
                'bbox_min', 'bbox_max', 'dims', 'aggregation', 'sparse_storage', 'compression'
            }
            other_metadata = {k: v for k, v in metadata.items() if k not in excluded_keys}
            if other_metadata:
                for key, value in sorted(other_metadata.items()):
                    if value is not None:
                        # Format value for display
                        if isinstance(value, (list, tuple)) and len(value) > 5:
                            print(f"  {key}: {value[:3]}... ({len(value)} items)")
                        elif isinstance(value, dict) and len(value) > 3:
                            print(f"  {key}: {dict(list(value.items())[:2])}... ({len(value)} keys)")
                        else:
                            print(f"  {key}: {value}")
            else:
                print(f"  (No additional metadata fields)")
            
            # Available signals
            available_signals = grid.get('available_signals', [])
            print(f"\n📊 Available Signals:")
            if available_signals:
                print(f"  {len(available_signals)} signal(s): {', '.join(available_signals)}")
            else:
                print(f"  No signals stored")
            
            # GridFS references (stored at document level, not in metadata)
            if verify_gridfs:
                print(f"\n💾 GridFS Storage:")
                # Check document-level fields (correct location)
                signal_references = grid.get('signal_references', {})
                # Also check metadata as fallback (for backwards compatibility)
                if not signal_references:
                    signal_references = metadata.get('signal_arrays_gridfs_ids', {})
                
                voxel_data_ref = grid.get('voxel_data_reference')
                # Also check metadata as fallback
                if not voxel_data_ref:
                    voxel_data_ref = metadata.get('voxel_data_gridfs_id')
                
                if signal_references:
                    print(f"  Signal Arrays: {len(signal_references)} signal(s) in GridFS")
                    for signal_name, file_id in signal_references.items():
                        file_info = check_gridfs_file(mongo_client, file_id, f"signal_{signal_name}")
                        if file_info.get('exists'):
                            size_mb = file_info.get('size_bytes', 0) / (1024 * 1024)
                            shape = file_info.get('data_shape', 'N/A')
                            print(f"    ✅ {signal_name}: {file_id[:24]}... ({size_mb:.2f} MB, shape: {shape})")
                        else:
                            print(f"    ❌ {signal_name}: {file_id[:24]}... (file not found)")
                else:
                    print(f"  Signal Arrays: None (no signals mapped to this grid)")
                
                if voxel_data_ref:
                    file_info = check_gridfs_file(mongo_client, voxel_data_ref, "voxel_data")
                    if file_info.get('exists'):
                        size_mb = file_info.get('size_bytes', 0) / (1024 * 1024)
                        shape = file_info.get('data_shape', 'N/A')
                        print(f"  ✅ Voxel Data: {voxel_data_ref[:24]}... ({size_mb:.2f} MB, shape: {shape})")
                    else:
                        print(f"  ❌ Voxel Data: {voxel_data_ref[:24]}... (file not found)")
                else:
                    print(f"  Voxel Data: None")
            
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
        description='Check voxel grids stored in MongoDB',
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
        help='Only list unique model IDs'
    )
    parser.add_argument(
        '--no-verify-gridfs',
        action='store_true',
        help='Skip GridFS file verification'
    )
    
    args = parser.parse_args()
    
    check_voxel_grids(
        model_id=args.model_id,
        grid_id=args.grid_id,
        summary_only=args.summary_only,
        list_models=args.list_models,
        verify_gridfs=not args.no_verify_gridfs
    )


if __name__ == '__main__':
    main()

