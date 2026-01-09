"""
Check Stored Data in MongoDB

This script checks what data has been stored in MongoDB collections
for a specific model.
"""

import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

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
    
    # Create a get_mongodb_config function for compatibility
    def get_mongodb_config():
        """Get MongoDB config from environment."""
        return MongoDBConfig.from_env()
except Exception as e:
    print(f"❌ Error loading MongoDB client: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


def check_stored_data(model_id: Optional[str] = None):
    """
    Check what data is stored in MongoDB for a model.
    
    Args:
        model_id: Optional model ID to filter (if None, shows all models)
    """
    print("=" * 80)
    print("🔍 Checking Stored Data in MongoDB")
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
        print(f"✅ Connected to MongoDB: {config.database}\n")
    except Exception as e:
        print(f"❌ Failed to connect to MongoDB: {e}")
        return
    
    # Collections to check
    collections = [
        'stl_models',
        'hatching_layers',
        'laser_parameters',
        'ct_scan_data',
        'ispm_monitoring_data',
        'voxel_grids'
    ]
    
    # First, show all model_ids across all collections
    print("\n📋 Model IDs in Database:")
    print("-" * 80)
    all_model_ids = set()
    for collection_name in collections:
        try:
            collection = mongo_client.get_collection(collection_name)
            # Get distinct model_ids
            model_ids = collection.distinct('model_id')
            all_model_ids.update(model_ids)
            if model_ids:
                print(f"   {collection_name}: {len(model_ids)} unique model_id(s)")
                for mid in model_ids[:5]:  # Show first 5
                    count = collection.count_documents({'model_id': mid})
                    print(f"      - {mid[:36]}... ({count} docs)")
                if len(model_ids) > 5:
                    print(f"      ... and {len(model_ids) - 5} more")
        except Exception as e:
            print(f"   {collection_name}: Error - {e}")
    
    print(f"\n   Total unique model_ids: {len(all_model_ids)}")
    if all_model_ids:
        print(f"   Model IDs: {list(all_model_ids)[:3]}...")
    
    # Show which models have complete data
    print(f"\n📊 Data Completeness Summary:")
    print("-" * 80)
    stl_collection = mongo_client.get_collection('stl_models')
    voxel_grids_collection = mongo_client.get_collection('voxel_grids')
    
    for mid in sorted(all_model_ids):
        stl_doc = stl_collection.find_one({'model_id': mid})
        model_name = stl_doc.get('model_name', stl_doc.get('filename', 'Unknown')) if stl_doc else 'Unknown'
        
        has_stl = stl_collection.count_documents({'model_id': mid}) > 0
        has_hatching = mongo_client.get_collection('hatching_layers').count_documents({'model_id': mid}) > 0
        has_laser = mongo_client.get_collection('laser_parameters').count_documents({'model_id': mid}) > 0
        has_ct = mongo_client.get_collection('ct_scan_data').count_documents({'model_id': mid}) > 0
        has_ispm = mongo_client.get_collection('ispm_monitoring_data').count_documents({'model_id': mid}) > 0
        has_voxel_grids = voxel_grids_collection.count_documents({'model_id': mid}) > 0
        
        # Count voxel grids for this model
        grid_count = voxel_grids_collection.count_documents({'model_id': mid})
        
        complete = has_stl and has_hatching and has_laser and has_ct and has_ispm
        status = "✅ Complete" if complete else "⚠️  Incomplete"
        
        print(f"   {mid[:36]}... ({model_name[:30]}...)")
        print(f"      Status: {status}")
        print(f"      STL: {'✓' if has_stl else '✗'}, Hatching: {'✓' if has_hatching else '✗'}, "
              f"Laser: {'✓' if has_laser else '✗'}, CT: {'✓' if has_ct else '✗'}, ISPM: {'✓' if has_ispm else '✗'}, "
              f"Voxel Grids: {'✓' if has_voxel_grids else '✗'} ({grid_count} grid(s))")
    
    filter_query = {'model_id': model_id} if model_id else {}
    
    for collection_name in collections:
        print(f"\n📦 Collection: {collection_name}")
        print("-" * 80)
        
        try:
            # Get collection and count documents using pymongo directly
            collection = mongo_client.get_collection(collection_name)
            total_count = collection.count_documents({})
            filtered_count = collection.count_documents(filter_query)
            print(f"   Total documents in collection: {total_count}")
            if model_id:
                print(f"   Documents matching model_id '{model_id[:36]}...': {filtered_count}")
            
            if filtered_count == 0:
                if model_id:
                    print(f"   ⚠️  No documents found for model_id: {model_id[:36]}...")
                    # Show what model_ids exist
                    existing_ids = collection.distinct('model_id')
                    if existing_ids:
                        print(f"   Available model_ids in this collection:")
                        for mid in existing_ids[:5]:
                            count = collection.count_documents({'model_id': mid})
                            print(f"      - {mid} ({count} docs)")
                else:
                    print("   ⚠️  No documents found")
                continue
            
            # Show breakdown by model_id if not filtering
            if not model_id:
                print(f"\n   📊 Breakdown by model_id:")
                pipeline = [
                    {"$group": {"_id": "$model_id", "count": {"$sum": 1}}},
                    {"$sort": {"count": -1}}
                ]
                for result in collection.aggregate(pipeline):
                    mid = result['_id']
                    count = result['count']
                    print(f"      {mid[:36]}...: {count} documents")
            
            # Get a sample document
            sample = collection.find_one(filter_query)
            if sample:
                print(f"\n   📄 Sample Document Structure:")
                
                # Remove _id for cleaner output
                sample_clean = {k: v for k, v in sample.items() if k != '_id'}
                
                # Show key fields
                for key, value in sample_clean.items():
                    if isinstance(value, dict):
                        print(f"      {key}:")
                        # For voxel_grids metadata, show all fields; for others, show first 5
                        max_fields = len(value) if collection_name == 'voxel_grids' and key == 'metadata' else 5
                        for sub_key, sub_value in list(value.items())[:max_fields]:
                            if isinstance(sub_value, (list, dict)) and len(str(sub_value)) > 100:
                                if isinstance(sub_value, dict):
                                    print(f"         {sub_key}: dict (size: {len(sub_value)})")
                                    # Show first 3 items of nested dict
                                    for nested_key, nested_val in list(sub_value.items())[:3]:
                                        print(f"            {nested_key}: {nested_val}")
                                    if len(sub_value) > 3:
                                        print(f"            ... and {len(sub_value) - 3} more")
                                else:
                                    print(f"         {sub_key}: {type(sub_value).__name__} (size: {len(sub_value) if hasattr(sub_value, '__len__') else 'N/A'})")
                            else:
                                print(f"         {sub_key}: {sub_value}")
                        if len(value) > max_fields and not (collection_name == 'voxel_grids' and key == 'metadata'):
                            print(f"         ... and {len(value) - max_fields} more fields")
                    elif isinstance(value, list):
                        if len(value) > 0 and isinstance(value[0], dict):
                            print(f"      {key}: List[{len(value)} items]")
                            if len(value) > 0:
                                print(f"         First item keys: {list(value[0].keys())[:5]}")
                        elif len(str(value)) > 100:
                            print(f"      {key}: List[{len(value)} items] (truncated)")
                        else:
                            print(f"      {key}: {value}")
                    else:
                        value_str = str(value)
                        if len(value_str) > 100:
                            value_str = value_str[:100] + "..."
                        print(f"      {key}: {value_str}")
                
                # Special handling for voxel_grids collection
                if collection_name == 'voxel_grids':
                    print(f"\n   📊 Grid Information:")
                    if 'grid_name' in sample_clean:
                        print(f"      Grid Name: {sample_clean['grid_name']}")
                    if 'model_name' in sample_clean:
                        print(f"      Model Name: {sample_clean['model_name']}")
                    if 'metadata' in sample_clean:
                        meta = sample_clean['metadata']
                        
                        # Basic grid properties
                        if 'grid_type' in meta:
                            print(f"      Grid Type: {meta['grid_type']}")
                        if 'resolution' in meta:
                            print(f"      Resolution: {meta['resolution']} mm")
                        if 'resolution_mode' in meta:
                            print(f"      Resolution Mode: {meta['resolution_mode']}")
                        if 'bbox_min' in meta and 'bbox_max' in meta:
                            print(f"      Bounding Box: {meta['bbox_min']} to {meta['bbox_max']}")
                        if 'bbox_mode' in meta:
                            print(f"      BBox Source: {meta['bbox_mode']}")
                        
                        # Coordinate system
                        if 'coordinate_system' in meta:
                            coord = meta['coordinate_system']
                            if isinstance(coord, dict):
                                print(f"      Coordinate System: {coord.get('type', 'N/A')}")
                                if 'origin' in coord:
                                    print(f"         Origin: {coord['origin']}")
                                if 'rotation' in coord:
                                    print(f"         Rotation: {coord['rotation']}")
                        
                        # Grid properties
                        if 'aggregation_method' in meta:
                            print(f"      Aggregation: {meta['aggregation_method']}")
                        if 'sparse_storage' in meta:
                            print(f"      Sparse Storage: {meta['sparse_storage']}")
                        if 'compression' in meta:
                            print(f"      Compression: {meta['compression']}")
                        
                        # Per-axis resolutions (if applicable)
                        if 'x_resolution' in meta or 'y_resolution' in meta or 'z_resolution' in meta:
                            print(f"      Per-Axis Resolution:")
                            if 'x_resolution' in meta:
                                print(f"         X: {meta['x_resolution']} mm")
                            if 'y_resolution' in meta:
                                print(f"         Y: {meta['y_resolution']} mm")
                            if 'z_resolution' in meta:
                                print(f"         Z: {meta['z_resolution']} mm")
                        
                        # Adaptive settings
                        if 'adaptive_strategy' in meta:
                            print(f"      Adaptive Strategy: {meta['adaptive_strategy']}")
                    
                    if 'available_signals' in sample_clean:
                        signals = sample_clean['available_signals']
                        print(f"      Available Signals: {len(signals)} signal(s)")
                        if signals:
                            print(f"         Signals: {', '.join(signals[:5])}")
                            if len(signals) > 5:
                                print(f"         ... and {len(signals) - 5} more")
                    
                    # Check for duplicates in this collection
                    if not model_id:  # Only check duplicates when showing all grids
                        from collections import defaultdict
                        all_grids = list(collection.find({}))
                        
                        # Group by model_id and grid_name
                        exact_dups = defaultdict(list)
                        for g in all_grids:
                            key = f"{g.get('model_id')}::{g.get('grid_name')}"
                            exact_dups[key].append(g)
                        
                        # Group by model_id and configuration
                        config_dups = defaultdict(list)
                        for g in all_grids:
                            model_id_g = g.get('model_id')
                            meta = g.get('metadata', {})
                            bbox_min = tuple(meta.get('bbox_min', []))
                            bbox_max = tuple(meta.get('bbox_max', []))
                            resolution = meta.get('resolution')
                            key = f"{model_id_g}::bbox_{bbox_min}_{bbox_max}::res_{resolution}"
                            config_dups[key].append(g)
                        
                        # Find actual duplicates
                        exact_duplicate_groups = {k: v for k, v in exact_dups.items() if len(v) > 1}
                        config_duplicate_groups = {k: v for k, v in config_dups.items() 
                                                   if len(v) > 1 and len(set(g.get('grid_name') for g in v)) > 1}
                        
                        if exact_duplicate_groups or config_duplicate_groups:
                            print(f"\n   ⚠️  Duplicate Grids Detected:")
                            if exact_duplicate_groups:
                                print(f"      Exact duplicates (same model_id + grid_name): {len(exact_duplicate_groups)} group(s)")
                                for key, grids in list(exact_duplicate_groups.items())[:3]:
                                    model_id_dup, grid_name = key.split('::', 1)
                                    print(f"         - {grid_name[:30]}... ({len(grids)} copies) for model {model_id_dup[:8]}...")
                            if config_duplicate_groups:
                                print(f"      Config duplicates (same model_id + config): {len(config_duplicate_groups)} group(s)")
                                for key, grids in list(config_duplicate_groups.items())[:3]:
                                    parts = key.split('::')
                                    model_id_dup = parts[0]
                                    print(f"         - {len(grids)} grids with same config for model {model_id_dup[:8]}...")
                            print(f"      💡 Run 'python generation/scripts/cleanup_duplicate_grids.py' to remove duplicates")
                
                # Check for coordinate_system
                if 'coordinate_system' in sample_clean:
                    print(f"\n   🗺️  Coordinate System Found:")
                    coord_sys = sample_clean['coordinate_system']
                    if isinstance(coord_sys, dict):
                        print(f"      Type: {coord_sys.get('type', 'N/A')}")
                        if 'origin' in coord_sys:
                            print(f"      Origin: {coord_sys['origin']}")
                        if 'rotation' in coord_sys:
                            print(f"      Rotation: {coord_sys['rotation']}")
                        if 'bounding_box' in coord_sys:
                            print(f"      Bounding Box: {coord_sys['bounding_box']}")
                
        except Exception as e:
            print(f"   ❌ Error checking collection: {e}")
            import traceback
            traceback.print_exc()
    
    # Check GridFS for CT scan data
    print(f"\n📦 GridFS Files (CT Scan Data)")
    print("-" * 80)
    try:
        from gridfs import GridFS
        db = mongo_client.database  # Use database property, not get_database()
        fs = GridFS(db, collection='fs')
        files = list(fs.find())
        print(f"   Total GridFS files: {len(files)}")
        if files:
            print(f"\n   📄 Sample files:")
            for f in files[:5]:
                metadata = f.metadata or {}
                model_id_info = metadata.get('model_id', 'N/A')
                file_type = metadata.get('data_type', 'N/A')
                print(f"      - {f.filename} (type: {file_type}, model_id: {model_id_info[:36] if model_id_info != 'N/A' else 'N/A'}..., size: {f.length} bytes)")
            if len(files) > 5:
                print(f"      ... and {len(files) - 5} more files")
    except Exception as e:
        print(f"   ⚠️  Could not check GridFS: {e}")
        import traceback
        traceback.print_exc()
    
    # Disconnect
    mongo_client.close()
    print("\n" + "=" * 80)
    print("✅ Check complete!")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Check stored data in MongoDB')
    parser.add_argument('--model-id', type=str, help='Model ID to check (default: all models)')
    parser.add_argument('--list-models', action='store_true', help='List all model IDs and exit')
    
    args = parser.parse_args()
    
    if args.list_models:
        # Quick list mode
        print("=" * 80)
        print("📋 All Model IDs in Database")
        print("=" * 80)
        config = get_mongodb_config()
        if not config.username:
            config.username = os.getenv('MONGO_ROOT_USERNAME', 'admin')
        if not config.password:
            config.password = os.getenv('MONGO_ROOT_PASSWORD', 'password')
        
        try:
            mongo_client = MongoDBClient(config=config)
            collections = ['stl_models', 'hatching_layers', 'laser_parameters', 'ct_scan_data', 'ispm_monitoring_data', 'voxel_grids']
            all_model_ids = set()
            
            for collection_name in collections:
                try:
                    collection = mongo_client.get_collection(collection_name)
                    model_ids = collection.distinct('model_id')
                    all_model_ids.update(model_ids)
                    if model_ids:
                        print(f"\n{collection_name}:")
                        for mid in model_ids:
                            count = collection.count_documents({'model_id': mid})
                            print(f"  {mid} ({count} docs)")
                except Exception as e:
                    print(f"{collection_name}: Error - {e}")
            
            print(f"\n{'=' * 80}")
            print(f"Total unique model_ids: {len(all_model_ids)}")
            if all_model_ids:
                print("\nAll model_ids:")
                for mid in sorted(all_model_ids):
                    print(f"  {mid}")
            
            mongo_client.close()
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        check_stored_data(model_id=args.model_id)

