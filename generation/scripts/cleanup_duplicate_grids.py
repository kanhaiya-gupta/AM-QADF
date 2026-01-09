"""
Cleanup Duplicate Voxel Grids in MongoDB

This script identifies and removes duplicate voxel grids.
Duplicates are identified by:
- Same model_id and grid_name
- Same model_id and identical configuration (bbox, resolution)
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict

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
    from gridfs import GridFS
    from bson import ObjectId
    
    def get_mongodb_config():
        """Get MongoDB config from environment."""
        return MongoDBConfig.from_env()
except Exception as e:
    print(f"❌ Error loading MongoDB client: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


def find_duplicate_grids(mongo_client: MongoDBClient) -> Dict[str, List[Dict[str, Any]]]:
    """
    Find duplicate grids by model_id and grid_name, or by configuration.
    
    Returns:
        Dictionary mapping duplicate group keys to lists of grid documents
    """
    collection = mongo_client.get_collection('voxel_grids')
    
    # Get all grids
    all_grids = list(collection.find({}))
    
    # Group by model_id and grid_name (exact duplicates)
    exact_duplicates = defaultdict(list)
    for grid in all_grids:
        model_id = grid.get('model_id')
        grid_name = grid.get('grid_name')
        key = f"{model_id}::{grid_name}"
        exact_duplicates[key].append(grid)
    
    # Group by model_id and configuration (config duplicates)
    config_duplicates = defaultdict(list)
    for grid in all_grids:
        model_id = grid.get('model_id')
        metadata = grid.get('metadata', {})
        bbox_min = tuple(metadata.get('bbox_min', []))
        bbox_max = tuple(metadata.get('bbox_max', []))
        resolution = metadata.get('resolution')
        
        # Create a key based on configuration
        config_key = f"{model_id}::bbox_{bbox_min}_{bbox_max}::res_{resolution}"
        config_duplicates[config_key].append(grid)
    
    # Find actual duplicates (more than 1 grid with same key)
    duplicates = {}
    
    # Exact duplicates (same model_id and grid_name)
    for key, grids in exact_duplicates.items():
        if len(grids) > 1:
            duplicates[f"exact_{key}"] = grids
    
    # Config duplicates (same model_id and configuration, different names)
    for key, grids in config_duplicates.items():
        if len(grids) > 1:
            # Only consider if they have different grid_names (otherwise already in exact duplicates)
            grid_names = [g.get('grid_name') for g in grids]
            if len(set(grid_names)) > 1:
                duplicates[f"config_{key}"] = grids
    
    return duplicates


def delete_duplicate_grids(dry_run: bool = True, keep_oldest: bool = False):
    """
    Find and delete duplicate voxel grids.
    
    Args:
        dry_run: If True, only show what would be deleted
        keep_oldest: If True, keep the oldest grid; if False, keep the newest
    """
    print("=" * 80)
    if dry_run:
        print("🔍 Dry Run: Checking for Duplicate Voxel Grids")
    else:
        print("🗑️  Cleaning Up Duplicate Voxel Grids")
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
    
    # Find duplicates
    print("🔍 Searching for duplicate grids...")
    duplicates = find_duplicate_grids(mongo_client)
    
    if not duplicates:
        print("\n✅ No duplicate grids found!")
        mongo_client.close()
        return
    
    print(f"\n📋 Found {len(duplicates)} duplicate group(s):")
    print("-" * 80)
    
    total_duplicates = 0
    grids_to_delete = []
    
    for group_key, grids in duplicates.items():
        # Sort by created_at (oldest first if keep_oldest, newest first otherwise)
        grids_sorted = sorted(grids, key=lambda g: g.get('created_at', ''), reverse=not keep_oldest)
        
        # Keep the first one, mark others for deletion
        keep_grid = grids_sorted[0]
        delete_grids = grids_sorted[1:]
        
        total_duplicates += len(delete_grids)
        
        print(f"\n📦 Duplicate Group: {group_key[:50]}...")
        print(f"   Total grids: {len(grids)}")
        print(f"   Keeping: {keep_grid.get('grid_name', 'Unknown')} (ID: {str(keep_grid['_id'])[:8]}...)")
        print(f"   {'Would delete' if dry_run else 'Deleting'}: {len(delete_grids)} grid(s)")
        
        for grid in delete_grids:
            grid_id = str(grid['_id'])
            grid_name = grid.get('grid_name', 'Unknown')
            created_at = grid.get('created_at', 'Unknown')
            print(f"      - {grid_name} (ID: {grid_id[:8]}..., created: {created_at})")
            grids_to_delete.append(grid)
    
    if not grids_to_delete:
        print("\n✅ No grids to delete!")
        mongo_client.close()
        return
    
    if dry_run:
        print(f"\n🔍 Dry Run - {len(grids_to_delete)} grid(s) would be deleted.")
        print("   Use --execute to actually delete.")
    else:
        print(f"\n⚠️  WARNING: This will permanently delete {len(grids_to_delete)} grid(s)!")
        response = input("   Type 'DELETE' to confirm: ")
        if response != 'DELETE':
            print("   ❌ Cancelled. No grids were deleted.")
            mongo_client.close()
            return
    
    # Delete grids
    if not dry_run:
        print("\n🗑️  Deleting duplicate grids...")
        print("-" * 80)
        
        collection = mongo_client.get_collection('voxel_grids')
        db = mongo_client.database
        fs = GridFS(db, collection='fs')
        
        deleted_count = 0
        deleted_files = 0
        
        for grid in grids_to_delete:
            grid_id = grid['_id']
            grid_name = grid.get('grid_name', 'Unknown')
            
            try:
                # Delete GridFS files
                signal_refs = grid.get('signal_references', {})
                for signal_name, file_id in signal_refs.items():
                    try:
                        if file_id:
                            fs.delete(ObjectId(file_id))
                            deleted_files += 1
                    except Exception as e:
                        print(f"   ⚠️  Error deleting GridFS file {file_id}: {e}")
                
                voxel_data_ref = grid.get('voxel_data_reference')
                if voxel_data_ref:
                    try:
                        fs.delete(ObjectId(voxel_data_ref))
                        deleted_files += 1
                    except Exception as e:
                        print(f"   ⚠️  Error deleting voxel data file {voxel_data_ref}: {e}")
                
                # Delete grid document
                result = collection.delete_one({'_id': grid_id})
                if result.deleted_count > 0:
                    deleted_count += 1
                    print(f"   ✅ Deleted grid: {grid_name} (ID: {str(grid_id)[:8]}...)")
                else:
                    print(f"   ⚠️  Grid not found: {grid_name}")
                    
            except Exception as e:
                print(f"   ❌ Error deleting grid {grid_name}: {e}")
        
        print(f"\n📊 Summary:")
        print(f"   Deleted grids: {deleted_count}/{len(grids_to_delete)}")
        print(f"   Deleted GridFS files: {deleted_files}")
    else:
        print(f"\n📊 Summary: Would delete {len(grids_to_delete)} grid(s)")
    
    # Disconnect
    mongo_client.close()
    print("\n" + "=" * 80)
    print("✅ Cleanup complete!")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Clean up duplicate voxel grids in MongoDB')
    parser.add_argument('--execute', action='store_true', 
                       help='Actually delete duplicates (default is dry-run)')
    parser.add_argument('--keep-oldest', action='store_true',
                       help='Keep the oldest grid instead of newest (default: keep newest)')
    
    args = parser.parse_args()
    
    delete_duplicate_grids(dry_run=not args.execute, keep_oldest=args.keep_oldest)

