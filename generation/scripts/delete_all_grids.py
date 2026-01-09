"""
Delete All Voxel Grids from MongoDB

This script deletes all voxel grids from the database.
Useful for cleaning up before recreating grids with updated metadata.
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Any

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


def delete_all_grids(dry_run: bool = True):
    """
    Delete all voxel grids from MongoDB.
    
    Args:
        dry_run: If True, only show what would be deleted
    """
    print("=" * 80)
    if dry_run:
        print("🔍 Dry Run: Checking Voxel Grids for Deletion")
    else:
        print("🗑️  Deleting All Voxel Grids")
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
    
    # Get all grids
    collection = mongo_client.get_collection('voxel_grids')
    all_grids = list(collection.find({}))
    
    if not all_grids:
        print("\n✅ No grids found in database!")
        mongo_client.close()
        return
    
    print(f"📋 Found {len(all_grids)} grid(s) in database:")
    print("-" * 80)
    
    for grid in all_grids:
        grid_id = str(grid['_id'])
        grid_name = grid.get('grid_name', 'Unknown')
        model_id = grid.get('model_id', 'Unknown')
        model_name = grid.get('model_name', 'Unknown')
        created_at = grid.get('created_at', 'Unknown')
        
        print(f"   - {grid_name}")
        print(f"     ID: {grid_id[:8]}...")
        print(f"     Model: {model_name} ({model_id[:8]}...)")
        print(f"     Created: {created_at}")
        print()
    
    if dry_run:
        print(f"🔍 Dry Run - {len(all_grids)} grid(s) would be deleted.")
        print("   Use --execute to actually delete.")
    else:
        print(f"⚠️  WARNING: This will permanently delete {len(all_grids)} grid(s)!")
        response = input("   Type 'DELETE ALL' to confirm: ")
        if response != 'DELETE ALL':
            print("   ❌ Cancelled. No grids were deleted.")
            mongo_client.close()
            return
    
    # Delete grids
    if not dry_run:
        print("\n🗑️  Deleting grids...")
        print("-" * 80)
        
        db = mongo_client.database
        fs = GridFS(db, collection='fs')
        
        deleted_count = 0
        deleted_files = 0
        
        for grid in all_grids:
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
        print(f"   Deleted grids: {deleted_count}/{len(all_grids)}")
        print(f"   Deleted GridFS files: {deleted_files}")
    else:
        print(f"\n📊 Summary: Would delete {len(all_grids)} grid(s)")
    
    # Disconnect
    mongo_client.close()
    print("\n" + "=" * 80)
    print("✅ Operation complete!")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Delete all voxel grids from MongoDB')
    parser.add_argument('--execute', action='store_true', 
                       help='Actually delete grids (default is dry-run)')
    
    args = parser.parse_args()
    
    delete_all_grids(dry_run=not args.execute)

