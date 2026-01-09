"""
Cleanup Incomplete Data in MongoDB

This script removes models that have incomplete data (only STL, missing other sources).
"""

import sys
import os
from pathlib import Path
from typing import Set

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
    
    def get_mongodb_config():
        """Get MongoDB config from environment."""
        return MongoDBConfig.from_env()
except Exception as e:
    print(f"❌ Error loading MongoDB client: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


def find_incomplete_models(mongo_client: MongoDBClient) -> Set[str]:
    """
    Find model_ids that have incomplete data.
    
    Returns:
        Set of model_ids that are incomplete
    """
    incomplete = set()
    
    # Get all model_ids from STL models
    stl_collection = mongo_client.get_collection('stl_models')
    all_stl_models = stl_collection.distinct('model_id')
    
    # Collections that should have data for complete models
    required_collections = {
        'hatching_layers',
        'laser_parameters',
        'ct_scan_data',
        'ispm_monitoring_data'
    }
    
    for model_id in all_stl_models:
        is_complete = True
        
        # Check each required collection
        for collection_name in required_collections:
            collection = mongo_client.get_collection(collection_name)
            count = collection.count_documents({'model_id': model_id})
            if count == 0:
                is_complete = False
                break
        
        if not is_complete:
            incomplete.add(model_id)
    
    return incomplete


def delete_model_data(mongo_client: MongoDBClient, model_id: str, dry_run: bool = True):
    """
    Delete all data for a specific model_id.
    
    Args:
        mongo_client: MongoDB client
        model_id: Model ID to delete
        dry_run: If True, only show what would be deleted
    """
    collections = [
        'stl_models',
        'hatching_layers',
        'laser_parameters',
        'ct_scan_data',
        'ispm_monitoring_data'
    ]
    
    total_deleted = 0
    
    # Delete from collections
    for collection_name in collections:
        collection = mongo_client.get_collection(collection_name)
        count = collection.count_documents({'model_id': model_id})
        
        if count > 0:
            if dry_run:
                print(f"   Would delete {count} documents from {collection_name}")
            else:
                result = collection.delete_many({'model_id': model_id})
                print(f"   ✅ Deleted {result.deleted_count} documents from {collection_name}")
                total_deleted += result.deleted_count
    
    # Delete GridFS files
    try:
        db = mongo_client.database
        fs = GridFS(db, collection='fs')
        
        # Find files for this model
        files = list(fs.find({'metadata.model_id': model_id}))
        
        if files:
            if dry_run:
                print(f"   Would delete {len(files)} GridFS files")
            else:
                for f in files:
                    fs.delete(f._id)
                print(f"   ✅ Deleted {len(files)} GridFS files")
                total_deleted += len(files)
    except Exception as e:
        if not dry_run:
            print(f"   ⚠️  Error deleting GridFS files: {e}")
    
    return total_deleted


def cleanup_incomplete_data(dry_run: bool = True, model_id: str = None):
    """
    Clean up incomplete data from MongoDB.
    
    Args:
        dry_run: If True, only show what would be deleted
        model_id: Optional specific model_id to delete (if None, finds incomplete ones)
    """
    print("=" * 80)
    if dry_run:
        print("🔍 Dry Run: Checking for Incomplete Data")
    else:
        print("🗑️  Cleaning Up Incomplete Data")
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
    
    # Find incomplete models
    if model_id:
        # Check if this specific model is incomplete
        incomplete = find_incomplete_models(mongo_client)
        if model_id in incomplete:
            models_to_delete = [model_id]
        else:
            # Get model name for display
            stl_collection = mongo_client.get_collection('stl_models')
            stl_doc = stl_collection.find_one({'model_id': model_id})
            model_name = stl_doc.get('model_name', stl_doc.get('filename', 'Unknown')) if stl_doc else 'Unknown'
            
            print(f"\n⚠️  Model {model_id[:36]}... ({model_name}) appears to be complete.")
            print("   Use --force to delete it anyway.")
            mongo_client.close()
            return
    else:
        models_to_delete = list(find_incomplete_models(mongo_client))
    
    if not models_to_delete:
        print("\n✅ No incomplete models found. All models have complete data!")
        mongo_client.close()
        return
    
    print(f"\n📋 Found {len(models_to_delete)} incomplete model(s):")
    print("-" * 80)
    
    stl_collection = mongo_client.get_collection('stl_models')
    for mid in models_to_delete:
        stl_doc = stl_collection.find_one({'model_id': mid})
        model_name = stl_doc.get('model_name', stl_doc.get('filename', 'Unknown')) if stl_doc else 'Unknown'
        print(f"   - {mid} ({model_name})")
    
    if dry_run:
        print("\n🔍 Dry Run - No data will be deleted. Use --execute to actually delete.")
    else:
        print("\n⚠️  WARNING: This will permanently delete data!")
        response = input("   Type 'DELETE' to confirm: ")
        if response != 'DELETE':
            print("   ❌ Cancelled. No data was deleted.")
            mongo_client.close()
            return
    
    print("\n🗑️  Deleting incomplete models...")
    print("-" * 80)
    
    total_deleted = 0
    for mid in models_to_delete:
        stl_doc = stl_collection.find_one({'model_id': mid})
        model_name = stl_doc.get('model_name', stl_doc.get('filename', 'Unknown')) if stl_doc else 'Unknown'
        print(f"\n📦 Model: {mid[:36]}... ({model_name})")
        
        deleted = delete_model_data(mongo_client, mid, dry_run=dry_run)
        total_deleted += deleted
        
        if not dry_run:
            print(f"   ✅ Total deleted: {deleted} items")
    
    if dry_run:
        print(f"\n📊 Summary: Would delete data for {len(models_to_delete)} model(s)")
    else:
        print(f"\n📊 Summary: Deleted data for {len(models_to_delete)} model(s), {total_deleted} total items")
    
    # Disconnect
    mongo_client.close()
    print("\n" + "=" * 80)
    print("✅ Cleanup complete!")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Clean up incomplete data in MongoDB')
    parser.add_argument('--execute', action='store_true', 
                       help='Actually delete data (default is dry-run)')
    parser.add_argument('--model-id', type=str, 
                       help='Specific model_id to delete (must be incomplete unless --force is used)')
    parser.add_argument('--force', action='store_true',
                       help='Force delete even if model appears complete (requires --model-id)')
    
    args = parser.parse_args()
    
    if args.force and not args.model_id:
        print("❌ Error: --force requires --model-id")
        sys.exit(1)
    
    if args.force and args.model_id:
        # Force delete specific model
        print("=" * 80)
        print("🗑️  Force Delete Mode")
        print("=" * 80)
        config = get_mongodb_config()
        if not config.username:
            config.username = os.getenv('MONGO_ROOT_USERNAME', 'admin')
        if not config.password:
            config.password = os.getenv('MONGO_ROOT_PASSWORD', 'password')
        
        try:
            mongo_client = MongoDBClient(config=config)
            print(f"✅ Connected to MongoDB: {config.database}\n")
            
            if not args.execute:
                print("🔍 Dry Run - Use --execute to actually delete.")
            else:
                print("⚠️  WARNING: Force deleting model. This will permanently delete data!")
                response = input("   Type 'DELETE' to confirm: ")
                if response != 'DELETE':
                    print("   ❌ Cancelled. No data was deleted.")
                    mongo_client.close()
                    sys.exit(0)
            
            print(f"\n🗑️  {'Would delete' if not args.execute else 'Deleting'} model: {args.model_id}")
            delete_model_data(mongo_client, args.model_id, dry_run=not args.execute)
            mongo_client.close()
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        cleanup_incomplete_data(dry_run=not args.execute, model_id=args.model_id)


