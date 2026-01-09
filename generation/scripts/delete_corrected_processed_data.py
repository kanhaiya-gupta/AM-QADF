"""
Delete Corrected and Processed Data from MongoDB

This script deletes voxel grids that have been corrected or processed,
including their GridFS files (signal arrays and voxel data).
"""

import sys
import os
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

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


def delete_corrected_processed_data(
    model_id: Optional[str] = None,
    grid_id: Optional[str] = None,
    correction_only: bool = False,
    processing_only: bool = False,
    execute: bool = False,
    force: bool = False
):
    """
    Delete corrected and processed grids from MongoDB.
    
    Args:
        model_id: Optional model ID to filter by
        grid_id: Optional grid ID to filter by
        correction_only: If True, only delete corrected grids
        processing_only: If True, only delete processed grids
        execute: If True, actually delete (otherwise dry-run)
        force: If True, skip confirmation prompt
    """
    print("=" * 80)
    print("🗑️  Delete Corrected and Processed Data from MongoDB")
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
    
    # Get grids to delete
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
        
        # Filter for corrected/processed grids
        grids_to_delete = []
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
                'is_corrected': is_corrected,
                'is_processed': is_processed,
                'signal_references': doc.get('signal_references', {}),
                'voxel_data_reference': doc.get('voxel_data_reference'),
                'available_signals': doc.get('available_signals', [])
            }
            grids_to_delete.append(grid)
    except Exception as e:
        print(f"❌ Failed to list grids: {e}")
        import traceback
        traceback.print_exc()
        return
    
    if not grids_to_delete:
        print("\n⚠️ No corrected or processed grids found to delete")
        if correction_only:
            print("   (Filtered for corrected grids only)")
        if processing_only:
            print("   (Filtered for processed grids only)")
        return
    
    # Display grids to be deleted
    print(f"\n📋 Found {len(grids_to_delete)} grid(s) to delete:")
    print("-" * 80)
    
    corrected_count = sum(1 for g in grids_to_delete if g.get('is_corrected'))
    processed_count = sum(1 for g in grids_to_delete if g.get('is_processed'))
    both_count = sum(1 for g in grids_to_delete if g.get('is_corrected') and g.get('is_processed'))
    
    print(f"  Corrected: {corrected_count}")
    print(f"  Processed: {processed_count}")
    print(f"  Both: {both_count}")
    print()
    
    # Group by model
    grids_by_model = {}
    for grid in grids_to_delete:
        mid = grid.get('model_id', 'unknown')
        if mid not in grids_by_model:
            grids_by_model[mid] = []
        grids_by_model[mid].append(grid)
    
    for model_id_key, model_grids in sorted(grids_by_model.items()):
        model_name = model_grids[0].get('model_name', 'Unknown')
        print(f"  Model: {model_name} ({model_id_key[:36]}...)")
        for grid in model_grids:
            status = []
            if grid.get('is_corrected'):
                status.append('corrected')
            if grid.get('is_processed'):
                status.append('processed')
            status_str = ' & '.join(status) if status else 'unknown'
            n_signals = len(grid.get('available_signals', []))
            print(f"    - {grid.get('grid_name', 'Unknown')} ({status_str}, {n_signals} signal(s)) [{grid.get('grid_id', 'N/A')[:8]}...]")
    
    # Count GridFS files
    total_gridfs_files = 0
    for grid in grids_to_delete:
        total_gridfs_files += len(grid.get('signal_references', {}))
        if grid.get('voxel_data_reference'):
            total_gridfs_files += 1
    
    print(f"\n  Total GridFS files to delete: {total_gridfs_files}")
    
    if not execute:
        print("\n" + "=" * 80)
        print("⚠️  DRY RUN MODE - No data will be deleted")
        print("=" * 80)
        print("\nTo actually delete, run with --execute flag:")
        print(f"  python {Path(__file__).name} --execute")
        if model_id:
            print(f"  (with --model-id {model_id})")
        if grid_id:
            print(f"  (with --grid-id {grid_id})")
        if correction_only:
            print(f"  (with --correction-only)")
        if processing_only:
            print(f"  (with --processing-only)")
        return
    
    # Confirmation
    if not force:
        print("\n" + "=" * 80)
        print("⚠️  WARNING: This will permanently delete:")
        print(f"  - {len(grids_to_delete)} grid document(s)")
        print(f"  - {total_gridfs_files} GridFS file(s)")
        print("=" * 80)
        print("\nThis action CANNOT be undone!")
        confirmation = input("\nType 'DELETE' to confirm: ")
        if confirmation != 'DELETE':
            print("❌ Deletion cancelled")
            return
    
    # Delete grids
    print("\n🗑️  Deleting grids and GridFS files...")
    deleted_count = 0
    deleted_gridfs_count = 0
    errors = []
    
    for grid in grids_to_delete:
        grid_id = grid.get('grid_id')
        grid_name = grid.get('grid_name', 'Unknown')
        
        try:
            # Delete signal arrays from GridFS
            signal_references = grid.get('signal_references', {})
            for signal_name, file_id in signal_references.items():
                try:
                    if mongo_client.delete_file(file_id):
                        deleted_gridfs_count += 1
                    else:
                        errors.append(f"Failed to delete signal {signal_name} (file_id: {file_id[:8]}...)")
                except Exception as e:
                    errors.append(f"Error deleting signal {signal_name}: {e}")
            
            # Delete voxel data from GridFS
            voxel_data_ref = grid.get('voxel_data_reference')
            if voxel_data_ref:
                try:
                    if mongo_client.delete_file(voxel_data_ref):
                        deleted_gridfs_count += 1
                    else:
                        errors.append(f"Failed to delete voxel data (file_id: {voxel_data_ref[:8]}...)")
                except Exception as e:
                    errors.append(f"Error deleting voxel data: {e}")
            
            # Delete grid document
            from bson import ObjectId
            result = collection.delete_one({'_id': ObjectId(grid_id)})
            if result.deleted_count > 0:
                deleted_count += 1
                print(f"  ✅ Deleted: {grid_name} ({grid_id[:8]}...)")
            else:
                errors.append(f"Grid document not found: {grid_id}")
        
        except Exception as e:
            errors.append(f"Error deleting grid {grid_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 Deletion Summary")
    print("=" * 80)
    print(f"✅ Grids deleted: {deleted_count}/{len(grids_to_delete)}")
    print(f"✅ GridFS files deleted: {deleted_gridfs_count}/{total_gridfs_files}")
    
    if errors:
        print(f"\n⚠️  Errors encountered: {len(errors)}")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
    else:
        print("\n✅ All deletions completed successfully!")
    
    print("\n" + "=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Delete corrected and processed data from MongoDB',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (show what would be deleted)
  python delete_corrected_processed_data.py
  
  # Delete all corrected grids
  python delete_corrected_processed_data.py --correction-only --execute
  
  # Delete all processed grids
  python delete_corrected_processed_data.py --processing-only --execute
  
  # Delete all corrected and processed grids
  python delete_corrected_processed_data.py --execute
  
  # Delete for a specific model
  python delete_corrected_processed_data.py --model-id <model_id> --execute
  
  # Delete a specific grid
  python delete_corrected_processed_data.py --grid-id <grid_id> --execute
        """
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
        help='Only delete corrected grids'
    )
    parser.add_argument(
        '--processing-only',
        action='store_true',
        help='Only delete processed grids'
    )
    parser.add_argument(
        '--execute',
        action='store_true',
        help='Actually delete (default is dry-run)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Skip confirmation prompt (use with caution!)'
    )
    
    args = parser.parse_args()
    
    delete_corrected_processed_data(
        model_id=args.model_id,
        grid_id=args.grid_id,
        correction_only=args.correction_only,
        processing_only=args.processing_only,
        execute=args.execute,
        force=args.force
    )


if __name__ == '__main__':
    main()

