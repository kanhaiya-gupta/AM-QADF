"""
Delete Signal Mapped Data from MongoDB

This script deletes signal mapping data from voxel grids, including:
- Signal arrays stored in GridFS
- Voxel data structures in GridFS
- Optionally, the grid documents themselves

Use with caution - this operation cannot be undone!
"""

import sys
import os
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any
from bson import ObjectId

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


def delete_signal_mapped_data(
    model_id: Optional[str] = None,
    grid_id: Optional[str] = None,
    delete_grids: bool = False,
    dry_run: bool = True
):
    """
    Delete signal mapping data from MongoDB.
    
    Args:
        model_id: Optional model ID to filter by
        grid_id: Optional grid ID to filter by
        delete_grids: If True, delete the grid documents entirely. If False, only clear signal references.
        dry_run: If True, only show what would be deleted without actually deleting
    """
    print("=" * 80)
    print("🗑️  Delete Signal Mapped Data from MongoDB")
    print("=" * 80)
    
    if dry_run:
        print("\n⚠️  DRY RUN MODE - No data will be deleted")
    else:
        print("\n⚠️  DELETION MODE - Data will be permanently deleted!")
    
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
    
    # Get grids with signal mappings
    try:
        collection = mongo_client.get_collection('voxel_grids')
        query = {}
        if model_id:
            query['model_id'] = model_id
        if grid_id:
            try:
                query['_id'] = ObjectId(grid_id)
            except:
                print(f"⚠️ Invalid grid_id format: {grid_id}")
                return
        
        # Find grids with signal references
        # First, get all grids matching the filter, then filter in Python
        # because MongoDB doesn't easily check for non-empty dictionaries
        all_grids = list(collection.find(query).sort('created_at', -1).limit(1000))
        
        # Filter for grids that have signal_references that are not empty
        grid_docs = []
        for grid in all_grids:
            signal_refs = grid.get('signal_references', {})
            available_signals = grid.get('available_signals', [])
            # Include if has non-empty signal_references or has available_signals
            if (signal_refs and isinstance(signal_refs, dict) and len(signal_refs) > 0) or \
               (available_signals and isinstance(available_signals, list) and len(available_signals) > 0):
                grid_docs.append(grid)
        
        # Also check for orphaned GridFS files (files not referenced by any grid document)
        from gridfs import GridFS
        fs = GridFS(mongo_client.database, collection='fs')
        
        # Get all signal-related GridFS files
        orphaned_files = []
        all_gridfs_files = list(fs.find({'metadata.data_type': 'signal_array'}))
        
        # Collect all referenced file IDs from existing grids
        referenced_file_ids = set()
        for grid in all_grids:
            signal_refs = grid.get('signal_references', {})
            if signal_refs:
                for file_id in signal_refs.values():
                    try:
                        referenced_file_ids.add(str(file_id))
                    except:
                        pass
            voxel_data_ref = grid.get('voxel_data_reference')
            if voxel_data_ref:
                try:
                    referenced_file_ids.add(str(voxel_data_ref))
                except:
                    pass
        
        # Find orphaned files
        for gridfs_file in all_gridfs_files:
            file_id = str(gridfs_file._id)
            if file_id not in referenced_file_ids:
                metadata = gridfs_file.metadata or {}
                orphaned_files.append({
                    'file_id': file_id,
                    'filename': gridfs_file.filename,
                    'metadata': metadata,
                    'length': gridfs_file.length
                })
        
        if not grid_docs and not orphaned_files:
            print("\n✅ No grids with signal mappings found")
            print("✅ No orphaned GridFS signal files found")
            return
        
        if grid_docs:
            print(f"\n📋 Found {len(grid_docs)} grid(s) with signal mappings")
            print("-" * 80)
        
        if orphaned_files:
            print(f"\n📋 Found {len(orphaned_files)} orphaned GridFS signal file(s)")
            print("-" * 80)
        
        # Group by model
        grids_by_model = {}
        for doc in grid_docs:
            mid = doc.get('model_id', 'unknown')
            if mid not in grids_by_model:
                grids_by_model[mid] = []
            grids_by_model[mid].append(doc)
        
        # Display what will be deleted
        total_signals = 0
        total_gridfs_files = 0
        
        if grid_docs:
            for model_id_key, model_grids in sorted(grids_by_model.items()):
                print(f"\nModel: {model_id_key}")
                for grid_doc in model_grids:
                    grid_id = str(grid_doc['_id'])
                    grid_name = grid_doc.get('grid_name', 'N/A')
                    signal_refs = grid_doc.get('signal_references', {})
                    voxel_data_ref = grid_doc.get('voxel_data_reference')
                    
                    num_signals = len(signal_refs)
                    total_signals += num_signals
                    total_gridfs_files += num_signals
                    if voxel_data_ref:
                        total_gridfs_files += 1
                    
                    print(f"  Grid: {grid_name} (ID: {grid_id[:24]}...)")
                    print(f"    Signals: {num_signals}")
                    if voxel_data_ref:
                        print(f"    Voxel Data: Yes")
                    if delete_grids:
                        print(f"    Grid Document: Will be deleted")
                    else:
                        print(f"    Grid Document: Will be updated (signals cleared)")
        
        if orphaned_files:
            print(f"\nOrphaned GridFS Files:")
            for orphan in orphaned_files[:10]:  # Show first 10
                signal_name = orphan['metadata'].get('signal_name', 'unknown')
                grid_id_meta = orphan['metadata'].get('grid_id', 'unknown')
                size_mb = orphan['length'] / (1024 * 1024)
                print(f"  - {orphan['filename']}")
                print(f"    Signal: {signal_name}, Grid ID (from metadata): {grid_id_meta}")
                print(f"    Size: {size_mb:.2f} MB")
            if len(orphaned_files) > 10:
                print(f"  ... and {len(orphaned_files) - 10} more orphaned files")
            total_gridfs_files += len(orphaned_files)
        
        print("\n" + "=" * 80)
        print("📊 Summary")
        print("=" * 80)
        print(f"Total Grids: {len(grid_docs)}")
        print(f"Total Signal Arrays (from grids): {total_signals}")
        print(f"Orphaned GridFS Files: {len(orphaned_files)}")
        print(f"Total GridFS Files to Delete: {total_gridfs_files}")
        if grid_docs:
            print(f"Grid Documents: {'Will be deleted' if delete_grids else 'Will be updated'}")
        
        if dry_run:
            print("\n💡 This was a dry run. Use --execute to actually delete the data.")
            return
        
        # Confirm deletion
        print("\n" + "=" * 80)
        print("⚠️  CONFIRMATION REQUIRED")
        print("=" * 80)
        print("This will PERMANENTLY DELETE:")
        print(f"  - {total_gridfs_files} GridFS file(s) ({total_signals} from grids + {len(orphaned_files)} orphaned)")
        if grid_docs:
            if delete_grids:
                print(f"  - {len(grid_docs)} grid document(s)")
            else:
                print(f"  - Signal references from {len(grid_docs)} grid document(s)")
        print("\nType 'DELETE' to confirm, or anything else to cancel:")
        
        confirmation = input("> ").strip()
        if confirmation != 'DELETE':
            print("\n❌ Deletion cancelled")
            return
        
        # Perform deletion
        print("\n🗑️  Deleting signal mapping data...")
        print("-" * 80)
        
        deleted_files = 0
        deleted_grids = 0
        deleted_orphaned = 0
        errors = []
        
        # Delete orphaned GridFS files first
        if orphaned_files:
            print(f"\nDeleting {len(orphaned_files)} orphaned GridFS file(s)...")
            for orphan in orphaned_files:
                try:
                    if mongo_client.delete_file(orphan['file_id']):
                        deleted_orphaned += 1
                        deleted_files += 1
                        if len(orphaned_files) <= 10:  # Show details for small numbers
                            print(f"  ✅ Deleted orphaned file: {orphan['filename']} ({orphan['file_id'][:24]}...)")
                    else:
                        errors.append(f"Failed to delete orphaned file {orphan['file_id'][:24]}...")
                except Exception as e:
                    errors.append(f"Error deleting orphaned file {orphan['file_id'][:24]}...: {e}")
        
        # Delete signal mappings from grids
        for grid_doc in grid_docs:
            grid_id = str(grid_doc['_id'])
            grid_name = grid_doc.get('grid_name', 'N/A')
            
            try:
                # Delete signal arrays from GridFS
                signal_refs = grid_doc.get('signal_references', {})
                for signal_name, file_id in signal_refs.items():
                    try:
                        if mongo_client.delete_file(file_id):
                            deleted_files += 1
                            if len(signal_refs) <= 5:  # Only show details for small numbers
                                print(f"  ✅ Deleted signal: {signal_name} ({file_id[:24]}...)")
                        else:
                            errors.append(f"Failed to delete signal {signal_name} ({file_id[:24]}...)")
                    except Exception as e:
                        errors.append(f"Error deleting signal {signal_name} ({file_id[:24]}...): {e}")
                
                # Delete voxel data from GridFS
                voxel_data_ref = grid_doc.get('voxel_data_reference')
                if voxel_data_ref:
                    try:
                        if mongo_client.delete_file(voxel_data_ref):
                            deleted_files += 1
                            print(f"  ✅ Deleted voxel data: {voxel_data_ref[:24]}...")
                        else:
                            errors.append(f"Failed to delete voxel data ({voxel_data_ref[:24]}...)")
                    except Exception as e:
                        errors.append(f"Error deleting voxel data ({voxel_data_ref[:24]}...): {e}")
                
                # Update or delete grid document
                if delete_grids:
                    # Delete the entire grid document
                    result = collection.delete_one({'_id': grid_doc['_id']})
                    if result.deleted_count > 0:
                        deleted_grids += 1
                        print(f"  ✅ Deleted grid document: {grid_name} ({grid_id[:24]}...)")
                    else:
                        errors.append(f"Failed to delete grid document {grid_id[:24]}...")
                else:
                    # Clear signal references but keep the grid
                    collection.update_one(
                        {'_id': grid_doc['_id']},
                        {
                            '$unset': {
                                'signal_references': '',
                                'voxel_data_reference': ''
                            },
                            '$set': {
                                'available_signals': []
                            }
                        }
                    )
                    deleted_grids += 1
                    print(f"  ✅ Cleared signals from grid: {grid_name} ({grid_id[:24]}...)")
                
            except Exception as e:
                errors.append(f"Error processing grid {grid_id[:24]}...: {e}")
        
        # Summary
        print("\n" + "=" * 80)
        print("✅ Deletion Complete")
        print("=" * 80)
        print(f"Deleted GridFS Files: {deleted_files}/{total_gridfs_files}")
        if orphaned_files:
            print(f"  - Orphaned files: {deleted_orphaned}/{len(orphaned_files)}")
        if grid_docs:
            if delete_grids:
                print(f"Deleted Grid Documents: {deleted_grids}/{len(grid_docs)}")
            else:
                print(f"Updated Grid Documents: {deleted_grids}/{len(grid_docs)}")
        
        if errors:
            print(f"\n⚠️  Errors ({len(errors)}):")
            for error in errors[:10]:  # Show first 10 errors
                print(f"  - {error}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more errors")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Delete signal mapping data from MongoDB',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (show what would be deleted)
  python delete_signal_mapped_data.py

  # Delete signals from all grids (keep grid documents)
  python delete_signal_mapped_data.py --execute

  # Delete signals and grid documents
  python delete_signal_mapped_data.py --execute --delete-grids

  # Delete signals for a specific model
  python delete_signal_mapped_data.py --execute --model-id <model_id>

  # Delete signals for a specific grid
  python delete_signal_mapped_data.py --execute --grid-id <grid_id>
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
        '--delete-grids',
        action='store_true',
        help='Delete grid documents entirely (default: only clear signal references)'
    )
    parser.add_argument(
        '--execute',
        action='store_true',
        help='Actually perform deletion (default: dry run)'
    )
    
    args = parser.parse_args()
    
    delete_signal_mapped_data(
        model_id=args.model_id,
        grid_id=args.grid_id,
        delete_grids=args.delete_grids,
        dry_run=not args.execute
    )


if __name__ == '__main__':
    main()

