"""
Delete All Alignment Data from MongoDB

This script deletes all alignment results and associated GridFS files.
Use with caution - this operation cannot be undone!
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Any

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
    from am_qadf.synchronization import AlignmentStorage
    
    def get_mongodb_config():
        """Get MongoDB config from environment."""
        return MongoDBConfig.from_env()
except Exception as e:
    print(f"❌ Error loading MongoDB client: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


def delete_all_alignments(execute: bool = False, force: bool = False):
    """
    Delete all alignment data from MongoDB.
    
    Args:
        execute: If True, actually delete (default: False for dry-run)
        force: If True, skip confirmation prompt
    """
    print("=" * 80)
    print("🗑️  Delete All Alignment Data from MongoDB")
    print("=" * 80)
    
    if not execute:
        print("\n⚠️  DRY RUN MODE - No data will be deleted")
        print("   Use --execute flag to actually delete data\n")
    
    # Connect to MongoDB
    print("🔌 Connecting to MongoDB...")
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
        
        alignment_storage = AlignmentStorage(mongo_client=mongo_client)
        print(f"✅ Connected to MongoDB: {config.database}\n")
    except Exception as e:
        print(f"❌ Failed to connect to MongoDB: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Get all alignments
    print("📋 Finding all alignments...")
    alignments = alignment_storage.list_alignments(limit=10000)
    
    if not alignments:
        print("✅ No alignments found in database. Nothing to delete.")
        return
    
    print(f"   Found {len(alignments)} alignment(s)\n")
    
    # Group by model
    by_model = {}
    for align in alignments:
        model_id = align.get('model_id', 'Unknown')
        model_name = align.get('model_name', 'Unknown')
        if model_id not in by_model:
            by_model[model_id] = {'name': model_name, 'count': 0, 'alignments': []}
        by_model[model_id]['count'] += 1
        by_model[model_id]['alignments'].append(align)
    
    print("📊 Alignments to delete:")
    print("-" * 80)
    for model_id, info in sorted(by_model.items()):
        print(f"   {info['name']} ({model_id[:36]}...): {info['count']} alignment(s)")
        for align in info['alignments']:
            align_id = align.get('alignment_id', 'N/A')
            mode = align.get('alignment_mode', 'N/A')
            created = align.get('created_at', 'N/A')
            print(f"      - {align_id[:36]}... ({mode}, created: {created})")
    
    # Count GridFS files
    gridfs_count = 0
    for align in alignments:
        data_refs = align.get('aligned_data_references', {})
        for source_name, refs in data_refs.items():
            if isinstance(refs, dict):
                if 'points_gridfs_id' in refs:
                    gridfs_count += 1
                if 'signals_gridfs_id' in refs:
                    gridfs_count += 1
                if 'signals_gridfs_ids' in refs:
                    gridfs_count += len(refs.get('signals_gridfs_ids', {}))
    
    if gridfs_count > 0:
        print(f"\n   GridFS Files: {gridfs_count} file(s) to delete")
    
    if not execute:
        print("\n" + "=" * 80)
        print("⚠️  This is a DRY RUN. No data was deleted.")
        print("   To actually delete, run with --execute flag")
        print("=" * 80)
        return
    
    # Confirmation
    if not force:
        print("\n" + "=" * 80)
        print("⚠️  WARNING: This will permanently delete ALL alignment data!")
        print("   This includes:")
        print(f"   - {len(alignments)} alignment document(s)")
        print(f"   - {gridfs_count} GridFS file(s)")
        print("   This operation CANNOT be undone!")
        print("=" * 80)
        
        confirmation = input("\nType 'DELETE' to confirm: ")
        if confirmation != 'DELETE':
            print("❌ Deletion cancelled. Type 'DELETE' exactly to confirm.")
            return
    
    # Delete alignments
    print("\n🗑️  Deleting alignments...")
    deleted_count = 0
    failed_count = 0
    gridfs_deleted = 0
    
    for align in alignments:
        align_id = align.get('alignment_id', 'N/A')
        model_name = align.get('model_name', 'Unknown')
        
        try:
            # Delete GridFS files first
            data_refs = align.get('aligned_data_references', {})
            for source_name, refs in data_refs.items():
                if isinstance(refs, dict):
                    # Delete points file
                    if 'points_gridfs_id' in refs:
                        try:
                            if mongo_client.delete_file(refs['points_gridfs_id']):
                                gridfs_deleted += 1
                        except Exception as e:
                            print(f"   ⚠️  Warning: Failed to delete GridFS file for {source_name} points: {e}")
                    
                    # Delete signals files
                    if 'signals_gridfs_id' in refs:
                        try:
                            if mongo_client.delete_file(refs['signals_gridfs_id']):
                                gridfs_deleted += 1
                        except Exception as e:
                            print(f"   ⚠️  Warning: Failed to delete GridFS file for {source_name} signals: {e}")
                    
                    if 'signals_gridfs_ids' in refs:
                        for signal_name, sig_file_id in refs.get('signals_gridfs_ids', {}).items():
                            try:
                                if mongo_client.delete_file(sig_file_id):
                                    gridfs_deleted += 1
                            except Exception as e:
                                print(f"   ⚠️  Warning: Failed to delete GridFS file for {source_name}/{signal_name}: {e}")
            
            # Delete alignment document
            if alignment_storage.delete_alignment(align_id):
                deleted_count += 1
                print(f"   ✅ Deleted: {model_name} ({align_id[:36]}...)")
            else:
                failed_count += 1
                print(f"   ❌ Failed to delete: {align_id[:36]}...")
                
        except Exception as e:
            failed_count += 1
            print(f"   ❌ Error deleting {align_id[:36]}...: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 Deletion Summary:")
    print("-" * 80)
    print(f"   Alignments Deleted: {deleted_count}/{len(alignments)}")
    print(f"   GridFS Files Deleted: {gridfs_deleted}")
    if failed_count > 0:
        print(f"   Failed: {failed_count}")
    print("=" * 80)
    
    if deleted_count == len(alignments):
        print("✅ All alignments deleted successfully!")
    else:
        print(f"⚠️  Some alignments could not be deleted ({failed_count} failed)")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Delete all alignment data from MongoDB',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (safe, shows what would be deleted)
  python generation/scripts/delete_all_alignments.py
  
  # Actually delete (requires confirmation)
  python generation/scripts/delete_all_alignments.py --execute
  
  # Delete without confirmation prompt
  python generation/scripts/delete_all_alignments.py --execute --force
        """
    )
    parser.add_argument('--execute', action='store_true',
                       help='Actually delete data (default: dry-run)')
    parser.add_argument('--force', action='store_true',
                       help='Skip confirmation prompt (use with --execute)')
    
    args = parser.parse_args()
    
    delete_all_alignments(execute=args.execute, force=args.force)


if __name__ == '__main__':
    main()

