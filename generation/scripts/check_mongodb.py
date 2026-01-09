"""
MongoDB Database Checker

Connects to MongoDB and displays database information, collections, and sample data.
"""

# Standard library imports first
import sys
import os
import json
from pathlib import Path
from datetime import datetime

# Add project root to path AFTER standard library imports
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Load environment variables from development.env if it exists
env_file = project_root / 'development.env'
if env_file.exists():
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                # Remove quotes if present
                value = value.strip('"\'')
                os.environ[key] = value

# Set default MongoDB credentials if not in environment
if 'MONGO_ROOT_USERNAME' not in os.environ:
    os.environ['MONGO_ROOT_USERNAME'] = 'admin'
if 'MONGO_ROOT_PASSWORD' not in os.environ:
    os.environ['MONGO_ROOT_PASSWORD'] = 'password'

# Import from infrastructure package
try:
    from src.infrastructure.config import MongoDBConfig
    from src.infrastructure.database import MongoDBClient
except ImportError as e:
    print(f"❌ Error importing infrastructure modules: {e}")
    print(f"   Project root: {project_root}")
    print(f"   Python path: {sys.path[:3]}")
    raise

# Third-party imports after path setup
from pymongo import MongoClient

# Create a get_mongodb_config function for compatibility
def get_mongodb_config():
    """Get MongoDB config from environment."""
    return MongoDBConfig.from_env()


def format_size(size_bytes: int) -> str:
    """Format bytes to human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def check_mongodb():
    """Check MongoDB connection and display database information."""
    print("=" * 80)
    print("🔍 MongoDB Database Checker")
    print("=" * 80)
    
    # Get configuration
    config = get_mongodb_config()
    print(f"\n📋 Configuration:")
    print(f"   Host: {config.host if config.host else 'localhost'}")
    print(f"   Port: {config.port}")
    print(f"   Database: {config.database}")
    print(f"   Username: {config.username if config.username else 'None (no auth)'}")
    
    # Create client with proper authentication
    try:
        client = MongoDBClient(config=config)
        print(f"\n🔌 Connecting to MongoDB...")
        print("✅ Connected successfully!")
    except Exception as e:
        print(f"\n❌ Failed to connect to MongoDB: {e}")
        print("   Please check:")
        print("   1. MongoDB is running (docker ps)")
        print("   2. Connection settings are correct")
        print("   3. Network/firewall allows connection")
        return
    
    # Get MongoDB client for direct access
    mongo_client = client.client
    
    # Authentication is handled via connection string
    
    # List all databases
    print(f"\n📚 Available Databases:")
    print("-" * 80)
    try:
        db_list = mongo_client.list_database_names()
    except Exception as e:
        print(f"   ⚠️  Could not list databases (requires admin privileges): {e}")
        print(f"   Continuing with main database only...")
        db_list = [config.database]  # Just use the main database
    for db_name in sorted(db_list):
        db = mongo_client[db_name]
        stats = db.command("dbStats")
        size = stats.get('dataSize', 0)
        collections = stats.get('collections', 0)
        print(f"   📦 {db_name}")
        print(f"      Collections: {collections}")
        print(f"      Size: {format_size(size)}")
    
    # Check main database
    main_db = mongo_client[config.database]
    print(f"\n📊 Main Database: {config.database}")
    print("-" * 80)
    
    # List collections
    collections = main_db.list_collection_names()
    if collections:
        print(f"   Found {len(collections)} collection(s):\n")
        for coll_name in sorted(collections):
            coll = main_db[coll_name]
            count = coll.count_documents({})
            stats = main_db.command("collStats", coll_name)
            size = stats.get('size', 0)
            storage_size = stats.get('storageSize', 0)
            
            print(f"   📁 {coll_name}")
            print(f"      Documents: {count:,}")
            print(f"      Size: {format_size(size)}")
            print(f"      Storage Size: {format_size(storage_size)}")
            
            # Show indexes
            indexes = coll.list_indexes()
            index_list = list(indexes)
            if len(index_list) > 1:  # More than just _id index
                print(f"      Indexes: {len(index_list)}")
                for idx in index_list[1:]:  # Skip _id
                    keys = idx.get('key', {})
                    print(f"         - {', '.join([f'{k}:{v}' for k, v in keys.items()])}")
            
            # Show sample document (if any)
            if count > 0:
                sample = coll.find_one()
                if sample:
                    # Remove _id for cleaner display
                    sample_display = {k: v for k, v in sample.items() if k != '_id'}
                    print(f"      Sample fields: {', '.join(list(sample_display.keys())[:10])}")
                    if len(sample_display) > 10:
                        print(f"         ... and {len(sample_display) - 10} more fields")
            print()
    else:
        print("   ⚠️  No collections found in this database")
    
    # Check GridFS (if available)
    print(f"\n📦 GridFS")
    print("-" * 80)
    try:
        from gridfs import GridFS
        fs = GridFS(mongo_client[config.database])
        files = list(fs.find().limit(10))
        if files:
            print(f"   Found {len(files)} file(s) (showing first 10):\n")
            for f in files:
                print(f"   📄 {f.filename}")
                print(f"      Size: {format_size(f.length)}")
                print(f"      Upload Date: {f.upload_date}")
                if f.metadata:
                    print(f"      Metadata: {list(f.metadata.keys())}")
                print()
        else:
            print("   ⚠️  No files in GridFS")
    except Exception as e:
        print(f"   ⚠️  Could not check GridFS: {e}")
    
    # Server info
    print(f"\n🖥️  Server Information:")
    print("-" * 80)
    try:
        server_info = mongo_client.server_info()
        print(f"   Version: {server_info.get('version', 'Unknown')}")
        print(f"   Git Version: {server_info.get('gitVersion', 'Unknown')}")
    except Exception as e:
        print(f"   ⚠️  Could not get server info: {e}")
    
    # Disconnect (client cleanup handled automatically)
    print(f"\n✅ Check complete!")
    print("=" * 80)


if __name__ == "__main__":
    try:
        check_mongodb()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

