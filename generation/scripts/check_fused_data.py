#!/usr/bin/env python3
"""
Check Fused Data in MongoDB

This script checks fused grids saved from Notebook 06 (Multi-Source Data Fusion).
It verifies that:
1. Fusion metadata is properly stored
2. Fused signals are stored in GridFS
3. All source grid references are present
4. Fusion metrics are saved
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
src_dir = project_root / 'src'

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import os
from bson import ObjectId
from gridfs import GridFS

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

from src.infrastructure.config import MongoDBConfig
from src.infrastructure.database import MongoDBClient
from am_qadf.voxel_domain import VoxelGridStorage

def check_gridfs_file(mongo_client, file_id, bucket_name='fs'):
    """Check if a GridFS file exists and get its metadata."""
    try:
        fs = GridFS(mongo_client.database, collection=bucket_name)
        grid_file = fs.get(ObjectId(file_id))
        
        metadata = grid_file.metadata or {}
        size_mb = grid_file.length / (1024 * 1024)
        
        return {
            'exists': True,
            'size_mb': size_mb,
            'filename': grid_file.filename,
            'metadata': metadata
        }
    except Exception as e:
        return {
            'exists': False,
            'error': str(e)
        }

def main():
    print("=" * 80)
    print("🔍 Checking Fused Data in MongoDB")
    print("=" * 80)
    print()
    
    # Connect to MongoDB
    print("🔌 Connecting to MongoDB...")
    try:
        config = MongoDBConfig.from_env()
        if not config.username:
            config.username = os.getenv('MONGO_ROOT_USERNAME', 'admin')
        if not config.password:
            config.password = os.getenv('MONGO_ROOT_PASSWORD', 'password')
        
        mongo_client = MongoDBClient(config=config)
        if not mongo_client.is_connected():
            print("❌ Failed to connect to MongoDB")
            return
        
        print(f"✅ Connected to MongoDB: {config.database}")
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        return
    
    # Initialize VoxelGridStorage
    try:
        voxel_storage = VoxelGridStorage(mongo_client=mongo_client)
    except Exception as e:
        print(f"❌ Failed to initialize VoxelGridStorage: {e}")
        return
    
    # Get all grids
    collection = mongo_client.get_collection('voxel_grids')
    all_grids = list(collection.find({}).sort('created_at', -1).limit(1000))
    
    # Filter for fused grids
    fused_grids = []
    for grid in all_grids:
        metadata = grid.get('metadata', {})
        config_meta = metadata.get('configuration_metadata', {})
        if not config_meta:
            config_meta = metadata
        
        if config_meta.get('fusion_applied', False):
            fused_grids.append(grid)
    
    print()
    print("📊 Summary Statistics")
    print("-" * 80)
    print(f"Total Grids: {len(all_grids)}")
    print(f"Fused Grids: {len(fused_grids)}")
    print()
    
    if not fused_grids:
        print("⚠️ No fused grids found in MongoDB")
        return
    
    print("=" * 80)
    print("📋 Detailed Fused Grid Information")
    print("=" * 80)
    print()
    
    for idx, grid in enumerate(fused_grids, 1):
        grid_id = str(grid.get('_id', ''))
        grid_name = grid.get('grid_name', 'Unknown')
        model_id = grid.get('model_id', 'Unknown')
        model_name = grid.get('model_name', 'Unknown')
        created_at = grid.get('created_at', 'Unknown')
        
        metadata = grid.get('metadata', {})
        config_meta = metadata.get('configuration_metadata', {})
        if not config_meta:
            config_meta = metadata
        
        print("-" * 80)
        print(f"📦 Grid {idx}/{len(fused_grids)}")
        print("-" * 80)
        print()
        print("📌 Basic Information:")
        print(f"  Grid ID: {grid_id}")
        print(f"  Grid Name: {grid_name}")
        print(f"  Model ID: {model_id}")
        print(f"  Model Name: {model_name}")
        print(f"  Created: {created_at}")
        print()
        
        # Fusion metadata
        print("🔀 Fusion Details:")
        fusion_strategy = config_meta.get('fusion_strategy', 'Unknown')
        fusion_timestamp = config_meta.get('fusion_timestamp', 'Unknown')
        num_sources = config_meta.get('num_sources', 0)
        source_grids = config_meta.get('source_grids', [])
        source_names = config_meta.get('source_names', [])
        
        print(f"  Fusion Strategy: {fusion_strategy}")
        print(f"  Fusion Timestamp: {fusion_timestamp}")
        print(f"  Number of Sources: {num_sources}")
        print(f"  Source Grid IDs: {len(source_grids)} grid(s)")
        for i, (grid_id_ref, source_name) in enumerate(zip(source_grids, source_names), 1):
            print(f"    {i}. {source_name} (Grid ID: {grid_id_ref[:8]}...)")
        print()
        
        # Strategy-specific parameters
        if fusion_strategy == 'weighted_average':
            normalize = config_meta.get('normalize_weights', 'N/A')
            auto_weight = config_meta.get('auto_weight_quality', 'N/A')
            print(f"  Normalize Weights: {normalize}")
            print(f"  Auto-weight by Quality: {auto_weight}")
        elif fusion_strategy == 'quality_based':
            threshold = config_meta.get('quality_threshold', 'N/A')
            print(f"  Quality Threshold: {threshold}")
        print()
        
        # Fusion metrics
        fusion_metrics = config_meta.get('fusion_metrics', {})
        if fusion_metrics:
            print("📊 Fusion Metrics:")
            print(f"  Fusion Score: {fusion_metrics.get('fusion_score', 'N/A')}")
            print(f"  Coverage: {fusion_metrics.get('coverage', 'N/A')}")
            print(f"  Quality Score: {fusion_metrics.get('quality_score', 'N/A')}")
            print(f"  Consistency Score: {fusion_metrics.get('consistency_score', 'N/A')}")
            print()
        
        # Available signals
        available_signals = grid.get('available_signals', [])
        print(f"📊 Available Signals: {len(available_signals)} signal(s)")
        if available_signals:
            for signal in available_signals:
                print(f"  - {signal}")
        print()
        
        # Signal storage verification
        signal_references = grid.get('signal_references', {})
        print("💾 Signal Storage (GridFS):")
        if signal_references:
            total_size = 0
            valid_signals = 0
            for signal_name, file_id in signal_references.items():
                file_info = check_gridfs_file(mongo_client, file_id)
                if file_info['exists']:
                    total_size += file_info['size_mb']
                    valid_signals += 1
                    file_meta = file_info.get('metadata', {})
                    grid_id_from_meta = file_meta.get('grid_id', 'N/A')
                    
                    print(f"  ✅ {signal_name}:")
                    print(f"     File ID: {file_id[:24]}...")
                    print(f"     Size: {file_info['size_mb']:.2f} MB")
                    print(f"     Grid ID (from metadata): {grid_id_from_meta}")
                    
                    # Check for grid_id mismatch
                    if grid_id_from_meta != 'N/A' and grid_id_from_meta != grid_id:
                        print(f"     ⚠️ Grid ID mismatch! Expected: {grid_id[:8]}..., Got: {grid_id_from_meta[:8]}...")
                else:
                    print(f"  ❌ {signal_name}: File not found ({file_info.get('error', 'Unknown error')})")
            
            print()
            print(f"  Storage Summary:")
            print(f"    Valid Signals: {valid_signals}/{len(signal_references)}")
            print(f"    Total Storage: {total_size:.2f} MB")
        else:
            print("  ⚠️ No signal arrays stored in GridFS")
        print()
        
        # Description
        description = grid.get('description', '')
        if description:
            print(f"📝 Description: {description}")
            print()
    
    print("=" * 80)
    print("✅ Check complete!")
    print("=" * 80)

if __name__ == '__main__':
    main()

