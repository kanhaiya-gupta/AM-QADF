"""
Basic Usage Example

Demonstrates basic usage of the AM-QADF framework:
1. Initialize clients
2. Query data
3. Create voxel grid
4. Map signals to voxels
"""

import numpy as np
from am_qadf.query import UnifiedQueryClient
from am_qadf.voxel_domain import VoxelDomainClient
from am_qadf.query.mongodb_client import MongoDBClient


def main():
    """Basic usage example."""
    print("=" * 60)
    print("AM-QADF Basic Usage Example")
    print("=" * 60)
    
    # Step 1: Initialize MongoDB client
    print("\n1. Initializing MongoDB client...")
    mongo_client = MongoDBClient(
        connection_string="mongodb://localhost:27017",
        database_name="am_qadf"
    )
    
    # Step 2: Initialize unified query client
    print("2. Initializing unified query client...")
    query_client = UnifiedQueryClient(mongo_client=mongo_client)
    
    # Step 3: Initialize voxel domain client
    print("3. Initializing voxel domain client...")
    voxel_client = VoxelDomainClient(
        unified_query_client=query_client,
        base_resolution=1.0,  # 1mm voxel resolution
        adaptive=False,
        mongo_client=mongo_client
    )
    
    # Step 4: Query data for a model
    print("\n4. Querying data...")
    model_id = "example_model_id"  # Replace with actual model ID
    
    try:
        # Get all data for the model
        all_data = query_client.get_all_data(model_id)
        print(f"   Retrieved data for model: {model_id}")
        print(f"   Available sources: {list(all_data.keys())}")
        
        # Step 5: Create voxel grid
        print("\n5. Creating voxel grid...")
        voxel_grid = voxel_client.create_voxel_grid(
            model_id=model_id,
            resolution=1.0
        )
        print(f"   Grid dimensions: {voxel_grid.dims}")
        print(f"   Grid size: {voxel_grid.size} mm")
        
        # Step 6: Map signals to voxels
        print("\n6. Mapping signals to voxels...")
        voxel_grid = voxel_client.map_signals_to_voxels(
            model_id=model_id,
            voxel_grid=voxel_grid,
            sources=['hatching', 'laser'],
            method='nearest',
            n_workers=4
        )
        print(f"   Available signals: {voxel_grid.available_signals}")
        
        # Step 7: Access signal data
        print("\n7. Accessing signal data...")
        if 'power' in voxel_grid.available_signals:
            power_array = voxel_grid.get_signal_array('power', default=0.0)
            print(f"   Power signal shape: {power_array.shape}")
            print(f"   Power signal range: [{np.min(power_array):.2f}, {np.max(power_array):.2f}]")
        
        print("\n" + "=" * 60)
        print("Basic usage example completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("   Make sure MongoDB is running and contains data for the model.")
        return


if __name__ == "__main__":
    main()

