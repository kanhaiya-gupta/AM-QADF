"""
Example Usage of Infrastructure Layer

Demonstrates how to use the infrastructure layer for database connections.
"""

from src.infrastructure.database import get_connection_manager, check_all_connections
from src.infrastructure.config import get_database_configs


def example_basic_usage():
    """Basic usage example."""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)

    # Get connection manager (loads from development.env by default)
    manager = get_connection_manager(env_name="development")

    # Get MongoDB client
    mongodb_client = manager.get_mongodb_client()

    if mongodb_client:
        # Check connection
        if mongodb_client.is_connected():
            print("✅ MongoDB connected successfully")

            # Get collection
            collection = mongodb_client.get_collection("hatching_layers")
            print(f"✅ Accessing collection: {collection.name}")

            # Example query
            count = collection.count_documents({})
            print(f"✅ Collection has {count} documents")
        else:
            print("❌ MongoDB not connected")
    else:
        print("❌ MongoDB client not available")


def example_health_check():
    """Health check example."""
    print("\n" + "=" * 60)
    print("Example 2: Health Check")
    print("=" * 60)

    # Check all connections
    health_status = check_all_connections(env_name="development")

    for db_type, status in health_status.items():
        print(f"\n{db_type.upper()}:")
        print(f"  Status: {status.get('status', 'unknown')}")
        if "error" in status:
            print(f"  Error: {status['error']}")
        if "server_version" in status:
            print(f"  Version: {status['server_version']}")


def example_config_loading():
    """Configuration loading example."""
    print("\n" + "=" * 60)
    print("Example 3: Configuration Loading")
    print("=" * 60)

    # Load database configurations
    config = get_database_configs(env_name="development")

    if config.mongodb:
        print(f"\nMongoDB Config:")
        print(f"  URL: {config.mongodb.url}")
        print(f"  Database: {config.mongodb.database}")
        print(f"  Host: {config.mongodb.host}")
        print(f"  Port: {config.mongodb.port}")
        print(f"  Max Pool Size: {config.mongodb.max_pool_size}")


def example_with_query_client():
    """Example using with AM-QADF query clients."""
    print("\n" + "=" * 60)
    print("Example 4: Using with AM-QADF Query Clients")
    print("=" * 60)

    try:
        from src.infrastructure.database import get_connection_manager
        from am_qadf.query import UnifiedQueryClient

        # Get MongoDB connection
        manager = get_connection_manager()
        mongodb_client = manager.get_mongodb_client()

        if mongodb_client:
            # Create unified query client
            query_client = UnifiedQueryClient(mongo_client=mongodb_client)
            print("✅ UnifiedQueryClient created with MongoDB connection")

            # Example: Query data
            # result = query_client.query(
            #     model_id="my_model",
            #     sources=['hatching', 'laser']
            # )
            print("✅ Ready to query data")
        else:
            print("❌ MongoDB client not available")

    except ImportError as e:
        print(f"⚠️  Could not import AM-QADF modules: {e}")
        print("   This is expected if running infrastructure examples standalone")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Infrastructure Layer - Usage Examples")
    print("=" * 60)

    # Run examples
    example_basic_usage()
    example_health_check()
    example_config_loading()
    example_with_query_client()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
