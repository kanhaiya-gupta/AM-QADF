"""
Database Configuration

Parses environment variables into database connection configurations.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from .env_loader import get_env_var, get_env_bool, get_env_int, get_env_float


@dataclass
class MongoDBConfig:
    """MongoDB connection configuration."""

    url: str
    database: str
    username: Optional[str] = None
    password: Optional[str] = None
    host: Optional[str] = None
    port: int = 27017

    # Connection pool settings
    max_pool_size: int = 100
    min_pool_size: int = 0
    max_idle_time_ms: int = 30000
    connect_timeout_ms: int = 20000
    socket_timeout_ms: int = 20000
    server_selection_timeout_ms: int = 30000

    # Retry settings
    retry_writes: bool = True
    retry_reads: bool = True
    max_retry_attempts: int = 3

    # SSL/TLS
    ssl: bool = False
    ssl_certfile: Optional[str] = None
    ssl_keyfile: Optional[str] = None
    ssl_ca_certs: Optional[str] = None

    @classmethod
    def from_env(cls) -> "MongoDBConfig":
        """Create MongoDB config from environment variables."""
        # Try to get full URL first
        url = get_env_var("MONGODB_URL")

        if url:
            # Parse URL if provided
            return cls(
                url=url,
                database=get_env_var("MONGO_DATABASE", "am_qadf_data"),
                username=get_env_var("MONGO_ROOT_USERNAME"),
                password=get_env_var("MONGO_ROOT_PASSWORD"),
            )

        # Otherwise build from individual components
        host = get_env_var("MONGODB_HOST", "localhost")
        port = get_env_int("MONGODB_PORT", 27017)
        database = get_env_var("MONGO_DATABASE", "am_qadf_data")
        username = get_env_var("MONGO_ROOT_USERNAME")
        password = get_env_var("MONGO_ROOT_PASSWORD")

        # Build URL
        if username and password:
            # For root/admin users, authenticate against admin database
            url = f"mongodb://{username}:{password}@{host}:{port}/{database}?authSource=admin"
        else:
            url = f"mongodb://{host}:{port}/{database}"

        return cls(
            url=url,
            database=database,
            username=username,
            password=password,
            host=host,
            port=port,
            max_pool_size=get_env_int("MONGODB_MAX_POOL_SIZE", 100),
            min_pool_size=get_env_int("MONGODB_MIN_POOL_SIZE", 0),
            connect_timeout_ms=get_env_int("MONGODB_CONNECT_TIMEOUT_MS", 20000),
            socket_timeout_ms=get_env_int("MONGODB_SOCKET_TIMEOUT_MS", 20000),
            ssl=get_env_bool("MONGODB_SSL", False),
        )


@dataclass
class CassandraConfig:
    """Cassandra connection configuration."""

    hosts: List[str]
    keyspace: str
    port: int = 9042
    username: Optional[str] = None
    password: Optional[str] = None
    cluster_name: Optional[str] = None
    datacenter: Optional[str] = None

    # Connection pool
    max_connections: int = 50
    max_requests_per_connection: int = 32768
    connection_timeout: int = 10
    request_timeout: int = 10

    # SSL/TLS
    ssl_enabled: bool = False
    ssl_certfile: Optional[str] = None
    ssl_keyfile: Optional[str] = None
    ssl_ca_certs: Optional[str] = None

    @classmethod
    def from_env(cls) -> "CassandraConfig":
        """Create Cassandra config from environment variables."""
        hosts_str = get_env_var("CASSANDRA_HOSTS", "localhost")
        hosts = [h.strip() for h in hosts_str.split(",")]

        return cls(
            hosts=hosts,
            port=get_env_int("CASSANDRA_PORT", 9042),
            keyspace=get_env_var("CASSANDRA_KEYSPACE", "am_qadf_timeseries"),
            username=get_env_var("CASSANDRA_USERNAME"),
            password=get_env_var("CASSANDRA_PASSWORD"),
            cluster_name=get_env_var("CASSANDRA_CLUSTER_NAME"),
            datacenter=get_env_var("CASSANDRA_DATACENTER"),
            max_connections=get_env_int("CASSANDRA_MAX_CONNECTIONS", 50),
            connection_timeout=get_env_int("CASSANDRA_CONNECTION_TIMEOUT", 10),
            request_timeout=get_env_int("CASSANDRA_REQUEST_TIMEOUT", 10),
            ssl_enabled=get_env_bool("CASSANDRA_SSL_ENABLED", False),
        )


@dataclass
class Neo4jConfig:
    """Neo4j connection configuration."""

    uri: str
    username: str
    password: str
    database: str = "neo4j"

    # Connection pool
    max_connection_lifetime: int = 3600
    max_connection_pool_size: int = 50
    connection_acquisition_timeout: int = 60
    connection_timeout: int = 30

    # SSL/TLS
    encrypted: bool = False
    trust: str = "TRUST_SYSTEM_CA_SIGNED_CERTIFICATES"

    @classmethod
    def from_env(cls) -> "Neo4jConfig":
        """Create Neo4j config from environment variables."""
        return cls(
            uri=get_env_var("NEO4J_URI", "bolt://localhost:7687"),
            username=get_env_var("NEO4J_USERNAME", "neo4j"),
            password=get_env_var("NEO4J_PASSWORD", "password"),
            database=get_env_var("NEO4J_DEFAULT_DATABASE", "neo4j"),
            max_connection_lifetime=get_env_int("NEO4J_MAX_CONNECTION_LIFETIME", 3600),
            max_connection_pool_size=get_env_int("NEO4J_MAX_CONNECTION_POOL_SIZE", 50),
            encrypted=get_env_bool("NEO4J_ENCRYPTED", False),
        )


@dataclass
class ElasticsearchConfig:
    """Elasticsearch connection configuration."""

    url: str
    host: str
    port: int = 9200
    username: Optional[str] = None
    password: Optional[str] = None

    # Connection settings
    max_connections: int = 20
    max_retries: int = 3
    timeout: int = 30
    retry_on_timeout: bool = True

    # SSL/TLS
    verify_certs: bool = True

    @classmethod
    def from_env(cls) -> "ElasticsearchConfig":
        """Create Elasticsearch config from environment variables."""
        url = get_env_var("ELASTICSEARCH_URL", "http://localhost:9200")
        host = get_env_var("ELASTICSEARCH_HOST", "localhost")
        port = get_env_int("ELASTICSEARCH_PORT", 9200)

        return cls(
            url=url,
            host=host,
            port=port,
            username=get_env_var("ELASTICSEARCH_USERNAME"),
            password=get_env_var("ELASTICSEARCH_PASSWORD"),
            max_connections=get_env_int("ELASTICSEARCH_MAX_CONNECTIONS", 20),
            max_retries=get_env_int("ELASTICSEARCH_MAX_RETRIES", 3),
            timeout=get_env_int("ELASTICSEARCH_TIMEOUT", 30),
            retry_on_timeout=get_env_bool("ELASTICSEARCH_RETRY_ON_TIMEOUT", True),
            verify_certs=get_env_bool("ELASTICSEARCH_VERIFY_CERTS", True),
        )


@dataclass
class DatabaseConfig:
    """Container for all database configurations."""

    mongodb: Optional[MongoDBConfig] = None
    cassandra: Optional[CassandraConfig] = None
    neo4j: Optional[Neo4jConfig] = None
    elasticsearch: Optional[ElasticsearchConfig] = None

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Load all database configs from environment variables."""
        return cls(
            mongodb=MongoDBConfig.from_env(),
            cassandra=CassandraConfig.from_env(),
            neo4j=Neo4jConfig.from_env(),
            elasticsearch=ElasticsearchConfig.from_env(),
        )


def get_database_configs(env_name: str = "development") -> DatabaseConfig:
    """
    Get database configurations for specified environment.

    Args:
        env_name: Environment name (development, production, etc.)

    Returns:
        DatabaseConfig with all database configurations
    """
    from .env_loader import load_env_file

    # Load environment variables
    load_env_file(env_name)

    # Parse configurations
    return DatabaseConfig.from_env()
