"""
Production configuration management.

This module provides production configuration management with environment-based
settings, secrets management, and configuration validation.
"""

import os
import json
import yaml
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path


@dataclass
class ProductionConfig:
    """Production configuration settings."""

    # Environment settings
    environment: str = "production"  # 'development', 'staging', 'production'
    log_level: str = "INFO"  # 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    enable_metrics: bool = True  # Enable metrics collection
    enable_tracing: bool = True  # Enable distributed tracing
    enable_profiling: bool = False  # Enable performance profiling

    # Database configuration
    database_pool_size: int = 20  # Connection pool size
    database_max_overflow: int = 10  # Max overflow connections
    database_timeout: float = 30.0  # Connection timeout in seconds

    # Caching configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    redis_ttl: int = 3600  # Default TTL in seconds

    # Kafka configuration
    kafka_bootstrap_servers: List[str] = field(default_factory=lambda: ["localhost:9092"])
    kafka_max_poll_records: int = 500
    kafka_session_timeout_ms: int = 30000
    kafka_request_timeout_ms: int = 30000

    # Monitoring configuration
    metrics_port: int = 9090  # Prometheus metrics port
    health_check_port: int = 8080  # Health check endpoint port
    enable_health_checks: bool = True

    # Performance configuration
    worker_threads: int = 4  # Number of worker threads
    max_concurrent_requests: int = 100  # Max concurrent requests
    request_timeout: float = 60.0  # Request timeout in seconds

    # Secrets management
    secrets_manager: str = "env"  # 'env', 'vault', 'aws_secrets'
    secrets_path: Optional[str] = None  # Path to secrets (for file-based)

    # Feature flags
    enable_experimental_features: bool = False
    enable_debug_mode: bool = False

    @classmethod
    def from_env(cls) -> "ProductionConfig":
        """Load configuration from environment variables."""
        config_dict = {}

        # Environment settings
        config_dict["environment"] = os.getenv("AM_QADF_ENV", "production")
        config_dict["log_level"] = os.getenv("AM_QADF_LOG_LEVEL", "INFO")
        config_dict["enable_metrics"] = os.getenv("AM_QADF_ENABLE_METRICS", "true").lower() == "true"
        config_dict["enable_tracing"] = os.getenv("AM_QADF_ENABLE_TRACING", "true").lower() == "true"
        config_dict["enable_profiling"] = os.getenv("AM_QADF_ENABLE_PROFILING", "false").lower() == "true"

        # Database configuration
        config_dict["database_pool_size"] = int(os.getenv("AM_QADF_DB_POOL_SIZE", "20"))
        config_dict["database_max_overflow"] = int(os.getenv("AM_QADF_DB_MAX_OVERFLOW", "10"))
        config_dict["database_timeout"] = float(os.getenv("AM_QADF_DB_TIMEOUT", "30.0"))

        # Caching configuration
        config_dict["redis_host"] = os.getenv("AM_QADF_REDIS_HOST", "localhost")
        config_dict["redis_port"] = int(os.getenv("AM_QADF_REDIS_PORT", "6379"))
        config_dict["redis_db"] = int(os.getenv("AM_QADF_REDIS_DB", "0"))
        config_dict["redis_password"] = os.getenv("AM_QADF_REDIS_PASSWORD")
        config_dict["redis_ttl"] = int(os.getenv("AM_QADF_REDIS_TTL", "3600"))

        # Kafka configuration
        kafka_servers = os.getenv("AM_QADF_KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        config_dict["kafka_bootstrap_servers"] = [s.strip() for s in kafka_servers.split(",")]
        config_dict["kafka_max_poll_records"] = int(os.getenv("AM_QADF_KAFKA_MAX_POLL_RECORDS", "500"))
        config_dict["kafka_session_timeout_ms"] = int(os.getenv("AM_QADF_KAFKA_SESSION_TIMEOUT_MS", "30000"))
        config_dict["kafka_request_timeout_ms"] = int(os.getenv("AM_QADF_KAFKA_REQUEST_TIMEOUT_MS", "30000"))

        # Monitoring configuration
        config_dict["metrics_port"] = int(os.getenv("AM_QADF_METRICS_PORT", "9090"))
        config_dict["health_check_port"] = int(os.getenv("AM_QADF_HEALTH_CHECK_PORT", "8080"))
        config_dict["enable_health_checks"] = os.getenv("AM_QADF_ENABLE_HEALTH_CHECKS", "true").lower() == "true"

        # Performance configuration
        config_dict["worker_threads"] = int(os.getenv("AM_QADF_WORKER_THREADS", "4"))
        config_dict["max_concurrent_requests"] = int(os.getenv("AM_QADF_MAX_CONCURRENT_REQUESTS", "100"))
        config_dict["request_timeout"] = float(os.getenv("AM_QADF_REQUEST_TIMEOUT", "60.0"))

        # Secrets management
        config_dict["secrets_manager"] = os.getenv("AM_QADF_SECRETS_MANAGER", "env")
        config_dict["secrets_path"] = os.getenv("AM_QADF_SECRETS_PATH")

        # Feature flags
        config_dict["enable_experimental_features"] = os.getenv("AM_QADF_ENABLE_EXPERIMENTAL", "false").lower() == "true"
        config_dict["enable_debug_mode"] = os.getenv("AM_QADF_DEBUG", "false").lower() == "true"

        return cls(**config_dict)

    @classmethod
    def from_file(cls, config_path: str) -> "ProductionConfig":
        """Load configuration from YAML/JSON file."""
        config_path_obj = Path(config_path)

        if not config_path_obj.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path_obj, "r") as f:
            if config_path_obj.suffix in [".yaml", ".yml"]:
                try:
                    import yaml

                    config_dict = yaml.safe_load(f)
                except ImportError:
                    raise ImportError("PyYAML is required to load YAML configuration files. Install with: pip install pyyaml")
            elif config_path_obj.suffix == ".json":
                config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_path_obj.suffix}")

        if not isinstance(config_dict, dict):
            raise ValueError("Configuration file must contain a dictionary/object")

        # Filter config_dict to only include valid fields
        valid_fields = {k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__}
        return cls(**valid_fields)

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate configuration settings."""
        errors = []

        # Environment validation
        if self.environment not in ["development", "staging", "production"]:
            errors.append(f"Invalid environment: {self.environment}. Must be one of: development, staging, production")

        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            errors.append(f"Invalid log level: {self.log_level}. Must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL")

        # Database configuration validation
        if self.database_pool_size < 1:
            errors.append("database_pool_size must be at least 1")
        if self.database_max_overflow < 0:
            errors.append("database_max_overflow must be non-negative")
        if self.database_timeout <= 0:
            errors.append("database_timeout must be positive")

        # Redis configuration validation
        if self.redis_port < 1 or self.redis_port > 65535:
            errors.append("redis_port must be between 1 and 65535")
        if self.redis_db < 0:
            errors.append("redis_db must be non-negative")
        if self.redis_ttl < 0:
            errors.append("redis_ttl must be non-negative")

        # Kafka configuration validation
        if not self.kafka_bootstrap_servers:
            errors.append("kafka_bootstrap_servers cannot be empty")
        if self.kafka_max_poll_records < 1:
            errors.append("kafka_max_poll_records must be at least 1")
        if self.kafka_session_timeout_ms <= 0:
            errors.append("kafka_session_timeout_ms must be positive")
        if self.kafka_request_timeout_ms <= 0:
            errors.append("kafka_request_timeout_ms must be positive")

        # Monitoring configuration validation
        if self.metrics_port < 1 or self.metrics_port > 65535:
            errors.append("metrics_port must be between 1 and 65535")
        if self.health_check_port < 1 or self.health_check_port > 65535:
            errors.append("health_check_port must be between 1 and 65535")
        if self.metrics_port == self.health_check_port:
            errors.append("metrics_port and health_check_port cannot be the same")

        # Performance configuration validation
        if self.worker_threads < 1:
            errors.append("worker_threads must be at least 1")
        if self.max_concurrent_requests < 1:
            errors.append("max_concurrent_requests must be at least 1")
        if self.request_timeout <= 0:
            errors.append("request_timeout must be positive")

        # Secrets management validation
        if self.secrets_manager not in ["env", "vault", "aws_secrets", "file"]:
            errors.append(f"Invalid secrets_manager: {self.secrets_manager}. Must be one of: env, vault, aws_secrets, file")
        if self.secrets_manager == "file" and not self.secrets_path:
            errors.append("secrets_path must be provided when secrets_manager is 'file'")

        return len(errors) == 0, errors

    def get_secret(self, secret_name: str) -> Optional[str]:
        """Get secret value from secrets manager."""
        if self.secrets_manager == "env":
            # Try AM_QADF_SECRET_ prefix first, then direct name
            env_key = f"AM_QADF_SECRET_{secret_name.upper()}"
            return os.getenv(env_key) or os.getenv(secret_name)

        elif self.secrets_manager == "file" and self.secrets_path:
            # Load secrets from file (JSON format)
            secrets_path = Path(self.secrets_path)
            if secrets_path.exists():
                with open(secrets_path, "r") as f:
                    secrets = json.load(f)
                    return secrets.get(secret_name)

        elif self.secrets_manager == "vault":
            # TODO: Implement HashiCorp Vault integration
            raise NotImplementedError("Vault secrets manager not yet implemented")

        elif self.secrets_manager == "aws_secrets":
            # TODO: Implement AWS Secrets Manager integration
            raise NotImplementedError("AWS Secrets Manager not yet implemented")

        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        config_dict = asdict(self)
        # Don't expose sensitive information in to_dict
        if config_dict.get("redis_password"):
            config_dict["redis_password"] = "***"
        return config_dict

    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"ProductionConfig(environment='{self.environment}', log_level='{self.log_level}', ...)"
