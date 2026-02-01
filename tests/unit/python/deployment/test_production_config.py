"""
Unit tests for ProductionConfig.

Tests for production configuration management.
"""

import pytest
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open

from am_qadf.deployment.production_config import ProductionConfig


class TestProductionConfig:
    """Test suite for ProductionConfig class."""

    @pytest.mark.unit
    def test_production_config_creation_defaults(self):
        """Test creating ProductionConfig with defaults."""
        config = ProductionConfig()

        assert config.environment == "production"
        assert config.log_level == "INFO"
        assert config.enable_metrics is True
        assert config.enable_tracing is True
        assert config.enable_profiling is False
        assert config.database_pool_size == 20
        assert config.database_max_overflow == 10
        assert config.database_timeout == 30.0
        assert config.redis_host == "localhost"
        assert config.redis_port == 6379
        assert config.worker_threads == 4
        assert config.max_concurrent_requests == 100

    @pytest.mark.unit
    def test_production_config_creation_custom(self):
        """Test creating ProductionConfig with custom values."""
        config = ProductionConfig(
            environment="development",
            log_level="DEBUG",
            enable_metrics=False,
            enable_tracing=False,
            database_pool_size=50,
            redis_host="redis.example.com",
            redis_port=6380,
            worker_threads=8,
            max_concurrent_requests=200,
        )

        assert config.environment == "development"
        assert config.log_level == "DEBUG"
        assert config.enable_metrics is False
        assert config.enable_tracing is False
        assert config.database_pool_size == 50
        assert config.redis_host == "redis.example.com"
        assert config.redis_port == 6380
        assert config.worker_threads == 8
        assert config.max_concurrent_requests == 200

    @pytest.mark.unit
    def test_from_env(self):
        """Test loading configuration from environment variables."""
        env_vars = {
            "AM_QADF_ENV": "staging",
            "AM_QADF_LOG_LEVEL": "WARNING",
            "AM_QADF_ENABLE_METRICS": "false",
            "AM_QADF_DB_POOL_SIZE": "30",
            "AM_QADF_REDIS_HOST": "redis.staging.com",
            "AM_QADF_REDIS_PORT": "6380",
            "AM_QADF_WORKER_THREADS": "6",
            "AM_QADF_KAFKA_BOOTSTRAP_SERVERS": "kafka1:9092,kafka2:9092",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            config = ProductionConfig.from_env()

            assert config.environment == "staging"
            assert config.log_level == "WARNING"
            assert config.enable_metrics is False
            assert config.database_pool_size == 30
            assert config.redis_host == "redis.staging.com"
            assert config.redis_port == 6380
            assert config.worker_threads == 6
            assert len(config.kafka_bootstrap_servers) == 2
            assert "kafka1:9092" in config.kafka_bootstrap_servers
            assert "kafka2:9092" in config.kafka_bootstrap_servers

    @pytest.mark.unit
    def test_from_file_json(self):
        """Test loading configuration from JSON file."""
        config_data = {
            "environment": "production",
            "log_level": "INFO",
            "enable_metrics": True,
            "database_pool_size": 25,
            "redis_host": "redis.prod.com",
            "worker_threads": 8,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            config = ProductionConfig.from_file(temp_path)

            assert config.environment == "production"
            assert config.log_level == "INFO"
            assert config.enable_metrics is True
            assert config.database_pool_size == 25
            assert config.redis_host == "redis.prod.com"
            assert config.worker_threads == 8
        finally:
            os.unlink(temp_path)

    @pytest.mark.unit
    def test_from_file_yaml(self):
        """Test loading configuration from YAML file."""
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML not available")

        config_data = {
            "environment": "development",
            "log_level": "DEBUG",
            "enable_profiling": True,
            "worker_threads": 2,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            config = ProductionConfig.from_file(temp_path)

            assert config.environment == "development"
            assert config.log_level == "DEBUG"
            assert config.enable_profiling is True
            assert config.worker_threads == 2
        finally:
            os.unlink(temp_path)

    @pytest.mark.unit
    def test_from_file_not_found(self):
        """Test loading configuration from non-existent file."""
        with pytest.raises(FileNotFoundError):
            ProductionConfig.from_file("/nonexistent/path/config.json")

    @pytest.mark.unit
    def test_from_file_invalid_format(self):
        """Test loading configuration from unsupported format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("not a valid config format")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported configuration file format"):
                ProductionConfig.from_file(temp_path)
        finally:
            os.unlink(temp_path)

    @pytest.mark.unit
    def test_validate_success(self):
        """Test configuration validation with valid values."""
        config = ProductionConfig(
            environment="production",
            log_level="INFO",
            database_pool_size=20,
            redis_port=6379,
            metrics_port=9090,
            health_check_port=8080,
            worker_threads=4,
            secrets_manager="env",
        )

        is_valid, errors = config.validate()

        assert is_valid is True
        assert len(errors) == 0

    @pytest.mark.unit
    def test_validate_invalid_environment(self):
        """Test configuration validation with invalid environment."""
        config = ProductionConfig(environment="invalid")

        is_valid, errors = config.validate()

        assert is_valid is False
        assert len(errors) > 0
        assert any("environment" in error.lower() for error in errors)

    @pytest.mark.unit
    def test_validate_invalid_log_level(self):
        """Test configuration validation with invalid log level."""
        config = ProductionConfig(log_level="INVALID")

        is_valid, errors = config.validate()

        assert is_valid is False
        assert any("log level" in error.lower() for error in errors)

    @pytest.mark.unit
    def test_validate_invalid_database_pool_size(self):
        """Test configuration validation with invalid pool size."""
        config = ProductionConfig(database_pool_size=0)

        is_valid, errors = config.validate()

        assert is_valid is False
        assert any("pool_size" in error.lower() for error in errors)

    @pytest.mark.unit
    def test_validate_invalid_ports(self):
        """Test configuration validation with invalid ports."""
        config = ProductionConfig(metrics_port=0, health_check_port=70000)

        is_valid, errors = config.validate()

        assert is_valid is False
        assert len(errors) >= 2

    @pytest.mark.unit
    def test_validate_same_ports(self):
        """Test configuration validation with same metrics and health check ports."""
        config = ProductionConfig(metrics_port=8080, health_check_port=8080)

        is_valid, errors = config.validate()

        assert is_valid is False
        assert any("cannot be the same" in error.lower() for error in errors)

    @pytest.mark.unit
    def test_validate_empty_kafka_servers(self):
        """Test configuration validation with empty Kafka servers."""
        config = ProductionConfig(kafka_bootstrap_servers=[])

        is_valid, errors = config.validate()

        assert is_valid is False
        assert any("kafka" in error.lower() for error in errors)

    @pytest.mark.unit
    def test_get_secret_env(self):
        """Test getting secret from environment."""
        with patch.dict(os.environ, {"AM_QADF_SECRET_API_KEY": "secret123"}):
            config = ProductionConfig(secrets_manager="env")
            secret = config.get_secret("api_key")

            assert secret == "secret123"

    @pytest.mark.unit
    def test_get_secret_env_direct(self):
        """Test getting secret from environment with direct name."""
        with patch.dict(os.environ, {"DATABASE_PASSWORD": "pass123"}):
            config = ProductionConfig(secrets_manager="env")
            secret = config.get_secret("DATABASE_PASSWORD")

            assert secret == "pass123"

    @pytest.mark.unit
    def test_get_secret_file(self):
        """Test getting secret from file."""
        secrets_data = {
            "api_key": "secret_from_file",
            "database_password": "db_pass",
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(secrets_data, f)
            temp_path = f.name

        try:
            config = ProductionConfig(secrets_manager="file", secrets_path=temp_path)
            secret = config.get_secret("api_key")

            assert secret == "secret_from_file"
        finally:
            os.unlink(temp_path)

    @pytest.mark.unit
    def test_get_secret_file_not_found(self):
        """Test getting secret from non-existent file."""
        config = ProductionConfig(secrets_manager="file", secrets_path="/nonexistent/secrets.json")
        secret = config.get_secret("api_key")

        assert secret is None

    @pytest.mark.unit
    def test_get_secret_vault_not_implemented(self):
        """Test getting secret from Vault (not implemented)."""
        config = ProductionConfig(secrets_manager="vault")

        with pytest.raises(NotImplementedError, match="Vault secrets manager"):
            config.get_secret("api_key")

    @pytest.mark.unit
    def test_get_secret_aws_not_implemented(self):
        """Test getting secret from AWS Secrets Manager (not implemented)."""
        config = ProductionConfig(secrets_manager="aws_secrets")

        with pytest.raises(NotImplementedError, match="AWS Secrets Manager"):
            config.get_secret("api_key")

    @pytest.mark.unit
    def test_to_dict(self):
        """Test converting configuration to dictionary."""
        config = ProductionConfig(
            environment="production",
            redis_password="secret123",
        )

        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["environment"] == "production"
        assert config_dict["redis_password"] == "***"  # Sensitive data masked

    @pytest.mark.unit
    def test_repr(self):
        """Test string representation of configuration."""
        config = ProductionConfig(environment="development", log_level="DEBUG")
        repr_str = repr(config)

        assert "ProductionConfig" in repr_str
        assert "environment" in repr_str or "development" in repr_str
