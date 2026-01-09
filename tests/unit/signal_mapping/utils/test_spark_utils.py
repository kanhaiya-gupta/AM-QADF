"""
Unit tests for Spark utilities.

Tests for Spark session management and MongoDB loading (with mocking).
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from am_qadf.signal_mapping.utils.spark_utils import (
    load_points_from_mongodb_to_spark,
    optimize_spark_for_signal_mapping,
    PYSPARK_AVAILABLE,
    _load_via_pymongo,
    _create_point_schema,
)


class TestLoadPointsFromMongoDBToSpark:
    """Test suite for load_points_from_mongodb_to_spark function."""

    @pytest.mark.unit
    @patch("am_qadf.signal_mapping.utils.spark_utils.PYSPARK_AVAILABLE", False)
    def test_load_points_pyspark_not_available(self):
        """Test loading when PySpark is not available."""
        mock_spark = Mock()

        result = load_points_from_mongodb_to_spark(mock_spark, "mongodb://localhost:27017", "test_db", "test_collection")

        assert result is None

    @pytest.mark.unit
    @patch("am_qadf.signal_mapping.utils.spark_utils.PYSPARK_AVAILABLE", True)
    @patch("am_qadf.signal_mapping.utils.spark_utils.SparkSession")
    def test_load_points_success(self, mock_spark_session_class):
        """Test successful loading from MongoDB."""
        mock_spark = Mock()
        mock_df = Mock()
        mock_reader = Mock()
        mock_reader.format.return_value = mock_reader
        mock_reader.option.return_value = mock_reader
        mock_reader.load.return_value = mock_df
        mock_spark.read = mock_reader

        result = load_points_from_mongodb_to_spark(mock_spark, "mongodb://localhost:27017", "test_db", "test_collection")

        assert result is mock_df
        mock_reader.format.assert_called_with("mongo")

    @pytest.mark.unit
    @patch("am_qadf.signal_mapping.utils.spark_utils.PYSPARK_AVAILABLE", True)
    @patch("am_qadf.signal_mapping.utils.spark_utils.SparkSession")
    def test_load_points_with_query(self, mock_spark_session_class):
        """Test loading with query filter."""
        mock_spark = Mock()
        mock_df = Mock()
        mock_reader = Mock()
        mock_reader.format.return_value = mock_reader
        mock_reader.option.return_value = mock_reader
        mock_reader.load.return_value = mock_df
        mock_spark.read = mock_reader

        query = {"model_id": "test_123"}

        result = load_points_from_mongodb_to_spark(
            mock_spark,
            "mongodb://localhost:27017",
            "test_db",
            "test_collection",
            query=query,
        )

        assert result is mock_df

    @pytest.mark.unit
    @patch("am_qadf.signal_mapping.utils.spark_utils.PYSPARK_AVAILABLE", True)
    @patch("am_qadf.signal_mapping.utils.spark_utils.SparkSession")
    @patch("am_qadf.signal_mapping.utils.spark_utils._load_via_pymongo")
    def test_load_points_fallback_to_pymongo(self, mock_load_pymongo, mock_spark_session_class):
        """Test fallback to pymongo when Spark connector fails."""
        mock_spark = Mock()
        mock_df = Mock()
        mock_reader = Mock()
        mock_reader.format.return_value = mock_reader
        mock_reader.option.return_value = mock_reader
        mock_reader.load.side_effect = Exception("Connection failed")
        mock_spark.read = mock_reader
        mock_load_pymongo.return_value = mock_df

        result = load_points_from_mongodb_to_spark(mock_spark, "mongodb://localhost:27017", "test_db", "test_collection")

        assert result is mock_df
        mock_load_pymongo.assert_called_once()


class TestLoadViaPymongo:
    """Test suite for _load_via_pymongo function."""

    @pytest.mark.unit
    @patch("am_qadf.signal_mapping.utils.spark_utils.MongoClient")
    @patch("am_qadf.signal_mapping.utils.spark_utils.PYSPARK_AVAILABLE", True)
    @patch("am_qadf.signal_mapping.utils.spark_utils.PYMONGO_AVAILABLE", True)
    def test_load_via_pymongo_success(self, mock_mongo_client_class):
        """Test successful loading via pymongo."""
        mock_spark = Mock()
        mock_df = Mock()
        mock_spark.createDataFrame.return_value = mock_df

        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_client.__getitem__.return_value = mock_db
        mock_db.__getitem__.return_value = mock_collection
        mock_mongo_client_class.return_value = mock_client

        # Mock cursor with data
        test_data = [
            {"x": 1.0, "y": 2.0, "z": 3.0, "signals": {"power": 200.0}},
            {"x": 4.0, "y": 5.0, "z": 6.0, "signals": {"power": 250.0}},
        ]
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = MagicMock(return_value=iter(test_data))
        mock_collection.find.return_value = mock_cursor

        result = _load_via_pymongo(mock_spark, "mongodb://localhost:27017", "test_db", "test_collection")

        assert result is mock_df
        mock_spark.createDataFrame.assert_called_once()

    @pytest.mark.unit
    @patch("am_qadf.signal_mapping.utils.spark_utils.MongoClient")
    @patch("am_qadf.signal_mapping.utils.spark_utils.PYSPARK_AVAILABLE", True)
    @patch("am_qadf.signal_mapping.utils.spark_utils.PYMONGO_AVAILABLE", True)
    def test_load_via_pymongo_empty_collection(self, mock_mongo_client_class):
        """Test loading from empty collection."""
        mock_spark = Mock()
        mock_df = Mock()
        mock_spark.createDataFrame.return_value = mock_df

        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_client.__getitem__.return_value = mock_db
        mock_db.__getitem__.return_value = mock_collection
        mock_mongo_client_class.return_value = mock_client

        # Mock cursor with no data
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = MagicMock(return_value=iter([]))
        mock_collection.find.return_value = mock_cursor

        result = _load_via_pymongo(mock_spark, "mongodb://localhost:27017", "test_db", "test_collection")

        assert result is mock_df
        # Should create empty DataFrame with schema
        mock_spark.createDataFrame.assert_called_once()

    @pytest.mark.unit
    @patch("am_qadf.signal_mapping.utils.spark_utils.PYMONGO_AVAILABLE", False)
    def test_load_via_pymongo_import_error(self):
        """Test that ImportError is raised when pymongo is not available."""
        mock_spark = Mock()

        with pytest.raises(ImportError, match="pymongo required"):
            _load_via_pymongo(mock_spark, "mongodb://localhost:27017", "test_db", "test_collection")


class TestCreatePointSchema:
    """Test suite for _create_point_schema function."""

    @pytest.mark.unit
    @patch("am_qadf.signal_mapping.utils.spark_utils.PYSPARK_AVAILABLE", True)
    def test_create_point_schema(self):
        """Test creating point schema."""
        schema = _create_point_schema()

        assert schema is not None
        # Schema should have x, y, z fields
        field_names = [field.name for field in schema.fields]
        assert "x" in field_names
        assert "y" in field_names
        assert "z" in field_names


class TestOptimizeSparkForSignalMapping:
    """Test suite for optimize_spark_for_signal_mapping function."""

    @pytest.mark.unit
    @patch("am_qadf.signal_mapping.utils.spark_utils.PYSPARK_AVAILABLE", False)
    def test_optimize_spark_pyspark_not_available(self):
        """Test optimization when PySpark is not available."""
        mock_spark = Mock()

        result = optimize_spark_for_signal_mapping(mock_spark)

        assert result is mock_spark
        # Should not modify spark configuration

    @pytest.mark.unit
    @patch("am_qadf.signal_mapping.utils.spark_utils.PYSPARK_AVAILABLE", True)
    def test_optimize_spark_default(self):
        """Test Spark optimization with default parameters."""
        mock_spark = Mock()
        mock_spark.conf = Mock()
        mock_spark.conf.set = Mock()

        result = optimize_spark_for_signal_mapping(mock_spark)

        assert result is mock_spark
        # Should set configuration
        assert mock_spark.conf.set.called

    @pytest.mark.unit
    @patch("am_qadf.signal_mapping.utils.spark_utils.PYSPARK_AVAILABLE", True)
    def test_optimize_spark_custom_partitions(self):
        """Test Spark optimization with custom number of partitions."""
        mock_spark = Mock()
        mock_spark.conf = Mock()
        mock_spark.conf.set = Mock()

        result = optimize_spark_for_signal_mapping(mock_spark, num_partitions=500)

        assert result is mock_spark
        # Should set custom partition count
        # Check that set was called with the partition configuration
        assert mock_spark.conf.set.called

    @pytest.mark.unit
    @patch("am_qadf.signal_mapping.utils.spark_utils.PYSPARK_AVAILABLE", True)
    @patch("am_qadf.signal_mapping.utils.spark_utils.logger")
    def test_optimize_spark_logging(self, mock_logger):
        """Test that optimization logs information."""
        mock_spark = Mock()
        mock_spark.conf = Mock()
        mock_spark.conf.set = Mock()

        optimize_spark_for_signal_mapping(mock_spark)

        # Should log optimization
        mock_logger.info.assert_called()


class TestPysparkAvailable:
    """Test suite for PYSPARK_AVAILABLE flag."""

    @pytest.mark.unit
    def test_pyspark_available_flag(self):
        """Test that PYSPARK_AVAILABLE flag is set correctly."""
        # This flag depends on whether pyspark is installed
        # We just verify it's a boolean
        assert isinstance(PYSPARK_AVAILABLE, bool)
