"""
Spark Utilities for Signal Mapping

Utilities for loading data from MongoDB into Spark DataFrames
and managing Spark sessions for signal mapping operations.
"""

import numpy as np
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

# Try to import PySpark
try:
    from pyspark.sql import SparkSession, DataFrame
    from pyspark.sql.functions import col, struct
    from pyspark.sql.types import StructType, StructField, FloatType, ArrayType

    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False

# Try to import pymongo (for fallback loading)
try:
    import pymongo
    from pymongo import MongoClient

    PYMONGO_AVAILABLE = True
except ImportError:
    pymongo = None
    MongoClient = None
    PYMONGO_AVAILABLE = False


def load_points_from_mongodb_to_spark(
    spark: SparkSession,
    mongo_uri: str,
    database: str,
    collection: str,
    query: Optional[Dict] = None,
    projection: Optional[Dict] = None,
) -> Optional[DataFrame]:
    """
    Load points data from MongoDB into Spark DataFrame.

    Args:
        spark: SparkSession instance
        mongo_uri: MongoDB connection URI
        database: Database name
        collection: Collection name
        query: Optional MongoDB query filter
        projection: Optional projection to select fields

    Returns:
        Spark DataFrame with point data, or None if error
    """
    if not PYSPARK_AVAILABLE:
        logger.error("PySpark not available")
        return None

    try:
        # Read from MongoDB using Spark MongoDB connector
        # Note: Requires spark-mongodb connector or pymongo with manual conversion
        df = (
            spark.read.format("mongo")
            .option("uri", mongo_uri)
            .option("database", database)
            .option("collection", collection)
            .load()
        )

        if query:
            # Apply query filter (simplified - may need custom implementation)
            logger.warning("Query filtering in Spark MongoDB connector may need custom implementation")

        return df
    except Exception as e:
        logger.error(f"Failed to load data from MongoDB: {e}", exc_info=True)
        # Fallback: Load via pymongo and convert
        try:
            return _load_via_pymongo(spark, mongo_uri, database, collection, query, projection)
        except Exception as e2:
            logger.error(f"Fallback loading also failed: {e2}", exc_info=True)
            return None


def _load_via_pymongo(
    spark: SparkSession,
    mongo_uri: str,
    database: str,
    collection: str,
    query: Optional[Dict] = None,
    projection: Optional[Dict] = None,
) -> DataFrame:
    """
    Load data via pymongo and convert to Spark DataFrame.

    This is a fallback method when Spark MongoDB connector is not available.
    """
    if not PYMONGO_AVAILABLE or pymongo is None or MongoClient is None:
        raise ImportError("pymongo required for MongoDB loading")

    # Connect to MongoDB
    client = MongoClient(mongo_uri)
    db = client[database]
    coll = db[collection]

    # Query data
    cursor = coll.find(query or {}, projection)
    data = list(cursor)

    if not data:
        logger.warning(f"No data found in {database}.{collection}")
        return spark.createDataFrame([], schema=_create_point_schema())

    # Convert to list of dictionaries
    rows = []
    for doc in data:
        row = {}
        if "point" in doc:
            point = doc["point"]
            row["x"] = float(point[0]) if isinstance(point, (list, tuple)) else float(point.get("x", 0))
            row["y"] = float(point[1]) if isinstance(point, (list, tuple)) else float(point.get("y", 0))
            row["z"] = float(point[2]) if isinstance(point, (list, tuple)) else float(point.get("z", 0))
        elif "x" in doc and "y" in doc and "z" in doc:
            row["x"] = float(doc["x"])
            row["y"] = float(doc["y"])
            row["z"] = float(doc["z"])

        # Add signals
        if "signals" in doc:
            signals = doc["signals"]
            for signal_name, signal_value in signals.items():
                row[signal_name] = float(signal_value) if signal_value is not None else 0.0

        rows.append(row)

    # Create DataFrame
    df = spark.createDataFrame(rows)
    return df


def _create_point_schema() -> StructType:
    """Create schema for point data."""
    return StructType(
        [
            StructField("x", FloatType(), True),
            StructField("y", FloatType(), True),
            StructField("z", FloatType(), True),
        ]
    )


def create_spark_session(
    app_name: str = "SignalMapping",
    master: Optional[str] = None,
    config: Optional[Dict[str, str]] = None,
) -> Optional[Any]:
    """
    Create or get Spark session.

    Args:
        app_name: Application name
        master: Spark master URL (e.g., 'local[*]', 'spark://host:port')
                If None, uses default from environment
        config: Additional Spark configuration

    Returns:
        SparkSession instance, or None if PySpark not available
    """
    if not PYSPARK_AVAILABLE:
        logger.warning("PySpark not available. Cannot create Spark session.")
        return None

    try:
        builder = SparkSession.builder.appName(app_name)

        if master:
            builder = builder.master(master)

        if config:
            for key, value in config.items():
                builder = builder.config(key, value)

        spark = builder.getOrCreate()
        logger.info(f"Spark session created: {spark.sparkContext.appName}")
        return spark
    except Exception as e:
        logger.error(f"Failed to create Spark session: {e}", exc_info=True)
        return None


def optimize_spark_for_signal_mapping(spark: SparkSession, num_partitions: Optional[int] = None) -> SparkSession:
    """
    Optimize Spark configuration for signal mapping operations.

    Args:
        spark: SparkSession instance
        num_partitions: Number of partitions (if None, auto-calculated)

    Returns:
        SparkSession with optimized configuration
    """
    if not PYSPARK_AVAILABLE:
        return spark

    # Set optimal configuration for signal mapping
    spark.conf.set("spark.sql.shuffle.partitions", str(num_partitions or 200))
    spark.conf.set("spark.sql.adaptive.enabled", "true")
    spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")

    logger.info("Spark configuration optimized for signal mapping")
    return spark
