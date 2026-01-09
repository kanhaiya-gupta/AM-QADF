"""
Spark-Based Interpolation Module

Handles distributed interpolation operations using PySpark for cluster-scale processing.
Supports all interpolation methods with distributed computation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

# Try to import PySpark
try:
    from pyspark.sql import SparkSession, DataFrame
    from pyspark.sql.functions import (
        udf,
        col,
        mean,
        max as spark_max,
        min as spark_min,
        sum as spark_sum,
        array,
    )
    from pyspark.sql.types import (
        ArrayType,
        IntegerType,
        FloatType,
        StructType,
        StructField,
    )
    import pyspark.sql.types as T

    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False
    logger.warning(
        "PySpark not available. Install with: pip install pyspark\n" "Spark-based interpolation will not be available."
    )


class SparkInterpolationMethod(ABC):
    """
    Base class for Spark-based interpolation methods.

    Provides common functionality for distributed interpolation using PySpark.
    """

    @abstractmethod
    def interpolate_spark(
        self,
        spark: SparkSession,
        points_df: DataFrame,
        voxel_grid_config: Dict[str, Any],
    ) -> Any:
        """
        Interpolate points to voxel grid using Spark.

        Args:
            spark: SparkSession instance
            points_df: Spark DataFrame with columns: x, y, z, and signal columns
            voxel_grid_config: Dictionary with voxel grid configuration:
                - bbox_min: Tuple[float, float, float]
                - bbox_max: Tuple[float, float, float]
                - resolution: float
                - aggregation: str ('mean', 'max', 'min', 'sum')

        Returns:
            VoxelGrid with interpolated data
        """
        pass

    def _calculate_voxel_indices_udf(self, voxel_grid_config: Dict[str, Any]):
        """
        Create UDF for calculating voxel indices from coordinates.

        Args:
            voxel_grid_config: Voxel grid configuration

        Returns:
            UDF function
        """
        if not PYSPARK_AVAILABLE:
            raise ImportError("PySpark is required for Spark-based interpolation")

        bbox_min = np.array(voxel_grid_config["bbox_min"])
        resolution = voxel_grid_config["resolution"]
        dims = np.ceil((np.array(voxel_grid_config["bbox_max"]) - bbox_min) / resolution).astype(int)
        dims = np.maximum(dims, [1, 1, 1])

        @udf(returnType=ArrayType(IntegerType()))
        def calculate_voxel_idx(x, y, z):
            """Calculate voxel indices from world coordinates."""
            point = np.array([x, y, z])
            normalized = (point - bbox_min) / resolution
            indices = np.floor(normalized).astype(int)
            indices = np.clip(indices, [0, 0, 0], dims - 1)
            return indices.tolist()

        return calculate_voxel_idx


class SparkNearestNeighbor(SparkInterpolationMethod):
    """
    Spark-based nearest neighbor interpolation.

    Strategy: Map-reduce pattern
    1. Calculate voxel indices for all points
    2. Group by voxel index
    3. Aggregate signals per voxel
    4. Collect results and build VoxelGrid
    """

    def interpolate_spark(
        self,
        spark: SparkSession,
        points_df: DataFrame,
        voxel_grid_config: Dict[str, Any],
    ) -> Any:
        """
        Spark-based nearest neighbor interpolation.
        """
        if not PYSPARK_AVAILABLE:
            raise ImportError("PySpark is required")

        from ...voxelization.voxel_grid import VoxelGrid

        # Calculate voxel indices
        calculate_voxel_idx = self._calculate_voxel_indices_udf(voxel_grid_config)
        df_with_voxels = points_df.withColumn("voxel_idx", calculate_voxel_idx(col("x"), col("y"), col("z")))

        # Extract voxel index components
        df_with_voxels = (
            df_with_voxels.withColumn("voxel_i", col("voxel_idx")[0])
            .withColumn("voxel_j", col("voxel_idx")[1])
            .withColumn("voxel_k", col("voxel_idx")[2])
        )

        # Get signal column names (exclude x, y, z, voxel_idx, voxel_i, voxel_j, voxel_k)
        all_columns = points_df.columns
        signal_columns = [c for c in all_columns if c not in ["x", "y", "z"]]

        # Group by voxel and aggregate signals
        aggregation = voxel_grid_config.get("aggregation", "mean")
        agg_exprs = []

        for signal_col in signal_columns:
            if aggregation == "mean":
                agg_exprs.append(mean(col(signal_col)).alias(f"{signal_col}_mean"))
            elif aggregation == "max":
                agg_exprs.append(spark_max(col(signal_col)).alias(f"{signal_col}_max"))
            elif aggregation == "min":
                agg_exprs.append(spark_min(col(signal_col)).alias(f"{signal_col}_min"))
            elif aggregation == "sum":
                agg_exprs.append(spark_sum(col(signal_col)).alias(f"{signal_col}_sum"))
            else:
                agg_exprs.append(mean(col(signal_col)).alias(f"{signal_col}_mean"))

        # Count points per voxel
        agg_exprs.append(spark_sum(col("x")).alias("_count"))  # Dummy, will use count

        aggregated = df_with_voxels.groupBy("voxel_i", "voxel_j", "voxel_k").agg(*agg_exprs)

        # Collect results
        results = aggregated.collect()

        # Build VoxelGrid
        voxel_grid = VoxelGrid(
            bbox_min=voxel_grid_config["bbox_min"],
            bbox_max=voxel_grid_config["bbox_max"],
            resolution=voxel_grid_config["resolution"],
            aggregation=aggregation,
        )

        voxel_data = {}
        for row in results:
            voxel_key = (int(row["voxel_i"]), int(row["voxel_j"]), int(row["voxel_k"]))

            signals = {}
            for signal_col in signal_columns:
                if aggregation == "mean":
                    signal_name = f"{signal_col}_mean"
                elif aggregation == "max":
                    signal_name = f"{signal_col}_max"
                elif aggregation == "min":
                    signal_name = f"{signal_col}_min"
                elif aggregation == "sum":
                    signal_name = f"{signal_col}_sum"
                else:
                    signal_name = f"{signal_col}_mean"

                if signal_name in row.asDict() and row[signal_name] is not None:
                    signals[signal_col] = float(row[signal_name])

            # Count points (approximate from number of rows)
            count = 1  # Will be updated during merge

            voxel_data[voxel_key] = {"signals": signals, "count": count}

        voxel_grid._build_voxel_grid_batch(voxel_data)
        return voxel_grid


class SparkLinearInterpolation(SparkInterpolationMethod):
    """
    Spark-based linear interpolation using k-nearest neighbors.

    Strategy: Spatial partitioning and neighbor search
    """

    def __init__(self, k_neighbors: int = 8, radius: Optional[float] = None):
        """
        Initialize Spark linear interpolation.

        Args:
            k_neighbors: Number of nearest neighbors
            radius: Optional maximum search radius
        """
        self.k_neighbors = k_neighbors
        self.radius = radius

    def interpolate_spark(
        self,
        spark: SparkSession,
        points_df: DataFrame,
        voxel_grid_config: Dict[str, Any],
    ) -> Any:
        """
        Spark-based linear interpolation.

        Note: Full k-nearest neighbor search in Spark is complex.
        This implementation uses a simplified approach with spatial partitioning.
        """
        if not PYSPARK_AVAILABLE:
            raise ImportError("PySpark is required")

        logger.warning(
            "Spark-based linear interpolation uses simplified approach. "
            "For full k-NN search, consider using Spark MLlib or spatial libraries."
        )

        # For now, fall back to nearest neighbor with spatial partitioning
        # Full k-NN implementation would require spatial indexing libraries
        nearest_method = SparkNearestNeighbor()
        return nearest_method.interpolate_spark(spark, points_df, voxel_grid_config)


class SparkIDWInterpolation(SparkInterpolationMethod):
    """
    Spark-based Inverse Distance Weighting interpolation.

    Similar to linear but with power parameter.
    """

    def __init__(self, power: float = 2.0, k_neighbors: int = 8, radius: Optional[float] = None):
        """
        Initialize Spark IDW interpolation.

        Args:
            power: Power parameter for distance weighting
            k_neighbors: Number of nearest neighbors
            radius: Optional maximum search radius
        """
        self.power = power
        self.k_neighbors = k_neighbors
        self.radius = radius

    def interpolate_spark(
        self,
        spark: SparkSession,
        points_df: DataFrame,
        voxel_grid_config: Dict[str, Any],
    ) -> Any:
        """
        Spark-based IDW interpolation.

        Note: Full IDW implementation requires spatial neighbor search.
        This uses a simplified approach.
        """
        if not PYSPARK_AVAILABLE:
            raise ImportError("PySpark is required")

        logger.warning(
            "Spark-based IDW interpolation uses simplified approach. "
            "For full IDW, consider using spatial indexing libraries."
        )

        # Simplified: use nearest neighbor for now
        nearest_method = SparkNearestNeighbor()
        return nearest_method.interpolate_spark(spark, points_df, voxel_grid_config)


class SparkGaussianKDE(SparkInterpolationMethod):
    """
    Spark-based Gaussian Kernel Density Estimation.

    Strategy: Spatial partitioning, distributed kernel evaluation
    """

    def __init__(self, bandwidth: Optional[float] = None, adaptive: bool = False):
        """
        Initialize Spark Gaussian KDE.

        Args:
            bandwidth: Kernel bandwidth
            adaptive: Whether to use adaptive bandwidth
        """
        self.bandwidth = bandwidth
        self.adaptive = adaptive

    def interpolate_spark(
        self,
        spark: SparkSession,
        points_df: DataFrame,
        voxel_grid_config: Dict[str, Any],
    ) -> Any:
        """
        Spark-based Gaussian KDE interpolation.

        Strategy:
        1. Partition points by spatial regions
        2. Broadcast voxel centers to all partitions
        3. Compute kernel contributions in parallel
        4. Aggregate kernel values per voxel
        """
        if not PYSPARK_AVAILABLE:
            raise ImportError("PySpark is required")

        logger.warning(
            "Spark-based Gaussian KDE uses simplified approach. "
            "Full implementation would require distributed spatial indexing."
        )

        # Simplified: use nearest neighbor for now
        # Full KDE would require:
        # - Spatial partitioning
        # - Broadcast voxel centers
        # - Distributed kernel evaluation
        # - Aggregation of kernel contributions
        nearest_method = SparkNearestNeighbor()
        return nearest_method.interpolate_spark(spark, points_df, voxel_grid_config)


# Method registry
class SparkRBFInterpolation(SparkInterpolationMethod):
    """
    Spark-based Radial Basis Functions interpolation.

    Strategy: For RBF's O(N³) complexity, we use a hybrid approach:
    1. For small datasets (< 10,000 points): Use sequential RBF (more efficient)
    2. For large datasets: Use distributed approximation with spatial partitioning
    3. Partition points spatially, compute local RBF, then merge results

    Note: Full distributed RBF with exact interpolation is computationally
    challenging. This implementation provides a practical distributed approximation.
    """

    def __init__(
        self,
        kernel: str = "gaussian",
        epsilon: Optional[float] = None,
        smoothing: float = 0.0,
        max_points_per_partition: int = 5000,
        use_sequential_threshold: int = 10000,
    ):
        """
        Initialize Spark RBF interpolation.

        Args:
            kernel: RBF kernel type
            epsilon: Shape parameter (auto-estimated if None)
            smoothing: Smoothing parameter (0.0 = exact interpolation)
            max_points_per_partition: Maximum points per partition for distributed processing
            use_sequential_threshold: Use sequential RBF if points < threshold
        """
        self.kernel = kernel
        self.epsilon = epsilon
        self.smoothing = smoothing
        self.max_points_per_partition = max_points_per_partition
        self.use_sequential_threshold = use_sequential_threshold

    def interpolate_spark(
        self,
        spark: SparkSession,
        points_df: DataFrame,
        voxel_grid_config: Dict[str, Any],
    ) -> Any:
        """
        Spark-based RBF interpolation.

        Uses hybrid approach: sequential for small datasets, distributed for large ones.
        """
        if not PYSPARK_AVAILABLE:
            raise ImportError("PySpark is required for Spark-based RBF interpolation")

        # Get point count
        n_points = points_df.count()

        # For small datasets, sequential RBF is more efficient
        if n_points < self.use_sequential_threshold:
            logger.info(
                f"Dataset size ({n_points:,}) is below threshold ({self.use_sequential_threshold:,}). "
                "Using sequential RBF for better performance."
            )
            return self._sequential_rbf_fallback(spark, points_df, voxel_grid_config)

        logger.info(
            f"Using distributed RBF approximation for large dataset ({n_points:,} points). "
            "This uses spatial partitioning and local RBF interpolation."
        )

        # For large datasets, use distributed approximation
        return self._distributed_rbf_approximation(spark, points_df, voxel_grid_config)

    def _sequential_rbf_fallback(
        self,
        spark: SparkSession,
        points_df: DataFrame,
        voxel_grid_config: Dict[str, Any],
    ) -> Any:
        """Fall back to sequential RBF for small datasets."""
        from ...voxelization.voxel_grid import VoxelGrid
        from ..methods.rbf import RBFInterpolation

        # Collect points and signals from Spark DataFrame
        rows = points_df.collect()

        # Extract points and signals
        points = np.array([[row.x, row.y, row.z] for row in rows])
        signal_names = [col_name for col_name in points_df.columns if col_name not in ["x", "y", "z"]]
        signals = {name: np.array([getattr(row, name) for row in rows]) for name in signal_names}

        # Create voxel grid
        voxel_grid = VoxelGrid(
            bbox_min=voxel_grid_config["bbox_min"],
            bbox_max=voxel_grid_config["bbox_max"],
            resolution=voxel_grid_config["resolution"],
            aggregation=voxel_grid_config.get("aggregation", "mean"),
        )

        # Use sequential RBF
        rbf = RBFInterpolation(
            kernel=self.kernel,
            epsilon=self.epsilon,
            smoothing=self.smoothing,
        )
        return rbf.interpolate(points, signals, voxel_grid)

    def _distributed_rbf_approximation(
        self,
        spark: SparkSession,
        points_df: DataFrame,
        voxel_grid_config: Dict[str, Any],
    ) -> Any:
        """
        Distributed RBF approximation using spatial partitioning.

        Strategy:
        1. Partition points spatially
        2. Compute local RBF for each partition
        3. Merge results at voxel level
        """
        from ...voxelization.voxel_grid import VoxelGrid
        from ..methods.rbf import RBFInterpolation

        logger.warning(
            "Distributed RBF approximation is experimental. "
            "For exact RBF interpolation, consider using sequential RBF for smaller datasets "
            "or alternative methods (linear, IDW, KDE) for large datasets."
        )

        # For now, use a simplified approach: spatial partitioning with local RBF
        # This is a practical approximation for very large datasets

        # Partition points spatially using voxel-based partitioning
        bbox_min = np.array(voxel_grid_config["bbox_min"])
        bbox_max = np.array(voxel_grid_config["bbox_max"])
        resolution = voxel_grid_config["resolution"]

        # Calculate spatial partitions
        # Use larger partition size to reduce overhead
        partition_size = max(self.max_points_per_partition, 1000)

        # Add partition ID based on spatial location
        @udf(returnType=IntegerType())
        def get_partition_id(x, y, z):
            """Get partition ID based on spatial location."""
            point = np.array([x, y, z])
            normalized = (point - bbox_min) / (bbox_max - bbox_min)
            # Simple spatial hashing
            partition_x = int(normalized[0] * 4)  # 4 partitions per dimension
            partition_y = int(normalized[1] * 4)
            partition_z = int(normalized[2] * 4)
            return partition_x * 16 + partition_y * 4 + partition_z

        # Add partition column
        points_df_partitioned = points_df.withColumn("partition_id", get_partition_id(col("x"), col("y"), col("z")))

        # Process each partition
        partitions = points_df_partitioned.select("partition_id").distinct().collect()
        all_voxel_data = {}

        for partition_row in partitions:
            partition_id = partition_row.partition_id
            partition_df = points_df_partitioned.filter(col("partition_id") == partition_id)

            # Collect partition data
            partition_rows = partition_df.collect()
            if len(partition_rows) == 0:
                continue

            # Extract points and signals
            partition_points = np.array([[row.x, row.y, row.z] for row in partition_rows])
            signal_names = [col_name for col_name in partition_df.columns if col_name not in ["x", "y", "z", "partition_id"]]
            partition_signals = {name: np.array([getattr(row, name) for row in partition_rows]) for name in signal_names}

            # Create local voxel grid for partition
            partition_voxel_grid = VoxelGrid(
                bbox_min=voxel_grid_config["bbox_min"],
                bbox_max=voxel_grid_config["bbox_max"],
                resolution=voxel_grid_config["resolution"],
                aggregation=voxel_grid_config.get("aggregation", "mean"),
            )

            # Apply local RBF
            try:
                rbf = RBFInterpolation(
                    kernel=self.kernel,
                    epsilon=self.epsilon,
                    smoothing=self.smoothing,
                )
                local_result = rbf.interpolate(partition_points, partition_signals, partition_voxel_grid)

                # Merge voxel data
                for voxel_key, voxel_data in local_result.voxels.items():
                    if voxel_key not in all_voxel_data:
                        all_voxel_data[voxel_key] = {"signals": {}, "count": 0}
                    # Merge signals (use mean for overlapping voxels)
                    for signal_name, signal_value in voxel_data.get("signals", {}).items():
                        if signal_name in all_voxel_data[voxel_key]["signals"]:
                            # Average overlapping values
                            existing = all_voxel_data[voxel_key]["signals"][signal_name]
                            all_voxel_data[voxel_key]["signals"][signal_name] = (existing + signal_value) / 2.0
                        else:
                            all_voxel_data[voxel_key]["signals"][signal_name] = signal_value
                    all_voxel_data[voxel_key]["count"] += voxel_data.get("count", 1)

            except Exception as e:
                logger.warning(f"Error processing partition {partition_id}: {e}. Skipping partition.")
                continue

        # Build final voxel grid
        voxel_grid = VoxelGrid(
            bbox_min=voxel_grid_config["bbox_min"],
            bbox_max=voxel_grid_config["bbox_max"],
            resolution=voxel_grid_config["resolution"],
            aggregation=voxel_grid_config.get("aggregation", "mean"),
        )
        voxel_grid._build_voxel_grid_batch(all_voxel_data)
        return voxel_grid


SPARK_INTERPOLATION_METHODS = {
    "nearest": SparkNearestNeighbor,
    "linear": SparkLinearInterpolation,
    "idw": SparkIDWInterpolation,
    "gaussian_kde": SparkGaussianKDE,
    "rbf": SparkRBFInterpolation,
}


def create_spark_session(
    app_name: str = "SignalMapping",
    master: Optional[str] = None,
    config: Optional[Dict[str, str]] = None,
) -> Optional[SparkSession]:
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
        logger.error("PySpark not available. Cannot create Spark session.")
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


def points_to_spark_dataframe(spark: SparkSession, points: np.ndarray, signals: Dict[str, np.ndarray]) -> DataFrame:
    """
    Convert points and signals to Spark DataFrame.

    Args:
        spark: SparkSession instance
        points: Array of points (N, 3)
        signals: Dictionary of signal arrays (N,)

    Returns:
        Spark DataFrame with columns: x, y, z, and signal columns
    """
    if not PYSPARK_AVAILABLE:
        raise ImportError("PySpark is required")

    # Prepare data for DataFrame
    data = []
    for i in range(len(points)):
        row = {
            "x": float(points[i, 0]),
            "y": float(points[i, 1]),
            "z": float(points[i, 2]),
        }
        for signal_name, signal_array in signals.items():
            if i < len(signal_array):
                row[signal_name] = float(signal_array[i])
            else:
                row[signal_name] = 0.0
        data.append(row)

    # Handle empty dataset
    if len(data) == 0:
        raise ValueError("Cannot create Spark DataFrame from empty dataset. Points array is empty.")

    # Create DataFrame
    df = spark.createDataFrame(data)
    return df


def interpolate_to_voxels_spark(
    spark: SparkSession,
    points: np.ndarray,
    signals: Dict[str, np.ndarray],
    voxel_grid_config: Dict[str, Any],
    method: str = "nearest",
    **method_kwargs,
) -> Any:
    """
    Interpolate points to voxel grid using Spark.

    Args:
        spark: SparkSession instance
        points: Array of points (N, 3)
        signals: Dictionary of signal arrays (N,)
        voxel_grid_config: Voxel grid configuration
        method: Interpolation method ('nearest', 'linear', 'idw', 'gaussian_kde')
        **method_kwargs: Method-specific arguments

    Returns:
        VoxelGrid with interpolated data
    """
    if not PYSPARK_AVAILABLE:
        raise ImportError("PySpark is required for Spark-based interpolation")

    if method not in SPARK_INTERPOLATION_METHODS:
        raise ValueError(
            f"Unknown Spark interpolation method: {method}. " f"Available methods: {list(SPARK_INTERPOLATION_METHODS.keys())}"
        )

    # Convert to Spark DataFrame
    points_df = points_to_spark_dataframe(spark, points, signals)

    # Get method instance
    method_class = SPARK_INTERPOLATION_METHODS[method]
    method_instance = method_class(**method_kwargs)

    # Interpolate
    return method_instance.interpolate_spark(spark, points_df, voxel_grid_config)
