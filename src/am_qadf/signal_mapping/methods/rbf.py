"""
Radial Basis Functions (RBF) Interpolation

Vectorized RBF interpolation providing exact interpolation at data points
with smooth interpolation between points. Uses scipy.interpolate.RBFInterpolator.

Complexity: O(N³) - use Spark for large datasets (N > 10,000)
"""

import numpy as np
from typing import Dict, Optional
import logging

from .base import InterpolationMethod
from ..utils._performance import performance_monitor
from ...voxelization.voxel_grid import VoxelGrid

logger = logging.getLogger(__name__)


class RBFInterpolation(InterpolationMethod):
    """
    Radial Basis Functions (RBF) interpolation.

    Provides exact interpolation at data points with smooth interpolation between.
    Uses scipy.interpolate.RBFInterpolator for computation.

    Mathematical formulation:
        v(x) = sum(w_i * phi(||x - p_i||))
    where weights w_i are determined by solving a linear system to ensure
    exact interpolation at data points.

    Complexity: O(N³) for solving linear system
    Best for: High accuracy requirements, exact interpolation needed

    Args:
        kernel: RBF kernel type. Options:
            - 'gaussian': exp(-(epsilon*r)^2)
            - 'multiquadric': sqrt(1 + (epsilon*r)^2)
            - 'inverse_multiquadric': 1/sqrt(1 + (epsilon*r)^2)
            - 'thin_plate_spline': r^2 * log(r)
            - 'linear': r
            - 'cubic': r^3
            - 'quintic': r^5
        epsilon: Shape parameter for kernel (auto-estimated if None)
        smoothing: Smoothing parameter (0.0 = exact interpolation)
        use_sparse: Use sparse matrices for large N (experimental)
        max_points: Maximum points before warning (None = no limit)
    """

    def __init__(
        self,
        kernel: str = "gaussian",
        epsilon: Optional[float] = None,
        smoothing: float = 0.0,
        use_sparse: bool = False,
        max_points: Optional[int] = None,
    ):
        """
        Initialize RBF interpolation.

        Args:
            kernel: RBF kernel type (default: 'gaussian')
            epsilon: Shape parameter (auto-estimated if None)
            smoothing: Smoothing parameter (0.0 = exact interpolation)
            use_sparse: Use sparse matrices (experimental)
            max_points: Maximum points before warning
        """
        self.kernel = kernel
        self.epsilon = epsilon
        self.smoothing = smoothing
        self.use_sparse = use_sparse
        self.max_points = max_points

        # Validate kernel type
        valid_kernels = [
            "gaussian",
            "multiquadric",
            "inverse_multiquadric",
            "thin_plate_spline",
            "linear",
            "cubic",
            "quintic",
        ]
        if kernel not in valid_kernels:
            raise ValueError(
                f"Invalid kernel '{kernel}'. "
                f"Valid options: {valid_kernels}"
            )

    def _estimate_epsilon(self, points: np.ndarray) -> float:
        """
        Estimate optimal epsilon parameter based on point distribution.

        Uses average nearest neighbor distance as a heuristic.

        Args:
            points: Array of points (N, 3)

        Returns:
            Estimated epsilon value
        """
        if len(points) < 2:
            return 1.0

        try:
            from scipy.spatial import cKDTree

            # Find average distance to nearest neighbor
            tree = cKDTree(points)
            distances, _ = tree.query(points, k=2)  # k=2 to exclude self
            if len(distances.shape) > 1:
                nn_distances = distances[:, 1]  # Second nearest (first is self)
            else:
                nn_distances = distances

            avg_nn_distance = np.mean(nn_distances[nn_distances > 0])
            if avg_nn_distance > 0:
                # Use inverse of average distance as epsilon
                epsilon = 1.0 / avg_nn_distance
            else:
                epsilon = 1.0

            # Clamp to reasonable range
            epsilon = np.clip(epsilon, 0.1, 10.0)

            return float(epsilon)

        except ImportError:
            # Fallback if scipy not available
            logger.warning("scipy not available for epsilon estimation, using default")
            return 1.0

    def _warn_large_dataset(self, n_points: int):
        """Warn user about large dataset performance."""
        if self.max_points and n_points > self.max_points:
            logger.warning(
                f"Large dataset detected ({n_points:,} points). "
                f"RBF has O(N³) complexity and may be slow. "
                f"Consider using Spark backend or alternative methods "
                f"(linear, IDW, KDE) for better performance."
            )

    @performance_monitor
    def interpolate(
        self, points: np.ndarray, signals: Dict[str, np.ndarray], voxel_grid: VoxelGrid
    ) -> VoxelGrid:
        """
        Vectorized RBF interpolation.

        Interpolates signals from points to voxel grid using Radial Basis Functions.
        Provides exact interpolation at data points (when smoothing=0).

        Args:
            points: Array of points (N, 3) with (x, y, z) coordinates
            signals: Dictionary mapping signal names to arrays (N,) of values
            voxel_grid: Target voxel grid

        Returns:
            VoxelGrid with interpolated signals

        Raises:
            ImportError: If scipy is not available
            ValueError: If points or signals are invalid
        """
        if len(points) == 0:
            return voxel_grid

        # Check for scipy
        try:
            from scipy.interpolate import RBFInterpolator
        except ImportError:
            raise ImportError(
                "scipy is required for RBF interpolation. "
                "Install with: pip install scipy"
            )

        # Warn about large datasets
        n_points = len(points)
        self._warn_large_dataset(n_points)

        # Auto-estimate epsilon if not provided
        if self.epsilon is None:
            self.epsilon = self._estimate_epsilon(points)
            logger.info(f"Auto-estimated epsilon: {self.epsilon:.4f}")

        # Get unique voxel centers that need interpolation
        voxel_indices = self._world_to_voxel_batch(points, voxel_grid)
        unique_voxels = np.unique(voxel_indices, axis=0)
        voxel_centers = voxel_grid.bbox_min + (unique_voxels + 0.5) * voxel_grid.resolution

        # Interpolate each signal separately
        voxel_data = {}

        for signal_name, signal_values in signals.items():
            if len(signal_values) != len(points):
                logger.warning(
                    f"Signal '{signal_name}' length ({len(signal_values)}) "
                    f"does not match points length ({len(points)}). Skipping."
                )
                continue

            try:
                # Create RBF interpolator
                # Note: RBFInterpolator expects points and values as separate arrays
                rbf = RBFInterpolator(
                    points,
                    signal_values,
                    kernel=self.kernel,
                    epsilon=self.epsilon,
                    smoothing=self.smoothing,
                )

                # Interpolate at voxel centers
                interpolated_values = rbf(voxel_centers)

                # Store interpolated values in voxel data
                for voxel_idx, (voxel_key, value) in enumerate(
                    zip(unique_voxels, interpolated_values)
                ):
                    voxel_key_tuple = tuple(voxel_key)

                    if voxel_key_tuple not in voxel_data:
                        voxel_data[voxel_key_tuple] = {
                            "signals": {},
                            "count": 1,  # RBF uses all points, count = 1
                        }

                    voxel_data[voxel_key_tuple]["signals"][signal_name] = float(value)

            except Exception as e:
                logger.error(
                    f"Error interpolating signal '{signal_name}' with RBF: {e}",
                    exc_info=True,
                )
                # Continue with other signals
                continue

        # Build voxel grid from interpolated data
        self._build_voxel_grid_batch(voxel_grid, voxel_data)
        return voxel_grid
