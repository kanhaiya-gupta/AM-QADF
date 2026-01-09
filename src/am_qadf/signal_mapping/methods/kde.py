"""
Gaussian Kernel Density Estimation (KDE) Interpolation

Vectorized Gaussian KDE interpolation.
Each point contributes a Gaussian kernel, and voxel values are computed
as the weighted sum of overlapping kernels.
"""

import numpy as np
from typing import Dict, Optional

from .base import InterpolationMethod
from ..utils._performance import performance_monitor
from ...voxelization.voxel_grid import VoxelGrid


class GaussianKDEInterpolation(InterpolationMethod):
    """
    Vectorized Gaussian Kernel Density Estimation (KDE) interpolation.

    Each point contributes a Gaussian kernel, and voxel values are computed
    as the weighted sum of overlapping kernels.
    Formula: v = sum(s_i * K_h(||p - c||))
    where K_h is Gaussian kernel with bandwidth h
    """

    def __init__(self, bandwidth: Optional[float] = None, adaptive: bool = False):
        """
        Initialize Gaussian KDE interpolation.

        Args:
            bandwidth: Kernel bandwidth (if None, auto-estimated)
            adaptive: Whether to use adaptive bandwidth
        """
        self.bandwidth = bandwidth
        self.adaptive = adaptive

    def _estimate_bandwidth(self, points: np.ndarray) -> float:
        """
        Estimate optimal bandwidth using Silverman's rule of thumb.

        Args:
            points: Array of points (N, 3)

        Returns:
            Estimated bandwidth
        """
        # Silverman's rule of thumb
        n = len(points)
        if n < 2:
            return 1.0

        # Estimate standard deviation for each dimension
        std = np.std(points, axis=0)
        mean_std = np.mean(std)

        # Silverman's formula
        bandwidth = 1.06 * mean_std * (n ** (-1.0 / 5.0))

        return max(bandwidth, 0.1)  # Minimum bandwidth

    @performance_monitor
    def interpolate(self, points: np.ndarray, signals: Dict[str, np.ndarray], voxel_grid: VoxelGrid) -> VoxelGrid:
        """
        Vectorized Gaussian KDE interpolation.

        For large datasets, uses spatial indexing to limit kernel evaluation
        to nearby points only.
        """
        if len(points) == 0:
            return voxel_grid

        try:
            from scipy.spatial import cKDTree
        except ImportError:
            raise ImportError("scipy is required for Gaussian KDE interpolation. " "Install with: pip install scipy")

        # Auto-select bandwidth if not provided
        if self.bandwidth is None:
            self.bandwidth = self._estimate_bandwidth(points)

        # Get unique voxel centers
        voxel_indices = self._world_to_voxel_batch(points, voxel_grid)
        unique_voxels = np.unique(voxel_indices, axis=0)
        voxel_centers = voxel_grid.bbox_min + (unique_voxels + 0.5) * voxel_grid.resolution

        # For large datasets, use spatial indexing to limit computation
        # Only evaluate kernels for points within 3*bandwidth
        search_radius = 3.0 * self.bandwidth
        tree = cKDTree(points)

        # Aggregate signals per voxel
        voxel_data = {}
        for voxel_idx, voxel_center in enumerate(voxel_centers):
            voxel_key = tuple(unique_voxels[voxel_idx])

            # Find points within search radius
            nearby_indices = tree.query_ball_point(voxel_center, search_radius)

            if len(nearby_indices) == 0:
                continue

            nearby_points = points[nearby_indices]
            distances = np.linalg.norm(nearby_points - voxel_center, axis=1)

            # Gaussian kernel: K_h(d) = exp(-d^2 / (2*h^2)) / (h * sqrt(2*pi))
            kernel_values = np.exp(-(distances**2) / (2 * self.bandwidth**2))
            kernel_values = kernel_values / (self.bandwidth * np.sqrt(2 * np.pi))

            # Normalize kernels
            kernel_sum = kernel_values.sum()
            if kernel_sum > 0:
                kernel_values = kernel_values / kernel_sum

            # Aggregate signals
            aggregated_signals = {}
            for signal_name, signal_array in signals.items():
                if len(signal_array) == len(points):
                    nearby_signals = signal_array[nearby_indices]
                    weighted_sum = np.sum(nearby_signals * kernel_values)
                    aggregated_signals[signal_name] = float(weighted_sum)

            voxel_data[voxel_key] = {
                "signals": aggregated_signals,
                "count": int(np.sum(kernel_values > 1e-6)),  # Effective point count
            }

        self._build_voxel_grid_batch(voxel_grid, voxel_data)
        return voxel_grid
