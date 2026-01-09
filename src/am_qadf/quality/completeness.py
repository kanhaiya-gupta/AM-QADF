"""
Completeness Checks

Checks data completeness for voxel domain data:
- Missing Data Detection: Identify missing voxels
- Coverage Analysis: Analyze spatial/temporal coverage
- Gap Filling: Strategies for handling missing data
- Data Validation: Validate data integrity
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from enum import Enum


class GapFillingStrategy(Enum):
    """Strategies for filling missing data gaps."""

    NONE = "none"  # Don't fill gaps
    ZERO = "zero"  # Fill with zeros
    NEAREST = "nearest"  # Nearest neighbor interpolation
    LINEAR = "linear"  # Linear interpolation
    MEAN = "mean"  # Fill with mean value
    MEDIAN = "median"  # Fill with median value


@dataclass
class CompletenessMetrics:
    """Completeness metrics for voxel domain data."""

    completeness_ratio: float  # Overall completeness (0-1)
    spatial_coverage: float  # Spatial coverage ratio (0-1)
    temporal_coverage: float  # Temporal coverage ratio (0-1)
    missing_voxels_count: int  # Number of missing voxels
    missing_regions_count: int  # Number of missing regions
    gap_fillable_ratio: float  # Ratio of gaps that can be filled (0-1)

    # Detailed information
    missing_voxel_indices: Optional[np.ndarray] = None  # Indices of missing voxels
    missing_regions: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None  # Missing region bboxes

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        result = {
            "completeness_ratio": self.completeness_ratio,
            "spatial_coverage": self.spatial_coverage,
            "temporal_coverage": self.temporal_coverage,
            "missing_voxels_count": self.missing_voxels_count,
            "missing_regions_count": self.missing_regions_count,
            "gap_fillable_ratio": self.gap_fillable_ratio,
        }
        if self.missing_voxel_indices is not None:
            result["missing_voxel_indices_count"] = len(self.missing_voxel_indices)
        if self.missing_regions is not None:
            result["missing_regions"] = len(self.missing_regions)
        return result


class CompletenessAnalyzer:
    """Analyzes completeness for voxel domain data."""

    def __init__(self):
        """Initialize the completeness analyzer."""
        pass

    def detect_missing_data(self, signal_array: np.ndarray, store_indices: bool = True) -> Tuple[int, Optional[np.ndarray]]:
        """
        Detect missing data in signal array.

        Args:
            signal_array: Signal array
            store_indices: Whether to store indices of missing voxels

        Returns:
            (missing_count, missing_indices)
        """
        # Missing = NaN or zero
        missing_mask = np.isnan(signal_array) | (signal_array == 0.0)
        missing_count = np.sum(missing_mask)

        missing_indices = None
        if store_indices and missing_count > 0:
            missing_indices = np.where(missing_mask)

        return missing_count, missing_indices

    def analyze_coverage(self, voxel_data: Any, signals: Optional[List[str]] = None) -> Tuple[float, float]:
        """
        Analyze spatial and temporal coverage.

        Args:
            voxel_data: Voxel domain data object
            signals: List of signal names to check (None = all signals)

        Returns:
            (spatial_coverage, temporal_coverage)
        """
        if signals is None:
            signals = list(voxel_data.available_signals) if hasattr(voxel_data, "available_signals") else []

        if not signals:
            return 0.0, 0.0

        # Spatial coverage: ratio of spatial voxels with data
        total_voxels = np.prod(voxel_data.dims) if hasattr(voxel_data, "dims") else 0
        if total_voxels == 0:
            return 0.0, 0.0

        filled_voxels = 0
        for signal in signals:
            try:
                signal_array = voxel_data.get_signal_array(signal, default=0.0)
                filled_voxels += np.sum((signal_array != 0.0) & ~np.isnan(signal_array))
            except Exception:
                continue

        spatial_coverage = min(1.0, filled_voxels / total_voxels)

        # Temporal coverage: ratio of layers with data
        dims = voxel_data.dims if hasattr(voxel_data, "dims") else (0, 0, 0)
        if len(dims) < 3:
            temporal_coverage = 0.0
        else:
            z_dim = dims[2]
            if z_dim == 0:
                temporal_coverage = 0.0
            else:
                layers_with_data = set()
                for signal in signals:
                    try:
                        signal_array = voxel_data.get_signal_array(signal, default=0.0)
                        for z in range(z_dim):
                            slice_data = signal_array[:, :, z]
                            if np.any((slice_data != 0.0) & ~np.isnan(slice_data)):
                                layers_with_data.add(z)
                    except Exception:
                        continue
                temporal_coverage = len(layers_with_data) / z_dim if z_dim > 0 else 0.0

        return spatial_coverage, temporal_coverage

    def identify_missing_regions(
        self,
        signal_array: np.ndarray,
        bbox_min: Tuple[float, float, float],
        resolution: float,
        min_region_size: int = 10,
    ) -> List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
        """
        Identify missing data regions.

        Args:
            signal_array: Signal array
            bbox_min: Minimum bounding box coordinates
            resolution: Voxel resolution
            min_region_size: Minimum voxel count for a region

        Returns:
            List of (bbox_min, bbox_max) tuples for missing regions
        """
        # Create mask of missing voxels
        missing_mask = np.isnan(signal_array) | (signal_array == 0.0)

        if not np.any(missing_mask):
            return []

        # Find connected components
        try:
            from scipy import ndimage

            labeled, num_features = ndimage.label(missing_mask)

            missing_regions = []
            for label_id in range(1, num_features + 1):
                region_mask = labeled == label_id
                region_size = np.sum(region_mask)

                if region_size < min_region_size:
                    continue

                # Find bounding box of region
                coords = np.where(region_mask)
                if len(coords[0]) == 0:
                    continue

                min_coords = [np.min(coords[i]) for i in range(3)]
                max_coords = [np.max(coords[i]) for i in range(3)]

                # Convert to world coordinates
                bbox_min_world = tuple(bbox_min[i] + min_coords[i] * resolution for i in range(3))
                bbox_max_world = tuple(bbox_min[i] + (max_coords[i] + 1) * resolution for i in range(3))

                missing_regions.append((bbox_min_world, bbox_max_world))

            return missing_regions
        except ImportError:
            # Fallback: return single region covering all missing voxels
            coords = np.where(missing_mask)
            if len(coords[0]) > 0:
                min_coords = [np.min(coords[i]) for i in range(3)]
                max_coords = [np.max(coords[i]) for i in range(3)]
                bbox_min_world = tuple(bbox_min[i] + min_coords[i] * resolution for i in range(3))
                bbox_max_world = tuple(bbox_min[i] + (max_coords[i] + 1) * resolution for i in range(3))
                return [(bbox_min_world, bbox_max_world)]
            return []

    def fill_gaps(
        self,
        signal_array: np.ndarray,
        strategy: GapFillingStrategy = GapFillingStrategy.LINEAR,
    ) -> np.ndarray:
        """
        Fill missing data gaps using specified strategy.

        Args:
            signal_array: Signal array with missing data
            strategy: Gap filling strategy

        Returns:
            Filled signal array
        """
        filled_array = signal_array.copy()
        missing_mask = np.isnan(filled_array) | (filled_array == 0.0)

        if not np.any(missing_mask):
            return filled_array

        if strategy == GapFillingStrategy.NONE:
            return filled_array
        elif strategy == GapFillingStrategy.ZERO:
            filled_array[missing_mask] = 0.0
        elif strategy == GapFillingStrategy.MEAN:
            mean_value = np.nanmean(filled_array[~missing_mask])
            filled_array[missing_mask] = mean_value if not np.isnan(mean_value) else 0.0
        elif strategy == GapFillingStrategy.MEDIAN:
            median_value = np.nanmedian(filled_array[~missing_mask])
            filled_array[missing_mask] = median_value if not np.isnan(median_value) else 0.0
        elif strategy == GapFillingStrategy.NEAREST:
            from scipy.ndimage import distance_transform_edt

            # Find nearest non-missing value
            distances, indices = distance_transform_edt(missing_mask, return_indices=True)
            # Handle different array dimensions
            ndim = filled_array.ndim
            for i in range(ndim):
                indices[i] = np.clip(indices[i], 0, filled_array.shape[i] - 1)

            # Get values at nearest indices
            if ndim == 1:
                filled_array[missing_mask] = filled_array[indices[0][missing_mask]]
            elif ndim == 2:
                filled_array[missing_mask] = filled_array[indices[0][missing_mask], indices[1][missing_mask]]
            else:  # 3D or higher
                filled_array[missing_mask] = filled_array[tuple(indices[i][missing_mask] for i in range(ndim))]
        elif strategy == GapFillingStrategy.LINEAR:
            # Use scipy interpolation
            valid_mask = ~missing_mask
            if not np.any(valid_mask):
                # No valid data, fill with zeros
                filled_array[missing_mask] = 0.0
            elif filled_array.ndim == 1:
                # 1D case: use interp1d
                from scipy.interpolate import interp1d

                valid_indices = np.where(valid_mask)[0]
                valid_values = filled_array[valid_indices]
                if len(valid_indices) > 1:
                    interp = interp1d(
                        valid_indices,
                        valid_values,
                        kind="linear",
                        bounds_error=False,
                        fill_value="extrapolate",
                    )
                    missing_indices = np.where(missing_mask)[0]
                    filled_array[missing_indices] = interp(missing_indices)
                else:
                    # Only one valid point, use it for all missing
                    filled_array[missing_mask] = valid_values[0] if len(valid_values) > 0 else 0.0
            else:
                # Multi-dimensional case: use RegularGridInterpolator
                from scipy.interpolate import RegularGridInterpolator

                # Create coordinate arrays
                coords = [np.arange(s) for s in filled_array.shape]
                # Create interpolator
                interp = RegularGridInterpolator(
                    coords,
                    filled_array,
                    method="linear",
                    bounds_error=False,
                    fill_value=0.0,
                )
                # Interpolate missing points
                missing_coords = np.where(missing_mask)
                if len(missing_coords[0]) > 0:
                    missing_points = np.column_stack(missing_coords)
                    filled_values = interp(missing_points)
                    filled_array[missing_mask] = filled_values

        return filled_array

    def assess_completeness(
        self,
        voxel_data: Any,
        signals: Optional[List[str]] = None,
        store_details: bool = True,
    ) -> CompletenessMetrics:
        """
        Assess overall completeness.

        Args:
            voxel_data: Voxel domain data object
            signals: List of signal names to check (None = all signals)
            store_details: Whether to store detailed information

        Returns:
            CompletenessMetrics object
        """
        if signals is None:
            signals = list(voxel_data.available_signals) if hasattr(voxel_data, "available_signals") else []

        if not signals:
            return CompletenessMetrics(
                completeness_ratio=0.0,
                spatial_coverage=0.0,
                temporal_coverage=0.0,
                missing_voxels_count=0,
                missing_regions_count=0,
                gap_fillable_ratio=0.0,
            )

        # Analyze coverage
        spatial_coverage, temporal_coverage = self.analyze_coverage(voxel_data, signals)

        # Detect missing data
        total_voxels = np.prod(voxel_data.dims) if hasattr(voxel_data, "dims") else 0
        missing_count = 0
        missing_indices = None

        # Combine missing data from all signals
        combined_missing_mask = None
        for signal in signals:
            try:
                signal_array = voxel_data.get_signal_array(signal, default=0.0)
                signal_missing = np.isnan(signal_array) | (signal_array == 0.0)
                if combined_missing_mask is None:
                    combined_missing_mask = signal_missing
                else:
                    combined_missing_mask &= signal_missing
            except Exception:
                continue

        if combined_missing_mask is not None:
            missing_count = np.sum(combined_missing_mask)
            if store_details:
                missing_indices = np.where(combined_missing_mask)

        completeness_ratio = 1.0 - (missing_count / total_voxels) if total_voxels > 0 else 0.0

        # Identify missing regions
        missing_regions = []
        if len(signals) > 0:
            try:
                signal_array = voxel_data.get_signal_array(signals[0], default=0.0)
                bbox_min = voxel_data.bbox_min if hasattr(voxel_data, "bbox_min") else (0, 0, 0)
                resolution = voxel_data.resolution if hasattr(voxel_data, "resolution") else 1.0
                missing_regions = self.identify_missing_regions(signal_array, bbox_min, resolution)
            except Exception:
                pass

        # Estimate gap fillable ratio (simplified: assume 80% can be filled)
        gap_fillable_ratio = 0.8 if missing_count > 0 else 1.0

        return CompletenessMetrics(
            completeness_ratio=completeness_ratio,
            spatial_coverage=spatial_coverage,
            temporal_coverage=temporal_coverage,
            missing_voxels_count=missing_count,
            missing_regions_count=len(missing_regions),
            gap_fillable_ratio=gap_fillable_ratio,
            missing_voxel_indices=missing_indices,
            missing_regions=missing_regions,
        )
