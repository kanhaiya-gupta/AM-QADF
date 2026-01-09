"""
Temporal Alignment

Map timestamps to build layers and handle temporal data alignment.
"""

from typing import Optional, Dict, List, Tuple, Any
import numpy as np
from scipy.interpolate import interp1d
from dataclasses import dataclass


@dataclass
class TimePoint:
    """Represents a point in time with associated data."""

    timestamp: float  # seconds since build start
    layer_index: Optional[int] = None
    z_height: Optional[float] = None
    data: Optional[Dict[str, Any]] = None


class LayerTimeMapper:
    """
    Map build layers to timestamps and vice versa.

    Handles the relationship between:
    - Layer indices (0, 1, 2, ...)
    - Z heights (mm)
    - Timestamps (seconds)
    """

    def __init__(
        self,
        layer_thickness: float = 0.04,  # mm
        base_z: float = 0.0,  # mm
        time_per_layer: Optional[float] = None,  # seconds per layer
    ):
        """
        Initialize layer-time mapper.

        Args:
            layer_thickness: Thickness of each layer (mm)
            base_z: Base Z height (mm)
            time_per_layer: Average time per layer (seconds)
        """
        self.layer_thickness = layer_thickness
        self.base_z = base_z
        self.time_per_layer = time_per_layer
        self._layer_times: Dict[int, float] = {}  # layer_index -> timestamp
        self._z_heights: Dict[int, float] = {}  # layer_index -> z_height

    def add_layer_time(self, layer_index: int, timestamp: float, z_height: Optional[float] = None):
        """
        Add a known layer-time mapping.

        Args:
            layer_index: Layer index
            timestamp: Timestamp (seconds)
            z_height: Z height (mm), if None, computed from layer_index
        """
        if z_height is None:
            z_height = self.base_z + layer_index * self.layer_thickness

        self._layer_times[layer_index] = timestamp
        self._z_heights[layer_index] = z_height

    def layer_to_z(self, layer_index: int) -> float:
        """
        Convert layer index to Z height.

        Args:
            layer_index: Layer index

        Returns:
            Z height (mm)
        """
        if layer_index in self._z_heights:
            return self._z_heights[layer_index]

        return self.base_z + layer_index * self.layer_thickness

    def z_to_layer(self, z_height: float) -> int:
        """
        Convert Z height to layer index.

        Args:
            z_height: Z height (mm)

        Returns:
            Layer index
        """
        layer_index = int((z_height - self.base_z) / self.layer_thickness)
        return max(0, layer_index)

    def layer_to_time(self, layer_index: int) -> Optional[float]:
        """
        Convert layer index to timestamp.

        Args:
            layer_index: Layer index

        Returns:
            Timestamp (seconds) or None if not mapped
        """
        if layer_index in self._layer_times:
            return self._layer_times[layer_index]

        # Interpolate if we have other layer times
        if len(self._layer_times) > 1:
            layers = sorted(self._layer_times.keys())
            times = [self._layer_times[layer] for layer in layers]

            if layer_index < layers[0]:
                # Extrapolate backward
                if self.time_per_layer:
                    return times[0] - (layers[0] - layer_index) * self.time_per_layer
            elif layer_index > layers[-1]:
                # Extrapolate forward
                if self.time_per_layer:
                    return times[-1] + (layer_index - layers[-1]) * self.time_per_layer
            else:
                # Interpolate
                interp_func = interp1d(layers, times, kind="linear", fill_value="extrapolate")
                return float(interp_func(layer_index))

        # Use time_per_layer if available
        if self.time_per_layer and len(self._layer_times) > 0:
            base_layer = min(self._layer_times.keys())
            base_time = self._layer_times[base_layer]
            return base_time + (layer_index - base_layer) * self.time_per_layer

        return None

    def time_to_layer(self, timestamp: float) -> Optional[int]:
        """
        Convert timestamp to layer index.

        Args:
            timestamp: Timestamp (seconds)

        Returns:
            Layer index or None if not mapped
        """
        if len(self._layer_times) == 0:
            return None

        # Find closest layer time
        layers = sorted(self._layer_times.keys())
        times = [self._layer_times[layer] for layer in layers]

        # Check for exact match first
        for layer, time in zip(layers, times):
            if abs(timestamp - time) < 1e-9:  # Floating point comparison
                return layer

        if timestamp < times[0]:
            # Before first layer
            if self.time_per_layer:
                delta_layers = (times[0] - timestamp) / self.time_per_layer
                return max(0, int(layers[0] - delta_layers))
            return None
        elif timestamp > times[-1]:
            # After last layer
            if self.time_per_layer:
                delta_layers = (timestamp - times[-1]) / self.time_per_layer
                return int(layers[-1] + delta_layers)
            return None
        else:
            # Interpolate (need at least 2 points for interpolation)
            if len(times) >= 2:
                interp_func = interp1d(times, layers, kind="linear", fill_value="extrapolate")
                result = interp_func(timestamp)
                if not np.isnan(result):
                    return int(result)
            # If only one point or interpolation failed, use time_per_layer if available
            if self.time_per_layer and len(times) == 1:
                delta_layers = (timestamp - times[0]) / self.time_per_layer
                return int(layers[0] + delta_layers)

        return None


class TemporalAligner:
    """
    Align temporal data with build process.

    Handles:
    - Mapping timestamps to layers
    - Temporal interpolation
    - Handling missing temporal data
    """

    def __init__(self, layer_mapper: Optional[LayerTimeMapper] = None):
        """
        Initialize temporal aligner.

        Args:
            layer_mapper: LayerTimeMapper instance
        """
        self.layer_mapper = layer_mapper or LayerTimeMapper()
        self._time_points: List[TimePoint] = []

    def add_time_point(
        self,
        timestamp: float,
        layer_index: Optional[int] = None,
        z_height: Optional[float] = None,
        data: Optional[Dict[str, Any]] = None,
    ):
        """
        Add a time point with associated data.

        Args:
            timestamp: Timestamp (seconds)
            layer_index: Layer index (if None, computed from timestamp)
            z_height: Z height (if None, computed from layer_index)
            data: Associated data dictionary
        """
        if layer_index is None:
            layer_index = self.layer_mapper.time_to_layer(timestamp)

        if z_height is None and layer_index is not None:
            z_height = self.layer_mapper.layer_to_z(layer_index)

        time_point = TimePoint(timestamp=timestamp, layer_index=layer_index, z_height=z_height, data=data)

        self._time_points.append(time_point)

    def align_to_layers(self, target_layers: List[int], interpolation_method: str = "linear") -> List[Dict[str, Any]]:
        """
        Align temporal data to specific layers.

        Args:
            target_layers: List of target layer indices
            interpolation_method: Interpolation method ('linear', 'nearest', 'zero')

        Returns:
            List of data dictionaries, one per target layer
        """
        if len(self._time_points) == 0:
            return [{} for _ in target_layers]

        # Sort time points by timestamp
        sorted_points = sorted(self._time_points, key=lambda tp: tp.timestamp)

        # Get timestamps and layer indices
        timestamps = [tp.timestamp for tp in sorted_points]
        layer_indices = [tp.layer_index for tp in sorted_points if tp.layer_index is not None]

        if len(layer_indices) == 0:
            return [{} for _ in target_layers]

        # Get target timestamps
        target_times = []
        for layer_idx in target_layers:
            time = self.layer_mapper.layer_to_time(layer_idx)
            if time is not None:
                target_times.append(time)
            else:
                target_times.append(None)

        # Interpolate data for each target layer
        aligned_data: List[Dict[str, Any]] = []
        for i, target_time in enumerate(target_times):
            if target_time is None:
                aligned_data.append({})
                continue

            # Find closest time points
            if target_time <= timestamps[0]:
                # Before first point
                aligned_data.append(sorted_points[0].data or {})
            elif target_time >= timestamps[-1]:
                # After last point
                aligned_data.append(sorted_points[-1].data or {})
            else:
                # Interpolate
                if interpolation_method == "nearest":
                    idx = np.argmin(np.abs(np.array(timestamps) - target_time))
                    aligned_data.append(sorted_points[idx].data or {})
                else:
                    # Linear interpolation (simplified - would need to interpolate each data field)
                    idx = np.searchsorted(timestamps, target_time)
                    if idx == 0:
                        aligned_data.append(sorted_points[0].data or {})
                    else:
                        # Use nearest for now (full implementation would interpolate each field)
                        aligned_data.append(sorted_points[idx - 1].data or {})

        return aligned_data

    def get_layer_data(self, layer_index: int) -> Optional[Dict[str, Any]]:
        """
        Get data for a specific layer.

        Args:
            layer_index: Layer index

        Returns:
            Data dictionary or None
        """
        layer_time = self.layer_mapper.layer_to_time(layer_index)
        if layer_time is None:
            return None

        # Find closest time point
        if len(self._time_points) == 0:
            return None

        closest_point = min(self._time_points, key=lambda tp: abs(tp.timestamp - layer_time))

        return closest_point.data

    def handle_missing_temporal_data(
        self, required_layers: List[int], default_data: Optional[Dict[str, Any]] = None
    ) -> Dict[int, Dict[str, Any]]:
        """
        Handle missing temporal data by filling with defaults or interpolation.

        Args:
            required_layers: List of required layer indices
            default_data: Default data to use for missing layers

        Returns:
            Dictionary mapping layer_index to data
        """
        result = {}

        for layer_idx in required_layers:
            data = self.get_layer_data(layer_idx)
            if data is None:
                # Try interpolation
                aligned = self.align_to_layers([layer_idx])
                if aligned and aligned[0]:
                    data = aligned[0]
                else:
                    data = default_data or {}

            result[layer_idx] = data

        return result
