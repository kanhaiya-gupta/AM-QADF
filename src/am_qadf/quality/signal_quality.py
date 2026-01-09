"""
Signal Quality Assessment

Calculates signal quality metrics for voxel domain signals:
- Signal-to-Noise Ratio (SNR): Per signal, per voxel
- Uncertainty Quantification: Uncertainty propagation
- Confidence Scores: Confidence in each voxel value
- Data Quality Maps: Visualize quality across voxel grid
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class SignalQualityMetrics:
    """Signal quality metrics for a single signal."""

    signal_name: str
    snr_mean: float  # Mean SNR across all voxels
    snr_std: float  # Standard deviation of SNR
    snr_min: float  # Minimum SNR
    snr_max: float  # Maximum SNR
    uncertainty_mean: float  # Mean uncertainty
    confidence_mean: float  # Mean confidence score (0-1)
    quality_score: float  # Overall quality score (0-1)

    # Per-voxel arrays (optional, can be None for memory efficiency)
    snr_map: Optional[np.ndarray] = None
    uncertainty_map: Optional[np.ndarray] = None
    confidence_map: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        result = {
            "signal_name": self.signal_name,
            "snr_mean": self.snr_mean,
            "snr_std": self.snr_std,
            "snr_min": self.snr_min,
            "snr_max": self.snr_max,
            "uncertainty_mean": self.uncertainty_mean,
            "confidence_mean": self.confidence_mean,
            "quality_score": self.quality_score,
        }
        if self.snr_map is not None:
            result["snr_map_shape"] = self.snr_map.shape
        if self.uncertainty_map is not None:
            result["uncertainty_map_shape"] = self.uncertainty_map.shape
        if self.confidence_map is not None:
            result["confidence_map_shape"] = self.confidence_map.shape
        return result


class SignalQualityAnalyzer:
    """Analyzes signal quality for voxel domain signals."""

    def __init__(self, noise_floor: float = 1e-6):
        """
        Initialize the signal quality analyzer.

        Args:
            noise_floor: Minimum noise level for SNR calculation
        """
        self.noise_floor = noise_floor

    def calculate_snr(
        self,
        signal_array: np.ndarray,
        noise_estimate: Optional[np.ndarray] = None,
        store_map: bool = True,
    ) -> Tuple[float, float, float, float, Optional[np.ndarray]]:
        """
        Calculate Signal-to-Noise Ratio (SNR).

        Args:
            signal_array: Signal array
            noise_estimate: Optional noise estimate array (if None, estimates from signal)
            store_map: Whether to store per-voxel SNR map

        Returns:
            (mean_snr, std_snr, min_snr, max_snr, snr_map)
        """
        # Mask out invalid values
        valid_mask = (~np.isnan(signal_array)) & (signal_array != 0.0)

        if not np.any(valid_mask):
            return 0.0, 0.0, 0.0, 0.0, None

        signal_clean = signal_array[valid_mask]

        if noise_estimate is None:
            # Estimate noise from signal variation
            # Use local standard deviation as noise estimate
            from scipy import ndimage

            signal_smooth = ndimage.gaussian_filter(signal_array.astype(float), sigma=1.0)
            noise_estimate = np.abs(signal_array - signal_smooth)
            noise_estimate = np.maximum(noise_estimate, self.noise_floor)
        else:
            noise_estimate = np.maximum(noise_estimate, self.noise_floor)

        # Calculate SNR: SNR = 20 * log10(signal / noise)
        snr_map = np.zeros_like(signal_array, dtype=float)
        snr_map[valid_mask] = 20 * np.log10(
            np.maximum(signal_array[valid_mask], self.noise_floor) / noise_estimate[valid_mask]
        )
        snr_map[~valid_mask] = np.nan

        # Calculate statistics
        snr_clean = snr_map[valid_mask]
        if len(snr_clean) > 0:
            mean_snr = np.mean(snr_clean)
            std_snr = np.std(snr_clean)
            min_snr = np.min(snr_clean)
            max_snr = np.max(snr_clean)
        else:
            mean_snr = std_snr = min_snr = max_snr = 0.0

        snr_map_result = snr_map if store_map else None

        return mean_snr, std_snr, min_snr, max_snr, snr_map_result

    def calculate_uncertainty(
        self,
        signal_array: np.ndarray,
        measurement_uncertainty: Optional[float] = None,
        interpolation_uncertainty: Optional[np.ndarray] = None,
        store_map: bool = True,
    ) -> Tuple[float, Optional[np.ndarray]]:
        """
        Calculate uncertainty in signal values.

        Args:
            signal_array: Signal array
            measurement_uncertainty: Base measurement uncertainty (fraction or absolute)
            interpolation_uncertainty: Optional interpolation uncertainty map
            store_map: Whether to store per-voxel uncertainty map

        Returns:
            (mean_uncertainty, uncertainty_map)
        """
        valid_mask = (~np.isnan(signal_array)) & (signal_array != 0.0)

        if not np.any(valid_mask):
            return 0.0, None

        # Start with measurement uncertainty
        if measurement_uncertainty is None:
            # Default: 5% relative uncertainty
            uncertainty_map = np.abs(signal_array) * 0.05
        else:
            if measurement_uncertainty < 1.0:
                # Relative uncertainty
                uncertainty_map = np.abs(signal_array) * measurement_uncertainty
            else:
                # Absolute uncertainty
                uncertainty_map = np.full_like(signal_array, measurement_uncertainty)

        # Add interpolation uncertainty if provided
        if interpolation_uncertainty is not None:
            uncertainty_map = np.sqrt(uncertainty_map**2 + interpolation_uncertainty**2)

        # Set invalid regions to NaN
        uncertainty_map[~valid_mask] = np.nan

        # Calculate mean uncertainty
        uncertainty_clean = uncertainty_map[valid_mask]
        mean_uncertainty = np.mean(uncertainty_clean) if len(uncertainty_clean) > 0 else 0.0

        uncertainty_map_result = uncertainty_map if store_map else None

        return mean_uncertainty, uncertainty_map_result

    def calculate_confidence(
        self,
        signal_array: np.ndarray,
        snr_map: Optional[np.ndarray] = None,
        uncertainty_map: Optional[np.ndarray] = None,
        store_map: bool = True,
    ) -> Tuple[float, Optional[np.ndarray]]:
        """
        Calculate confidence scores for signal values.

        Args:
            signal_array: Signal array
            snr_map: Optional SNR map
            uncertainty_map: Optional uncertainty map
            store_map: Whether to store per-voxel confidence map

        Returns:
            (mean_confidence, confidence_map)
        """
        valid_mask = (~np.isnan(signal_array)) & (signal_array != 0.0)

        if not np.any(valid_mask):
            return 0.0, None

        confidence_map = np.zeros_like(signal_array, dtype=float)

        # Base confidence from signal strength
        signal_strength = np.abs(signal_array)
        signal_max = np.max(signal_strength[valid_mask]) if np.any(valid_mask) else 1.0
        if signal_max > 0:
            confidence_map[valid_mask] = np.clip(signal_strength[valid_mask] / signal_max, 0.0, 1.0)

        # Adjust based on SNR
        if snr_map is not None:
            # Normalize SNR to 0-1 range (assuming SNR range of -20 to 60 dB)
            snr_normalized = np.clip((snr_map + 20) / 80, 0.0, 1.0)
            confidence_map[valid_mask] = confidence_map[valid_mask] * 0.5 + snr_normalized[valid_mask] * 0.5

        # Adjust based on uncertainty
        if uncertainty_map is not None:
            uncertainty_normalized = 1.0 - np.clip(
                uncertainty_map / (np.max(uncertainty_map[valid_mask]) + 1e-10),
                0.0,
                1.0,
            )
            confidence_map[valid_mask] = confidence_map[valid_mask] * 0.7 + uncertainty_normalized[valid_mask] * 0.3

        confidence_map[~valid_mask] = 0.0

        # Calculate mean confidence
        confidence_clean = confidence_map[valid_mask]
        mean_confidence = np.mean(confidence_clean) if len(confidence_clean) > 0 else 0.0

        confidence_map_result = confidence_map if store_map else None

        return mean_confidence, confidence_map_result

    def assess_signal_quality(
        self,
        signal_name: str,
        signal_array: np.ndarray,
        noise_estimate: Optional[np.ndarray] = None,
        measurement_uncertainty: Optional[float] = None,
        store_maps: bool = True,
    ) -> SignalQualityMetrics:
        """
        Assess quality for a single signal.

        Args:
            signal_name: Name of the signal
            signal_array: Signal array
            noise_estimate: Optional noise estimate
            measurement_uncertainty: Optional measurement uncertainty
            store_maps: Whether to store per-voxel quality maps

        Returns:
            SignalQualityMetrics object
        """
        # Calculate SNR
        snr_mean, snr_std, snr_min, snr_max, snr_map = self.calculate_snr(signal_array, noise_estimate, store_map=store_maps)

        # Calculate uncertainty
        uncertainty_mean, uncertainty_map = self.calculate_uncertainty(
            signal_array, measurement_uncertainty, store_map=store_maps
        )

        # Calculate confidence
        confidence_mean, confidence_map = self.calculate_confidence(
            signal_array, snr_map, uncertainty_map, store_map=store_maps
        )

        # Overall quality score (weighted combination)
        quality_score = (
            0.4 * np.clip(confidence_mean, 0.0, 1.0)
            + 0.3 * np.clip((snr_mean + 20) / 80, 0.0, 1.0)
            + 0.3
            * np.clip(
                1.0 - (uncertainty_mean / (np.max(np.abs(signal_array)) + 1e-10)),
                0.0,
                1.0,
            )
        )

        return SignalQualityMetrics(
            signal_name=signal_name,
            snr_mean=snr_mean,
            snr_std=snr_std,
            snr_min=snr_min,
            snr_max=snr_max,
            uncertainty_mean=uncertainty_mean,
            confidence_mean=confidence_mean,
            quality_score=quality_score,
            snr_map=snr_map,
            uncertainty_map=uncertainty_map,
            confidence_map=confidence_map,
        )
