"""
Baseline Calculation for SPC

Calculates baseline statistics and control limits from historical data.
Provides adaptive control limit calculation for dynamic processes.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class BaselineStatistics:
    """Baseline statistics for SPC."""

    mean: float
    std: float
    median: float
    min: float
    max: float
    range: float
    sample_size: int
    subgroup_size: int
    within_subgroup_std: Optional[float] = None
    between_subgroup_std: Optional[float] = None
    overall_std: Optional[float] = None
    calculated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaselineCalculator:
    """
    Calculate baseline statistics for SPC.

    Provides methods for:
    - Calculating baseline statistics from historical data
    - Estimating within-subgroup and between-subgroup standard deviation
    - Calculating control limits for different chart types
    - Updating baselines with new data
    """

    def __init__(self):
        """Initialize baseline calculator."""
        pass

    def calculate_baseline(
        self,
        data: np.ndarray,
        subgroup_size: Optional[int] = None,
        method: str = "standard",
        config: Optional[Any] = None,  # SPCConfig type, avoiding circular import
    ) -> BaselineStatistics:
        """
        Calculate baseline statistics from data.

        Args:
            data: Input data array (1D for individual measurements, 2D for subgrouped data)
            subgroup_size: Subgroup size for X-bar/R/S charts (None for individual data)
            method: Calculation method ('standard', 'robust')
            config: Optional SPCConfig for parameters

        Returns:
            BaselineStatistics object
        """
        # Flatten if needed and remove invalid values
        data_flat = np.asarray(data).flatten()
        valid_mask = np.isfinite(data_flat)
        valid_data = data_flat[valid_mask]

        if len(valid_data) == 0:
            raise ValueError("No valid data points found for baseline calculation")

        if len(valid_data) < 10:
            logger.warning(f"Baseline calculated from only {len(valid_data)} samples (recommended: >100)")

        # Basic statistics
        mean = np.mean(valid_data)
        std = np.std(valid_data, ddof=1) if len(valid_data) > 1 else 0.0
        median = np.median(valid_data)
        min_val = np.min(valid_data)
        max_val = np.max(valid_data)
        range_val = max_val - min_val

        # Subgroup-specific statistics
        within_subgroup_std = None
        between_subgroup_std = None
        overall_std = std

        if subgroup_size is not None and subgroup_size > 1:
            # Calculate within-subgroup and between-subgroup statistics
            n_groups = len(valid_data) // subgroup_size
            if n_groups > 0:
                # Reshape into subgroups
                subgroups = valid_data[: n_groups * subgroup_size].reshape(n_groups, subgroup_size)

                # Within-subgroup standard deviation (pooled)
                subgroup_stds = np.std(subgroups, axis=1, ddof=1)
                within_subgroup_std = np.mean(subgroup_stds) if len(subgroup_stds) > 0 else std

                # Between-subgroup statistics
                subgroup_means = np.mean(subgroups, axis=1)
                between_subgroup_std = np.std(subgroup_means, ddof=1) if len(subgroup_means) > 1 else 0.0

                # Overall std (combines within and between)
                if within_subgroup_std is not None and between_subgroup_std is not None:
                    # Combined estimate: sqrt(var_within + var_between)
                    overall_std = np.sqrt(within_subgroup_std**2 + between_subgroup_std**2)
        else:
            subgroup_size = 1

        return BaselineStatistics(
            mean=mean,
            std=std,
            median=median,
            min=min_val,
            max=max_val,
            range=range_val,
            sample_size=len(valid_data),
            subgroup_size=subgroup_size if subgroup_size is not None else 1,
            within_subgroup_std=within_subgroup_std,
            between_subgroup_std=between_subgroup_std,
            overall_std=overall_std,
            calculated_at=datetime.now(),
            metadata={"method": method},
        )

    def estimate_within_subgroup_std(self, data: np.ndarray, subgroup_size: int) -> float:
        """
        Estimate within-subgroup standard deviation.

        Args:
            data: Input data array
            subgroup_size: Size of each subgroup

        Returns:
            Pooled within-subgroup standard deviation
        """
        data_flat = np.asarray(data).flatten()
        valid_data = data_flat[np.isfinite(data_flat)]

        if len(valid_data) < subgroup_size:
            return np.std(valid_data, ddof=1) if len(valid_data) > 1 else 0.0

        n_groups = len(valid_data) // subgroup_size
        if n_groups == 0:
            return np.std(valid_data, ddof=1) if len(valid_data) > 1 else 0.0

        subgroups = valid_data[: n_groups * subgroup_size].reshape(n_groups, subgroup_size)
        subgroup_stds = np.std(subgroups, axis=1, ddof=1)

        # Pooled standard deviation (mean of subgroup stds)
        return np.mean(subgroup_stds) if len(subgroup_stds) > 0 else 0.0

    def estimate_between_subgroup_std(self, subgroup_means: np.ndarray) -> float:
        """
        Estimate between-subgroup standard deviation.

        Args:
            subgroup_means: Array of subgroup means

        Returns:
            Standard deviation of subgroup means
        """
        valid_means = subgroup_means[np.isfinite(subgroup_means)]
        if len(valid_means) < 2:
            return 0.0
        return np.std(valid_means, ddof=1)

    def calculate_control_limits(
        self, baseline: BaselineStatistics, chart_type: str, subgroup_size: int = 5, sigma_multiplier: float = 3.0
    ) -> Dict[str, float]:
        """
        Calculate control limits for a given chart type.

        Args:
            baseline: Baseline statistics
            chart_type: Type of control chart ('xbar', 'r', 's', 'individual', 'moving_range')
            subgroup_size: Subgroup size (for X-bar, R, S charts)
            sigma_multiplier: Multiplier for control limits (default: 3.0 for 3-sigma)

        Returns:
            Dictionary with 'ucl', 'lcl', 'cl' keys
        """
        # Control limit constants (A2, D3, D4, B3, B4 depend on subgroup size)
        # Standard values for common subgroup sizes
        constants = self._get_control_chart_constants(subgroup_size)

        if chart_type == "xbar":
            # X-bar chart: CL = mean, UCL = mean + A2 * Rbar, LCL = mean - A2 * Rbar
            # Using std-based approximation: UCL = mean + 3*sigma_within/sqrt(n)
            if baseline.within_subgroup_std is not None:
                sigma_xbar = baseline.within_subgroup_std / np.sqrt(subgroup_size)
            else:
                sigma_xbar = baseline.std / np.sqrt(subgroup_size)

            cl = baseline.mean
            ucl = cl + sigma_multiplier * sigma_xbar
            lcl = cl - sigma_multiplier * sigma_xbar

        elif chart_type == "r":
            # R chart: CL = Rbar, UCL = D4 * Rbar, LCL = D3 * Rbar
            # Estimate Rbar from std: Rbar ≈ d2 * sigma (d2 depends on subgroup size)
            d2 = constants.get("d2", 2.326)  # Default for n=5
            rbar = d2 * (baseline.within_subgroup_std if baseline.within_subgroup_std is not None else baseline.std)

            cl = rbar
            ucl = constants.get("D4", 2.114) * rbar  # Default for n=5
            lcl = constants.get("D3", 0.0) * rbar

        elif chart_type == "s":
            # S chart: CL = Sbar, UCL = B4 * Sbar, LCL = B3 * Sbar
            sbar = baseline.within_subgroup_std if baseline.within_subgroup_std is not None else baseline.std

            cl = sbar
            ucl = constants.get("B4", 2.089) * sbar  # Default for n=5
            lcl = constants.get("B3", 0.0) * sbar

        elif chart_type == "individual":
            # Individual chart: CL = mean, UCL = mean + 3*sigma, LCL = mean - 3*sigma
            cl = baseline.mean
            ucl = cl + sigma_multiplier * baseline.std
            lcl = cl - sigma_multiplier * baseline.std

        elif chart_type == "moving_range":
            # Moving Range chart: CL = MRbar, UCL = 3.267 * MRbar (for n=2)
            # Estimate MRbar: MRbar ≈ 1.128 * sigma
            mrbar = 1.128 * baseline.std
            cl = mrbar
            ucl = 3.267 * mrbar  # D4 for n=2
            lcl = 0.0  # D3 for n=2 is 0

        else:
            raise ValueError(f"Unknown chart type: {chart_type}")

        return {"ucl": ucl, "lcl": lcl, "cl": cl}

    def _get_control_chart_constants(self, subgroup_size: int) -> Dict[str, float]:
        """
        Get control chart constants for given subgroup size.

        Standard values for subgroup sizes 2-10.

        Args:
            subgroup_size: Subgroup size

        Returns:
            Dictionary of constants (A2, D3, D4, B3, B4, d2)
        """
        # Standard SPC constants
        constants_table = {
            2: {"A2": 1.880, "D3": 0.0, "D4": 3.267, "B3": 0.0, "B4": 3.267, "d2": 1.128},
            3: {"A2": 1.023, "D3": 0.0, "D4": 2.574, "B3": 0.0, "B4": 2.568, "d2": 1.693},
            4: {"A2": 0.729, "D3": 0.0, "D4": 2.282, "B3": 0.0, "B4": 2.266, "d2": 2.059},
            5: {"A2": 0.577, "D3": 0.0, "D4": 2.114, "B3": 0.0, "B4": 2.089, "d2": 2.326},
            6: {"A2": 0.483, "D3": 0.0, "D4": 2.004, "B3": 0.076, "B4": 1.970, "d2": 2.534},
            7: {"A2": 0.419, "D3": 0.076, "D4": 1.924, "B3": 0.136, "B4": 1.882, "d2": 2.704},
            8: {"A2": 0.373, "D3": 0.136, "D4": 1.864, "B3": 0.184, "B4": 1.815, "d2": 2.847},
            9: {"A2": 0.337, "D3": 0.184, "D4": 1.816, "B3": 0.223, "B4": 1.761, "d2": 2.970},
            10: {"A2": 0.308, "D3": 0.223, "D4": 1.777, "B3": 0.256, "B4": 1.716, "d2": 3.078},
        }

        if subgroup_size in constants_table:
            return constants_table[subgroup_size]
        elif 2 <= subgroup_size <= 10:
            # Interpolate for intermediate values (simplified - use nearest)
            nearest = min(constants_table.keys(), key=lambda x: abs(x - subgroup_size))
            return constants_table[nearest]
        else:
            # For subgroup sizes outside range, use approximations
            logger.warning(f"Subgroup size {subgroup_size} outside standard range, using approximations")
            # Approximate formulas
            d2 = subgroup_size * 0.864 - 0.6 if subgroup_size <= 25 else 3.0
            A2 = 3.0 / (d2 * np.sqrt(subgroup_size))
            D4 = 1 + 3 * d2 / subgroup_size
            D3 = max(0.0, 1 - 3 * d2 / subgroup_size)
            B4 = 1 + 3 / np.sqrt(2 * (subgroup_size - 1))
            B3 = max(0.0, 1 - 3 / np.sqrt(2 * (subgroup_size - 1)))

            return {"A2": A2, "D3": D3, "D4": D4, "B3": B3, "B4": B4, "d2": d2}

    def update_baseline(
        self,
        current_baseline: BaselineStatistics,
        new_data: np.ndarray,
        method: str = "exponential_smoothing",
        alpha: float = 0.1,
    ) -> BaselineStatistics:
        """
        Update baseline statistics with new data.

        Args:
            current_baseline: Current baseline statistics
            new_data: New data points to incorporate
            method: Update method ('exponential_smoothing', 'cumulative', 'window')
            alpha: Smoothing parameter for exponential smoothing (0 < alpha < 1)

        Returns:
            Updated BaselineStatistics
        """
        new_data_flat = np.asarray(new_data).flatten()
        valid_new = new_data_flat[np.isfinite(new_data_flat)]

        if len(valid_new) == 0:
            logger.warning("No valid new data points, returning current baseline")
            return current_baseline

        if method == "exponential_smoothing":
            # Exponential smoothing: new_mean = alpha * new_sample_mean + (1-alpha) * old_mean
            new_mean = np.mean(valid_new)
            updated_mean = alpha * new_mean + (1 - alpha) * current_baseline.mean

            # For std, use pooled estimate
            new_std = np.std(valid_new, ddof=1) if len(valid_new) > 1 else 0.0
            updated_std = np.sqrt(alpha * new_std**2 + (1 - alpha) * current_baseline.std**2)

            # Update other statistics
            all_data = np.concatenate([valid_new, [current_baseline.mean] * len(valid_new)])
            # Simplified update - in practice would need full data
            updated_min = min(current_baseline.min, np.min(valid_new))
            updated_max = max(current_baseline.max, np.max(valid_new))
            updated_median = np.median(np.concatenate([valid_new, [current_baseline.median]]))

        elif method == "cumulative":
            # Cumulative update (would need full historical data)
            total_samples = current_baseline.sample_size + len(valid_new)
            new_mean = np.mean(valid_new)

            # Weighted average
            updated_mean = (current_baseline.mean * current_baseline.sample_size + new_mean * len(valid_new)) / total_samples

            # For std, would need full data - simplified here
            updated_std = current_baseline.std  # Placeholder
            updated_min = min(current_baseline.min, np.min(valid_new))
            updated_max = max(current_baseline.max, np.max(valid_new))
            updated_median = np.median(valid_new)  # Simplified

        else:  # window
            # Window-based (would need full data)
            logger.warning("Window method requires full historical data, using exponential smoothing")
            return self.update_baseline(current_baseline, new_data, method="exponential_smoothing", alpha=alpha)

        return BaselineStatistics(
            mean=updated_mean,
            std=updated_std,
            median=updated_median,
            min=updated_min,
            max=updated_max,
            range=updated_max - updated_min,
            sample_size=current_baseline.sample_size + len(valid_new),
            subgroup_size=current_baseline.subgroup_size,
            within_subgroup_std=current_baseline.within_subgroup_std,
            between_subgroup_std=current_baseline.between_subgroup_std,
            overall_std=updated_std,
            calculated_at=datetime.now(),
            metadata={**current_baseline.metadata, "update_method": method, "update_alpha": alpha},
        )


class AdaptiveLimitsCalculator:
    """
    Calculate adaptive control limits for dynamic processes.

    Provides methods for:
    - Adaptive control limit calculation
    - Drift detection
    - Recommendation for limit updates
    """

    def __init__(self):
        """Initialize adaptive limits calculator."""
        pass

    def calculate_adaptive_limits(
        self,
        historical_data: np.ndarray,
        current_data: np.ndarray,
        window_size: int = 100,
        update_frequency: int = 50,
        config: Optional[Any] = None,  # SPCConfig type
    ) -> Dict[str, Any]:
        """
        Calculate adaptive control limits.

        Args:
            historical_data: Historical baseline data
            current_data: Current monitoring data
            window_size: Size of sliding window for limit calculation
            update_frequency: Number of samples between updates
            config: Optional SPCConfig

        Returns:
            Dictionary with adaptive limits and metadata
        """
        baseline_calc = BaselineCalculator()

        # Use recent window of historical data for baseline
        if len(historical_data) > window_size:
            baseline_data = historical_data[-window_size:]
        else:
            baseline_data = historical_data

        baseline = baseline_calc.calculate_baseline(baseline_data, subgroup_size=config.subgroup_size if config else 5)

        # Check for drift
        drift_info = self.detect_drift(baseline, current_data)

        # Calculate limits
        sigma_mult = config.control_limit_sigma if config else 3.0
        limits = baseline_calc.calculate_control_limits(
            baseline, "xbar", subgroup_size=config.subgroup_size if config else 5, sigma_multiplier=sigma_mult
        )

        return {
            "limits": limits,
            "baseline": baseline,
            "drift_detected": drift_info.get("drift_detected", False),
            "drift_magnitude": drift_info.get("drift_magnitude", 0.0),
            "recommend_update": self.recommend_limit_update(baseline, current_data),
        }

    def detect_drift(self, baseline: BaselineStatistics, recent_data: np.ndarray) -> Dict[str, Any]:
        """
        Detect drift in process mean.

        Args:
            baseline: Baseline statistics
            recent_data: Recent process data

        Returns:
            Dictionary with drift detection results
        """
        recent_flat = np.asarray(recent_data).flatten()
        valid_recent = recent_flat[np.isfinite(recent_flat)]

        if len(valid_recent) < 10:
            return {"drift_detected": False, "drift_magnitude": 0.0, "insufficient_data": True}

        recent_mean = np.mean(valid_recent)
        recent_std = np.std(valid_recent, ddof=1) if len(valid_recent) > 1 else baseline.std

        # Z-test for mean shift
        n = len(valid_recent)
        z_score = (recent_mean - baseline.mean) / (baseline.std / np.sqrt(n)) if baseline.std > 0 else 0.0

        # Detect drift if |z| > 2 (approximately 2-sigma shift)
        drift_detected = abs(z_score) > 2.0
        drift_magnitude = recent_mean - baseline.mean

        return {
            "drift_detected": drift_detected,
            "drift_magnitude": drift_magnitude,
            "z_score": z_score,
            "recent_mean": recent_mean,
            "baseline_mean": baseline.mean,
            "shift_sigma": abs(z_score),
        }

    def recommend_limit_update(self, baseline: BaselineStatistics, recent_data: np.ndarray) -> bool:
        """
        Recommend whether to update control limits.

        Args:
            baseline: Current baseline statistics
            recent_data: Recent process data

        Returns:
            True if limits should be updated, False otherwise
        """
        drift_info = self.detect_drift(baseline, recent_data)

        # Recommend update if:
        # 1. Drift detected (significant shift)
        # 2. Sufficient data for stable estimate (>30 samples)
        recent_flat = np.asarray(recent_data).flatten()
        valid_recent = recent_flat[np.isfinite(recent_flat)]

        sufficient_data = len(valid_recent) >= 30
        significant_drift = drift_info.get("shift_sigma", 0.0) > 2.0

        return sufficient_data and significant_drift
