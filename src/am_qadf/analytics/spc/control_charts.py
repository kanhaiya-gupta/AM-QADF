"""
Control Charts for SPC

Generates and analyzes control charts for univariate SPC:
- X-bar (mean) control chart
- R (range) control chart
- S (standard deviation) control chart
- Individual measurements chart
- Moving range chart
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging

from .baseline_calculation import BaselineCalculator, BaselineStatistics

logger = logging.getLogger(__name__)


@dataclass
class ControlChartResult:
    """Result of control chart analysis."""

    chart_type: str  # 'xbar', 'r', 's', 'individual', 'moving_range'
    center_line: float  # Center line (CL)
    upper_control_limit: float  # UCL
    lower_control_limit: float  # LCL
    upper_warning_limit: Optional[float] = None  # UWL
    lower_warning_limit: Optional[float] = None  # LWL
    sample_values: np.ndarray = field(default_factory=lambda: np.array([]))  # Sample values
    sample_indices: np.ndarray = field(default_factory=lambda: np.array([]))  # Sample indices/timestamps
    out_of_control_points: List[int] = field(default_factory=list)  # Indices of OOC points
    rule_violations: Dict[str, List[int]] = field(default_factory=dict)  # Rule violations by rule name
    baseline_stats: Dict[str, float] = field(default_factory=dict)  # Baseline statistics
    metadata: Dict[str, Any] = field(default_factory=dict)


class ControlChartGenerator:
    """
    Generate control charts for SPC.

    Provides methods for:
    - Creating X-bar, R, S charts for subgrouped data
    - Creating Individual and Moving Range charts
    - Detecting out-of-control points
    - Updating control limits
    """

    def __init__(self):
        """Initialize control chart generator."""
        self.baseline_calc = BaselineCalculator()

    def create_xbar_chart(
        self, data: np.ndarray, subgroup_size: int = 5, config: Optional[Any] = None  # SPCConfig type
    ) -> ControlChartResult:
        """
        Create X-bar (mean) control chart.

        Args:
            data: Input data (1D array will be grouped into subgroups)
            subgroup_size: Size of each subgroup
            config: Optional SPCConfig

        Returns:
            ControlChartResult for X-bar chart
        """
        data_flat = np.asarray(data).flatten()
        valid_mask = np.isfinite(data_flat)
        valid_data = data_flat[valid_mask]

        if len(valid_data) < subgroup_size:
            raise ValueError(f"Insufficient data for subgroups of size {subgroup_size}")

        # Form subgroups
        n_groups = len(valid_data) // subgroup_size
        if n_groups == 0:
            raise ValueError(f"Need at least {subgroup_size} samples for X-bar chart")

        subgroups = valid_data[: n_groups * subgroup_size].reshape(n_groups, subgroup_size)
        subgroup_means = np.mean(subgroups, axis=1)

        # Calculate baseline and control limits
        baseline = self.baseline_calc.calculate_baseline(valid_data, subgroup_size=subgroup_size, config=config)

        sigma_mult = config.control_limit_sigma if config else 3.0
        limits = self.baseline_calc.calculate_control_limits(
            baseline, "xbar", subgroup_size=subgroup_size, sigma_multiplier=sigma_mult
        )

        # Calculate warning limits (2-sigma) if enabled
        warning_ucl = None
        warning_lcl = None
        if config and config.enable_warnings:
            warning_sigma = config.warning_sigma if hasattr(config, "warning_sigma") else 2.0
            if baseline.within_subgroup_std is not None:
                sigma_xbar = baseline.within_subgroup_std / np.sqrt(subgroup_size)
            else:
                sigma_xbar = baseline.std / np.sqrt(subgroup_size)
            warning_ucl = baseline.mean + warning_sigma * sigma_xbar
            warning_lcl = baseline.mean - warning_sigma * sigma_xbar

        # Detect out-of-control points
        ooc_indices = self._detect_out_of_control(subgroup_means, limits["ucl"], limits["lcl"])

        # Store baseline stats
        baseline_stats = {
            "mean": baseline.mean,
            "std": baseline.std,
            "within_subgroup_std": baseline.within_subgroup_std,
            "sample_size": baseline.sample_size,
            "subgroup_size": subgroup_size,
        }

        return ControlChartResult(
            chart_type="xbar",
            center_line=limits["cl"],
            upper_control_limit=limits["ucl"],
            lower_control_limit=limits["lcl"],
            upper_warning_limit=warning_ucl,
            lower_warning_limit=warning_lcl,
            sample_values=subgroup_means,
            sample_indices=np.arange(len(subgroup_means)),
            out_of_control_points=ooc_indices,
            baseline_stats=baseline_stats,
            metadata={"subgroup_size": subgroup_size, "n_subgroups": n_groups},
        )

    def create_r_chart(self, data: np.ndarray, subgroup_size: int = 5, config: Optional[Any] = None) -> ControlChartResult:
        """
        Create R (range) control chart.

        Args:
            data: Input data (1D array will be grouped into subgroups)
            subgroup_size: Size of each subgroup
            config: Optional SPCConfig

        Returns:
            ControlChartResult for R chart
        """
        data_flat = np.asarray(data).flatten()
        valid_mask = np.isfinite(data_flat)
        valid_data = data_flat[valid_mask]

        if len(valid_data) < subgroup_size:
            raise ValueError(f"Insufficient data for subgroups of size {subgroup_size}")

        # Form subgroups
        n_groups = len(valid_data) // subgroup_size
        if n_groups == 0:
            raise ValueError(f"Need at least {subgroup_size} samples for R chart")

        subgroups = valid_data[: n_groups * subgroup_size].reshape(n_groups, subgroup_size)
        subgroup_ranges = np.ptp(subgroups, axis=1)  # Peak-to-peak (range)

        # Calculate baseline
        baseline = self.baseline_calc.calculate_baseline(valid_data, subgroup_size=subgroup_size, config=config)

        # Calculate control limits for R chart
        sigma_mult = config.control_limit_sigma if config else 3.0
        limits = self.baseline_calc.calculate_control_limits(
            baseline, "r", subgroup_size=subgroup_size, sigma_multiplier=sigma_mult
        )

        # Detect out-of-control points
        ooc_indices = self._detect_out_of_control(subgroup_ranges, limits["ucl"], limits["lcl"])

        baseline_stats = {
            "mean": np.mean(subgroup_ranges),
            "std": np.std(subgroup_ranges, ddof=1),
            "sample_size": baseline.sample_size,
            "subgroup_size": subgroup_size,
        }

        return ControlChartResult(
            chart_type="r",
            center_line=limits["cl"],
            upper_control_limit=limits["ucl"],
            lower_control_limit=limits["lcl"],
            sample_values=subgroup_ranges,
            sample_indices=np.arange(len(subgroup_ranges)),
            out_of_control_points=ooc_indices,
            baseline_stats=baseline_stats,
            metadata={"subgroup_size": subgroup_size, "n_subgroups": n_groups},
        )

    def create_s_chart(self, data: np.ndarray, subgroup_size: int = 5, config: Optional[Any] = None) -> ControlChartResult:
        """
        Create S (standard deviation) control chart.

        Args:
            data: Input data (1D array will be grouped into subgroups)
            subgroup_size: Size of each subgroup
            config: Optional SPCConfig

        Returns:
            ControlChartResult for S chart
        """
        data_flat = np.asarray(data).flatten()
        valid_mask = np.isfinite(data_flat)
        valid_data = data_flat[valid_mask]

        if len(valid_data) < subgroup_size:
            raise ValueError(f"Insufficient data for subgroups of size {subgroup_size}")

        # Form subgroups
        n_groups = len(valid_data) // subgroup_size
        if n_groups == 0:
            raise ValueError(f"Need at least {subgroup_size} samples for S chart")

        subgroups = valid_data[: n_groups * subgroup_size].reshape(n_groups, subgroup_size)
        subgroup_stds = np.std(subgroups, axis=1, ddof=1)  # Sample standard deviation

        # Calculate baseline
        baseline = self.baseline_calc.calculate_baseline(valid_data, subgroup_size=subgroup_size, config=config)

        # Calculate control limits for S chart
        sigma_mult = config.control_limit_sigma if config else 3.0
        limits = self.baseline_calc.calculate_control_limits(
            baseline, "s", subgroup_size=subgroup_size, sigma_multiplier=sigma_mult
        )

        # Detect out-of-control points
        ooc_indices = self._detect_out_of_control(subgroup_stds, limits["ucl"], limits["lcl"])

        baseline_stats = {
            "mean": np.mean(subgroup_stds),
            "std": np.std(subgroup_stds, ddof=1),
            "sample_size": baseline.sample_size,
            "subgroup_size": subgroup_size,
        }

        return ControlChartResult(
            chart_type="s",
            center_line=limits["cl"],
            upper_control_limit=limits["ucl"],
            lower_control_limit=limits["lcl"],
            sample_values=subgroup_stds,
            sample_indices=np.arange(len(subgroup_stds)),
            out_of_control_points=ooc_indices,
            baseline_stats=baseline_stats,
            metadata={"subgroup_size": subgroup_size, "n_subgroups": n_groups},
        )

    def create_individual_chart(self, data: np.ndarray, config: Optional[Any] = None) -> ControlChartResult:
        """
        Create Individual measurements control chart.

        Args:
            data: Input data (1D array of individual measurements)
            config: Optional SPCConfig

        Returns:
            ControlChartResult for Individual chart
        """
        data_flat = np.asarray(data).flatten()
        valid_mask = np.isfinite(data_flat)
        valid_data = data_flat[valid_mask]

        if len(valid_data) < 10:
            raise ValueError("Need at least 10 samples for Individual chart")

        # Calculate baseline
        baseline = self.baseline_calc.calculate_baseline(valid_data, subgroup_size=1, config=config)

        # Calculate control limits
        sigma_mult = config.control_limit_sigma if config else 3.0
        limits = self.baseline_calc.calculate_control_limits(
            baseline, "individual", subgroup_size=1, sigma_multiplier=sigma_mult
        )

        # Warning limits if enabled
        warning_ucl = None
        warning_lcl = None
        if config and config.enable_warnings:
            warning_sigma = config.warning_sigma if hasattr(config, "warning_sigma") else 2.0
            warning_ucl = baseline.mean + warning_sigma * baseline.std
            warning_lcl = baseline.mean - warning_sigma * baseline.std

        # Detect out-of-control points
        ooc_indices = self._detect_out_of_control(valid_data, limits["ucl"], limits["lcl"])

        baseline_stats = {"mean": baseline.mean, "std": baseline.std, "sample_size": baseline.sample_size}

        return ControlChartResult(
            chart_type="individual",
            center_line=limits["cl"],
            upper_control_limit=limits["ucl"],
            lower_control_limit=limits["lcl"],
            upper_warning_limit=warning_ucl,
            lower_warning_limit=warning_lcl,
            sample_values=valid_data,
            sample_indices=np.arange(len(valid_data)),
            out_of_control_points=ooc_indices,
            baseline_stats=baseline_stats,
            metadata={"n_samples": len(valid_data)},
        )

    def create_moving_range_chart(
        self, data: np.ndarray, window_size: int = 2, config: Optional[Any] = None
    ) -> ControlChartResult:
        """
        Create Moving Range (MR) control chart.

        Args:
            data: Input data (1D array of individual measurements)
            window_size: Size of moving window (typically 2)
            config: Optional SPCConfig

        Returns:
            ControlChartResult for Moving Range chart
        """
        data_flat = np.asarray(data).flatten()
        valid_mask = np.isfinite(data_flat)
        valid_data = data_flat[valid_mask]

        if len(valid_data) < window_size + 1:
            raise ValueError(f"Need at least {window_size + 1} samples for Moving Range chart")

        # Calculate moving ranges
        moving_ranges = np.abs(np.diff(valid_data))

        # Calculate baseline for MR
        baseline = self.baseline_calc.calculate_baseline(moving_ranges, subgroup_size=1, config=config)

        # Calculate control limits for MR chart
        sigma_mult = config.control_limit_sigma if config else 3.0
        limits = self.baseline_calc.calculate_control_limits(
            baseline, "moving_range", subgroup_size=window_size, sigma_multiplier=sigma_mult
        )

        # Detect out-of-control points (MR chart typically only has UCL)
        ooc_indices = []
        for i, mr in enumerate(moving_ranges):
            if mr > limits["ucl"]:
                ooc_indices.append(i)

        baseline_stats = {"mean": baseline.mean, "std": baseline.std, "sample_size": len(moving_ranges)}

        return ControlChartResult(
            chart_type="moving_range",
            center_line=limits["cl"],
            upper_control_limit=limits["ucl"],
            lower_control_limit=limits["lcl"],  # Typically 0 for MR
            sample_values=moving_ranges,
            sample_indices=np.arange(len(moving_ranges)),
            out_of_control_points=ooc_indices,
            baseline_stats=baseline_stats,
            metadata={"window_size": window_size, "n_samples": len(moving_ranges)},
        )

    def create_xbar_r_charts(
        self, data: np.ndarray, subgroup_size: int = 5, config: Optional[Any] = None
    ) -> Tuple[ControlChartResult, ControlChartResult]:
        """
        Create paired X-bar and R charts.

        Args:
            data: Input data (1D array will be grouped into subgroups)
            subgroup_size: Size of each subgroup
            config: Optional SPCConfig

        Returns:
            Tuple of (XBarChartResult, RChartResult)
        """
        xbar_result = self.create_xbar_chart(data, subgroup_size, config)
        r_result = self.create_r_chart(data, subgroup_size, config)
        return (xbar_result, r_result)

    def create_xbar_s_charts(
        self, data: np.ndarray, subgroup_size: int = 5, config: Optional[Any] = None
    ) -> Tuple[ControlChartResult, ControlChartResult]:
        """
        Create paired X-bar and S charts.

        Args:
            data: Input data (1D array will be grouped into subgroups)
            subgroup_size: Size of each subgroup
            config: Optional SPCConfig

        Returns:
            Tuple of (XBarChartResult, SChartResult)
        """
        xbar_result = self.create_xbar_chart(data, subgroup_size, config)
        s_result = self.create_s_chart(data, subgroup_size, config)
        return (xbar_result, s_result)

    def update_control_limits(
        self, chart_result: ControlChartResult, new_data: np.ndarray, method: str = "adaptive"
    ) -> ControlChartResult:
        """
        Update control limits based on new data.

        Args:
            chart_result: Existing chart result
            new_data: New data points
            method: Update method ('adaptive', 'recalculate')

        Returns:
            Updated ControlChartResult
        """
        if method == "recalculate":
            # Recalculate from all data (would need historical data)
            logger.warning("Recalculate method requires full historical data, using adaptive method")
            method = "adaptive"

        # Adaptive update: update baseline and recalculate limits
        new_data_flat = np.asarray(new_data).flatten()
        valid_new = new_data_flat[np.isfinite(new_data_flat)]

        if len(valid_new) == 0:
            logger.warning("No valid new data, returning original chart")
            return chart_result

        # Create baseline from current stats
        current_baseline = BaselineStatistics(
            mean=chart_result.baseline_stats.get("mean", chart_result.center_line),
            std=chart_result.baseline_stats.get("std", 0.0),
            median=chart_result.center_line,
            min=0.0,
            max=0.0,
            range=0.0,
            sample_size=chart_result.baseline_stats.get("sample_size", 0),
            subgroup_size=chart_result.metadata.get("subgroup_size", 1),
        )

        # Update baseline
        updated_baseline = self.baseline_calc.update_baseline(current_baseline, valid_new, method="exponential_smoothing")

        # Recalculate limits
        subgroup_size = chart_result.metadata.get("subgroup_size", 5)
        limits = self.baseline_calc.calculate_control_limits(
            updated_baseline,
            chart_result.chart_type,
            subgroup_size=subgroup_size,
            sigma_multiplier=3.0,  # Default, should come from config
        )

        # Update chart result
        chart_result.center_line = limits["cl"]
        chart_result.upper_control_limit = limits["ucl"]
        chart_result.lower_control_limit = limits["lcl"]
        chart_result.baseline_stats.update(
            {"mean": updated_baseline.mean, "std": updated_baseline.std, "sample_size": updated_baseline.sample_size}
        )
        chart_result.metadata["last_updated"] = updated_baseline.calculated_at.isoformat()

        return chart_result

    def detect_out_of_control(self, values: np.ndarray, ucl: float, lcl: float) -> List[int]:
        """
        Detect out-of-control points.

        Args:
            values: Sample values
            ucl: Upper control limit
            lcl: Lower control limit

        Returns:
            List of indices of out-of-control points
        """
        ooc_indices = []
        for i, val in enumerate(values):
            if np.isfinite(val) and (val > ucl or val < lcl):
                ooc_indices.append(i)
        return ooc_indices

    def _detect_out_of_control(self, values: np.ndarray, ucl: float, lcl: float) -> List[int]:
        """Internal helper for out-of-control detection."""
        return self.detect_out_of_control(values, ucl, lcl)


# Convenience classes (can be extended later)
class XBarChart(ControlChartGenerator):
    """X-bar chart generator."""

    def create(self, data: np.ndarray, subgroup_size: int = 5, config: Optional[Any] = None) -> ControlChartResult:
        return self.create_xbar_chart(data, subgroup_size, config)


class RChart(ControlChartGenerator):
    """R chart generator."""

    def create(self, data: np.ndarray, subgroup_size: int = 5, config: Optional[Any] = None) -> ControlChartResult:
        return self.create_r_chart(data, subgroup_size, config)


class SChart(ControlChartGenerator):
    """S chart generator."""

    def create(self, data: np.ndarray, subgroup_size: int = 5, config: Optional[Any] = None) -> ControlChartResult:
        return self.create_s_chart(data, subgroup_size, config)


class IndividualChart(ControlChartGenerator):
    """Individual measurements chart generator."""

    def create(self, data: np.ndarray, config: Optional[Any] = None) -> ControlChartResult:
        return self.create_individual_chart(data, config)


class MovingRangeChart(ControlChartGenerator):
    """Moving range chart generator."""

    def create(self, data: np.ndarray, window_size: int = 2, config: Optional[Any] = None) -> ControlChartResult:
        return self.create_moving_range_chart(data, window_size, config)
