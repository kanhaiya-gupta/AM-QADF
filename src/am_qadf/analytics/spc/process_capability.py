"""
Process Capability Analysis

Calculates process capability indices (Cp, Cpk, Pp, Ppk) and assesses process performance.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging

from .baseline_calculation import BaselineCalculator

logger = logging.getLogger(__name__)


@dataclass
class ProcessCapabilityResult:
    """Result of process capability analysis."""

    cp: float  # Process capability index
    cpk: float  # Process capability index (centered)
    pp: float  # Process performance index
    ppk: float  # Process performance index (centered)
    cpu: float  # Upper capability index
    cpl: float  # Lower capability index
    specification_limits: Tuple[float, float]  # USL, LSL
    target_value: Optional[float]  # Target value
    process_mean: float  # Actual process mean
    process_std: float  # Actual process standard deviation
    within_subgroup_std: Optional[float] = None  # Within-subgroup std (for Cp)
    overall_std: Optional[float] = None  # Overall std (for Pp)
    capability_rating: str = "Unknown"  # 'Excellent', 'Adequate', 'Inadequate', etc.
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProcessCapabilityAnalyzer:
    """
    Analyze process capability.

    Provides methods for:
    - Calculating process capability indices (Cp, Cpk, Pp, Ppk)
    - Rating process capability
    - Estimating parts per million (PPM) out of specification
    - Comparing short-term vs. long-term capability
    """

    def __init__(self):
        """Initialize process capability analyzer."""
        self.baseline_calc = BaselineCalculator()

    def calculate_capability(
        self,
        data: np.ndarray,
        specification_limits: Tuple[float, float],
        target_value: Optional[float] = None,
        subgroup_size: Optional[int] = None,
        config: Optional[Any] = None,  # SPCConfig type
    ) -> ProcessCapabilityResult:
        """
        Calculate process capability indices.

        Args:
            data: Process data (1D array)
            specification_limits: (USL, LSL) tuple - upper and lower specification limits
            target_value: Optional target/center value
            subgroup_size: Optional subgroup size for within-subgroup std estimation
            config: Optional SPCConfig

        Returns:
            ProcessCapabilityResult object
        """
        usl, lsl = specification_limits

        if usl <= lsl:
            raise ValueError(f"USL ({usl}) must be greater than LSL ({lsl})")

        data_flat = np.asarray(data).flatten()
        valid_mask = np.isfinite(data_flat)
        valid_data = data_flat[valid_mask]

        if len(valid_data) < 30:
            logger.warning(f"Capability analysis recommended with >= 30 samples, got {len(valid_data)}")

        # Calculate process statistics
        process_mean = np.mean(valid_data)
        process_std = np.std(valid_data, ddof=1) if len(valid_data) > 1 else 0.0

        # Calculate baseline to get within-subgroup and overall std
        baseline = None
        within_subgroup_std = None
        overall_std = process_std

        if subgroup_size is not None and subgroup_size > 1:
            try:
                baseline = self.baseline_calc.calculate_baseline(valid_data, subgroup_size=subgroup_size, config=config)
                within_subgroup_std = baseline.within_subgroup_std
                overall_std = baseline.overall_std if baseline.overall_std is not None else process_std
            except Exception as e:
                logger.warning(f"Could not calculate within-subgroup std: {e}, using overall std")
                within_subgroup_std = None

        # If within_subgroup_std not available, use overall std as approximation
        if within_subgroup_std is None:
            within_subgroup_std = process_std

        # Calculate capability indices
        cp = self.calculate_cp(usl, lsl, within_subgroup_std)
        cpu, cpl, cpk = self.calculate_cpk(usl, lsl, process_mean, within_subgroup_std)
        pp = self.calculate_pp(usl, lsl, overall_std)
        ppu, ppl, ppk = self.calculate_ppk(usl, lsl, process_mean, overall_std)

        # Rate capability
        capability_rating = self.rate_capability(cpk)

        # Use target value if provided, otherwise use process mean
        if target_value is None:
            target_value = process_mean

        result = ProcessCapabilityResult(
            cp=cp,
            cpk=cpk,
            pp=pp,
            ppk=ppk,
            cpu=cpu,
            cpl=cpl,
            specification_limits=(usl, lsl),
            target_value=target_value,
            process_mean=process_mean,
            process_std=process_std,
            within_subgroup_std=within_subgroup_std,
            overall_std=overall_std,
            capability_rating=capability_rating,
            metadata={"sample_size": len(valid_data), "subgroup_size": subgroup_size, "ppu": ppu, "ppl": ppl},
        )

        return result

    def calculate_cp(self, usl: float, lsl: float, within_std: float) -> float:
        """
        Calculate Cp (process capability index).

        Cp = (USL - LSL) / (6 * σ_within)

        Args:
            usl: Upper specification limit
            lsl: Lower specification limit
            within_std: Within-subgroup standard deviation

        Returns:
            Cp index
        """
        if within_std <= 0:
            return 0.0
        return (usl - lsl) / (6.0 * within_std)

    def calculate_cpk(self, usl: float, lsl: float, mean: float, within_std: float) -> Tuple[float, float, float]:
        """
        Calculate Cpk (centered process capability index) and components.

        Cpk = min[(USL - μ) / (3σ_within), (μ - LSL) / (3σ_within)]
        CPU = (USL - μ) / (3σ_within)
        CPL = (μ - LSL) / (3σ_within)

        Args:
            usl: Upper specification limit
            lsl: Lower specification limit
            mean: Process mean
            within_std: Within-subgroup standard deviation

        Returns:
            Tuple of (CPU, CPL, CPK)
        """
        if within_std <= 0:
            return (0.0, 0.0, 0.0)

        cpu = (usl - mean) / (3.0 * within_std) if usl != float("inf") and not np.isinf(usl) else float("inf")
        cpl = (mean - lsl) / (3.0 * within_std) if lsl != float("-inf") and not np.isinf(lsl) else float("inf")
        # Cpk is min of CPU and CPL, but if one is inf, use the other
        if np.isinf(cpu) and np.isinf(cpl):
            cpk = float("inf")
        elif np.isinf(cpu):
            cpk = cpl
        elif np.isinf(cpl):
            cpk = cpu
        else:
            cpk = min(cpu, cpl)

        return (cpu, cpl, cpk)

    def calculate_pp(self, usl: float, lsl: float, overall_std: float) -> float:
        """
        Calculate Pp (process performance index).

        Pp = (USL - LSL) / (6 * σ_overall)

        Args:
            usl: Upper specification limit
            lsl: Lower specification limit
            overall_std: Overall process standard deviation

        Returns:
            Pp index
        """
        if overall_std <= 0:
            return 0.0
        return (usl - lsl) / (6.0 * overall_std)

    def calculate_ppk(self, usl: float, lsl: float, mean: float, overall_std: float) -> Tuple[float, float, float]:
        """
        Calculate Ppk (centered process performance index) and components.

        Ppk = min[(USL - μ) / (3σ_overall), (μ - LSL) / (3σ_overall)]
        PPU = (USL - μ) / (3σ_overall)
        PPL = (μ - LSL) / (3σ_overall)

        Args:
            usl: Upper specification limit
            lsl: Lower specification limit
            mean: Process mean
            overall_std: Overall process standard deviation

        Returns:
            Tuple of (PPU, PPL, PPK)
        """
        if overall_std <= 0:
            return (0.0, 0.0, 0.0)

        ppu = (usl - mean) / (3.0 * overall_std) if usl != float("inf") else float("inf")
        ppl = (mean - lsl) / (3.0 * overall_std) if lsl != float("-inf") else float("inf")
        ppk = min(ppu, ppl) if not (np.isinf(ppu) or np.isinf(ppl)) else float("inf")

        return (ppu, ppl, ppk)

    def rate_capability(self, cpk: float) -> str:
        """
        Rate process capability based on Cpk value.

        Rating guidelines:
        - Excellent: Cpk >= 1.67 (6-sigma process)
        - Adequate: 1.33 <= Cpk < 1.67 (5-sigma process)
        - Marginal: 1.0 <= Cpk < 1.33 (4-sigma process)
        - Inadequate: Cpk < 1.0

        Args:
            cpk: Cpk value

        Returns:
            Capability rating string
        """
        if np.isinf(cpk) or np.isnan(cpk):
            return "Invalid"
        elif cpk >= 1.67:
            return "Excellent"
        elif cpk >= 1.33:
            return "Adequate"
        elif cpk >= 1.0:
            return "Marginal"
        else:
            return "Inadequate"

    def estimate_ppm(self, cpk: float, process_mean: float, usl: float, lsl: float, process_std: float) -> float:
        """
        Estimate parts per million (PPM) out of specification.

        Uses normal distribution to estimate proportion beyond specification limits.

        Args:
            cpk: Cpk value (used for validation)
            process_mean: Process mean
            usl: Upper specification limit
            lsl: Lower specification limit
            process_std: Process standard deviation

        Returns:
            Estimated PPM out of specification
        """
        if process_std <= 0:
            return float("inf") if process_mean < lsl or process_mean > usl else 0.0

        from scipy import stats

        # Calculate proportion beyond USL
        if usl != float("inf"):
            z_upper = (usl - process_mean) / process_std
            p_above_usl = 1 - stats.norm.cdf(z_upper)
        else:
            p_above_usl = 0.0

        # Calculate proportion beyond LSL
        if lsl != float("-inf"):
            z_lower = (lsl - process_mean) / process_std
            p_below_lsl = stats.norm.cdf(z_lower)
        else:
            p_below_lsl = 0.0

        # Total proportion out of spec
        p_out_of_spec = p_above_usl + p_below_lsl

        # Convert to PPM
        ppm = p_out_of_spec * 1_000_000

        return ppm

    def compare_short_term_long_term(self, cp: float, cpk: float, pp: float, ppk: float) -> Dict[str, Any]:
        """
        Compare short-term (Cp/Cpk) vs. long-term (Pp/Ppk) capability.

        Args:
            cp: Short-term capability (Cp)
            cpk: Short-term centered capability (Cpk)
            pp: Long-term performance (Pp)
            ppk: Long-term centered performance (Ppk)

        Returns:
            Dictionary with comparison results
        """
        # Capability gap indicates process instability
        cp_pp_gap = cp - pp if not (np.isinf(cp) or np.isinf(pp)) else 0.0
        cpk_ppk_gap = cpk - ppk if not (np.isinf(cpk) or np.isinf(ppk)) else 0.0

        # Interpretation
        if cpk_ppk_gap > 0.5:
            stability = "Unstable - significant shift/drift"
        elif cpk_ppk_gap > 0.2:
            stability = "Moderate instability"
        elif cpk_ppk_gap > 0.0:
            stability = "Minor instability"
        else:
            stability = "Stable process"

        return {
            "cp": cp,
            "cpk": cpk,
            "pp": pp,
            "ppk": ppk,
            "capability_gap": cp_pp_gap,
            "centered_gap": cpk_ppk_gap,
            "stability_assessment": stability,
            "improvement_potential": max(0.0, cpk_ppk_gap),
        }
