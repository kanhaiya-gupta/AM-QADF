"""
Fusion Quality Assessment

Assesses the quality of fused voxel signals.
Core computation is in C++ (am_qadf_native) only; no Python fallback.
Requires am_qadf_native to be built with fusion quality bindings.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# C++ fusion quality (required)
try:
    from am_qadf_native.fusion import (
        FusionQualityAssessor as CppFusionQualityAssessor,
        FusionQualityResult as CppFusionQualityResult,
    )
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False
    CppFusionQualityAssessor = None
    CppFusionQualityResult = None


@dataclass
class FusionQualityMetrics:
    """Fusion quality metrics."""

    fusion_accuracy: float  # Accuracy of fusion (0-1, higher is better)
    signal_consistency: float  # Consistency with source signals (0-1)
    fusion_completeness: float  # Coverage of fused signal (0-1)
    quality_score: float  # Overall quality score (0-1)

    # Detailed metrics
    per_signal_accuracy: Dict[str, float]  # Accuracy per source signal
    coverage_ratio: float  # Ratio of voxels with fused data
    residual_errors: Optional[np.ndarray] = None  # Residual errors per voxel

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        result = {
            "fusion_accuracy": self.fusion_accuracy,
            "signal_consistency": self.signal_consistency,
            "fusion_completeness": self.fusion_completeness,
            "quality_score": self.quality_score,
            "per_signal_accuracy": self.per_signal_accuracy,
            "coverage_ratio": self.coverage_ratio,
        }
        if self.residual_errors is not None:
            result["residual_errors_shape"] = self.residual_errors.shape
        return result

    @classmethod
    def from_cpp_result(cls, cpp: Any) -> "FusionQualityMetrics":
        """Build FusionQualityMetrics from C++ FusionQualityResult."""
        return cls(
            fusion_accuracy=float(cpp.fusion_accuracy),
            signal_consistency=float(cpp.signal_consistency),
            fusion_completeness=float(cpp.fusion_completeness),
            quality_score=float(cpp.quality_score),
            per_signal_accuracy=dict(cpp.per_signal_accuracy),
            coverage_ratio=float(cpp.coverage_ratio),
            residual_errors=None,  # C++ returns summary only
        )


class FusionQualityAssessor:
    """Assesses quality of fused voxel signals. C++ only; no Python fallback."""

    def __init__(self):
        """Initialize the fusion quality assessor. Requires am_qadf_native."""
        if not CPP_AVAILABLE or CppFusionQualityAssessor is None:
            raise ImportError(
                "Fusion quality requires C++ bindings. "
                "Build am_qadf_native with pybind11 and fusion quality (FusionQualityAssessor)."
            )
        self._cpp_assessor = CppFusionQualityAssessor()

    def assess_fusion_quality(
        self,
        fused_array: np.ndarray,
        source_arrays: Dict[str, np.ndarray],
        fusion_weights: Optional[Dict[str, float]] = None,
    ) -> FusionQualityMetrics:
        """
        Assess quality of fused signal.
        Numpy array input is not supported (no Python fallback). Use assess_from_grids()
        with OpenVDB grids from VoxelGrid.get_grid().
        """
        raise NotImplementedError(
            "Numpy array input is not supported. Use assess_from_grids(fused_grid, source_grids, fusion_weights) "
            "with OpenVDB FloatGrid objects (e.g. from VoxelGrid.get_grid(signal_name))."
        )

    def assess_from_grids(
        self,
        fused_grid: Any,
        source_grids: Dict[str, Any],
        fusion_weights: Optional[Dict[str, float]] = None,
    ) -> FusionQualityMetrics:
        """
        Assess quality using OpenVDB grids (C++ only).

        Args:
            fused_grid: OpenVDB FloatGrid (e.g. from VoxelGrid.get_grid(signal_name))
            source_grids: Dict mapping signal name to OpenVDB FloatGrid
            fusion_weights: Optional per-source weights

        Returns:
            FusionQualityMetrics
        """
        weights = fusion_weights if fusion_weights is not None else {}
        cpp_result = self._cpp_assessor.assess(fused_grid, source_grids, weights)
        return FusionQualityMetrics.from_cpp_result(cpp_result)

    def compare_fusion_strategies(
        self,
        voxel_data: Any,
        signals: List[str],
        strategies: List[str],
        quality_scores: Optional[Dict[str, float]] = None,
    ) -> Dict[str, FusionQualityMetrics]:
        """
        Compare different fusion strategies.
        Requires VoxelFusion and grid-based fusion; use assess_from_grids() with
        fused and source grids for quality assessment.
        """
        raise NotImplementedError(
            "compare_fusion_strategies is not implemented without Python fallback. "
            "Use assess_from_grids() with OpenVDB grids directly."
        )
