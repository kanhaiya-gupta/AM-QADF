"""
Validation

Validation metrics and correction quality assessment.
"""

from typing import Optional, Dict, List, Tuple, Any
import numpy as np
from dataclasses import dataclass
from enum import Enum


class AlignmentQuality(Enum):
    """Quality levels for alignment."""

    EXCELLENT = "excellent"  # < 0.05 mm error
    GOOD = "good"  # < 0.1 mm error
    ACCEPTABLE = "acceptable"  # < 0.2 mm error
    POOR = "poor"  # >= 0.2 mm error


@dataclass
class ValidationMetrics:
    """Validation metrics for correction quality."""

    mean_error: float  # Mean alignment error (mm)
    max_error: float  # Maximum alignment error (mm)
    rms_error: float  # Root mean square error (mm)
    std_error: float  # Standard deviation of error (mm)
    median_error: float  # Median error (mm)
    num_points: int  # Number of validation points
    quality: AlignmentQuality  # Overall quality assessment

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mean_error": self.mean_error,
            "max_error": self.max_error,
            "rms_error": self.rms_error,
            "std_error": self.std_error,
            "median_error": self.median_error,
            "num_points": self.num_points,
            "quality": self.quality.value,
        }


class CorrectionValidator:
    """
    Validate geometric corrections and assess alignment quality.
    """

    def __init__(
        self,
        excellent_threshold: float = 0.05,  # mm
        good_threshold: float = 0.1,  # mm
        acceptable_threshold: float = 0.2,  # mm
    ):
        """
        Initialize correction validator.

        Args:
            excellent_threshold: Error threshold for excellent quality (mm)
            good_threshold: Error threshold for good quality (mm)
            acceptable_threshold: Error threshold for acceptable quality (mm)
        """
        self.excellent_threshold = excellent_threshold
        self.good_threshold = good_threshold
        self.acceptable_threshold = acceptable_threshold

    def compute_alignment_error(self, corrected_points: np.ndarray, reference_points: np.ndarray) -> np.ndarray:
        """
        Compute alignment error between corrected and reference points.

        Args:
            corrected_points: Corrected points (N, 3)
            reference_points: Reference (ground truth) points (N, 3)

        Returns:
            Array of errors (N,)
        """
        corrected_points = np.asarray(corrected_points)
        reference_points = np.asarray(reference_points)

        if corrected_points.shape != reference_points.shape:
            raise ValueError("Corrected and reference points must have same shape")

        # Compute Euclidean distance for each point
        errors = np.linalg.norm(corrected_points - reference_points, axis=1)

        return errors

    def assess_quality(self, mean_error: float) -> AlignmentQuality:
        """
        Assess alignment quality based on mean error.

        Args:
            mean_error: Mean alignment error (mm)

        Returns:
            AlignmentQuality enum
        """
        if mean_error < self.excellent_threshold:
            return AlignmentQuality.EXCELLENT
        elif mean_error < self.good_threshold:
            return AlignmentQuality.GOOD
        elif mean_error < self.acceptable_threshold:
            return AlignmentQuality.ACCEPTABLE
        else:
            return AlignmentQuality.POOR

    def validate_correction(self, corrected_points: np.ndarray, reference_points: np.ndarray) -> ValidationMetrics:
        """
        Validate correction quality.

        Args:
            corrected_points: Corrected points (N, 3)
            reference_points: Reference (ground truth) points (N, 3)

        Returns:
            ValidationMetrics object
        """
        errors = self.compute_alignment_error(corrected_points, reference_points)

        mean_error = float(np.mean(errors))
        max_error = float(np.max(errors))
        rms_error = float(np.sqrt(np.mean(errors**2)))
        std_error = float(np.std(errors))
        median_error = float(np.median(errors))
        num_points = len(errors)

        quality = self.assess_quality(mean_error)

        return ValidationMetrics(
            mean_error=mean_error,
            max_error=max_error,
            rms_error=rms_error,
            std_error=std_error,
            median_error=median_error,
            num_points=num_points,
            quality=quality,
        )

    def compare_corrections(
        self,
        corrections: Dict[str, Tuple[np.ndarray, np.ndarray]],
        reference_points: np.ndarray,
    ) -> Dict[str, ValidationMetrics]:
        """
        Compare multiple correction methods.

        Args:
            corrections: Dictionary mapping method names to (corrected_points, description)
            reference_points: Reference (ground truth) points (N, 3)

        Returns:
            Dictionary mapping method names to ValidationMetrics
        """
        results = {}

        for method_name, (corrected_points, _) in corrections.items():
            metrics = self.validate_correction(corrected_points, reference_points)
            results[method_name] = metrics

        return results

    def generate_validation_report(self, metrics: ValidationMetrics, include_details: bool = True) -> str:
        """
        Generate human-readable validation report.

        Args:
            metrics: ValidationMetrics object
            include_details: Whether to include detailed statistics

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 60)
        report.append("CORRECTION VALIDATION REPORT")
        report.append("=" * 60)
        report.append("")
        report.append(f"Overall Quality: {metrics.quality.value.upper()}")
        report.append(f"Number of Points: {metrics.num_points}")
        report.append("")

        if include_details:
            report.append("Error Statistics:")
            report.append(f"  Mean Error:   {metrics.mean_error:.4f} mm")
            report.append(f"  Max Error:    {metrics.max_error:.4f} mm")
            report.append(f"  RMS Error:    {metrics.rms_error:.4f} mm")
            report.append(f"  Std Error:    {metrics.std_error:.4f} mm")
            report.append(f"  Median Error: {metrics.median_error:.4f} mm")
            report.append("")

        # Quality assessment
        if metrics.quality == AlignmentQuality.EXCELLENT:
            report.append("✅ Excellent alignment - correction is highly accurate")
        elif metrics.quality == AlignmentQuality.GOOD:
            report.append("✅ Good alignment - correction is accurate")
        elif metrics.quality == AlignmentQuality.ACCEPTABLE:
            report.append("⚠️  Acceptable alignment - correction may need improvement")
        else:
            report.append("❌ Poor alignment - correction needs significant improvement")

        report.append("=" * 60)

        return "\n".join(report)

    def validate_distortion_correction(
        self,
        original_points: np.ndarray,
        corrected_points: np.ndarray,
        reference_points: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Validate distortion correction improvement.

        Compares alignment before and after correction.

        Args:
            original_points: Original (distorted) points (N, 3)
            corrected_points: Corrected points (N, 3)
            reference_points: Reference (ground truth) points (N, 3)

        Returns:
            Dictionary with before/after metrics
        """
        # Compute errors before correction
        original_errors = self.compute_alignment_error(original_points, reference_points)
        original_mean = float(np.mean(original_errors))
        original_max = float(np.max(original_errors))
        original_rms = float(np.sqrt(np.mean(original_errors**2)))

        # Compute errors after correction
        corrected_errors = self.compute_alignment_error(corrected_points, reference_points)
        corrected_mean = float(np.mean(corrected_errors))
        corrected_max = float(np.max(corrected_errors))
        corrected_rms = float(np.sqrt(np.mean(corrected_errors**2)))

        # Compute improvement
        improvement_mean = original_mean - corrected_mean
        improvement_percent = (improvement_mean / original_mean * 100) if original_mean > 0 else 0.0

        return {
            "before": {
                "mean_error": original_mean,
                "max_error": original_max,
                "rms_error": original_rms,
                "quality": self.assess_quality(original_mean).value,
            },
            "after": {
                "mean_error": corrected_mean,
                "max_error": corrected_max,
                "rms_error": corrected_rms,
                "quality": self.assess_quality(corrected_mean).value,
            },
            "improvement": {
                "mean_error_reduction": improvement_mean,
                "improvement_percent": improvement_percent,
                "max_error_reduction": original_max - corrected_max,
                "rms_error_reduction": original_rms - corrected_rms,
            },
        }
