"""
Unit tests for validation.

Tests for AlignmentQuality, ValidationMetrics, and CorrectionValidator.
"""

import pytest
import numpy as np
from am_qadf.correction.validation import (
    AlignmentQuality,
    ValidationMetrics,
    CorrectionValidator,
)


class TestAlignmentQuality:
    """Test suite for AlignmentQuality enum."""

    @pytest.mark.unit
    def test_alignment_quality_values(self):
        """Test AlignmentQuality enum values."""
        assert AlignmentQuality.EXCELLENT.value == "excellent"
        assert AlignmentQuality.GOOD.value == "good"
        assert AlignmentQuality.ACCEPTABLE.value == "acceptable"
        assert AlignmentQuality.POOR.value == "poor"

    @pytest.mark.unit
    def test_alignment_quality_enumeration(self):
        """Test that AlignmentQuality can be enumerated."""
        qualities = list(AlignmentQuality)
        assert len(qualities) == 4
        assert AlignmentQuality.EXCELLENT in qualities


class TestValidationMetrics:
    """Test suite for ValidationMetrics dataclass."""

    @pytest.mark.unit
    def test_validation_metrics_creation(self):
        """Test creating ValidationMetrics."""
        metrics = ValidationMetrics(
            mean_error=0.05,
            max_error=0.1,
            rms_error=0.06,
            std_error=0.02,
            median_error=0.05,
            num_points=100,
            quality=AlignmentQuality.EXCELLENT,
        )

        assert metrics.mean_error == 0.05
        assert metrics.max_error == 0.1
        assert metrics.num_points == 100
        assert metrics.quality == AlignmentQuality.EXCELLENT

    @pytest.mark.unit
    def test_validation_metrics_to_dict(self):
        """Test converting ValidationMetrics to dictionary."""
        metrics = ValidationMetrics(
            mean_error=0.05,
            max_error=0.1,
            rms_error=0.06,
            std_error=0.02,
            median_error=0.05,
            num_points=100,
            quality=AlignmentQuality.EXCELLENT,
        )

        result = metrics.to_dict()

        assert isinstance(result, dict)
        assert result["mean_error"] == 0.05
        assert result["quality"] == "excellent"
        assert result["num_points"] == 100


class TestCorrectionValidator:
    """Test suite for CorrectionValidator class."""

    @pytest.fixture
    def validator(self):
        """Create a CorrectionValidator instance."""
        return CorrectionValidator()

    @pytest.mark.unit
    def test_correction_validator_creation_default(self):
        """Test creating CorrectionValidator with default parameters."""
        validator = CorrectionValidator()

        assert validator.excellent_threshold == 0.05
        assert validator.good_threshold == 0.1
        assert validator.acceptable_threshold == 0.2

    @pytest.mark.unit
    def test_correction_validator_creation_custom(self):
        """Test creating CorrectionValidator with custom parameters."""
        validator = CorrectionValidator(excellent_threshold=0.01, good_threshold=0.05, acceptable_threshold=0.1)

        assert validator.excellent_threshold == 0.01
        assert validator.good_threshold == 0.05
        assert validator.acceptable_threshold == 0.1

    @pytest.mark.unit
    def test_compute_alignment_error(self, validator):
        """Test computing alignment error."""
        corrected_points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        reference_points = np.array([[1.05, 2.05, 3.05], [4.1, 5.1, 6.1]])

        errors = validator.compute_alignment_error(corrected_points, reference_points)

        assert len(errors) == 2
        assert all(errors > 0)
        assert errors[0] < 0.1  # Small error
        assert errors[1] < 0.2  # Small error

    @pytest.mark.unit
    def test_compute_alignment_error_perfect(self, validator):
        """Test computing alignment error with perfect alignment."""
        points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        errors = validator.compute_alignment_error(points, points)

        assert np.allclose(errors, 0.0)

    @pytest.mark.unit
    def test_compute_alignment_error_shape_mismatch(self, validator):
        """Test computing alignment error with shape mismatch."""
        corrected_points = np.array([[1.0, 2.0, 3.0]])
        reference_points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        with pytest.raises(ValueError, match="Corrected and reference points must have same shape"):
            validator.compute_alignment_error(corrected_points, reference_points)

    @pytest.mark.unit
    def test_assess_quality_excellent(self, validator):
        """Test assessing quality as excellent."""
        quality = validator.assess_quality(0.03)  # Below excellent threshold

        assert quality == AlignmentQuality.EXCELLENT

    @pytest.mark.unit
    def test_assess_quality_good(self, validator):
        """Test assessing quality as good."""
        quality = validator.assess_quality(0.08)  # Between excellent and good

        assert quality == AlignmentQuality.GOOD

    @pytest.mark.unit
    def test_assess_quality_acceptable(self, validator):
        """Test assessing quality as acceptable."""
        quality = validator.assess_quality(0.15)  # Between good and acceptable

        assert quality == AlignmentQuality.ACCEPTABLE

    @pytest.mark.unit
    def test_assess_quality_poor(self, validator):
        """Test assessing quality as poor."""
        quality = validator.assess_quality(0.25)  # Above acceptable threshold

        assert quality == AlignmentQuality.POOR

    @pytest.mark.unit
    def test_validate_correction(self, validator):
        """Test validating correction."""
        corrected_points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        reference_points = np.array([[1.02, 2.02, 3.02], [4.03, 5.03, 6.03], [7.01, 8.01, 9.01]])

        metrics = validator.validate_correction(corrected_points, reference_points)

        assert isinstance(metrics, ValidationMetrics)
        assert metrics.num_points == 3
        assert metrics.mean_error > 0
        assert metrics.max_error > 0
        assert metrics.rms_error > 0
        assert metrics.quality in AlignmentQuality

    @pytest.mark.unit
    def test_validate_correction_perfect(self, validator):
        """Test validating perfect correction."""
        points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        metrics = validator.validate_correction(points, points)

        assert metrics.mean_error == 0.0
        assert metrics.max_error == 0.0
        assert metrics.quality == AlignmentQuality.EXCELLENT

    @pytest.mark.unit
    def test_compare_corrections(self, validator):
        """Test comparing multiple correction methods."""
        reference_points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        corrections = {
            "method1": (np.array([[1.01, 2.01, 3.01], [4.01, 5.01, 6.01]]), "Method 1"),
            "method2": (np.array([[1.02, 2.02, 3.02], [4.02, 5.02, 6.02]]), "Method 2"),
        }

        results = validator.compare_corrections(corrections, reference_points)

        assert len(results) == 2
        assert "method1" in results
        assert "method2" in results
        assert isinstance(results["method1"], ValidationMetrics)
        assert isinstance(results["method2"], ValidationMetrics)
        # method1 should have lower error than method2
        assert results["method1"].mean_error < results["method2"].mean_error

    @pytest.mark.unit
    def test_generate_validation_report(self, validator):
        """Test generating validation report."""
        metrics = ValidationMetrics(
            mean_error=0.05,
            max_error=0.1,
            rms_error=0.06,
            std_error=0.02,
            median_error=0.05,
            num_points=100,
            quality=AlignmentQuality.EXCELLENT,
        )

        report = validator.generate_validation_report(metrics)

        assert isinstance(report, str)
        assert "CORRECTION VALIDATION REPORT" in report
        assert "EXCELLENT" in report
        assert "100" in report

    @pytest.mark.unit
    def test_generate_validation_report_no_details(self, validator):
        """Test generating validation report without details."""
        metrics = ValidationMetrics(
            mean_error=0.05,
            max_error=0.1,
            rms_error=0.06,
            std_error=0.02,
            median_error=0.05,
            num_points=100,
            quality=AlignmentQuality.EXCELLENT,
        )

        report = validator.generate_validation_report(metrics, include_details=False)

        assert isinstance(report, str)
        # Should not include detailed statistics
        assert "Mean Error:" not in report or "Error Statistics:" not in report

    @pytest.mark.unit
    def test_validate_distortion_correction(self, validator):
        """Test validating distortion correction improvement."""
        original_points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        corrected_points = np.array([[1.02, 2.02, 3.02], [4.03, 5.03, 6.03]])
        reference_points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        result = validator.validate_distortion_correction(original_points, corrected_points, reference_points)

        assert "before" in result
        assert "after" in result
        assert "improvement" in result
        assert "mean_error" in result["before"]
        assert "mean_error" in result["after"]
        assert "mean_error_reduction" in result["improvement"]

    @pytest.mark.unit
    def test_validate_distortion_correction_improvement(self, validator):
        """Test that correction shows improvement."""
        # Original points have large error
        original_points = np.array([[1.1, 2.1, 3.1], [4.2, 5.2, 6.2]])  # 0.1 error  # 0.2 error
        # Corrected points have small error
        corrected_points = np.array([[1.01, 2.01, 3.01], [4.02, 5.02, 6.02]])  # 0.01 error  # 0.02 error
        reference_points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        result = validator.validate_distortion_correction(original_points, corrected_points, reference_points)

        # After should have lower error than before
        assert result["after"]["mean_error"] < result["before"]["mean_error"]
        assert result["improvement"]["mean_error_reduction"] > 0
        assert result["improvement"]["improvement_percent"] > 0
