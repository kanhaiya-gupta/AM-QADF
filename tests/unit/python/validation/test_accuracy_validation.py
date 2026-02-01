"""
Unit tests for accuracy validation module.

Tests for AccuracyValidationResult and AccuracyValidator.
"""

import pytest
import numpy as np
from unittest.mock import Mock

try:
    from am_qadf.validation.accuracy_validation import (
        AccuracyValidationResult,
        AccuracyValidator,
    )
except ImportError:
    pytest.skip("Validation module not available", allow_module_level=True)


class TestAccuracyValidationResult:
    """Test suite for AccuracyValidationResult dataclass."""

    @pytest.mark.unit
    def test_accuracy_result_creation(self):
        """Test creating AccuracyValidationResult with all fields."""
        result = AccuracyValidationResult(
            signal_name="test_signal",
            rmse=0.05,
            mae=0.04,
            r2_score=0.95,
            max_error=0.1,
            within_tolerance=True,
            ground_truth_size=1000,
            validated_points=950,
            metadata={"test": "value"},
        )

        assert result.signal_name == "test_signal"
        assert result.rmse == 0.05
        assert result.mae == 0.04
        assert result.r2_score == 0.95
        assert result.max_error == 0.1
        assert result.within_tolerance is True
        assert result.ground_truth_size == 1000
        assert result.validated_points == 950

    @pytest.mark.unit
    def test_accuracy_result_to_dict(self):
        """Test converting AccuracyValidationResult to dictionary."""
        result = AccuracyValidationResult(
            signal_name="test",
            rmse=0.05,
            mae=0.04,
            r2_score=0.95,
            max_error=0.1,
            within_tolerance=True,
            ground_truth_size=1000,
            validated_points=950,
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["signal_name"] == "test"
        assert result_dict["rmse"] == 0.05
        assert result_dict["within_tolerance"] is True
        assert "timestamp" in result_dict


class TestAccuracyValidator:
    """Test suite for AccuracyValidator class."""

    @pytest.fixture
    def validator(self):
        """Create AccuracyValidator with default parameters."""
        return AccuracyValidator()

    @pytest.fixture
    def custom_validator(self):
        """Create AccuracyValidator with custom tolerance."""
        return AccuracyValidator(max_acceptable_error=0.05, tolerance_percent=3.0)

    @pytest.fixture
    def ground_truth_signal(self):
        """Ground truth signal array."""
        np.random.seed(42)
        return np.random.rand(50, 50, 10)

    @pytest.fixture
    def mapped_signal(self):
        """Framework-mapped signal (slightly different)."""
        np.random.seed(42)
        base = np.random.rand(50, 50, 10)
        return base + np.random.rand(50, 50, 10) * 0.05  # Small error

    @pytest.fixture
    def ground_truth_coords(self):
        """Ground truth coordinates."""
        np.random.seed(42)
        return np.random.rand(100, 3) * 10

    @pytest.fixture
    def framework_coords(self):
        """Framework-calculated coordinates (with small error)."""
        np.random.seed(42)
        base = np.random.rand(100, 3) * 10
        return base + np.random.rand(100, 3) * 0.01

    @pytest.mark.unit
    def test_validator_creation_default(self, validator):
        """Test creating AccuracyValidator with default parameters."""
        assert validator.max_acceptable_error == 0.1
        assert validator.tolerance_percent == 5.0

    @pytest.mark.unit
    def test_validator_creation_custom(self, custom_validator):
        """Test creating AccuracyValidator with custom parameters."""
        assert custom_validator.max_acceptable_error == 0.05
        assert custom_validator.tolerance_percent == 3.0

    @pytest.mark.unit
    def test_validate_signal_mapping_matching(self, validator, ground_truth_signal):
        """Test validating signal mapping with matching data."""
        result = validator.validate_signal_mapping(ground_truth_signal, ground_truth_signal.copy(), signal_name="test_signal")

        assert isinstance(result, AccuracyValidationResult)
        assert result.signal_name == "test_signal"
        assert result.rmse < 1e-10  # Should be very small for identical arrays
        assert result.r2_score >= 0.99  # Should be very high
        assert result.validated_points == result.ground_truth_size

    @pytest.mark.unit
    def test_validate_signal_mapping_different(self, validator, ground_truth_signal, mapped_signal):
        """Test validating signal mapping with different data."""
        result = validator.validate_signal_mapping(mapped_signal, ground_truth_signal, signal_name="test_signal")

        assert isinstance(result, AccuracyValidationResult)
        assert result.rmse > 0
        assert result.mae > 0
        assert result.max_error > 0
        assert 0 <= result.r2_score <= 1

    @pytest.mark.unit
    def test_validate_signal_mapping_size_mismatch(self, validator):
        """Test validating with size mismatch."""
        arr1 = np.random.rand(50, 50)
        arr2 = np.random.rand(60, 60)

        result = validator.validate_signal_mapping(arr1, arr2, "test")

        assert isinstance(result, AccuracyValidationResult)
        assert result.validated_points > 0  # Should use minimum size

    @pytest.mark.unit
    def test_validate_signal_mapping_with_nan(self, validator):
        """Test validating with NaN values."""
        arr1 = np.random.rand(50, 50)
        arr2 = arr1.copy()
        arr1[10:20, 10:20] = np.nan
        arr2[15:25, 15:25] = np.nan

        result = validator.validate_signal_mapping(arr1, arr2, "test")

        assert isinstance(result, AccuracyValidationResult)
        assert result.validated_points < 2500  # Should exclude NaN

    @pytest.mark.unit
    def test_validate_signal_mapping_empty(self, validator):
        """Test validating with empty arrays."""
        arr1 = np.array([])
        arr2 = np.array([])

        result = validator.validate_signal_mapping(arr1, arr2, "test")

        assert isinstance(result, AccuracyValidationResult)
        assert result.is_valid is False
        assert "error" in result.metadata

    @pytest.mark.unit
    def test_validate_spatial_alignment(self, validator, framework_coords, ground_truth_coords):
        """Test validating spatial alignment."""
        result = validator.validate_spatial_alignment(framework_coords, ground_truth_coords)

        assert isinstance(result, AccuracyValidationResult)
        assert result.signal_name == "spatial_alignment"
        assert result.rmse >= 0
        assert result.mae >= 0
        assert result.max_error >= 0

    @pytest.mark.unit
    def test_validate_spatial_alignment_shape_mismatch(self, validator):
        """Test spatial alignment with shape mismatch."""
        coords1 = np.random.rand(100, 3)
        coords2 = np.random.rand(100, 2)  # Different dimension

        result = validator.validate_spatial_alignment(coords1, coords2)

        assert isinstance(result, AccuracyValidationResult)
        # Should handle gracefully

    @pytest.mark.unit
    def test_validate_temporal_alignment(self, validator):
        """Test validating temporal alignment."""
        times1 = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        times2 = np.array([0.01, 1.01, 2.01, 3.01, 4.01])  # Small offset

        result = validator.validate_temporal_alignment(times1, times2)

        assert isinstance(result, AccuracyValidationResult)
        assert result.signal_name == "temporal_alignment"
        assert result.rmse > 0
        assert result.mae > 0

    @pytest.mark.unit
    def test_validate_temporal_alignment_matching(self, validator):
        """Test temporal alignment with matching times."""
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

        result = validator.validate_temporal_alignment(times, times.copy())

        assert isinstance(result, AccuracyValidationResult)
        assert result.rmse < 1e-10  # Should be very small

    @pytest.mark.unit
    def test_validate_quality_metrics(self, validator):
        """Test validating quality metrics."""
        framework_metrics = {
            "completeness": 0.9,
            "snr": 25.5,
            "alignment_accuracy": 0.95,
        }
        ground_truth_metrics = {
            "completeness": 0.88,
            "snr": 24.8,
            "alignment_accuracy": 0.94,
        }

        results = validator.validate_quality_metrics(framework_metrics, ground_truth_metrics)

        assert isinstance(results, dict)
        assert "completeness" in results
        assert "snr" in results
        assert "alignment_accuracy" in results
        for result in results.values():
            assert isinstance(result, AccuracyValidationResult)

    @pytest.mark.unit
    def test_validate_quality_metrics_partial_overlap(self, validator):
        """Test validating metrics with partial overlap."""
        framework = {"metric1": 0.9, "metric2": 0.8}
        ground_truth = {"metric1": 0.88}  # Missing metric2

        results = validator.validate_quality_metrics(framework, ground_truth)

        assert "metric1" in results
        assert "metric2" not in results  # Only common metrics

    @pytest.mark.unit
    def test_calculate_rmse(self, validator):
        """Test calculating RMSE."""
        predicted = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        actual = np.array([1.1, 2.1, 3.1, 4.1, 5.1])

        rmse = validator.calculate_rmse(predicted, actual)

        assert isinstance(rmse, float)
        assert rmse > 0
        assert rmse < 1.0  # Should be small for this case

    @pytest.mark.unit
    def test_calculate_mae(self, validator):
        """Test calculating MAE."""
        predicted = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        actual = np.array([1.1, 2.1, 3.1, 4.1, 5.1])

        mae = validator.calculate_mae(predicted, actual)

        assert isinstance(mae, float)
        assert mae > 0
        assert mae < 1.0

    @pytest.mark.unit
    def test_calculate_r2_score_perfect_match(self, validator):
        """Test calculating R² score with perfect match."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        r2 = validator.calculate_r2_score(data, data)

        assert abs(r2 - 1.0) < 1e-10  # Should be 1.0 for perfect match

    @pytest.mark.unit
    def test_calculate_r2_score_no_correlation(self, validator):
        """Test calculating R² score with no correlation."""
        predicted = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        actual = np.random.rand(5) * 100  # Random, uncorrelated

        r2 = validator.calculate_r2_score(predicted, actual)

        assert isinstance(r2, float)
        assert r2 <= 1.0  # R² can be negative for poor fits

    @pytest.mark.unit
    def test_validate_within_tolerance_valid(self, validator):
        """Test validate_within_tolerance with valid errors."""
        errors = np.array([0.01, 0.02, 0.03, 0.04, 0.05])  # All < 0.1

        is_valid = validator.validate_within_tolerance(errors, tolerance=0.1)

        assert is_valid is True

    @pytest.mark.unit
    def test_validate_within_tolerance_invalid(self, validator):
        """Test validate_within_tolerance with invalid errors."""
        errors = np.array([0.01, 0.02, 0.15, 0.04, 0.05])  # One > 0.1

        is_valid = validator.validate_within_tolerance(errors, tolerance=0.1)

        assert is_valid is False

    @pytest.mark.unit
    def test_validate_within_tolerance_custom(self, validator):
        """Test validate_within_tolerance with custom tolerance."""
        errors = np.array([0.02, 0.03, 0.04])

        is_valid = validator.validate_within_tolerance(errors, tolerance=0.05)

        assert is_valid is True

    @pytest.mark.unit
    def test_validate_within_tolerance_default(self, validator):
        """Test validate_within_tolerance with default tolerance."""
        errors = np.array([0.05, 0.06, 0.07])  # All < 0.1 (default)

        is_valid = validator.validate_within_tolerance(errors)

        assert is_valid is True
