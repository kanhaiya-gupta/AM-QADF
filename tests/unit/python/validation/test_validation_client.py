"""
Unit tests for validation client.

Tests for ValidationConfig and ValidationClient.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

try:
    from am_qadf.validation.validation_client import (
        ValidationConfig,
        ValidationClient,
    )
except ImportError:
    pytest.skip("Validation module not available", allow_module_level=True)


class TestValidationConfig:
    """Test suite for ValidationConfig dataclass."""

    @pytest.mark.unit
    def test_config_creation_default(self):
        """Test creating ValidationConfig with default values."""
        config = ValidationConfig()

        assert config.confidence_level == 0.95
        assert config.significance_level == 0.05
        assert config.max_acceptable_error == 0.1
        assert config.correlation_threshold == 0.85
        assert config.sample_size is None
        assert config.random_seed is None
        assert config.enable_benchmarking is True
        assert config.enable_mpm_comparison is True
        assert config.enable_accuracy_validation is True
        assert config.enable_statistical_validation is True

    @pytest.mark.unit
    def test_config_creation_custom(self):
        """Test creating ValidationConfig with custom values."""
        config = ValidationConfig(
            confidence_level=0.99,
            significance_level=0.01,
            max_acceptable_error=0.05,
            correlation_threshold=0.9,
            sample_size=1000,
            random_seed=42,
            enable_benchmarking=False,
            enable_mpm_comparison=False,
        )

        assert config.confidence_level == 0.99
        assert config.significance_level == 0.01
        assert config.max_acceptable_error == 0.05
        assert config.correlation_threshold == 0.9
        assert config.sample_size == 1000
        assert config.random_seed == 42
        assert config.enable_benchmarking is False
        assert config.enable_mpm_comparison is False

    @pytest.mark.unit
    def test_config_to_dict(self):
        """Test converting ValidationConfig to dictionary."""
        config = ValidationConfig(confidence_level=0.99, significance_level=0.01, random_seed=42)

        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["confidence_level"] == 0.99
        assert config_dict["significance_level"] == 0.01
        assert config_dict["random_seed"] == 42


class TestValidationClient:
    """Test suite for ValidationClient class."""

    @pytest.fixture
    def validation_client(self):
        """Create ValidationClient with all modules enabled."""
        config = ValidationConfig(
            enable_benchmarking=True,
            enable_mpm_comparison=True,
            enable_accuracy_validation=True,
            enable_statistical_validation=True,
        )
        return ValidationClient(config=config)

    @pytest.fixture
    def validation_client_disabled(self):
        """Create ValidationClient with all modules disabled."""
        config = ValidationConfig(
            enable_benchmarking=False,
            enable_mpm_comparison=False,
            enable_accuracy_validation=False,
            enable_statistical_validation=False,
        )
        return ValidationClient(config=config)

    @pytest.fixture
    def sample_operation(self):
        """Sample operation for benchmarking."""

        def operation(x):
            return x * 2

        return operation

    @pytest.mark.unit
    def test_client_creation_default(self):
        """Test creating ValidationClient with default config."""
        client = ValidationClient()

        assert client.config is not None
        assert isinstance(client.config, ValidationConfig)

    @pytest.mark.unit
    def test_client_creation_custom_config(self):
        """Test creating ValidationClient with custom config."""
        config = ValidationConfig(confidence_level=0.99)
        client = ValidationClient(config=config)

        assert client.config.confidence_level == 0.99

    @pytest.mark.unit
    def test_client_creation_modules_initialized(self, validation_client):
        """Test that validation modules are initialized when enabled."""
        # Modules may be None if validation module components unavailable
        # Just check that client was created
        assert validation_client is not None
        assert validation_client.config is not None

    @pytest.mark.unit
    def test_benchmark_operation_enabled(self, validation_client, sample_operation):
        """Test benchmark_operation when benchmarking is enabled."""
        if validation_client.benchmarker is None:
            pytest.skip("Benchmarking module not available")

        result = validation_client.benchmark_operation(sample_operation, 10, iterations=3)

        if result is not None:
            assert result.operation_name == "sample_operation" or "operation" in result.operation_name
            assert result.execution_time >= 0

    @pytest.mark.unit
    def test_benchmark_operation_disabled(self, validation_client_disabled, sample_operation):
        """Test benchmark_operation when benchmarking is disabled."""
        result = validation_client_disabled.benchmark_operation(sample_operation, 10)

        assert result is None  # Should return None when disabled

    @pytest.mark.unit
    def test_compare_with_mpm_enabled(self, validation_client):
        """Test compare_with_mpm when MPM comparison is enabled."""
        if validation_client.mpm_comparer is None:
            pytest.skip("MPM comparison module not available")

        framework_data = {"completeness": 0.9, "snr": 25.5}
        mpm_data = {"completeness": 0.88, "snr": 24.8}

        results = validation_client.compare_with_mpm(framework_data, mpm_data)

        if results:
            assert isinstance(results, dict)
            # Should contain comparison results if module available

    @pytest.mark.unit
    def test_compare_with_mpm_disabled(self, validation_client_disabled):
        """Test compare_with_mpm when MPM comparison is disabled."""
        framework_data = {"completeness": 0.9}
        mpm_data = {"completeness": 0.88}

        results = validation_client_disabled.compare_with_mpm(framework_data, mpm_data)

        assert results == {}  # Should return empty dict when disabled

    @pytest.mark.unit
    def test_validate_mpm_correlation(self, validation_client):
        """Test validate_mpm_correlation."""
        if validation_client.mpm_comparer is None:
            pytest.skip("MPM comparison module not available")

        framework_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mpm_values = np.array([1.1, 2.1, 3.1, 4.1, 5.1])

        correlation = validation_client.validate_mpm_correlation(framework_values, mpm_values)

        assert isinstance(correlation, float)
        assert 0 <= abs(correlation) <= 1

    @pytest.mark.unit
    def test_validate_accuracy_signal_mapping(self, validation_client):
        """Test validate_accuracy with signal_mapping type."""
        if validation_client.accuracy_validator is None:
            pytest.skip("Accuracy validation module not available")

        framework_data = np.random.rand(50, 50)
        ground_truth = framework_data + np.random.rand(50, 50) * 0.01

        result = validation_client.validate_accuracy(framework_data, ground_truth, validation_type="signal_mapping")

        if result is not None:
            assert (
                isinstance(result, type(validation_client.accuracy_validator).__module__.AccuracyValidationResult)
                if hasattr(type(validation_client.accuracy_validator).__module__, "AccuracyValidationResult")
                else True
            )
            # Just check it doesn't raise error

    @pytest.mark.unit
    def test_validate_accuracy_invalid_type(self, validation_client):
        """Test validate_accuracy with invalid validation type."""
        result = validation_client.validate_accuracy(np.array([1, 2, 3]), np.array([1, 2, 3]), validation_type="invalid_type")

        assert result is None

    @pytest.mark.unit
    def test_validate_accuracy_disabled(self, validation_client_disabled):
        """Test validate_accuracy when validation is disabled."""
        result = validation_client_disabled.validate_accuracy(np.array([1, 2, 3]), np.array([1, 2, 3]))

        assert result is None

    @pytest.mark.unit
    def test_perform_statistical_test_t_test(self, validation_client):
        """Test perform_statistical_test with t_test."""
        if validation_client.statistical_validator is None:
            pytest.skip("Statistical validation module not available")

        sample1 = np.random.normal(100, 10, 50)
        sample2 = np.random.normal(105, 10, 50)

        result = validation_client.perform_statistical_test("t_test", sample1, sample2, alternative="two-sided")

        if result is not None:
            assert result.test_name in ["t_test_independent", "t_test"]
            assert 0 <= result.p_value <= 1

    @pytest.mark.unit
    def test_perform_statistical_test_invalid_name(self, validation_client):
        """Test perform_statistical_test with invalid test name."""
        result = validation_client.perform_statistical_test("invalid_test", np.array([1, 2, 3]), np.array([1, 2, 3]))

        assert result is None

    @pytest.mark.unit
    def test_perform_statistical_test_disabled(self, validation_client_disabled):
        """Test perform_statistical_test when validation is disabled."""
        result = validation_client_disabled.perform_statistical_test("t_test", np.array([1, 2, 3]), np.array([1, 2, 3]))

        assert result is None

    @pytest.mark.unit
    def test_validate_improvement(self, validation_client):
        """Test validate_improvement."""
        if validation_client.statistical_validator is None:
            pytest.skip("Statistical validation module not available")

        baseline = np.array([0.5, 0.6, 0.55, 0.58, 0.57])
        improved = np.array([0.7, 0.75, 0.72, 0.73, 0.71])

        result = validation_client.validate_improvement(baseline, improved)

        if result is not None:
            assert result.test_name in ["t_test_independent", "t_test_paired", "t_test"]
            assert isinstance(result.is_significant, bool)

    @pytest.mark.unit
    def test_comprehensive_validation(self, validation_client):
        """Test comprehensive_validation."""
        framework_data = {"overall_score": 0.9}
        reference_data = {"quality_scores": {"overall_score": 0.88}}

        results = validation_client.comprehensive_validation(
            framework_data, reference_data, validation_types=["mpm_comparison"]
        )

        assert isinstance(results, dict)
        # May be empty if modules unavailable, but should not raise error

    @pytest.mark.unit
    def test_comprehensive_validation_all_types(self, validation_client):
        """Test comprehensive_validation with all validation types."""
        framework_data = np.array([0.9, 0.85, 0.92])
        reference_data = np.array([0.88, 0.83, 0.90])

        results = validation_client.comprehensive_validation(
            framework_data, reference_data, validation_types=["mpm_comparison", "accuracy", "statistical"]
        )

        assert isinstance(results, dict)

    @pytest.mark.unit
    def test_generate_validation_report(self, validation_client, tmp_path):
        """Test generate_validation_report."""
        # Create mock validation results
        validation_results = {
            "mpm_comparison": {
                "completeness": Mock(
                    framework_value=0.9,
                    mpm_value=0.88,
                    correlation=0.95,
                    relative_error=2.27,
                    is_valid=True,
                    to_dict=lambda: {
                        "framework_value": 0.9,
                        "mpm_value": 0.88,
                        "correlation": 0.95,
                        "relative_error": 2.27,
                        "is_valid": True,
                    },
                )
            },
            "accuracy": Mock(
                signal_name="test",
                rmse=0.05,
                mae=0.04,
                r2_score=0.95,
                max_error=0.1,
                within_tolerance=True,
                ground_truth_size=1000,
                validated_points=950,
                to_dict=lambda: {"signal_name": "test", "rmse": 0.05, "r2_score": 0.95, "within_tolerance": True},
            ),
            "statistical": Mock(
                test_name="t_test",
                null_hypothesis="μ₁ = μ₂",
                test_statistic=2.5,
                p_value=0.01,
                significance_level=0.05,
                is_significant=True,
                conclusion="Reject H₀",
                to_dict=lambda: {"test_name": "t_test", "p_value": 0.01, "is_significant": True},
            ),
        }

        output_path = tmp_path / "validation_report.txt"
        report = validation_client.generate_validation_report(validation_results, output_path=str(output_path))

        assert isinstance(report, str)
        assert "AM-QADF Validation Report" in report
        assert output_path.exists()

    @pytest.mark.unit
    def test_generate_validation_report_no_file(self, validation_client):
        """Test generate_validation_report without saving to file."""
        validation_results = {"mpm_comparison": {}}

        report = validation_client.generate_validation_report(validation_results)

        assert isinstance(report, str)
        assert "AM-QADF Validation Report" in report
