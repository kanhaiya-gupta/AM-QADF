"""
Unit tests for statistical validation module.

Tests for StatisticalValidationResult and StatisticalValidator.
"""

import pytest
import numpy as np
from unittest.mock import Mock

try:
    from am_qadf.validation.statistical_validation import (
        StatisticalValidationResult,
        StatisticalValidator,
    )
except ImportError:
    pytest.skip("Validation module not available", allow_module_level=True)


class TestStatisticalValidationResult:
    """Test suite for StatisticalValidationResult dataclass."""

    @pytest.mark.unit
    def test_statistical_result_creation(self):
        """Test creating StatisticalValidationResult with all fields."""
        result = StatisticalValidationResult(
            test_name="t_test",
            null_hypothesis="μ₁ = μ₂",
            alternative_hypothesis="μ₁ ≠ μ₂",
            test_statistic=2.5,
            p_value=0.01,
            significance_level=0.05,
            is_significant=True,
            conclusion="Reject H₀",
            metadata={"df": 18},
        )

        assert result.test_name == "t_test"
        assert result.null_hypothesis == "μ₁ = μ₂"
        assert result.test_statistic == 2.5
        assert result.p_value == 0.01
        assert result.significance_level == 0.05
        assert result.is_significant is True
        assert result.conclusion == "Reject H₀"

    @pytest.mark.unit
    def test_statistical_result_to_dict(self):
        """Test converting StatisticalValidationResult to dictionary."""
        result = StatisticalValidationResult(
            test_name="test",
            null_hypothesis="H₀",
            alternative_hypothesis="H₁",
            test_statistic=1.5,
            p_value=0.1,
            significance_level=0.05,
            is_significant=False,
            conclusion="Fail to reject",
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["test_name"] == "test"
        assert result_dict["p_value"] == 0.1
        assert result_dict["is_significant"] is False
        assert "timestamp" in result_dict


class TestStatisticalValidator:
    """Test suite for StatisticalValidator class."""

    @pytest.fixture
    def validator(self):
        """Create StatisticalValidator with default parameters."""
        return StatisticalValidator()

    @pytest.fixture
    def custom_validator(self):
        """Create StatisticalValidator with custom parameters."""
        return StatisticalValidator(significance_level=0.01, confidence_level=0.99)

    @pytest.fixture
    def normal_sample1(self):
        """Normally distributed sample 1."""
        np.random.seed(42)
        return np.random.normal(100, 10, 50)

    @pytest.fixture
    def normal_sample2(self):
        """Normally distributed sample 2 (different mean)."""
        np.random.seed(43)
        return np.random.normal(105, 10, 50)

    @pytest.fixture
    def identical_sample(self, normal_sample1):
        """Sample identical to sample1."""
        return normal_sample1.copy()

    @pytest.mark.unit
    def test_validator_creation_default(self, validator):
        """Test creating StatisticalValidator with default parameters."""
        assert validator.significance_level == 0.05
        assert validator.confidence_level == 0.95

    @pytest.mark.unit
    def test_validator_creation_custom(self, custom_validator):
        """Test creating StatisticalValidator with custom parameters."""
        assert custom_validator.significance_level == 0.01
        assert custom_validator.confidence_level == 0.99

    @pytest.mark.unit
    def test_t_test_independent(self, validator, normal_sample1, normal_sample2):
        """Test independent t-test."""
        result = validator.t_test(normal_sample1, normal_sample2, alternative="two-sided", paired=False)

        assert isinstance(result, StatisticalValidationResult)
        assert result.test_name == "t_test_independent"
        assert result.null_hypothesis == "μ₁ = μ₂ (means are equal)"
        assert 0 <= result.p_value <= 1
        assert isinstance(result.test_statistic, float)
        assert isinstance(result.is_significant, bool)

    @pytest.mark.unit
    def test_t_test_paired(self, validator, normal_sample1):
        """Test paired t-test."""
        sample2 = normal_sample1 + np.random.rand(len(normal_sample1)) * 2  # Similar but different

        result = validator.t_test(normal_sample1, sample2, alternative="two-sided", paired=True)

        assert isinstance(result, StatisticalValidationResult)
        assert result.test_name == "t_test_paired"
        assert len(normal_sample1) == len(sample2)

    @pytest.mark.unit
    def test_t_test_one_sided(self, validator, normal_sample1, normal_sample2):
        """Test one-sided t-test."""
        result = validator.t_test(normal_sample1, normal_sample2, alternative="greater")

        assert isinstance(result, StatisticalValidationResult)
        assert ">" in result.alternative_hypothesis or "greater" in result.alternative_hypothesis.lower()

    @pytest.mark.unit
    def test_t_test_insufficient_data(self, validator):
        """Test t-test with insufficient data."""
        sample1 = np.array([1.0])
        sample2 = np.array([2.0])

        result = validator.t_test(sample1, sample2)

        assert isinstance(result, StatisticalValidationResult)
        assert "Error" in result.conclusion or result.p_value != result.p_value  # NaN check

    @pytest.mark.unit
    def test_t_test_with_nan(self, validator):
        """Test t-test with NaN values."""
        sample1 = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        sample2 = np.array([1.1, 2.1, 3.1, np.nan, 5.1])

        result = validator.t_test(sample1, sample2)

        assert isinstance(result, StatisticalValidationResult)
        # Should handle NaN gracefully

    @pytest.mark.unit
    def test_t_test_identical_samples(self, validator, normal_sample1, identical_sample):
        """Test t-test with identical samples (should not be significant)."""
        result = validator.t_test(normal_sample1, identical_sample)

        assert isinstance(result, StatisticalValidationResult)
        # Identical samples should have high p-value (not significant)

    @pytest.mark.unit
    def test_mann_whitney_u_test(self, validator, normal_sample1, normal_sample2):
        """Test Mann-Whitney U test."""
        result = validator.mann_whitney_u_test(normal_sample1, normal_sample2)

        assert isinstance(result, StatisticalValidationResult)
        assert result.test_name == "mann_whitney_u_test"
        assert 0 <= result.p_value <= 1
        assert isinstance(result.test_statistic, float)

    @pytest.mark.unit
    def test_mann_whitney_insufficient_data(self, validator):
        """Test Mann-Whitney with insufficient data."""
        sample1 = np.array([1.0])
        sample2 = np.array([2.0])

        result = validator.mann_whitney_u_test(sample1, sample2)

        assert isinstance(result, StatisticalValidationResult)
        assert "Error" in result.conclusion or result.conclusion.startswith("Error")

    @pytest.mark.unit
    def test_correlation_test_pearson(self, validator):
        """Test Pearson correlation test."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([1.1, 2.1, 3.1, 4.1, 5.1])  # Highly correlated

        result = validator.correlation_test(x, y, method="pearson")

        assert isinstance(result, StatisticalValidationResult)
        assert result.test_name == "correlation_test_pearson"
        assert abs(result.test_statistic) > 0.9  # High correlation
        assert 0 <= result.p_value <= 1

    @pytest.mark.unit
    def test_correlation_test_spearman(self, validator):
        """Test Spearman correlation test."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([1.1, 2.1, 3.1, 4.1, 5.1])

        result = validator.correlation_test(x, y, method="spearman")

        assert isinstance(result, StatisticalValidationResult)
        assert result.test_name == "correlation_test_spearman"
        assert "spearman" in result.metadata["method"]

    @pytest.mark.unit
    def test_correlation_test_no_correlation(self, validator):
        """Test correlation test with no correlation."""
        x = np.random.rand(100)
        y = np.random.rand(100)  # Uncorrelated

        result = validator.correlation_test(x, y)

        assert isinstance(result, StatisticalValidationResult)
        assert abs(result.test_statistic) < 0.3  # Low correlation

    @pytest.mark.unit
    def test_correlation_test_insufficient_data(self, validator):
        """Test correlation test with insufficient data."""
        x = np.array([1.0, 2.0])
        y = np.array([1.1, 2.1])

        result = validator.correlation_test(x, y)

        assert isinstance(result, StatisticalValidationResult)
        assert "Error" in result.conclusion or result.conclusion.startswith("Error")

    @pytest.mark.unit
    def test_chi_square_test(self, validator):
        """Test chi-square test."""
        observed = np.array([10, 20, 30, 40])
        expected = np.array([25, 25, 25, 25])

        result = validator.chi_square_test(observed, expected)

        assert isinstance(result, StatisticalValidationResult)
        assert result.test_name == "chi_square_test"
        assert result.test_statistic > 0
        assert 0 <= result.p_value <= 1

    @pytest.mark.unit
    def test_chi_square_test_uniform(self, validator):
        """Test chi-square test with uniform expected (default)."""
        observed = np.array([10, 20, 30, 40])

        result = validator.chi_square_test(observed)

        assert isinstance(result, StatisticalValidationResult)
        assert "categories" in result.metadata

    @pytest.mark.unit
    def test_chi_square_insufficient_data(self, validator):
        """Test chi-square with insufficient data."""
        observed = np.array([10])

        result = validator.chi_square_test(observed)

        assert isinstance(result, StatisticalValidationResult)
        assert "Error" in result.conclusion or result.conclusion.startswith("Error")

    @pytest.mark.unit
    def test_anova_test_two_groups(self, validator):
        """Test ANOVA with two groups."""
        group1 = np.random.normal(100, 10, 50)
        group2 = np.random.normal(105, 10, 50)

        result = validator.anova_test([group1, group2])

        assert isinstance(result, StatisticalValidationResult)
        assert result.test_name == "anova_test"
        assert result.metadata["num_groups"] == 2
        assert 0 <= result.p_value <= 1

    @pytest.mark.unit
    def test_anova_test_three_groups(self, validator):
        """Test ANOVA with three groups."""
        group1 = np.random.normal(100, 10, 50)
        group2 = np.random.normal(105, 10, 50)
        group3 = np.random.normal(110, 10, 50)

        result = validator.anova_test([group1, group2, group3])

        assert isinstance(result, StatisticalValidationResult)
        assert result.metadata["num_groups"] == 3

    @pytest.mark.unit
    def test_anova_test_insufficient_groups(self, validator):
        """Test ANOVA with insufficient groups."""
        result = validator.anova_test([np.array([1, 2, 3])])

        assert isinstance(result, StatisticalValidationResult)
        assert "Error" in result.conclusion or result.conclusion.startswith("Error")

    @pytest.mark.unit
    def test_normality_test_shapiro(self, validator):
        """Test normality test with Shapiro-Wilk."""
        normal_data = np.random.normal(100, 10, 30)  # Small sample

        result = validator.normality_test(normal_data, method="shapiro")

        assert isinstance(result, StatisticalValidationResult)
        assert "shapiro" in result.test_name or "normaltest" in result.test_name
        assert 0 <= result.p_value <= 1

    @pytest.mark.unit
    def test_normality_test_large_sample(self, validator):
        """Test normality test with large sample (uses normaltest)."""
        normal_data = np.random.normal(100, 10, 6000)  # Large sample

        result = validator.normality_test(normal_data, method="shapiro")

        assert isinstance(result, StatisticalValidationResult)
        # Should automatically use normaltest for large samples

    @pytest.mark.unit
    def test_normality_test_non_normal(self, validator):
        """Test normality test with non-normal data."""
        non_normal = np.random.exponential(2, 50)  # Exponential distribution

        result = validator.normality_test(non_normal)

        assert isinstance(result, StatisticalValidationResult)
        # Non-normal data should have lower p-value

    @pytest.mark.unit
    def test_validate_improvement_greater(self, validator):
        """Test validate_improvement with 'greater' alternative."""
        baseline = np.array([0.5, 0.6, 0.55, 0.58, 0.57])
        improved = np.array([0.7, 0.75, 0.72, 0.73, 0.71])  # Better

        result = validator.validate_improvement(baseline, improved, alternative="greater")

        assert isinstance(result, StatisticalValidationResult)
        assert ">" in result.alternative_hypothesis or "greater" in result.alternative_hypothesis.lower()

    @pytest.mark.unit
    def test_validate_improvement_paired(self, validator):
        """Test validate_improvement with paired samples."""
        baseline = np.array([0.5, 0.6, 0.55, 0.58, 0.57])
        improved = np.array([0.7, 0.75, 0.72, 0.73, 0.71])  # Same length

        result = validator.validate_improvement(baseline, improved)

        assert isinstance(result, StatisticalValidationResult)
        # Should detect improvement if significant
