"""
Statistical Validation

Statistical significance testing and validation utilities.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging
from datetime import datetime
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class StatisticalValidationResult:
    """Result of statistical validation test."""

    test_name: str
    null_hypothesis: str
    alternative_hypothesis: str
    test_statistic: float
    p_value: float
    significance_level: float
    is_significant: bool
    conclusion: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "test_name": self.test_name,
            "null_hypothesis": self.null_hypothesis,
            "alternative_hypothesis": self.alternative_hypothesis,
            "test_statistic": self.test_statistic,
            "p_value": self.p_value,
            "significance_level": self.significance_level,
            "is_significant": self.is_significant,
            "conclusion": self.conclusion,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class StatisticalValidator:
    """
    Validator for statistical significance testing.

    Provides:
    - T-tests (paired, unpaired)
    - Non-parametric tests (Mann-Whitney U, Wilcoxon)
    - Correlation significance tests
    - ANOVA for multiple groups
    - Normality testing
    """

    def __init__(self, significance_level: float = 0.05, confidence_level: float = 0.95):
        """
        Initialize statistical validator.

        Args:
            significance_level: Significance level (alpha) for hypothesis tests
            confidence_level: Confidence level for intervals
        """
        self.significance_level = significance_level
        self.confidence_level = confidence_level

    def t_test(
        self, sample1: np.ndarray, sample2: np.ndarray, alternative: str = "two-sided", paired: bool = False
    ) -> StatisticalValidationResult:
        """
        Perform t-test for two samples.

        Args:
            sample1: First sample
            sample2: Second sample
            alternative: Alternative hypothesis ('two-sided', 'less', 'greater')
            paired: Whether samples are paired

        Returns:
            StatisticalValidationResult
        """
        sample1_flat = sample1.flatten()
        sample2_flat = sample2.flatten()

        # Remove NaN values
        valid_mask1 = ~np.isnan(sample1_flat)
        valid_mask2 = ~np.isnan(sample2_flat)
        sample1_valid = sample1_flat[valid_mask1]
        sample2_valid = sample2_flat[valid_mask2]

        if len(sample1_valid) < 2 or len(sample2_valid) < 2:
            logger.warning("Insufficient valid data for t-test")
            return self._create_error_result("t_test", "Insufficient data")

        try:
            if paired and len(sample1_valid) == len(sample2_valid):
                statistic, p_value = stats.ttest_rel(sample1_valid, sample2_valid, alternative=alternative)
            else:
                statistic, p_value = stats.ttest_ind(sample1_valid, sample2_valid, alternative=alternative)

            null_hyp = "μ₁ = μ₂ (means are equal)"
            alt_hyp = self._get_alternative_hypothesis(alternative, "means")
            is_significant = bool(p_value < self.significance_level)
            conclusion = self._get_conclusion(is_significant, null_hyp)

            return StatisticalValidationResult(
                test_name="t_test" + ("_paired" if paired else "_independent"),
                null_hypothesis=null_hyp,
                alternative_hypothesis=alt_hyp,
                test_statistic=float(statistic),
                p_value=float(p_value),
                significance_level=self.significance_level,
                is_significant=is_significant,
                conclusion=conclusion,
                metadata={
                    "sample1_size": len(sample1_valid),
                    "sample2_size": len(sample2_valid),
                    "sample1_mean": float(np.mean(sample1_valid)),
                    "sample2_mean": float(np.mean(sample2_valid)),
                    "alternative": alternative,
                    "paired": paired,
                },
            )
        except Exception as e:
            logger.error(f"T-test failed: {e}")
            return self._create_error_result("t_test", str(e))

    def mann_whitney_u_test(
        self, sample1: np.ndarray, sample2: np.ndarray, alternative: str = "two-sided"
    ) -> StatisticalValidationResult:
        """
        Perform Mann-Whitney U test (non-parametric).

        Args:
            sample1: First sample
            sample2: Second sample
            alternative: Alternative hypothesis

        Returns:
            StatisticalValidationResult
        """
        sample1_flat = sample1.flatten()
        sample2_flat = sample2.flatten()

        valid_mask1 = ~np.isnan(sample1_flat)
        valid_mask2 = ~np.isnan(sample2_flat)
        sample1_valid = sample1_flat[valid_mask1]
        sample2_valid = sample2_flat[valid_mask2]

        if len(sample1_valid) < 2 or len(sample2_valid) < 2:
            return self._create_error_result("mann_whitney_u_test", "Insufficient data")

        try:
            statistic, p_value = stats.mannwhitneyu(sample1_valid, sample2_valid, alternative=alternative)

            null_hyp = "Distributions are equal"
            alt_hyp = self._get_alternative_hypothesis(alternative, "distributions")
            is_significant = bool(p_value < self.significance_level)
            conclusion = self._get_conclusion(is_significant, null_hyp)

            return StatisticalValidationResult(
                test_name="mann_whitney_u_test",
                null_hypothesis=null_hyp,
                alternative_hypothesis=alt_hyp,
                test_statistic=float(statistic),
                p_value=float(p_value),
                significance_level=self.significance_level,
                is_significant=is_significant,
                conclusion=conclusion,
                metadata={
                    "sample1_size": len(sample1_valid),
                    "sample2_size": len(sample2_valid),
                    "alternative": alternative,
                },
            )
        except Exception as e:
            logger.error(f"Mann-Whitney U test failed: {e}")
            return self._create_error_result("mann_whitney_u_test", str(e))

    def correlation_test(self, x: np.ndarray, y: np.ndarray, method: str = "pearson") -> StatisticalValidationResult:
        """
        Test significance of correlation.

        Args:
            x: First variable
            y: Second variable
            method: Correlation method ('pearson' or 'spearman')

        Returns:
            StatisticalValidationResult
        """
        x_flat = x.flatten()
        y_flat = y.flatten()

        min_len = min(len(x_flat), len(y_flat))
        x_valid = x_flat[:min_len]
        y_valid = y_flat[:min_len]

        valid_mask = ~(np.isnan(x_valid) | np.isnan(y_valid))
        x_clean = x_valid[valid_mask]
        y_clean = y_valid[valid_mask]

        if len(x_clean) < 3:
            return self._create_error_result("correlation_test", "Insufficient data")

        try:
            if method.lower() == "pearson":
                statistic, p_value = stats.pearsonr(x_clean, y_clean)
            elif method.lower() == "spearman":
                statistic, p_value = stats.spearmanr(x_clean, y_clean)
            else:
                logger.warning(f"Unknown correlation method: {method}, using pearson")
                statistic, p_value = stats.pearsonr(x_clean, y_clean)

            null_hyp = f"ρ = 0 (no {method} correlation)"
            alt_hyp = f"ρ ≠ 0 ({method} correlation exists)"
            is_significant = bool(p_value < self.significance_level)
            conclusion = self._get_conclusion(is_significant, null_hyp)

            return StatisticalValidationResult(
                test_name=f"correlation_test_{method}",
                null_hypothesis=null_hyp,
                alternative_hypothesis=alt_hyp,
                test_statistic=float(statistic),
                p_value=float(p_value),
                significance_level=self.significance_level,
                is_significant=is_significant,
                conclusion=conclusion,
                metadata={
                    "correlation": float(statistic),
                    "method": method,
                    "sample_size": len(x_clean),
                },
            )
        except Exception as e:
            logger.error(f"Correlation test failed: {e}")
            return self._create_error_result("correlation_test", str(e))

    def chi_square_test(self, observed: np.ndarray, expected: Optional[np.ndarray] = None) -> StatisticalValidationResult:
        """
        Perform chi-square test.

        Args:
            observed: Observed frequencies
            expected: Expected frequencies (uniform if None)

        Returns:
            StatisticalValidationResult
        """
        observed_flat = observed.flatten()
        observed_clean = observed_flat[~np.isnan(observed_flat)]

        if len(observed_clean) < 2:
            return self._create_error_result("chi_square_test", "Insufficient data")

        try:
            if expected is None:
                expected_flat = np.full_like(observed_clean, np.mean(observed_clean))
            else:
                expected_flat = expected.flatten()
                expected_clean = expected_flat[~np.isnan(expected_flat)]
                if len(expected_clean) != len(observed_clean):
                    expected_clean = np.full_like(observed_clean, np.mean(observed_clean))
                expected_flat = expected_clean

            statistic, p_value = stats.chisquare(observed_clean, expected_flat)

            null_hyp = "Observed frequencies match expected"
            alt_hyp = "Observed frequencies differ from expected"
            is_significant = bool(p_value < self.significance_level)
            conclusion = self._get_conclusion(is_significant, null_hyp)

            return StatisticalValidationResult(
                test_name="chi_square_test",
                null_hypothesis=null_hyp,
                alternative_hypothesis=alt_hyp,
                test_statistic=float(statistic),
                p_value=float(p_value),
                significance_level=self.significance_level,
                is_significant=is_significant,
                conclusion=conclusion,
                metadata={
                    "observed_sum": float(np.sum(observed_clean)),
                    "expected_sum": float(np.sum(expected_flat)),
                    "categories": len(observed_clean),
                },
            )
        except Exception as e:
            logger.error(f"Chi-square test failed: {e}")
            return self._create_error_result("chi_square_test", str(e))

    def anova_test(self, groups: List[np.ndarray]) -> StatisticalValidationResult:
        """
        Perform one-way ANOVA test.

        Args:
            groups: List of sample groups

        Returns:
            StatisticalValidationResult
        """
        if len(groups) < 2:
            return self._create_error_result("anova_test", "Need at least 2 groups")

        try:
            # Clean groups
            clean_groups = []
            for group in groups:
                group_flat = group.flatten()
                group_clean = group_flat[~np.isnan(group_flat)]
                if len(group_clean) > 0:
                    clean_groups.append(group_clean)

            if len(clean_groups) < 2:
                return self._create_error_result("anova_test", "Insufficient valid groups")

            statistic, p_value = stats.f_oneway(*clean_groups)

            null_hyp = "All group means are equal"
            alt_hyp = "At least one group mean differs"
            is_significant = bool(p_value < self.significance_level)
            conclusion = self._get_conclusion(is_significant, null_hyp)

            return StatisticalValidationResult(
                test_name="anova_test",
                null_hypothesis=null_hyp,
                alternative_hypothesis=alt_hyp,
                test_statistic=float(statistic),
                p_value=float(p_value),
                significance_level=self.significance_level,
                is_significant=is_significant,
                conclusion=conclusion,
                metadata={
                    "num_groups": len(clean_groups),
                    "group_sizes": [len(g) for g in clean_groups],
                    "group_means": [float(np.mean(g)) for g in clean_groups],
                },
            )
        except Exception as e:
            logger.error(f"ANOVA test failed: {e}")
            return self._create_error_result("anova_test", str(e))

    def normality_test(self, sample: np.ndarray, method: str = "shapiro") -> StatisticalValidationResult:
        """
        Test for normality.

        Args:
            sample: Sample data
            method: Test method ('shapiro' or 'normaltest')

        Returns:
            StatisticalValidationResult
        """
        sample_flat = sample.flatten()
        sample_clean = sample_flat[~np.isnan(sample_flat)]

        if len(sample_clean) < 3:
            return self._create_error_result("normality_test", "Insufficient data")

        try:
            if method.lower() == "shapiro":
                # Shapiro-Wilk test (recommended for small samples)
                if len(sample_clean) > 5000:
                    logger.warning("Sample size > 5000, using normaltest instead")
                    statistic, p_value = stats.normaltest(sample_clean)
                    method = "normaltest"
                else:
                    statistic, p_value = stats.shapiro(sample_clean)
            elif method.lower() == "normaltest":
                statistic, p_value = stats.normaltest(sample_clean)
            else:
                logger.warning(f"Unknown normality test method: {method}, using shapiro")
                statistic, p_value = (
                    stats.shapiro(sample_clean) if len(sample_clean) <= 5000 else stats.normaltest(sample_clean)
                )

            null_hyp = "Sample is normally distributed"
            alt_hyp = "Sample is not normally distributed"
            is_significant = bool(p_value < self.significance_level)
            conclusion = self._get_conclusion(is_significant, null_hyp, invert=True)  # Invert: significant = not normal

            return StatisticalValidationResult(
                test_name=f"normality_test_{method}",
                null_hypothesis=null_hyp,
                alternative_hypothesis=alt_hyp,
                test_statistic=float(statistic),
                p_value=float(p_value),
                significance_level=self.significance_level,
                is_significant=is_significant,
                conclusion=conclusion,
                metadata={
                    "sample_size": len(sample_clean),
                    "method": method,
                    "sample_mean": float(np.mean(sample_clean)),
                    "sample_std": float(np.std(sample_clean)),
                },
            )
        except Exception as e:
            logger.error(f"Normality test failed: {e}")
            return self._create_error_result("normality_test", str(e))

    def validate_improvement(
        self, baseline: np.ndarray, improved: np.ndarray, alternative: str = "greater"
    ) -> StatisticalValidationResult:
        """
        Validate that improved results are statistically better than baseline.

        Args:
            baseline: Baseline performance values
            improved: Improved performance values
            alternative: Alternative hypothesis ('greater' means improved > baseline)

        Returns:
            StatisticalValidationResult
        """
        # Use paired t-test if same length, otherwise independent
        paired = len(baseline.flatten()) == len(improved.flatten())
        return self.t_test(baseline, improved, alternative=alternative, paired=paired)

    def _get_alternative_hypothesis(self, alternative: str, parameter: str) -> str:
        """Get formatted alternative hypothesis."""
        alternatives = {
            "two-sided": f"{parameter} differ",
            "less": f"{parameter} of first sample < second sample",
            "greater": f"{parameter} of first sample > second sample",
        }
        return alternatives.get(alternative, f"{parameter} differ")

    def _get_conclusion(self, is_significant: bool, null_hyp: str, invert: bool = False) -> str:
        """Get conclusion string."""
        if invert:
            is_significant = not is_significant

        if is_significant:
            return f"Reject H₀: {null_hyp} (p < {self.significance_level})"
        else:
            return f"Fail to reject H₀: {null_hyp} (p ≥ {self.significance_level})"

    def _create_error_result(self, test_name: str, error_message: str) -> StatisticalValidationResult:
        """Create error result."""
        return StatisticalValidationResult(
            test_name=test_name,
            null_hypothesis="N/A",
            alternative_hypothesis="N/A",
            test_statistic=float("nan"),
            p_value=float("nan"),
            significance_level=self.significance_level,
            is_significant=False,
            conclusion=f"Error: {error_message}",
            metadata={"error": error_message},
        )
