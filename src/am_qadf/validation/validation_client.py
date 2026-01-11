"""
Validation Client

Main client for comprehensive validation operations including benchmarking,
MPM comparison, accuracy validation, and statistical testing.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import logging
from datetime import datetime
import time

logger = logging.getLogger(__name__)

# Import validation components with fallback for direct loading
try:
    from .benchmarking import PerformanceBenchmarker, BenchmarkResult
    from .mpm_comparison import MPMComparisonEngine, MPMComparisonResult
    from .accuracy_validation import AccuracyValidator, AccuracyValidationResult
    from .statistical_validation import StatisticalValidator, StatisticalValidationResult
except ImportError:
    # Fallback for direct module loading (e.g., in notebooks)
    import sys
    import importlib.util
    from pathlib import Path

    current_file = Path(__file__)
    module_dir = current_file.parent

    # Load benchmarking
    try:
        benchmarking_path = module_dir / "benchmarking.py"
        if benchmarking_path.exists():
            spec = importlib.util.spec_from_file_location("benchmarking", benchmarking_path)
            benchmark_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(benchmark_module)
            PerformanceBenchmarker = getattr(benchmark_module, "PerformanceBenchmarker", None)
            BenchmarkResult = getattr(benchmark_module, "BenchmarkResult", None)
        else:
            PerformanceBenchmarker = None
            BenchmarkResult = None
    except Exception as e:
        logger.warning(f"Could not load benchmarking module: {e}")
        PerformanceBenchmarker = None
        BenchmarkResult = None

    # Load MPM comparison
    try:
        mpm_path = module_dir / "mpm_comparison.py"
        if mpm_path.exists():
            spec = importlib.util.spec_from_file_location("mpm_comparison", mpm_path)
            mpm_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mpm_module)
            MPMComparisonEngine = getattr(mpm_module, "MPMComparisonEngine", None)
            MPMComparisonResult = getattr(mpm_module, "MPMComparisonResult", None)
        else:
            MPMComparisonEngine = None
            MPMComparisonResult = None
    except Exception as e:
        logger.warning(f"Could not load MPM comparison module: {e}")
        MPMComparisonEngine = None
        MPMComparisonResult = None

    # Load accuracy validation
    try:
        accuracy_path = module_dir / "accuracy_validation.py"
        if accuracy_path.exists():
            spec = importlib.util.spec_from_file_location("accuracy_validation", accuracy_path)
            accuracy_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(accuracy_module)
            AccuracyValidator = getattr(accuracy_module, "AccuracyValidator", None)
            AccuracyValidationResult = getattr(accuracy_module, "AccuracyValidationResult", None)
        else:
            AccuracyValidator = None
            AccuracyValidationResult = None
    except Exception as e:
        logger.warning(f"Could not load accuracy validation module: {e}")
        AccuracyValidator = None
        AccuracyValidationResult = None

    # Load statistical validation
    try:
        statistical_path = module_dir / "statistical_validation.py"
        if statistical_path.exists():
            spec = importlib.util.spec_from_file_location("statistical_validation", statistical_path)
            statistical_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(statistical_module)
            StatisticalValidator = getattr(statistical_module, "StatisticalValidator", None)
            StatisticalValidationResult = getattr(statistical_module, "StatisticalValidationResult", None)
        else:
            StatisticalValidator = None
            StatisticalValidationResult = None
    except Exception as e:
        logger.warning(f"Could not load statistical validation module: {e}")
        StatisticalValidator = None
        StatisticalValidationResult = None


@dataclass
class ValidationConfig:
    """Configuration for validation operations."""

    confidence_level: float = 0.95
    significance_level: float = 0.05
    max_acceptable_error: float = 0.1
    correlation_threshold: float = 0.85
    sample_size: Optional[int] = None
    random_seed: Optional[int] = None
    enable_benchmarking: bool = True
    enable_mpm_comparison: bool = True
    enable_accuracy_validation: bool = True
    enable_statistical_validation: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "confidence_level": self.confidence_level,
            "significance_level": self.significance_level,
            "max_acceptable_error": self.max_acceptable_error,
            "correlation_threshold": self.correlation_threshold,
            "sample_size": self.sample_size,
            "random_seed": self.random_seed,
            "enable_benchmarking": self.enable_benchmarking,
            "enable_mpm_comparison": self.enable_mpm_comparison,
            "enable_accuracy_validation": self.enable_accuracy_validation,
            "enable_statistical_validation": self.enable_statistical_validation,
        }


class ValidationClient:
    """
    Client for comprehensive validation operations.

    Provides:
    - Performance benchmarking of framework operations
    - Comparison with MPM (Melt Pool Monitoring) system outputs
    - Accuracy validation against ground truth data
    - Statistical significance testing
    """

    def __init__(self, config: Optional[ValidationConfig] = None):
        """
        Initialize the validation client.

        Args:
            config: Validation configuration. If None, uses default config.
        """
        self.config = config or ValidationConfig()

        # Initialize sub-modules if available
        if PerformanceBenchmarker and self.config.enable_benchmarking:
            self.benchmarker = PerformanceBenchmarker()
        else:
            self.benchmarker = None
            logger.warning("Benchmarking module not available")

        if MPMComparisonEngine and self.config.enable_mpm_comparison:
            self.mpm_comparer = MPMComparisonEngine(correlation_threshold=self.config.correlation_threshold)
        else:
            self.mpm_comparer = None
            logger.warning("MPM comparison module not available")

        if AccuracyValidator and self.config.enable_accuracy_validation:
            self.accuracy_validator = AccuracyValidator(max_acceptable_error=self.config.max_acceptable_error)
        else:
            self.accuracy_validator = None
            logger.warning("Accuracy validation module not available")

        if StatisticalValidator and self.config.enable_statistical_validation:
            self.statistical_validator = StatisticalValidator(
                significance_level=self.config.significance_level, confidence_level=self.config.confidence_level
            )
        else:
            self.statistical_validator = None
            logger.warning("Statistical validation module not available")

    # Benchmarking methods
    def benchmark_operation(self, operation: Callable, *args, iterations: int = 1, **kwargs) -> Optional[BenchmarkResult]:
        """
        Benchmark a framework operation.

        Args:
            operation: Function or method to benchmark
            *args: Positional arguments for the operation
            iterations: Number of iterations to run (for averaging)
            **kwargs: Keyword arguments for the operation

        Returns:
            BenchmarkResult if benchmarking is available, None otherwise
        """
        if not self.benchmarker:
            logger.error("Benchmarking not available")
            return None

        operation_name = getattr(operation, "__name__", str(operation))
        return self.benchmarker.benchmark_operation(operation_name, operation, *args, iterations=iterations, **kwargs)

    # MPM Comparison methods
    def compare_with_mpm(
        self, framework_data: Any, mpm_data: Any, metrics: Optional[List[str]] = None
    ) -> Dict[str, Optional[MPMComparisonResult]]:
        """
        Compare framework outputs with MPM system outputs.

        Args:
            framework_data: Framework-generated data (dict, array, or object)
            mpm_data: MPM system data (dict, array, or object)
            metrics: List of metric names to compare. If None, compares all available.

        Returns:
            Dictionary mapping metric names to MPMComparisonResult
        """
        if not self.mpm_comparer:
            logger.error("MPM comparison not available")
            return {}

        return self.mpm_comparer.compare_all_metrics(framework_data, mpm_data, metrics)

    def validate_mpm_correlation(self, framework_values: np.ndarray, mpm_values: np.ndarray, method: str = "pearson") -> float:
        """
        Calculate correlation between framework and MPM values.

        Args:
            framework_values: Framework output values
            mpm_values: MPM system output values
            method: Correlation method ('pearson' or 'spearman')

        Returns:
            Correlation coefficient (0 to 1)
        """
        if not self.mpm_comparer:
            logger.error("MPM comparison not available")
            return 0.0

        return self.mpm_comparer.calculate_correlation(framework_values, mpm_values, method)

    # Accuracy Validation methods
    def validate_accuracy(
        self, framework_data: Any, ground_truth: Any, validation_type: str = "signal_mapping"
    ) -> Optional[AccuracyValidationResult]:
        """
        Validate framework accuracy against ground truth.

        Args:
            framework_data: Framework-generated data
            ground_truth: Ground truth reference data
            validation_type: Type of validation ('signal_mapping', 'spatial', 'temporal', 'quality')

        Returns:
            AccuracyValidationResult if validation is available, None otherwise
        """
        if not self.accuracy_validator:
            logger.error("Accuracy validation not available")
            return None

        validation_methods = {
            "signal_mapping": self.accuracy_validator.validate_signal_mapping,
            "spatial": self.accuracy_validator.validate_spatial_alignment,
            "temporal": self.accuracy_validator.validate_temporal_alignment,
            "quality": self.accuracy_validator.validate_quality_metrics,
        }

        if validation_type not in validation_methods:
            logger.error(f"Unknown validation type: {validation_type}")
            return None

        return validation_methods[validation_type](framework_data, ground_truth)

    # Statistical Validation methods
    def perform_statistical_test(self, test_name: str, *args, **kwargs) -> Optional[StatisticalValidationResult]:
        """
        Perform a statistical significance test.

        Args:
            test_name: Name of test ('t_test', 'mann_whitney', 'correlation', etc.)
            *args: Positional arguments for the test
            **kwargs: Keyword arguments for the test

        Returns:
            StatisticalValidationResult if testing is available, None otherwise
        """
        if not self.statistical_validator:
            logger.error("Statistical validation not available")
            return None

        test_methods = {
            "t_test": self.statistical_validator.t_test,
            "mann_whitney": self.statistical_validator.mann_whitney_u_test,
            "correlation": self.statistical_validator.correlation_test,
            "chi_square": self.statistical_validator.chi_square_test,
            "anova": self.statistical_validator.anova_test,
            "normality": self.statistical_validator.normality_test,
        }

        if test_name not in test_methods:
            logger.error(f"Unknown test name: {test_name}")
            return None

        return test_methods[test_name](*args, **kwargs)

    def validate_improvement(
        self, baseline: np.ndarray, improved: np.ndarray, alternative: str = "greater"
    ) -> Optional[StatisticalValidationResult]:
        """
        Validate that improved results are statistically better than baseline.

        Args:
            baseline: Baseline performance values
            improved: Improved performance values
            alternative: Alternative hypothesis ('greater', 'less', 'two-sided')

        Returns:
            StatisticalValidationResult indicating if improvement is significant
        """
        if not self.statistical_validator:
            logger.error("Statistical validation not available")
            return None

        return self.statistical_validator.validate_improvement(baseline, improved, alternative)

    # Comprehensive Validation
    def comprehensive_validation(
        self, framework_data: Any, reference_data: Any, validation_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive validation with multiple validation types.

        Args:
            framework_data: Framework-generated data
            reference_data: Reference data (MPM or ground truth)
            validation_types: List of validation types to perform.
                            If None, performs all available validations.

        Returns:
            Dictionary with validation results for each type
        """
        if validation_types is None:
            validation_types = ["mpm_comparison", "accuracy", "statistical"]

        results = {}

        # MPM Comparison
        if "mpm_comparison" in validation_types and self.mpm_comparer:
            results["mpm_comparison"] = self.compare_with_mpm(framework_data, reference_data)

        # Accuracy Validation
        if "accuracy" in validation_types and self.accuracy_validator:
            results["accuracy"] = self.validate_accuracy(framework_data, reference_data, validation_type="signal_mapping")

        # Statistical Validation
        if "statistical" in validation_types and self.statistical_validator:
            # Convert data to arrays for statistical testing
            if hasattr(framework_data, "flatten"):
                framework_array = framework_data.flatten()
            elif isinstance(framework_data, (list, dict)):
                framework_array = np.array(
                    list(framework_data.values()) if isinstance(framework_data, dict) else framework_data
                )
            else:
                framework_array = np.array([framework_data])

            if hasattr(reference_data, "flatten"):
                reference_array = reference_data.flatten()
            elif isinstance(reference_data, (list, dict)):
                reference_array = np.array(
                    list(reference_data.values()) if isinstance(reference_data, dict) else reference_data
                )
            else:
                reference_array = np.array([reference_data])

            results["statistical"] = self.statistical_validator.t_test(
                framework_array, reference_array, alternative="two-sided"
            )

        return results

    def generate_validation_report(self, results: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive validation report.

        Args:
            results: Dictionary of validation results
            output_path: Optional path to save report. If None, returns report as string.

        Returns:
            Validation report as formatted string
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("AM-QADF Validation Report")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        # MPM Comparison Results
        if "mpm_comparison" in results:
            report_lines.append("MPM System Comparison")
            report_lines.append("-" * 80)
            mpm_results = results["mpm_comparison"]
            if isinstance(mpm_results, dict):
                for metric_name, result in mpm_results.items():
                    if result and hasattr(result, "to_dict"):
                        report_lines.append(f"  Metric: {metric_name}")
                        report_lines.append(f"    Framework Value: {result.framework_value:.6f}")
                        report_lines.append(f"    MPM Value: {result.mpm_value:.6f}")
                        report_lines.append(f"    Correlation: {result.correlation:.4f}")
                        report_lines.append(f"    Relative Error: {result.relative_error:.4f}%")
                        report_lines.append(f"    Valid: {'✓' if result.is_valid else '✗'}")
                        report_lines.append("")
            report_lines.append("")

        # Accuracy Validation Results
        if "accuracy" in results and results["accuracy"]:
            report_lines.append("Accuracy Validation")
            report_lines.append("-" * 80)
            acc_result = results["accuracy"]
            if hasattr(acc_result, "to_dict"):
                report_lines.append(f"  Signal: {acc_result.signal_name}")
                report_lines.append(f"  RMSE: {acc_result.rmse:.6f}")
                report_lines.append(f"  MAE: {acc_result.mae:.6f}")
                report_lines.append(f"  R² Score: {acc_result.r2_score:.4f}")
                report_lines.append(f"  Max Error: {acc_result.max_error:.6f}")
                report_lines.append(f"  Within Tolerance: {'✓' if acc_result.within_tolerance else '✗'}")
                report_lines.append(f"  Validated Points: {acc_result.validated_points}/{acc_result.ground_truth_size}")
                report_lines.append("")

        # Statistical Validation Results
        if "statistical" in results and results["statistical"]:
            report_lines.append("Statistical Validation")
            report_lines.append("-" * 80)
            stat_result = results["statistical"]
            if hasattr(stat_result, "to_dict"):
                report_lines.append(f"  Test: {stat_result.test_name}")
                report_lines.append(f"  Null Hypothesis: {stat_result.null_hypothesis}")
                report_lines.append(f"  Test Statistic: {stat_result.test_statistic:.6f}")
                report_lines.append(f"  P-value: {stat_result.p_value:.6f}")
                report_lines.append(f"  Significance Level: {stat_result.significance_level:.4f}")
                report_lines.append(f"  Significant: {'✓' if stat_result.is_significant else '✗'}")
                report_lines.append(f"  Conclusion: {stat_result.conclusion}")
                report_lines.append("")

        report_lines.append("=" * 80)

        report_text = "\n".join(report_lines)

        if output_path:
            with open(output_path, "w") as f:
                f.write(report_text)
            logger.info(f"Validation report saved to {output_path}")

        return report_text
