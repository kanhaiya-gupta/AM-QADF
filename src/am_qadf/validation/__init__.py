"""
AM-QADF Validation Module

Validation and benchmarking modules for AM-QADF framework.
Handles performance benchmarking, MPM system comparison, accuracy validation,
and statistical significance testing.
"""

from .validation_client import ValidationClient

from .benchmarking import (
    BenchmarkResult,
    PerformanceBenchmarker,
    benchmark,
)

from .mpm_comparison import (
    MPMComparisonResult,
    MPMComparisonEngine,
)

from .accuracy_validation import (
    AccuracyValidationResult,
    AccuracyValidator,
)

from .statistical_validation import (
    StatisticalValidationResult,
    StatisticalValidator,
)

# Import config classes
try:
    from .validation_client import ValidationConfig
except ImportError:
    ValidationConfig = None

__all__ = [
    # Main client
    "ValidationClient",
    "ValidationConfig",
    # Benchmarking
    "BenchmarkResult",
    "PerformanceBenchmarker",
    "benchmark",
    # MPM Comparison
    "MPMComparisonResult",
    "MPMComparisonEngine",
    # Accuracy Validation
    "AccuracyValidationResult",
    "AccuracyValidator",
    # Statistical Validation
    "StatisticalValidationResult",
    "StatisticalValidator",
]
