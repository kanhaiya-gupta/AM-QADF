"""
AM-QADF Virtual Experiments Module

Virtual experiments for voxel domain data.
Handles experiment design, execution, result analysis, comparison, and parameter optimization.
"""

from .client import (
    VirtualExperimentConfig,
    VirtualExperimentClient,
)

from .query import (
    ExperimentQuery,
)

from .storage import (
    ExperimentResult,
    ExperimentStorage,
)

from .result_analyzer import (
    AnalysisResult,
    VirtualExperimentResultAnalyzer,
)

from .comparison_analyzer import (
    ComparisonResult,
    ComparisonAnalyzer,
)

from .parameter_optimizer import (
    OptimizationResult,
    ParameterOptimizer,
)

__all__ = [
    # Client
    "VirtualExperimentConfig",
    "VirtualExperimentClient",
    # Query
    "ExperimentQuery",
    # Storage
    "ExperimentResult",
    "ExperimentStorage",
    # Result analysis
    "AnalysisResult",
    "VirtualExperimentResultAnalyzer",
    # Comparison
    "ComparisonResult",
    "ComparisonAnalyzer",
    # Optimization
    "OptimizationResult",
    "ParameterOptimizer",
]
