"""
AM-QADF Analytics Module

Comprehensive analytics for voxel domain data.
Domain-driven organization with multiple analysis domains.
"""

# Sensitivity Analysis
from .sensitivity_analysis import (
    SensitivityAnalysisConfig,
    SensitivityAnalysisClient,
    SensitivityQuery,
    SensitivityResult,
    SensitivityStorage,
    GlobalSensitivityAnalyzer,
    SobolAnalyzer,
    MorrisAnalyzer,
    LocalSensitivityAnalyzer,
    DerivativeAnalyzer,
    ExperimentalDesigner,
    UncertaintyQuantifier,
)

# Statistical Analysis
from .statistical_analysis import (
    AdvancedAnalyticsClient,
    DescriptiveStatsAnalyzer,
    CorrelationAnalyzer,
    TrendAnalyzer,
    PatternAnalyzer,
    MultivariateAnalyzer,
    TimeSeriesAnalyzer,
    RegressionAnalyzer,
    NonparametricAnalyzer,
)

# Quality Assessment
from .quality_assessment import (
    QualityAssessmentClient,
    CompletenessAnalyzer,
    SignalQualityAnalyzer,
    AlignmentAccuracyAnalyzer,
    DataQualityAnalyzer,
)

# Process Analysis
from .process_analysis import (
    ParameterAnalyzer,
    QualityAnalyzer,
    SensorAnalyzer,
    ProcessOptimizer,
)

# Virtual Experiments
from .virtual_experiments import (
    VirtualExperimentClient,
    ExperimentQuery,
    ExperimentStorage,
    VirtualExperimentResultAnalyzer,
    ComparisonAnalyzer,
    ParameterOptimizer,
)

# Reporting
from .reporting import (
    AnalysisReportGenerator,
    AnalysisVisualizer,
    AnalysisDocumentation,
)

# Statistical Process Control
from .spc import (
    SPCClient,
    SPCConfig,
    ControlChartGenerator,
    ControlChartResult,
    BaselineCalculator,
    BaselineStatistics,
    AdaptiveLimitsCalculator,
    ProcessCapabilityAnalyzer,
    ProcessCapabilityResult,
    MultivariateSPCAnalyzer,
    MultivariateSPCResult,
    ControlRuleDetector,
    ControlRuleViolation,
    SPCStorage,
)

__all__ = [
    # Sensitivity Analysis
    "SensitivityAnalysisConfig",
    "SensitivityAnalysisClient",
    "SensitivityQuery",
    "SensitivityResult",
    "SensitivityStorage",
    "GlobalSensitivityAnalyzer",
    "SobolAnalyzer",
    "MorrisAnalyzer",
    "LocalSensitivityAnalyzer",
    "DerivativeAnalyzer",
    "ExperimentalDesigner",
    "UncertaintyQuantifier",
    # Statistical Analysis
    "AdvancedAnalyticsClient",
    "DescriptiveStatsAnalyzer",
    "CorrelationAnalyzer",
    "TrendAnalyzer",
    "PatternAnalyzer",
    "MultivariateAnalyzer",
    "TimeSeriesAnalyzer",
    "RegressionAnalyzer",
    "NonparametricAnalyzer",
    # Quality Assessment
    "QualityAssessmentClient",
    "CompletenessAnalyzer",
    "SignalQualityAnalyzer",
    "AlignmentAccuracyAnalyzer",
    "DataQualityAnalyzer",
    # Process Analysis
    "ParameterAnalyzer",
    "QualityAnalyzer",
    "SensorAnalyzer",
    "ProcessOptimizer",
    # Virtual Experiments
    "VirtualExperimentClient",
    "ExperimentQuery",
    "ExperimentStorage",
    "VirtualExperimentResultAnalyzer",
    "ComparisonAnalyzer",
    "ParameterOptimizer",
    # Reporting
    "AnalysisReportGenerator",
    "AnalysisVisualizer",
    "AnalysisDocumentation",
    # Statistical Process Control
    "SPCClient",
    "SPCConfig",
    "ControlChartGenerator",
    "ControlChartResult",
    "BaselineCalculator",
    "BaselineStatistics",
    "AdaptiveLimitsCalculator",
    "ProcessCapabilityAnalyzer",
    "ProcessCapabilityResult",
    "MultivariateSPCAnalyzer",
    "MultivariateSPCResult",
    "ControlRuleDetector",
    "ControlRuleViolation",
    "SPCStorage",
]
