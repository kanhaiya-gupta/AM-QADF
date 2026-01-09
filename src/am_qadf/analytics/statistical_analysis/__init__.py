"""
AM-QADF Statistical Analysis Module

Statistical analysis for voxel domain data.
Handles descriptive statistics, correlation, trends, patterns, multivariate analysis, time series, regression, and non-parametric methods.
"""

from .client import (
    AdvancedAnalyticsClient,
)

from .descriptive_stats import (
    DescriptiveStatistics,
    DescriptiveStatsAnalyzer,
)

from .correlation import (
    CorrelationResults,
    CorrelationAnalyzer,
)

from .trends import (
    TrendResults,
    TrendAnalyzer,
)

from .patterns import (
    PatternResults,
    PatternAnalyzer,
)

from .multivariate import (
    MultivariateConfig,
    MultivariateResult,
    MultivariateAnalyzer,
    PCAAnalyzer,
    ClusterAnalyzer,
)

from .time_series import (
    TimeSeriesConfig,
    TimeSeriesResult,
    TimeSeriesAnalyzer,
    TrendAnalyzer as TimeSeriesTrendAnalyzer,
    SeasonalityAnalyzer,
)

from .regression import (
    RegressionConfig,
    RegressionResult,
    RegressionAnalyzer,
    LinearRegression,
    PolynomialRegression,
)

from .nonparametric import (
    NonparametricConfig,
    NonparametricResult,
    NonparametricAnalyzer,
    KernelDensityAnalyzer,
)

__all__ = [
    # Client
    "AdvancedAnalyticsClient",
    # Descriptive statistics
    "DescriptiveStatistics",
    "DescriptiveStatsAnalyzer",
    # Correlation
    "CorrelationResults",
    "CorrelationAnalyzer",
    # Trends
    "TrendResults",
    "TrendAnalyzer",
    # Patterns
    "PatternResults",
    "PatternAnalyzer",
    # Multivariate
    "MultivariateConfig",
    "MultivariateResult",
    "MultivariateAnalyzer",
    "PCAAnalyzer",
    "ClusterAnalyzer",
    # Time series
    "TimeSeriesConfig",
    "TimeSeriesResult",
    "TimeSeriesAnalyzer",
    "TimeSeriesTrendAnalyzer",
    "SeasonalityAnalyzer",
    # Regression
    "RegressionConfig",
    "RegressionResult",
    "RegressionAnalyzer",
    "LinearRegression",
    "PolynomialRegression",
    # Non-parametric
    "NonparametricConfig",
    "NonparametricResult",
    "NonparametricAnalyzer",
    "KernelDensityAnalyzer",
]
