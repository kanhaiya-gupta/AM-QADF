"""
AM-QADF Quality Assessment Module (Analytics)

Quality assessment for voxel domain data.
Handles completeness, signal quality, alignment accuracy, and overall data quality metrics.
"""

from .client import (
    QualityAssessmentClient,
)

from .completeness import (
    GapFillingStrategy,
    CompletenessMetrics,
    CompletenessAnalyzer,
)

from .signal_quality import (
    SignalQualityMetrics,
    SignalQualityAnalyzer,
)

from .alignment_accuracy import (
    AlignmentAccuracyMetrics,
    AlignmentAccuracyAnalyzer,
)

from .data_quality import (
    DataQualityMetrics,
    DataQualityAnalyzer,
)

__all__ = [
    # Client
    "QualityAssessmentClient",
    # Completeness
    "GapFillingStrategy",
    "CompletenessMetrics",
    "CompletenessAnalyzer",
    # Signal quality
    "SignalQualityMetrics",
    "SignalQualityAnalyzer",
    # Alignment accuracy
    "AlignmentAccuracyMetrics",
    "AlignmentAccuracyAnalyzer",
    # Data quality
    "DataQualityMetrics",
    "DataQualityAnalyzer",
]
