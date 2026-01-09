"""
AM-QADF Quality Module

Quality assessment modules for voxel domain data.
Handles data completeness, signal quality, alignment accuracy, and overall data quality metrics.
"""

from .quality_assessment_client import QualityAssessmentClient

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
    # Main client
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
