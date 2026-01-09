"""
AM-QADF Anomaly Detection Detectors Module

All detector implementations organized by category.
"""

# Statistical detectors
from .statistical import (
    ZScoreDetector,
    IQRDetector,
    MahalanobisDetector,
    ModifiedZScoreDetector,
    GrubbsDetector,
)

# Clustering detectors
from .clustering import (
    DBSCANDetector,
    IsolationForestDetector,
    LOFDetector,
    OneClassSVMDetector,
    KMeansDetector,
)

# Ensemble detectors
try:
    from .ensemble import (
        VotingEnsembleDetector,
        WeightedEnsembleDetector,
    )

    ENSEMBLE_AVAILABLE = True
except ImportError:
    ENSEMBLE_AVAILABLE = False
    VotingEnsembleDetector = None
    WeightedEnsembleDetector = None

# Machine learning detectors
try:
    from .machine_learning import (
        AutoencoderDetector,
        RandomForestAnomalyDetector,
    )

    ML_AVAILABLE = True
    ML_DETECTORS = ["AutoencoderDetector", "RandomForestAnomalyDetector"]

    # Optional ML methods
    try:
        from .machine_learning import LSTMAutoencoderDetector

        ML_DETECTORS.append("LSTMAutoencoderDetector")
    except ImportError:
        LSTMAutoencoderDetector = None

    try:
        from .machine_learning import VAEDetector

        ML_DETECTORS.append("VAEDetector")
    except ImportError:
        VAEDetector = None
except ImportError:
    ML_AVAILABLE = False
    AutoencoderDetector = None
    RandomForestAnomalyDetector = None
    LSTMAutoencoderDetector = None
    VAEDetector = None
    ML_DETECTORS = []

# Rule-based detectors
try:
    from .rule_based import (
        ThresholdViolationDetector,
        PatternDeviationDetector,
        TemporalPatternDetector,
        SpatialPatternDetector,
        MultiSignalCorrelationDetector,
    )

    RULE_BASED_AVAILABLE = True
except ImportError:
    RULE_BASED_AVAILABLE = False
    ThresholdViolationDetector = None
    PatternDeviationDetector = None
    TemporalPatternDetector = None
    SpatialPatternDetector = None
    MultiSignalCorrelationDetector = None

__all__ = [
    # Statistical
    "ZScoreDetector",
    "IQRDetector",
    "MahalanobisDetector",
    "ModifiedZScoreDetector",
    "GrubbsDetector",
    # Clustering
    "DBSCANDetector",
    "IsolationForestDetector",
    "LOFDetector",
    "OneClassSVMDetector",
    "KMeansDetector",
]

# Add ensemble if available
if ENSEMBLE_AVAILABLE:
    __all__.extend(
        [
            "VotingEnsembleDetector",
            "WeightedEnsembleDetector",
        ]
    )

# Add ML if available
if ML_AVAILABLE:
    __all__.extend(ML_DETECTORS)

# Add rule-based if available
if RULE_BASED_AVAILABLE:
    __all__.extend(
        [
            "ThresholdViolationDetector",
            "PatternDeviationDetector",
            "TemporalPatternDetector",
            "SpatialPatternDetector",
            "MultiSignalCorrelationDetector",
        ]
    )
