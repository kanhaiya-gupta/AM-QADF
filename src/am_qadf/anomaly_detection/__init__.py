"""
AM-QADF Anomaly Detection Module

Comprehensive anomaly detection for voxel domain data.
Handles detection, querying, storage, and voxel-level anomaly analysis.
"""

# Core
from .core.base_detector import (
    BaseAnomalyDetector,
    AnomalyDetectionResult,
    AnomalyDetectionConfig,
)
from .core.types import AnomalyType

# Preprocessing
from .utils.preprocessing import (
    PreprocessingConfig,
    DataPreprocessor,
    extract_features_from_fused_data,
)

# Synthetic anomalies (optional)
try:
    from .utils.synthetic_anomalies import (
        SyntheticAnomalyGenerator,
        AnomalyInjectionConfig,
        AnomalyInjectionType,
    )

    SYNTHETIC_AVAILABLE = True
except ImportError:
    SYNTHETIC_AVAILABLE = False
    SyntheticAnomalyGenerator = None
    AnomalyInjectionConfig = None
    AnomalyInjectionType = None

# Statistical methods
from .detectors.statistical import (
    ZScoreDetector,
    IQRDetector,
    MahalanobisDetector,
    ModifiedZScoreDetector,
    GrubbsDetector,
)

# Clustering methods
from .detectors.clustering import (
    DBSCANDetector,
    IsolationForestDetector,
    LOFDetector,
    OneClassSVMDetector,
    KMeansDetector,
)

# Ensemble methods
try:
    from .detectors.ensemble import (
        VotingEnsembleDetector,
        WeightedEnsembleDetector,
    )

    ENSEMBLE_AVAILABLE = True
except ImportError:
    ENSEMBLE_AVAILABLE = False
    VotingEnsembleDetector = None
    WeightedEnsembleDetector = None

# Machine learning methods (optional - may require tensorflow)
try:
    from .detectors.machine_learning import (
        AutoencoderDetector,
        RandomForestAnomalyDetector,
    )

    ML_AVAILABLE = True
    ML_DETECTORS = ["AutoencoderDetector", "RandomForestAnomalyDetector"]

    # Optional ML methods
    try:
        from .detectors.machine_learning import LSTMAutoencoderDetector

        ML_DETECTORS.append("LSTMAutoencoderDetector")
    except ImportError:
        LSTMAutoencoderDetector = None

    try:
        from .detectors.machine_learning import VAEDetector

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

# Rule-based methods
try:
    from .detectors.rule_based import (
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

# Evaluation
try:
    from .evaluation import (
        AnomalyDetectionMetrics,
        calculate_classification_metrics,
        calculate_ranking_metrics,
        calculate_statistical_metrics,
        AnomalyDetectionCV,
        k_fold_cv,
        time_series_cv,
        spatial_cv,
        AnomalyDetectionComparison,
        compare_detectors,
        statistical_significance_test,
    )

    EVALUATION_AVAILABLE = True
except ImportError:
    EVALUATION_AVAILABLE = False
    AnomalyDetectionMetrics = None
    calculate_classification_metrics = None
    calculate_ranking_metrics = None
    calculate_statistical_metrics = None
    AnomalyDetectionCV = None
    k_fold_cv = None
    time_series_cv = None
    spatial_cv = None
    AnomalyDetectionComparison = None
    compare_detectors = None
    statistical_significance_test = None

# Integration
from .integration.client import (
    AnomalyDetectionConfig as IntegrationAnomalyDetectionConfig,
    AnomalyDetectionClient,
)
from .integration.query import AnomalyQuery
from .integration.storage import AnomalyResult, AnomalyStorage

# Utils
from .utils.voxel_detection import (
    VoxelAnomalyResult,
    VoxelAnomalyDetector,
)

__all__ = [
    # Base classes
    "BaseAnomalyDetector",
    "AnomalyDetectionResult",
    "AnomalyDetectionConfig",
    "AnomalyType",
    # Preprocessing
    "DataPreprocessor",
    "PreprocessingConfig",
    "extract_features_from_fused_data",
    # Synthetic anomalies
    "SyntheticAnomalyGenerator",
    "AnomalyInjectionConfig",
    "AnomalyInjectionType",
    # Statistical methods
    "ZScoreDetector",
    "IQRDetector",
    "MahalanobisDetector",
    "ModifiedZScoreDetector",
    "GrubbsDetector",
    # Clustering methods
    "DBSCANDetector",
    "IsolationForestDetector",
    "LOFDetector",
    "OneClassSVMDetector",
    "KMeansDetector",
    # Integration
    "AnomalyDetectionClient",
    "AnomalyQuery",
    "AnomalyResult",
    "AnomalyStorage",
    # Voxel detection
    "VoxelAnomalyResult",
    "VoxelAnomalyDetector",
]

# Add ensemble if available
if ENSEMBLE_AVAILABLE:
    __all__.extend(
        [
            "VotingEnsembleDetector",
            "WeightedEnsembleDetector",
        ]
    )

# Add ML methods if available
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

# Add evaluation if available
if EVALUATION_AVAILABLE:
    __all__.extend(
        [
            "AnomalyDetectionMetrics",
            "calculate_classification_metrics",
            "calculate_ranking_metrics",
            "calculate_statistical_metrics",
            "AnomalyDetectionCV",
            "k_fold_cv",
            "time_series_cv",
            "spatial_cv",
            "AnomalyDetectionComparison",
            "compare_detectors",
            "statistical_significance_test",
        ]
    )
