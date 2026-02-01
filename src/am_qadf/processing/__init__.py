"""
AM-QADF Processing Module

Signal processing and noise reduction.
Handles noise filtering, signal smoothing, and derived signal generation.
"""

# C++ wrappers (placeholder files - will be implemented)
try:
    from .noise_reduction import (
        OutlierDetector,
        SignalSmoother,
        SignalQualityMetrics,
        NoiseReductionPipeline,
    )
except (NotImplementedError, ImportError):
    OutlierDetector = None
    SignalSmoother = None
    SignalQualityMetrics = None
    NoiseReductionPipeline = None

try:
    from .signal_generation import (
        ThermalFieldGenerator,
        DensityFieldEstimator,
        StressFieldGenerator,
    )
except (NotImplementedError, ImportError):
    ThermalFieldGenerator = None
    DensityFieldEstimator = None
    StressFieldGenerator = None

__all__ = [
    # Noise reduction
    "OutlierDetector",
    "SignalSmoother",
    "SignalQualityMetrics",
    "NoiseReductionPipeline",
    # Signal generation
    "ThermalFieldGenerator",
    "DensityFieldEstimator",
    "StressFieldGenerator",
]
