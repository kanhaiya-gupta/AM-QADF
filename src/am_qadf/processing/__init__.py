"""
AM-QADF Processing Module

Signal processing and noise reduction.
Handles noise filtering, signal smoothing, and derived signal generation.
"""

from .noise_reduction import (
    OutlierDetector,
    SignalSmoother,
    SignalQualityMetrics,
    NoiseReductionPipeline,
)

from .signal_generation import (
    ThermalFieldGenerator,
    DensityFieldEstimator,
    StressFieldGenerator,
)

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
