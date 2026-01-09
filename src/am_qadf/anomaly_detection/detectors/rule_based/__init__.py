"""
Rule-Based Anomaly Detection Methods

Rule-based methods use domain knowledge and predefined rules to detect anomalies.
These methods are particularly useful for process-specific anomaly detection.
"""

# Try relative import first, then absolute import (for importlib compatibility)
try:
    from .threshold_violations import ThresholdViolationDetector
    from .pattern_deviation import PatternDeviationDetector
    from .temporal_pattern import TemporalPatternDetector
    from .spatial_pattern import SpatialPatternDetector
    from .multi_signal_correlation import MultiSignalCorrelationDetector
except ImportError:
    # Try absolute import (for when loaded via importlib)
    import sys
    from pathlib import Path
    import importlib.util

    current_dir = Path(__file__).parent

    # Load each module
    modules_to_load = [
        ("threshold_violations", "ThresholdViolationDetector"),
        ("pattern_deviation", "PatternDeviationDetector"),
        ("temporal_pattern", "TemporalPatternDetector"),
        ("spatial_pattern", "SpatialPatternDetector"),
        ("multi_signal_correlation", "MultiSignalCorrelationDetector"),
    ]

    for module_name, class_name in modules_to_load:
        module_path = current_dir / f"{module_name}.py"
        if module_path.exists():
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            globals()[class_name] = getattr(module, class_name)

__all__ = [
    "ThresholdViolationDetector",
    "PatternDeviationDetector",
    "TemporalPatternDetector",
    "SpatialPatternDetector",
    "MultiSignalCorrelationDetector",
]
