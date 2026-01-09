"""
Data Generation Module for PBF-LB/M Process Chain

This module provides data generation capabilities for demo and testing purposes.
In production, data comes from real sensors and experiments.

All data generation is kept external to the core framework to maintain
clear separation between framework code and data generation logic.
"""

__version__ = "0.1.0"

# Import main generators (lazy imports to avoid heavy dependencies)
__all__ = [
    # Sensor generators
    'ISPMGenerator',
    'CTScanGenerator',
    'LaserParameterGenerator',
    # Process generators
    'STLProcessor',
    'HatchingGenerator',
    'BuildSimulator',
    # Scripts
    'generate_all_data',
    'generate_for_demo',
]

# Lazy imports - actual imports happen when modules are accessed
def __getattr__(name):
    """Lazy import of generators to avoid heavy dependencies at import time."""
    if name == 'ISPMGenerator':
        from .sensors.ispm_generator import ISPMGenerator
        return ISPMGenerator
    elif name == 'CTScanGenerator':
        from .sensors.ct_scan_generator import CTScanGenerator
        return CTScanGenerator
    elif name == 'LaserParameterGenerator':
        from .sensors.laser_parameter_generator import LaserParameterGenerator
        return LaserParameterGenerator
    elif name == 'STLProcessor':
        from .process.stl_processor import STLProcessor
        return STLProcessor
    elif name == 'HatchingGenerator':
        from .process.hatching_generator import HatchingGenerator
        return HatchingGenerator
    elif name == 'BuildSimulator':
        from .process.build_simulator import BuildSimulator
        return BuildSimulator
    elif name == 'generate_all_data':
        from .scripts.generate_all_data import generate_all_data
        return generate_all_data
    elif name == 'generate_for_demo':
        from .scripts.generate_for_demo import generate_for_demo
        return generate_for_demo
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

