"""
Signal Generation - C++ Wrapper

Thin Python wrapper for C++ signal generation implementation.
All core computation is done in C++.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np

try:
    from am_qadf_native.processing import SignalGeneration
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False
    SignalGeneration = None


class ThermalFieldGenerator:
    """Thermal field generator - C++ wrapper."""
    
    def __init__(self):
        """Initialize thermal field generator."""
        if not CPP_AVAILABLE:
            raise ImportError("C++ bindings not available")
        self._generator = SignalGeneration()
    
    def generate(self, points: np.ndarray, **kwargs) -> np.ndarray:
        """
        Generate thermal field values for points.
        
        Args:
            points: Array of points (N, 3)
            **kwargs: Generation parameters
            
        Returns:
            Array of thermal values
        """
        points_cpp = np.asarray(points, dtype=np.float32).reshape(-1, 3).tolist()
        amplitude = kwargs.get("amplitude", 1.0)
        frequency = kwargs.get("frequency", 1.0)
        values = self._generator.generate_synthetic(points_cpp, "thermal", amplitude, frequency)
        return np.array(values, dtype=np.float32)


class DensityFieldEstimator:
    """Density field estimator - C++ wrapper."""
    
    def __init__(self):
        """Initialize density field estimator."""
        if not CPP_AVAILABLE:
            raise ImportError("C++ bindings not available")
        self._generator = SignalGeneration()
    
    def estimate(self, points: np.ndarray, **kwargs) -> np.ndarray:
        """
        Estimate density field values for points.
        
        Args:
            points: Array of points (N, 3)
            **kwargs: Estimation parameters
            
        Returns:
            Array of density values
        """
        points_cpp = np.asarray(points, dtype=np.float32).reshape(-1, 3).tolist()
        center = kwargs.get("center", (0.0, 0.0, 0.0))
        center_cpp = list(center)[:3]
        amplitude = kwargs.get("amplitude", 1.0)
        sigma = kwargs.get("sigma", 1.0)
        values = self._generator.generate_gaussian(points_cpp, center_cpp, amplitude, sigma)
        return np.array(values, dtype=np.float32)


class StressFieldGenerator:
    """Stress field generator - C++ wrapper."""
    
    def __init__(self):
        """Initialize stress field generator."""
        if not CPP_AVAILABLE:
            raise ImportError("C++ bindings not available")
        self._generator = SignalGeneration()
    
    def generate(self, points: np.ndarray, expression: Optional[str] = None, **kwargs) -> np.ndarray:
        """
        Generate stress field values for points.
        
        Args:
            points: Array of points (N, 3)
            expression: Optional expression string for generation
            **kwargs: Generation parameters
            
        Returns:
            Array of stress values
        """
        points_cpp = np.asarray(points, dtype=np.float32).reshape(-1, 3).tolist()
        if expression:
            values = self._generator.generate_from_expression(points_cpp, expression)
        else:
            amplitude = kwargs.get("amplitude", 1.0)
            frequency = kwargs.get("frequency", 1.0)
            values = self._generator.generate_synthetic(points_cpp, "stress", amplitude, frequency)
        return np.array(values, dtype=np.float32)
