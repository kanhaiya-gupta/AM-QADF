"""
Laser Parameter Generator

Generates realistic laser process parameters including:
- Laser power
- Scan speed
- Energy density
- Hatch spacing
- Exposure time
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class LaserParameterPoint:
    """Single laser parameter data point."""
    timestamp: datetime
    layer_index: int
    x: float
    y: float
    z: float
    laser_power: float  # W
    scan_speed: float  # mm/s
    hatch_spacing: float  # mm
    energy_density: float  # J/mm²
    exposure_time: float  # s
    region_type: str  # "contour", "hatch", "support"


@dataclass
class LaserParameterGeneratorConfig:
    """Configuration for laser parameter generation."""
    # Laser power ranges
    power_mean: float = 200.0  # W
    power_std: float = 10.0  # W
    power_min: float = 100.0  # W
    power_max: float = 400.0  # W
    
    # Scan speed ranges
    speed_mean: float = 500.0  # mm/s
    speed_std: float = 50.0  # mm/s
    speed_min: float = 100.0  # mm/s
    speed_max: float = 2000.0  # mm/s
    
    # Hatch spacing
    hatch_spacing_mean: float = 0.1  # mm
    hatch_spacing_std: float = 0.01  # mm
    
    # Region-specific parameters
    contour_power_multiplier: float = 1.2  # Contours use 20% more power
    contour_speed_multiplier: float = 0.8  # Contours use 20% less speed
    support_power_multiplier: float = 0.7  # Supports use 30% less power
    support_speed_multiplier: float = 1.5  # Supports use 50% more speed
    
    # Random seed
    random_seed: Optional[int] = None


class LaserParameterGenerator:
    """
    Generator for laser process parameters.
    
    Creates realistic laser parameters with spatial and temporal variations.
    """
    
    def __init__(self, config: Optional[LaserParameterGeneratorConfig] = None):
        """
        Initialize laser parameter generator.
        
        Args:
            config: Configuration for data generation
        """
        self.config = config or LaserParameterGeneratorConfig()
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
        
        logger.info(f"LaserParameterGenerator initialized with config: {self.config}")
    
    def generate_point(self,
                      timestamp: datetime,
                      layer_index: int,
                      x: float,
                      y: float,
                      z: float,
                      region_type: str = "hatch") -> LaserParameterPoint:
        """
        Generate laser parameters for a single point.
        
        Args:
            timestamp: Timestamp
            layer_index: Layer number
            x, y, z: Spatial coordinates (mm)
            region_type: Type of region ("contour", "hatch", "support")
            
        Returns:
            LaserParameterPoint
        """
        # Base power and speed
        power = np.random.normal(self.config.power_mean, self.config.power_std)
        speed = np.random.normal(self.config.speed_mean, self.config.speed_std)
        
        # Adjust for region type
        if region_type == "contour":
            power *= self.config.contour_power_multiplier
            speed *= self.config.contour_speed_multiplier
        elif region_type == "support":
            power *= self.config.support_power_multiplier
            speed *= self.config.support_speed_multiplier
        
        # Clip to valid ranges
        power = np.clip(power, self.config.power_min, self.config.power_max)
        speed = np.clip(speed, self.config.speed_min, self.config.speed_max)
        
        # Hatch spacing
        hatch_spacing = np.random.normal(
            self.config.hatch_spacing_mean,
            self.config.hatch_spacing_std
        )
        hatch_spacing = max(0.05, hatch_spacing)  # Minimum spacing
        
        # Calculate energy density: E = P / (v * h)
        # where P = power (W), v = speed (mm/s), h = hatch spacing (mm)
        # Energy density in J/mm²
        energy_density = power / (speed * hatch_spacing) if speed > 0 and hatch_spacing > 0 else 0.0
        
        # Exposure time (time to scan one hatch spacing)
        exposure_time = hatch_spacing / speed if speed > 0 else 0.0
        
        return LaserParameterPoint(
            timestamp=timestamp,
            layer_index=layer_index,
            x=x,
            y=y,
            z=z,
            laser_power=power,
            scan_speed=speed,
            hatch_spacing=hatch_spacing,
            energy_density=energy_density,
            exposure_time=exposure_time,
            region_type=region_type
        )
    
    def generate_for_layer(self,
                          layer_index: int,
                          layer_z: float,
                          n_points: int = 1000,
                          region_distribution: Optional[Dict[str, float]] = None,
                          bounding_box: Optional[Dict[str, Tuple[float, float]]] = None) -> List[LaserParameterPoint]:
        """
        Generate laser parameters for a layer.
        
        Args:
            layer_index: Layer number
            layer_z: Z position of layer (mm)
            n_points: Number of parameter points
            region_distribution: Distribution of region types {"contour": 0.1, "hatch": 0.8, "support": 0.1}
            bounding_box: Optional bounding box dict with 'min' and 'max' keys, each containing (x, y, z) tuples
            
        Returns:
            List of LaserParameterPoint objects
        """
        if region_distribution is None:
            region_distribution = {"contour": 0.1, "hatch": 0.8, "support": 0.1}
        
        # Determine X and Y ranges from bounding box or use default
        if bounding_box and 'min' in bounding_box and 'max' in bounding_box:
            bbox_min = bounding_box['min']
            bbox_max = bounding_box['max']
            x_min, y_min = bbox_min[0], bbox_min[1]
            x_max, y_max = bbox_max[0], bbox_max[1]
        else:
            # Default: 0-100 range (for backward compatibility)
            x_min, y_min = 0.0, 0.0
            x_max, y_max = 100.0, 100.0
        
        points = []
        start_time = datetime.now()
        
        for i in range(n_points):
            # Determine region type
            rand = np.random.random()
            cumulative = 0.0
            region_type = "hatch"  # default
            for rtype, prob in region_distribution.items():
                cumulative += prob
                if rand <= cumulative:
                    region_type = rtype
                    break
            
            # Random spatial position within bounding box
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)
            z = layer_z
            
            # Timestamp (distributed across layer build time)
            timestamp = start_time + timedelta(seconds=i * 0.1)
            
            point = self.generate_point(
                timestamp=timestamp,
                layer_index=layer_index,
                x=x,
                y=y,
                z=z,
                region_type=region_type
            )
            points.append(point)
        
        return points
    
    def generate_for_build(self,
                          build_id: str,
                          n_layers: int = 100,
                          layer_thickness: float = 0.05,
                          points_per_layer: int = 1000,
                          bounding_box: Optional[Dict[str, Tuple[float, float]]] = None) -> Dict[str, Any]:
        """
        Generate laser parameters for an entire build.
        
        Args:
            build_id: Build identifier
            n_layers: Number of layers
            layer_thickness: Thickness of each layer (mm)
            points_per_layer: Number of parameter points per layer
            bounding_box: Optional bounding box dict with 'min' and 'max' keys, each containing (x, y, z) tuples
            
        Returns:
            Dictionary containing build laser parameter data
        """
        all_points = []
        start_time = datetime.now()
        current_time = start_time
        
        for layer_idx in range(n_layers):
            layer_z = layer_idx * layer_thickness
            layer_points = self.generate_for_layer(
                layer_index=layer_idx,
                layer_z=layer_z,
                n_points=points_per_layer,
                bounding_box=bounding_box
            )
            
            # Update timestamps
            for point in layer_points:
                point.timestamp = current_time
                current_time += timedelta(seconds=0.1)
            
            all_points.extend(layer_points)
        
        # Calculate statistics
        powers = [p.laser_power for p in all_points]
        speeds = [p.scan_speed for p in all_points]
        energy_densities = [p.energy_density for p in all_points]
        
        return {
            'build_id': build_id,
            'n_layers': n_layers,
            'n_points': len(all_points),
            'start_time': start_time,
            'end_time': all_points[-1].timestamp if all_points else None,
            'points': all_points,
            'statistics': {
                'mean_power': float(np.mean(powers)),
                'mean_speed': float(np.mean(speeds)),
                'mean_energy_density': float(np.mean(energy_densities)),
                'std_power': float(np.std(powers)),
                'std_speed': float(np.std(speeds)),
                'std_energy_density': float(np.std(energy_densities))
            },
            'metadata': {
                'generator_config': self.config,
                'generated_at': datetime.now().isoformat()
            }
        }

