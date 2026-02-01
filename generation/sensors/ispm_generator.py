"""
ISPM_Thermal (In-Situ Process Monitoring - Thermal) Data Generator

Generates realistic thermal ISPM sensor data including:
- Melt pool temperature and size
- Thermal gradients
- Cooling rates
- Process events
- Layer-by-layer thermal monitoring data

Note: ISPM is a broad category. This generator specifically handles ISPM_Thermal
(thermal monitoring). Other ISPM types include:
- ISPM_Optical: Photodiodes, cameras, melt pool imaging
- ISPM_Acoustic: Acoustic emissions, sound sensors
- ISPM_Strain: Strain gauges, deformation sensors
- ISPM_Plume: Vapor plume monitoring
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class ISPMThermalDataPoint:
    """Single ISPM_Thermal data point."""
    timestamp: datetime
    layer_index: int
    x: float
    y: float
    z: float
    melt_pool_temperature: float  # Celsius (MPTmean - mean temperature)
    melt_pool_size: Dict[str, float]  # width, length, depth in mm
    peak_temperature: float  # Celsius (MPTmax - maximum temperature)
    cooling_rate: float  # K/s
    temperature_gradient: float  # K/mm
    process_event: Optional[str] = None  # e.g., "layer_start", "hatch_complete"
    
    # Additional fields from research (melt pool geometry)
    melt_pool_area: Optional[float] = None  # mm² (MPA - melt pool area)
    melt_pool_eccentricity: Optional[float] = None  # ratio (MPE - MPW/MPL)
    melt_pool_perimeter: Optional[float] = None  # mm (MPP - melt pool perimeter)
    
    # Time over threshold metrics (cooling behavior)
    time_over_threshold_1200K: Optional[float] = None  # ms (TOT1200K - above camera sensitivity)
    time_over_threshold_1680K: Optional[float] = None  # ms (TOT1680K - above solidification ~1660K)
    time_over_threshold_2400K: Optional[float] = None  # ms (TOT2400K - above upper threshold)


@dataclass
class ISPMThermalGeneratorConfig:
    """Configuration for ISPM_Thermal data generation."""
    # Temperature ranges
    base_temperature: float = 1500.0  # Base melt pool temperature (Celsius)
    temperature_variation: float = 50.0  # Temperature variation (Celsius)
    
    # Melt pool size ranges
    melt_pool_width_mean: float = 0.5  # mm
    melt_pool_width_std: float = 0.1  # mm
    melt_pool_length_mean: float = 0.8  # mm
    melt_pool_length_std: float = 0.15  # mm
    melt_pool_depth_mean: float = 0.3  # mm
    melt_pool_depth_std: float = 0.05  # mm
    
    # Thermal parameters
    cooling_rate_mean: float = 100.0  # K/s
    cooling_rate_std: float = 20.0  # K/s
    temperature_gradient_mean: float = 50.0  # K/mm
    temperature_gradient_std: float = 10.0  # K/mm
    
    # Sampling
    sampling_rate: float = 1000.0  # Hz
    points_per_layer: int = 1000  # Number of data points per layer
    
    # Random seed
    random_seed: Optional[int] = None


class ISPMThermalGenerator:
    """
    Generator for ISPM_Thermal (In-Situ Process Monitoring - Thermal) sensor data.
    
    Creates realistic thermal melt pool monitoring data with temporal and spatial variations.
    This generator specifically handles ISPM_Thermal (thermal monitoring).
    Other ISPM types include: ISPM_Optical, ISPM_Acoustic, ISPM_Strain, ISPM_Plume, etc.
    """
    
    def __init__(self, config: Optional[ISPMThermalGeneratorConfig] = None):
        """
        Initialize ISPM_Thermal generator.
        
        Args:
            config: Configuration for data generation
        """
        self.config = config or ISPMThermalGeneratorConfig()
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
        
        logger.info(f"ISPMThermalGenerator initialized with config: {self.config}")
    
    def generate_for_layer(self,
                             layer_index: int,
                             layer_z: float,
                             n_points: Optional[int] = None,
                             start_time: Optional[datetime] = None,
                             bounding_box: Optional[Dict[str, Tuple[float, float]]] = None) -> List[ISPMThermalDataPoint]:
        """
        Generate ISPM data for a single layer.
        
        Args:
            layer_index: Layer number
            layer_z: Z position of the layer
            n_points: Number of data points (defaults to config.points_per_layer)
            start_time: Start timestamp (defaults to now)
            bounding_box: Optional bounding box dict with 'min' and 'max' keys, each containing (x, y, z) tuples
            
        Returns:
            List of ISPM data points
        """
        if n_points is None:
            n_points = self.config.points_per_layer
        if start_time is None:
            start_time = datetime.now()
        
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
        
        data_points = []
        dt = 1.0 / self.config.sampling_rate  # Time step in seconds
        
        for i in range(n_points):
            timestamp = start_time + timedelta(seconds=i * dt)
            
            # Generate spatial position (random within bounding box)
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)
            z = layer_z
            
            # Generate melt pool temperature with variation
            temperature = np.random.normal(
                self.config.base_temperature,
                self.config.temperature_variation
            )
            
            # Generate melt pool size
            melt_pool_size = {
                'width': max(0.1, np.random.normal(
                    self.config.melt_pool_width_mean,
                    self.config.melt_pool_width_std
                )),
                'length': max(0.1, np.random.normal(
                    self.config.melt_pool_length_mean,
                    self.config.melt_pool_length_std
                )),
                'depth': max(0.05, np.random.normal(
                    self.config.melt_pool_depth_mean,
                    self.config.melt_pool_depth_std
                ))
            }
            
            # Generate thermal parameters
            peak_temperature = temperature + np.random.uniform(50, 200)  # Peak above melt pool
            cooling_rate = max(10.0, np.random.normal(
                self.config.cooling_rate_mean,
                self.config.cooling_rate_std
            ))
            temperature_gradient = max(10.0, np.random.normal(
                self.config.temperature_gradient_mean,
                self.config.temperature_gradient_std
            ))
            
            # Calculate additional geometric fields (from research article)
            # Melt pool area (MPA): approximate as ellipse area = π * (width/2) * (length/2)
            melt_pool_area = np.pi * (melt_pool_size['width'] / 2.0) * (melt_pool_size['length'] / 2.0)
            
            # Melt pool eccentricity (MPE): ratio of width to length (minor/major axis)
            # Eccentricity of ellipse: e = sqrt(1 - (b²/a²)) where a=length/2, b=width/2
            # But research uses simple ratio MPW/MPL
            melt_pool_eccentricity = melt_pool_size['width'] / melt_pool_size['length'] if melt_pool_size['length'] > 0 else 0.0
            
            # Melt pool perimeter (MPP): approximate as ellipse perimeter
            # Ramanujan's approximation: P ≈ π * [3(a+b) - sqrt((3a+b)(a+3b))]
            a = melt_pool_size['length'] / 2.0
            b = melt_pool_size['width'] / 2.0
            if a > 0 and b > 0:
                melt_pool_perimeter = np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))
            else:
                melt_pool_perimeter = 0.0
            
            # Time over threshold metrics (TOT) - cooling behavior
            # These represent time periods where temperature exceeds threshold
            # Simulate based on cooling rate and temperature
            temp_kelvin = temperature + 273.15  # Convert to Kelvin
            
            # TOT1200K: Time above camera sensitivity threshold
            if temp_kelvin > 1200.0:
                # Time to cool from current temp to 1200K: t = (T - 1200) / cooling_rate
                time_over_threshold_1200K = max(0.0, (temp_kelvin - 1200.0) / cooling_rate * 1000.0)  # Convert to ms
            else:
                time_over_threshold_1200K = 0.0
            
            # TOT1680K: Time above solidification temperature (~1660K)
            if temp_kelvin > 1680.0:
                time_over_threshold_1680K = max(0.0, (temp_kelvin - 1680.0) / cooling_rate * 1000.0)  # Convert to ms
            else:
                time_over_threshold_1680K = 0.0
            
            # TOT2400K: Time above upper threshold
            if temp_kelvin > 2400.0:
                time_over_threshold_2400K = max(0.0, (temp_kelvin - 2400.0) / cooling_rate * 1000.0)  # Convert to ms
            else:
                time_over_threshold_2400K = 0.0
            
            # Process events (occasional)
            process_event = None
            if np.random.random() < 0.01:  # 1% chance
                events = ["layer_start", "hatch_complete", "contour_complete", "layer_complete"]
                process_event = np.random.choice(events)
            
            data_point = ISPMThermalDataPoint(
                timestamp=timestamp,
                layer_index=layer_index,
                x=x,
                y=y,
                z=z,
                melt_pool_temperature=temperature,
                melt_pool_size=melt_pool_size,
                peak_temperature=peak_temperature,
                cooling_rate=cooling_rate,
                temperature_gradient=temperature_gradient,
                process_event=process_event,
                melt_pool_area=melt_pool_area,
                melt_pool_eccentricity=melt_pool_eccentricity,
                melt_pool_perimeter=melt_pool_perimeter,
                time_over_threshold_1200K=time_over_threshold_1200K,
                time_over_threshold_1680K=time_over_threshold_1680K,
                time_over_threshold_2400K=time_over_threshold_2400K
            )
            data_points.append(data_point)
        
        return data_points
    
    def generate_for_build(self,
                          build_id: str,
                          n_layers: int = 100,
                          layer_thickness: float = 0.05,  # mm
                          start_time: Optional[datetime] = None,
                          bounding_box: Optional[Dict[str, Tuple[float, float]]] = None) -> Dict[str, Any]:
        """
        Generate ISPM data for an entire build.
        
        Args:
            build_id: Build identifier
            n_layers: Number of layers
            layer_thickness: Thickness of each layer (mm)
            start_time: Build start time (defaults to now)
            bounding_box: Optional bounding box {'x': (min, max), 'y': (min, max), 'z': (min, max)}
            
        Returns:
            Dictionary containing build ISPM data
        """
        if start_time is None:
            start_time = datetime.now()
        
        all_data_points = []
        current_time = start_time
        
        # Determine spatial range from bounding box or use defaults
        if bounding_box:
            x_range = (bounding_box['x'][0], bounding_box['x'][1])
            y_range = (bounding_box['y'][0], bounding_box['y'][1])
            z_range = (bounding_box['z'][0], bounding_box['z'][1])
        else:
            x_range = (0.0, 100.0)
            y_range = (0.0, 100.0)
            z_range = (0.0, n_layers * layer_thickness)
        
        # Convert bounding_box format if needed (from {'x': (min, max), ...} to {'min': (x, y, z), 'max': (x, y, z)})
        bbox_for_layer = None
        if bounding_box:
            if 'x' in bounding_box:
                # Convert from {'x': (min, max), 'y': (min, max), 'z': (min, max)} format
                bbox_for_layer = {
                    'min': (bounding_box['x'][0], bounding_box['y'][0], bounding_box['z'][0]),
                    'max': (bounding_box['x'][1], bounding_box['y'][1], bounding_box['z'][1])
                }
            elif 'min' in bounding_box and 'max' in bounding_box:
                # Already in correct format
                bbox_for_layer = bounding_box
        
        for layer_idx in range(n_layers):
            layer_z = layer_idx * layer_thickness
            layer_data = self.generate_for_layer(
                layer_index=layer_idx,
                layer_z=layer_z,
                start_time=current_time,
                bounding_box=bbox_for_layer
            )
            
            all_data_points.extend(layer_data)
            
            # Update time for next layer
            if layer_data:
                last_timestamp = layer_data[-1].timestamp
                current_time = last_timestamp + timedelta(seconds=1)  # 1s gap between layers
        
        # Calculate bounding box from actual data points
        if all_data_points:
            x_coords = [p.x for p in all_data_points]
            y_coords = [p.y for p in all_data_points]
            z_coords = [p.z for p in all_data_points]
            bbox_min = (min(x_coords), min(y_coords), min(z_coords))
            bbox_max = (max(x_coords), max(y_coords), max(z_coords))
        else:
            bbox_min = (0.0, 0.0, 0.0)
            bbox_max = (100.0, 100.0, n_layers * layer_thickness)
        
        # Create coordinate system information (critical for merging with STL/hatching/CT data)
        # ISPM sensors are typically positioned on the machine and measure in build platform coordinates
        coordinate_system = {
            'type': 'ispm_sensor',  # ISPM sensor coordinate system
            'origin': {
                'x': 0.0,  # Typically aligned with build platform origin
                'y': 0.0,
                'z': 0.0
            },
            'rotation': {
                'x_deg': 0.0,  # Typically aligned with build platform (no rotation)
                'y_deg': 0.0,
                'z_deg': 0.0
            },
            'scale_factor': {
                'x': 1.0,  # No scaling
                'y': 1.0,
                'z': 1.0
            },
            'sensor_position': {
                'description': 'ISPM sensors are typically positioned at fixed locations on the machine',
                'coordinate_space': 'build_platform'  # Same as build platform coordinates
            },
            'measurement_coordinates': {
                'description': 'Measurement coordinates (x, y, z) are in build platform space',
                'spatial_range': {
                    'x': [float(bbox_min[0]), float(bbox_max[0])],
                    'y': [float(bbox_min[1]), float(bbox_max[1])],
                    'z': [float(bbox_min[2]), float(bbox_max[2])]
                }
            },
            'bounding_box': {
                'min': [float(bbox_min[0]), float(bbox_min[1]), float(bbox_min[2])],
                'max': [float(bbox_max[0]), float(bbox_max[1]), float(bbox_max[2])]
            },
            'description': 'ISPM sensor coordinate system. Measurement coordinates are in build platform space. Sensors are typically positioned at fixed locations on the machine and monitor the process at specific spatial locations.'
        }
        
        return {
            'build_id': build_id,
            'n_layers': n_layers,
            'n_data_points': len(all_data_points),
            'start_time': start_time,
            'end_time': all_data_points[-1].timestamp if all_data_points else None,
            'data_points': all_data_points,
            'coordinate_system': coordinate_system,  # Critical for merging with STL/hatching/CT data
            'metadata': {
                'generator_config': self.config,
                'generated_at': datetime.now().isoformat()
            }
        }
    
    def generate_anomalous_data_point(self,
                                     base_data_point: ISPMDataPoint,
                                     anomaly_type: str = "temperature_spike") -> ISPMDataPoint:
        """
        Generate an anomalous data point based on a normal one.
        
        Args:
            base_data_point: Base normal data point
            anomaly_type: Type of anomaly ("temperature_spike", "temperature_drop", "large_melt_pool")
            
        Returns:
            Anomalous data point
        """
        if anomaly_type == "temperature_spike":
            temperature = base_data_point.melt_pool_temperature + np.random.uniform(200, 500)
            peak_temperature = temperature + np.random.uniform(100, 300)
        elif anomaly_type == "temperature_drop":
            temperature = max(800, base_data_point.melt_pool_temperature - np.random.uniform(300, 600))
            peak_temperature = temperature + np.random.uniform(50, 150)
        elif anomaly_type == "large_melt_pool":
            temperature = base_data_point.melt_pool_temperature
            peak_temperature = base_data_point.peak_temperature
            melt_pool_size = {
                'width': base_data_point.melt_pool_size['width'] * np.random.uniform(2, 4),
                'length': base_data_point.melt_pool_size['length'] * np.random.uniform(2, 4),
                'depth': base_data_point.melt_pool_size['depth'] * np.random.uniform(1.5, 3)
            }
        else:
            # Default: slight variation
            temperature = base_data_point.melt_pool_temperature
            peak_temperature = base_data_point.peak_temperature
            melt_pool_size = base_data_point.melt_pool_size.copy()
        
        if anomaly_type != "large_melt_pool":
            melt_pool_size = base_data_point.melt_pool_size.copy()
        
        # Recalculate geometric fields for anomalous data point
        melt_pool_area = np.pi * (melt_pool_size['width'] / 2.0) * (melt_pool_size['length'] / 2.0)
        melt_pool_eccentricity = melt_pool_size['width'] / melt_pool_size['length'] if melt_pool_size['length'] > 0 else 0.0
        a = melt_pool_size['length'] / 2.0
        b = melt_pool_size['width'] / 2.0
        if a > 0 and b > 0:
            melt_pool_perimeter = np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))
        else:
            melt_pool_perimeter = 0.0
        
        # Recalculate TOT metrics
        temp_kelvin = temperature + 273.15
        time_over_threshold_1200K = max(0.0, (temp_kelvin - 1200.0) / base_data_point.cooling_rate * 1000.0) if temp_kelvin > 1200.0 else 0.0
        time_over_threshold_1680K = max(0.0, (temp_kelvin - 1680.0) / base_data_point.cooling_rate * 1000.0) if temp_kelvin > 1680.0 else 0.0
        time_over_threshold_2400K = max(0.0, (temp_kelvin - 2400.0) / base_data_point.cooling_rate * 1000.0) if temp_kelvin > 2400.0 else 0.0
        
        return ISPMThermalDataPoint(
            timestamp=base_data_point.timestamp,
            layer_index=base_data_point.layer_index,
            x=base_data_point.x,
            y=base_data_point.y,
            z=base_data_point.z,
            melt_pool_temperature=temperature,
            melt_pool_size=melt_pool_size,
            peak_temperature=peak_temperature,
            cooling_rate=base_data_point.cooling_rate,
            temperature_gradient=base_data_point.temperature_gradient,
            process_event=base_data_point.process_event,
            melt_pool_area=melt_pool_area,
            melt_pool_eccentricity=melt_pool_eccentricity,
            melt_pool_perimeter=melt_pool_perimeter,
            time_over_threshold_1200K=time_over_threshold_1200K,
            time_over_threshold_1680K=time_over_threshold_1680K,
            time_over_threshold_2400K=time_over_threshold_2400K
        )

