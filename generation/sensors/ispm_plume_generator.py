"""
ISPM_Plume (In-Situ Process Monitoring - Plume) Data Generator

Generates realistic vapor plume ISPM sensor data for PBF-LB/M processes including:
- Vapor plume characteristics (density, temperature, velocity, composition)
- Plume geometry (height, width, angle, spread)
- Process quality indicators (stability, intensity variations)
- Anomaly detection (excessive plume, unstable plume, contamination)
- Temporal and spatial plume dynamics

Note: ISPM is a broad category. This generator specifically handles ISPM_Plume
(vapor plume monitoring - monitoring the vapor plume above the melt pool).
Other ISPM types include:
- ISPM_Thermal: Thermal monitoring
- ISPM_Optical: Photodiodes, cameras, melt pool imaging
- ISPM_Acoustic: Acoustic emissions, sound sensors
- ISPM_Strain: Strain gauges, deformation sensors
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class ISPMPlumeDataPoint:
    """Single ISPM_Plume data point."""
    timestamp: datetime
    layer_index: int
    x: float
    y: float
    z: float
    
    # Plume characteristics (primary vapor plume monitoring in PBF-LB/M)
    plume_intensity: float  # Arbitrary units (0-10000 typical range) - overall plume intensity
    plume_density: float  # kg/m³ (vapor density in plume)
    plume_temperature: float  # Celsius (temperature of vapor plume)
    plume_velocity: float  # m/s (vertical velocity of plume)
    plume_velocity_x: float  # m/s (horizontal velocity in x-direction)
    plume_velocity_y: float  # m/s (horizontal velocity in y-direction)
    
    # Plume geometry
    plume_height: float  # mm (height of plume above melt pool)
    plume_width: float  # mm (width/diameter of plume at base)
    plume_angle: float  # degrees (angle of plume from vertical)
    plume_spread: float  # mm (spread/divergence of plume)
    plume_area: float  # mm² (cross-sectional area of plume)
    
    # Plume composition
    particle_concentration: float  # particles/m³ (particle concentration in plume)
    metal_vapor_concentration: float  # kg/m³ (metal vapor concentration)
    gas_composition_ratio: float  # ratio (ratio of metal vapor to inert gas)
    
    # Plume dynamics
    plume_fluctuation_rate: float  # Hz (rate of plume fluctuations)
    plume_instability_index: float  # 0-1 scale (instability of plume, 1 = very unstable)
    plume_turbulence: float  # 0-1 scale (turbulence level in plume)
    
    # Process quality indicators (required fields)
    process_stability: float  # 0-1 scale (1 = stable, 0 = unstable)
    plume_stability: float  # 0-1 scale (1 = stable plume, 0 = unstable)
    intensity_variation: float  # Coefficient of variation (std/mean)
    
    # Plume energy metrics (required fields)
    plume_energy: float  # Total plume energy (arbitrary units)
    energy_density: float  # Energy per unit volume (J/m³)
    
    # Signal-to-noise ratio (required field)
    signal_to_noise_ratio: float  # SNR in dB
    
    # Optional fields (with defaults - must come after required fields)
    # Event detection (critical for PBF-LB/M quality)
    excessive_plume_event: bool = False  # Plume intensity exceeds threshold
    unstable_plume_event: bool = False  # Significant plume instability detected
    contamination_event: bool = False  # Contamination detected in plume
    anomaly_detected: bool = False
    anomaly_type: Optional[str] = None  # e.g., "excessive_plume", "unstable_plume", "contamination", "process_instability"
    
    # Process events
    process_event: Optional[str] = None  # e.g., "layer_start", "excessive_plume_event", "unstable_plume_event", "contamination_event"
    
    # Additional plume features
    plume_momentum: Optional[float] = None  # kg·m/s (plume momentum)
    plume_pressure: Optional[float] = None  # Pa (pressure in plume region, -1 if not available)


@dataclass
class ISPMPlumeGeneratorConfig:
    """Configuration for ISPM_Plume data generation."""
    # Plume intensity ranges
    plume_intensity_mean: float = 5000.0  # Typical intensity value
    plume_intensity_std: float = 800.0  # Variation
    
    # Plume density
    plume_density_mean: float = 0.01  # kg/m³ (vapor density is typically low)
    plume_density_std: float = 0.002  # kg/m³
    
    # Plume temperature
    plume_temperature_mean: float = 2500.0  # Celsius (high temperature vapor)
    plume_temperature_std: float = 200.0  # Celsius
    
    # Plume velocity
    plume_velocity_mean: float = 5.0  # m/s (typical vertical velocity)
    plume_velocity_std: float = 1.5  # m/s
    
    # Plume geometry
    plume_height_mean: float = 2.0  # mm (height above melt pool)
    plume_height_std: float = 0.5  # mm
    plume_width_mean: float = 1.5  # mm (width at base)
    plume_width_std: float = 0.3  # mm
    plume_angle_mean: float = 15.0  # degrees (slight angle from vertical)
    plume_angle_std: float = 5.0  # degrees
    
    # Plume composition
    particle_concentration_mean: float = 1e12  # particles/m³
    particle_concentration_std: float = 2e11  # particles/m³
    metal_vapor_concentration_mean: float = 0.008  # kg/m³
    metal_vapor_concentration_std: float = 0.002  # kg/m³
    
    # Process stability
    process_stability_mean: float = 0.80  # Typically 0.7-0.9 for good processes
    process_stability_std: float = 0.12
    plume_stability_mean: float = 0.75  # Plume stability typically slightly lower
    plume_stability_std: float = 0.15
    
    # Event probabilities (per data point)
    excessive_plume_probability: float = 0.02  # 2% chance of excessive plume per point
    unstable_plume_probability: float = 0.015  # 1.5% chance of unstable plume per point
    contamination_probability: float = 0.005  # 0.5% chance of contamination per point
    anomaly_probability: float = 0.01  # 1% chance of anomaly per point
    
    # Signal-to-noise ratio
    snr_mean: float = 30.0  # dB (plume sensors typically good SNR)
    snr_std: float = 5.0  # dB
    
    # Sampling
    sampling_rate: float = 1000.0  # Hz (plume sensors typically ~1kHz)
    points_per_layer: int = 1000  # Number of data points per layer
    
    # Random seed
    random_seed: Optional[int] = None


class ISPMPlumeGenerator:
    """
    Generator for ISPM_Plume (In-Situ Process Monitoring - Plume) sensor data.
    
    Creates realistic vapor plume monitoring data with temporal and spatial variations.
    This generator specifically handles ISPM_Plume (vapor plume monitoring).
    Other ISPM types include: ISPM_Thermal, ISPM_Optical, ISPM_Acoustic, ISPM_Strain, etc.
    """
    
    def __init__(self, config: Optional[ISPMPlumeGeneratorConfig] = None):
        """
        Initialize ISPM_Plume generator.
        
        Args:
            config: Configuration for data generation
        """
        self.config = config or ISPMPlumeGeneratorConfig()
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
        
        logger.info(f"ISPMPlumeGenerator initialized with config: {self.config}")
    
    def generate_for_layer(self,
                             layer_index: int,
                             layer_z: float,
                             n_points: Optional[int] = None,
                             start_time: Optional[datetime] = None,
                             bounding_box: Optional[Dict[str, Tuple[float, float]]] = None) -> List[ISPMPlumeDataPoint]:
        """
        Generate ISPM_Plume data for a single layer.
        
        Args:
            layer_index: Layer index (0-based)
            layer_z: Z-coordinate of the layer (mm)
            n_points: Number of data points to generate (default: config.points_per_layer)
            start_time: Start time for the layer (default: current time)
            bounding_box: Bounding box for spatial distribution {'x': (min, max), 'y': (min, max)}
        
        Returns:
            List of ISPM_Plume data points
        """
        n_points = n_points or self.config.points_per_layer
        start_time = start_time or datetime.now()
        
        # Default bounding box if not provided
        if bounding_box is None:
            bounding_box = {
                'x': (-50.0, 50.0),
                'y': (-50.0, 50.0)
            }
        
        data_points = []
        time_step = timedelta(seconds=1.0 / self.config.sampling_rate)
        
        for i in range(n_points):
            current_time = start_time + i * time_step
            
            # Spatial distribution (uniform within bounding box)
            x = np.random.uniform(bounding_box['x'][0], bounding_box['x'][1])
            y = np.random.uniform(bounding_box['y'][0], bounding_box['y'][1])
            z = layer_z
            
            # Generate plume characteristics
            plume_intensity = np.random.normal(self.config.plume_intensity_mean, self.config.plume_intensity_std)
            plume_intensity = max(0.0, plume_intensity)  # Ensure non-negative
            
            plume_density = np.random.normal(self.config.plume_density_mean, self.config.plume_density_std)
            plume_density = max(0.0, plume_density)
            
            plume_temperature = np.random.normal(self.config.plume_temperature_mean, self.config.plume_temperature_std)
            plume_temperature = max(1000.0, plume_temperature)  # Minimum reasonable temperature
            
            # Plume velocity (vertical and horizontal components)
            plume_velocity = np.random.normal(self.config.plume_velocity_mean, self.config.plume_velocity_std)
            plume_velocity = max(0.0, plume_velocity)
            plume_velocity_x = np.random.normal(0.5, 0.3)  # Small horizontal component
            plume_velocity_y = np.random.normal(0.5, 0.3)
            
            # Plume geometry
            plume_height = np.random.normal(self.config.plume_height_mean, self.config.plume_height_std)
            plume_height = max(0.5, plume_height)  # Minimum height
            
            plume_width = np.random.normal(self.config.plume_width_mean, self.config.plume_width_std)
            plume_width = max(0.3, plume_width)
            
            plume_angle = np.random.normal(self.config.plume_angle_mean, self.config.plume_angle_std)
            plume_angle = np.clip(plume_angle, 0.0, 45.0)  # Reasonable angle range
            
            plume_spread = plume_width * (1.0 + np.random.normal(0.2, 0.1))  # Spread increases with height
            plume_area = np.pi * (plume_width / 2.0) ** 2  # Circular cross-section
            
            # Plume composition
            particle_concentration = np.random.normal(self.config.particle_concentration_mean, self.config.particle_concentration_std)
            particle_concentration = max(0.0, particle_concentration)
            
            metal_vapor_concentration = np.random.normal(self.config.metal_vapor_concentration_mean, self.config.metal_vapor_concentration_std)
            metal_vapor_concentration = max(0.0, metal_vapor_concentration)
            
            gas_composition_ratio = metal_vapor_concentration / (plume_density + 1e-6)  # Ratio of metal vapor to total density
            
            # Plume dynamics
            plume_fluctuation_rate = np.random.normal(10.0, 3.0)  # Hz (typical fluctuation rate)
            plume_fluctuation_rate = max(0.0, plume_fluctuation_rate)
            
            plume_instability_index = np.random.normal(0.2, 0.1)  # Typically low instability
            plume_instability_index = np.clip(plume_instability_index, 0.0, 1.0)
            
            plume_turbulence = np.random.normal(0.3, 0.15)
            plume_turbulence = np.clip(plume_turbulence, 0.0, 1.0)
            
            # Process stability
            process_stability = np.random.normal(self.config.process_stability_mean, self.config.process_stability_std)
            process_stability = np.clip(process_stability, 0.0, 1.0)
            
            plume_stability = np.random.normal(self.config.plume_stability_mean, self.config.plume_stability_std)
            plume_stability = np.clip(plume_stability, 0.0, 1.0)
            
            # Intensity variation
            intensity_variation = abs(np.random.normal(0.15, 0.05))  # 15% typical variation
            
            # Signal-to-noise ratio
            snr = np.random.normal(self.config.snr_mean, self.config.snr_std)
            snr = max(10.0, snr)  # Minimum reasonable SNR
            
            # Plume energy
            plume_energy = plume_intensity * plume_area * plume_height  # Proportional to intensity, area, height
            energy_density = plume_energy / (plume_area * plume_height + 1e-6)  # Energy per unit volume
            
            # Event detection
            excessive_plume_event = False
            unstable_plume_event = False
            contamination_event = False
            anomaly_detected = False
            anomaly_type = None
            
            # Random events
            if np.random.random() < self.config.excessive_plume_probability:
                excessive_plume_event = True
                plume_intensity = self.config.plume_intensity_mean * (1.5 + np.random.uniform(0.2, 0.8))
                plume_density = self.config.plume_density_mean * (1.3 + np.random.uniform(0.1, 0.5))
            
            if np.random.random() < self.config.unstable_plume_probability:
                unstable_plume_event = True
                plume_instability_index = np.random.uniform(0.6, 1.0)  # High instability
                plume_stability = np.random.uniform(0.2, 0.5)  # Low stability
                plume_fluctuation_rate = np.random.uniform(20.0, 50.0)  # High fluctuation rate
            
            if np.random.random() < self.config.contamination_probability:
                contamination_event = True
                particle_concentration = self.config.particle_concentration_mean * (2.0 + np.random.uniform(0.5, 1.5))
                gas_composition_ratio = np.random.uniform(0.5, 0.8)  # Altered composition
            
            if np.random.random() < self.config.anomaly_probability:
                anomaly_detected = True
                anomaly_types = ["excessive_plume", "unstable_plume", "contamination", "process_instability"]
                anomaly_type = np.random.choice(anomaly_types)
            
            # Process event
            process_event = None
            if i == 0:
                process_event = "layer_start"
            elif excessive_plume_event:
                process_event = "excessive_plume_event"
            elif unstable_plume_event:
                process_event = "unstable_plume_event"
            elif contamination_event:
                process_event = "contamination_event"
            elif anomaly_detected:
                process_event = "anomaly_event"
            
            # Plume momentum (optional)
            plume_momentum = plume_density * plume_velocity * plume_area  # kg·m/s
            plume_pressure = None  # Not typically measured directly
            
            data_point = ISPMPlumeDataPoint(
                timestamp=current_time,
                layer_index=layer_index,
                x=x,
                y=y,
                z=z,
                plume_intensity=plume_intensity,
                plume_density=plume_density,
                plume_temperature=plume_temperature,
                plume_velocity=plume_velocity,
                plume_velocity_x=plume_velocity_x,
                plume_velocity_y=plume_velocity_y,
                plume_height=plume_height,
                plume_width=plume_width,
                plume_angle=plume_angle,
                plume_spread=plume_spread,
                plume_area=plume_area,
                particle_concentration=particle_concentration,
                metal_vapor_concentration=metal_vapor_concentration,
                gas_composition_ratio=gas_composition_ratio,
                plume_fluctuation_rate=plume_fluctuation_rate,
                plume_instability_index=plume_instability_index,
                plume_turbulence=plume_turbulence,
                process_stability=process_stability,
                plume_stability=plume_stability,
                intensity_variation=intensity_variation,
                excessive_plume_event=excessive_plume_event,
                unstable_plume_event=unstable_plume_event,
                contamination_event=contamination_event,
                anomaly_detected=anomaly_detected,
                anomaly_type=anomaly_type,
                plume_energy=plume_energy,
                energy_density=energy_density,
                process_event=process_event,
                signal_to_noise_ratio=snr,
                plume_momentum=plume_momentum,
                plume_pressure=plume_pressure,
            )
            
            data_points.append(data_point)
        
        logger.info(f"Generated {len(data_points)} ISPM_Plume data points for layer {layer_index}")
        return data_points
    
    def generate_for_build(self,
                             build_id: str,
                             n_layers: int,
                             layer_thickness: float = 0.05,
                             start_time: Optional[datetime] = None,
                             bounding_box: Optional[Dict[str, Tuple[float, float]]] = None) -> Dict[str, Any]:
        """
        Generate ISPM_Plume data for an entire build.
        
        Args:
            build_id: Build identifier
            n_layers: Number of layers in the build
            layer_thickness: Thickness of each layer (mm)
            start_time: Start time for the build (default: current time)
            bounding_box: Bounding box for spatial distribution {'x': (min, max), 'y': (min, max), 'z': (min, max)}
        
        Returns:
            Dictionary containing build ISPM_Plume data with 'data_points' and 'coordinate_system'
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
                    'x': (bounding_box['x'][0], bounding_box['x'][1]),
                    'y': (bounding_box['y'][0], bounding_box['y'][1])
                }
            elif 'min' in bounding_box and 'max' in bounding_box:
                # Already in correct format, convert to layer format
                bbox_for_layer = {
                    'x': (bounding_box['min'][0], bounding_box['max'][0]),
                    'y': (bounding_box['min'][1], bounding_box['max'][1])
                }
        
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
            'axes': {
                'x': {'direction': [1, 0, 0], 'unit': 'mm'},
                'y': {'direction': [0, 1, 0], 'unit': 'mm'},
                'z': {'direction': [0, 0, 1], 'unit': 'mm'}
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
                'description': 'ISPM plume sensors are typically positioned above the build chamber to monitor vapor plume',
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
            'description': 'ISPM plume sensor coordinate system. Measurement coordinates are in build platform space. Sensors are typically positioned above the build chamber and monitor the vapor plume at specific spatial locations.'
        }
        
        logger.info(f"Generated {len(all_data_points)} total ISPM_Plume data points for {n_layers} layers")
        
        return {
            'build_id': build_id,
            'data_points': all_data_points,
            'coordinate_system': coordinate_system,
            'n_layers': n_layers,
            'n_points': len(all_data_points)
        }
    
    def generate_anomalous_data_point(self,
                                        layer_index: int,
                                        layer_z: float,
                                        timestamp: datetime,
                                        x: float,
                                        y: float,
                                        anomaly_type: str = "excessive_plume") -> ISPMPlumeDataPoint:
        """
        Generate an anomalous ISPM_Plume data point (for testing anomaly detection).
        
        Args:
            layer_index: Layer index
            layer_z: Z-coordinate
            timestamp: Timestamp
            x: X-coordinate
            y: Y-coordinate
            anomaly_type: Type of anomaly ("excessive_plume", "unstable_plume", "contamination", "process_instability")
        
        Returns:
            Anomalous ISPM_Plume data point
        """
        # Base plume values (high)
        plume_intensity = np.random.normal(8000.0, 1000.0)
        plume_density = np.random.normal(0.015, 0.003)
        plume_temperature = np.random.normal(2800.0, 300.0)
        plume_velocity = np.random.normal(7.0, 2.0)
        
        # Anomaly-specific modifications
        if anomaly_type == "excessive_plume":
            plume_intensity = 12000.0  # Very high intensity
            plume_density = 0.025  # Very high density
            plume_temperature = 3000.0  # Very high temperature
            excessive_plume_event = True
            unstable_plume_event = False
            contamination_event = False
        elif anomaly_type == "unstable_plume":
            plume_intensity = np.random.normal(6000.0, 2000.0)  # High variation
            plume_instability_index = 0.9  # Very unstable
            plume_stability = 0.2  # Low stability
            plume_fluctuation_rate = 50.0  # High fluctuation
            excessive_plume_event = False
            unstable_plume_event = True
            contamination_event = False
        elif anomaly_type == "contamination":
            particle_concentration = 3e12  # Very high particle concentration
            gas_composition_ratio = 0.6  # Altered composition
            excessive_plume_event = False
            unstable_plume_event = False
            contamination_event = True
        else:  # "process_instability"
            plume_intensity = np.random.normal(4000.0, 1500.0)
            process_stability = 0.3  # Low process stability
            plume_stability = 0.4  # Low plume stability
            excessive_plume_event = False
            unstable_plume_event = False
            contamination_event = False
        
        # Other fields
        plume_velocity_x = np.random.normal(1.0, 0.5)
        plume_velocity_y = np.random.normal(1.0, 0.5)
        plume_height = np.random.normal(3.0, 0.8)  # Higher plume
        plume_width = np.random.normal(2.0, 0.5)
        plume_angle = np.random.normal(20.0, 8.0)
        plume_spread = plume_width * 1.5
        plume_area = np.pi * (plume_width / 2.0) ** 2
        particle_concentration = particle_concentration if 'particle_concentration' in locals() else np.random.normal(1.5e12, 3e11)
        metal_vapor_concentration = np.random.normal(0.012, 0.003)
        gas_composition_ratio = gas_composition_ratio if 'gas_composition_ratio' in locals() else metal_vapor_concentration / (plume_density + 1e-6)
        plume_fluctuation_rate = plume_fluctuation_rate if 'plume_fluctuation_rate' in locals() else np.random.normal(25.0, 8.0)
        plume_instability_index = plume_instability_index if 'plume_instability_index' in locals() else np.random.normal(0.4, 0.2)
        plume_turbulence = np.random.normal(0.6, 0.2)
        process_stability = process_stability if 'process_stability' in locals() else np.random.uniform(0.4, 0.6)
        plume_stability = plume_stability if 'plume_stability' in locals() else np.random.uniform(0.3, 0.5)
        intensity_variation = np.random.uniform(0.3, 0.5)  # High variation
        snr = np.random.normal(20.0, 5.0)  # Lower SNR for anomalies
        plume_energy = plume_intensity * plume_area * plume_height
        energy_density = plume_energy / (plume_area * plume_height + 1e-6)
        anomaly_detected = True
        process_event = "anomaly_event"
        plume_momentum = plume_density * plume_velocity * plume_area
        plume_pressure = None
        
        return ISPMPlumeDataPoint(
            timestamp=timestamp,
            layer_index=layer_index,
            x=x,
            y=y,
            z=layer_z,
            plume_intensity=plume_intensity,
            plume_density=plume_density,
            plume_temperature=plume_temperature,
            plume_velocity=plume_velocity,
            plume_velocity_x=plume_velocity_x,
            plume_velocity_y=plume_velocity_y,
            plume_height=plume_height,
            plume_width=plume_width,
            plume_angle=plume_angle,
            plume_spread=plume_spread,
            plume_area=plume_area,
            particle_concentration=particle_concentration,
            metal_vapor_concentration=metal_vapor_concentration,
            gas_composition_ratio=gas_composition_ratio,
            plume_fluctuation_rate=plume_fluctuation_rate,
            plume_instability_index=plume_instability_index,
            plume_turbulence=plume_turbulence,
            process_stability=process_stability,
            plume_stability=plume_stability,
            intensity_variation=intensity_variation,
            excessive_plume_event=excessive_plume_event,
            unstable_plume_event=unstable_plume_event,
            contamination_event=contamination_event,
            anomaly_detected=anomaly_detected,
            anomaly_type=anomaly_type,
            plume_energy=plume_energy,
            energy_density=energy_density,
            process_event=process_event,
            signal_to_noise_ratio=snr,
            plume_momentum=plume_momentum,
            plume_pressure=plume_pressure,
        )
