"""
Laser Parameter Generator

Generates realistic laser process parameters and temporal sensor data including:
- Process Parameters (setpoints/commanded): commanded_power, commanded_scan_speed, hatch_spacing, energy_density, exposure_time
- Temporal Sensors (actual measurements): actual_power, power_error, power_stability, beam temporal characteristics
- Laser System Health: temperature, cooling, power supply metrics

Based on sensor_fields_research.md - Category 3: Temporal Sensors (For Anomaly Detection)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class LaserParameterPoint:
    """
    Single laser parameter data point with process parameters and temporal sensor measurements.
    
    Includes:
    - Process Parameters (setpoints/commanded values)
    - Temporal Sensors (actual measured values for anomaly detection)
    - Laser System Health (temporal metrics)
    """
    # Required fields
    timestamp: datetime  # Required for all temporal sensors
    layer_index: int
    x: float
    y: float
    z: float
    
    # Process Parameters (setpoints/commanded values)
    commanded_power: float  # W (setpoint/commanded power)
    commanded_scan_speed: float  # mm/s (setpoint/commanded speed)
    hatch_spacing: float  # mm
    energy_density: float  # J/mm² (calculated)
    exposure_time: float  # s (calculated)
    region_type: str  # "contour", "hatch", "support"
    
    # Temporal Sensors - Laser Power (Category 3.1)
    actual_power: Optional[float] = None  # W (measured actual power)
    power_setpoint: Optional[float] = None  # W (commanded power - same as commanded_power)
    power_error: Optional[float] = None  # W (actual_power - power_setpoint)
    power_stability: Optional[float] = None  # % (coefficient of variation)
    power_fluctuation_amplitude: Optional[float] = None  # W (peak-to-peak variation)
    power_fluctuation_frequency: Optional[float] = None  # Hz (dominant frequency)
    
    # Temporal Sensors - Beam Temporal Characteristics (Category 3.2)
    pulse_frequency: Optional[float] = None  # Hz (for pulsed lasers)
    pulse_duration: Optional[float] = None  # µs (pulse width FWHM)
    pulse_energy: Optional[float] = None  # mJ (energy per pulse)
    duty_cycle: Optional[float] = None  # % (on-time percentage)
    beam_modulation_frequency: Optional[float] = None  # Hz (power modulation frequency)
    
    # Temporal Sensors - Laser System Health (Category 3.3)
    laser_temperature: Optional[float] = None  # °C (laser head/cavity temperature)
    laser_cooling_water_temp: Optional[float] = None  # °C
    laser_cooling_flow_rate: Optional[float] = None  # L/min
    laser_power_supply_voltage: Optional[float] = None  # V
    laser_power_supply_current: Optional[float] = None  # A
    laser_diode_current: Optional[float] = None  # A
    laser_diode_temperature: Optional[float] = None  # °C
    laser_operating_hours: Optional[float] = None  # hours (total laser operating time)
    laser_pulse_count: Optional[int] = None  # count (total number of pulses for pulsed lasers)


@dataclass
class LaserParameterGeneratorConfig:
    """Configuration for laser parameter generation."""
    # Process Parameters - Laser power ranges (setpoints)
    power_mean: float = 200.0  # W
    power_std: float = 10.0  # W
    power_min: float = 100.0  # W
    power_max: float = 400.0  # W
    
    # Process Parameters - Scan speed ranges (setpoints)
    speed_mean: float = 500.0  # mm/s
    speed_std: float = 50.0  # mm/s
    speed_min: float = 100.0  # mm/s
    speed_max: float = 2000.0  # mm/s
    
    # Process Parameters - Hatch spacing
    hatch_spacing_mean: float = 0.15  # mm (0.15mm spacing allows gaps to be visible at 0.1mm voxel resolution)
    hatch_spacing_std: float = 0.01  # mm
    
    # Region-specific parameters
    contour_power_multiplier: float = 1.2  # Contours use 20% more power
    contour_speed_multiplier: float = 0.8  # Contours use 20% less speed
    support_power_multiplier: float = 0.7  # Supports use 30% less power
    support_speed_multiplier: float = 1.5  # Supports use 50% more speed
    
    # Temporal Sensors - Power measurement accuracy
    power_measurement_error_std: float = 2.0  # W (typical measurement error)
    power_stability_window: int = 10  # Number of points for stability calculation
    
    # Temporal Sensors - Beam temporal characteristics (for pulsed lasers)
    is_pulsed_laser: bool = False  # Set to True for pulsed laser simulation
    pulse_frequency_mean: float = 1000.0  # Hz
    pulse_frequency_std: float = 50.0  # Hz
    pulse_duration_mean: float = 100.0  # µs
    pulse_duration_std: float = 10.0  # µs
    
    # Temporal Sensors - Laser system health
    laser_temperature_mean: float = 25.0  # °C
    laser_temperature_std: float = 2.0  # °C
    cooling_water_temp_mean: float = 20.0  # °C
    cooling_water_temp_std: float = 1.0  # °C
    cooling_flow_rate_mean: float = 5.0  # L/min
    cooling_flow_rate_std: float = 0.5  # L/min
    
    # Enable temporal sensor generation
    generate_temporal_sensors: bool = True  # Set to False to only generate process parameters
    
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
        
        # Temporal Sensors - Generate actual measurements if enabled
        actual_power = None
        power_setpoint = power
        power_error = None
        power_stability = None
        power_fluctuation_amplitude = None
        power_fluctuation_frequency = None
        
        pulse_frequency = None
        pulse_duration = None
        pulse_energy = None
        duty_cycle = None
        beam_modulation_frequency = None
        
        laser_temperature = None
        laser_cooling_water_temp = None
        laser_cooling_flow_rate = None
        laser_power_supply_voltage = None
        laser_power_supply_current = None
        laser_diode_current = None
        laser_diode_temperature = None
        laser_operating_hours = None
        laser_pulse_count = None
        
        if self.config.generate_temporal_sensors:
            # Actual power measurement (commanded + measurement error)
            actual_power = power + np.random.normal(0, self.config.power_measurement_error_std)
            actual_power = np.clip(actual_power, self.config.power_min, self.config.power_max)
            power_setpoint = power  # Setpoint is the commanded power
            power_error = actual_power - power_setpoint
            
            # Power stability (simulated as coefficient of variation)
            power_stability = abs(self.config.power_measurement_error_std / power) * 100.0 if power > 0 else 0.0
            
            # Power fluctuation (simulated)
            power_fluctuation_amplitude = np.random.uniform(1.0, 5.0)  # W
            power_fluctuation_frequency = np.random.uniform(0.1, 10.0)  # Hz
            
            # Beam temporal characteristics (for pulsed lasers)
            if self.config.is_pulsed_laser:
                pulse_frequency = np.random.normal(self.config.pulse_frequency_mean, self.config.pulse_frequency_std)
                pulse_frequency = max(1.0, pulse_frequency)  # Minimum 1 Hz
                pulse_duration = np.random.normal(self.config.pulse_duration_mean, self.config.pulse_duration_std)
                pulse_duration = max(1.0, pulse_duration)  # Minimum 1 µs
                pulse_energy = (actual_power / 1000.0) * (pulse_duration / 1e6) if pulse_frequency > 0 else 0.0  # mJ
                duty_cycle = (pulse_duration / 1e6) * pulse_frequency * 100.0  # %
                duty_cycle = min(100.0, duty_cycle)  # Cap at 100%
            
            # Laser system health
            laser_temperature = np.random.normal(self.config.laser_temperature_mean, self.config.laser_temperature_std)
            laser_cooling_water_temp = np.random.normal(self.config.cooling_water_temp_mean, self.config.cooling_water_temp_std)
            laser_cooling_flow_rate = np.random.normal(self.config.cooling_flow_rate_mean, self.config.cooling_flow_rate_std)
            laser_cooling_flow_rate = max(0.0, laser_cooling_flow_rate)  # Non-negative
            
            # Power supply (typical values for industrial lasers)
            laser_power_supply_voltage = np.random.normal(400.0, 10.0)  # V
            laser_power_supply_current = actual_power / laser_power_supply_voltage if laser_power_supply_voltage > 0 else 0.0
            laser_diode_current = laser_power_supply_current * 0.8  # Typical efficiency
            laser_diode_temperature = laser_temperature + np.random.normal(5.0, 1.0)  # Diode runs hotter
            
            # Cumulative metrics (calculated from build time in generate_for_build)
            # laser_operating_hours: time since build start (calculated in generate_for_build)
            # laser_pulse_count: cumulative pulse count (calculated in generate_for_build)
            # These are set to None here and will be populated during build generation
            laser_operating_hours = None
            laser_pulse_count = None
        
        return LaserParameterPoint(
            timestamp=timestamp,
            layer_index=layer_index,
            x=x,
            y=y,
            z=z,
            commanded_power=power,
            commanded_scan_speed=speed,
            hatch_spacing=hatch_spacing,
            energy_density=energy_density,
            exposure_time=exposure_time,
            region_type=region_type,
            # Temporal Sensors
            actual_power=actual_power,
            power_setpoint=power_setpoint,
            power_error=power_error,
            power_stability=power_stability,
            power_fluctuation_amplitude=power_fluctuation_amplitude,
            power_fluctuation_frequency=power_fluctuation_frequency,
            # Beam Temporal
            pulse_frequency=pulse_frequency,
            pulse_duration=pulse_duration,
            pulse_energy=pulse_energy,
            duty_cycle=duty_cycle,
            beam_modulation_frequency=beam_modulation_frequency,
            # System Health
            laser_temperature=laser_temperature,
            laser_cooling_water_temp=laser_cooling_water_temp,
            laser_cooling_flow_rate=laser_cooling_flow_rate,
            laser_power_supply_voltage=laser_power_supply_voltage,
            laser_power_supply_current=laser_power_supply_current,
            laser_diode_current=laser_diode_current,
            laser_diode_temperature=laser_diode_temperature,
            laser_operating_hours=laser_operating_hours,
            laser_pulse_count=laser_pulse_count
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
        cumulative_pulse_count = 0
        
        for layer_idx in range(n_layers):
            layer_z = layer_idx * layer_thickness
            layer_points = self.generate_for_layer(
                layer_index=layer_idx,
                layer_z=layer_z,
                n_points=points_per_layer,
                bounding_box=bounding_box
            )
            
            # Update timestamps and calculate cumulative metrics
            for i, point in enumerate(layer_points):
                point.timestamp = current_time
                
                # Calculate cumulative metrics for temporal sensors
                if self.config.generate_temporal_sensors:
                    # laser_operating_hours: time since build start
                    time_delta = (current_time - start_time).total_seconds()
                    point.laser_operating_hours = time_delta / 3600.0  # Convert to hours
                    
                    # laser_pulse_count: cumulative pulse count (for pulsed lasers)
                    if self.config.is_pulsed_laser and point.pulse_frequency is not None:
                        # Calculate pulses since last point (0.1 seconds between points)
                        time_delta_seconds = 0.1 if i > 0 else 0.0
                        pulses_in_interval = point.pulse_frequency * time_delta_seconds
                        cumulative_pulse_count += int(pulses_in_interval)
                        point.laser_pulse_count = cumulative_pulse_count
                
                current_time += timedelta(seconds=0.1)
            
            all_points.extend(layer_points)
        
        # Calculate statistics
        powers = [p.commanded_power for p in all_points]
        speeds = [p.commanded_scan_speed for p in all_points]
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

