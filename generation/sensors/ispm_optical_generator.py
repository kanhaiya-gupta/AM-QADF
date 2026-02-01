"""
ISPM_Optical (In-Situ Process Monitoring - Optical) Data Generator

Generates realistic optical ISPM sensor data for PBF-LB/M processes including:
- Photodiode signals (coaxial and off-axis)
- Melt pool brightness/intensity
- Spatter detection
- Process stability metrics
- Melt pool imaging data

Note: ISPM is a broad category. This generator specifically handles ISPM_Optical
(optical monitoring - photodiodes, cameras, melt pool imaging).
Other ISPM types include:
- ISPM_Thermal: Thermal monitoring
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
class ISPMOpticalDataPoint:
    """Single ISPM_Optical data point."""
    timestamp: datetime
    layer_index: int
    x: float
    y: float
    z: float
    
    # Photodiode signals (primary optical monitoring in PBF-LB/M)
    photodiode_intensity: float  # Arbitrary units (0-10000 typical range)
    photodiode_frequency: float  # Hz (dominant frequency component)
    
    # Melt pool brightness/intensity
    melt_pool_brightness: float  # Arbitrary units (0-10000 typical range)
    melt_pool_intensity_mean: float  # Mean intensity value
    melt_pool_intensity_max: float  # Peak intensity value
    melt_pool_intensity_std: float  # Standard deviation (stability indicator)
    
    # Process stability metrics
    process_stability: float  # 0-1 scale (1 = stable, 0 = unstable)
    intensity_variation: float  # Coefficient of variation (std/mean)
    signal_to_noise_ratio: float  # SNR in dB
    
    # Optional fields (with defaults - must come after required fields)
    photodiode_coaxial: Optional[float] = None  # Coaxial photodiode signal (if available)
    photodiode_off_axis: Optional[float] = None  # Off-axis photodiode signal (if available)
    
    # Spatter detection (critical for PBF-LB/M quality)
    spatter_detected: bool = False
    spatter_intensity: Optional[float] = None  # Intensity spike when spatter occurs
    spatter_count: int = 0  # Number of spatter events in time window
    
    # Melt pool imaging (if camera-based system)
    melt_pool_image_available: bool = False
    melt_pool_area_pixels: Optional[int] = None  # Melt pool area in pixels (if imaged)
    melt_pool_centroid_x: Optional[float] = None  # Centroid X in pixels
    melt_pool_centroid_y: Optional[float] = None  # Centroid Y in pixels
    
    # Keyhole detection (important for porosity prediction)
    keyhole_detected: bool = False
    keyhole_intensity: Optional[float] = None  # Intensity spike indicating keyhole
    
    # Process events
    process_event: Optional[str] = None  # e.g., "layer_start", "hatch_complete", "spatter_event"
    
    # Frequency domain features (for advanced analysis)
    dominant_frequency: Optional[float] = None  # Hz
    frequency_bandwidth: Optional[float] = None  # Hz (spectral width)
    spectral_energy: Optional[float] = None  # Total spectral energy


@dataclass
class ISPMOpticalGeneratorConfig:
    """Configuration for ISPM_Optical data generation."""
    # Photodiode intensity ranges
    photodiode_intensity_mean: float = 5000.0  # Typical intensity value
    photodiode_intensity_std: float = 500.0  # Variation
    
    # Frequency characteristics
    photodiode_frequency_mean: float = 1000.0  # Hz (typical laser modulation frequency)
    photodiode_frequency_std: float = 50.0  # Hz
    
    # Melt pool brightness
    melt_pool_brightness_mean: float = 6000.0
    melt_pool_brightness_std: float = 600.0
    
    # Process stability
    process_stability_mean: float = 0.85  # Typically 0.7-0.95 for good processes
    process_stability_std: float = 0.1
    
    # Spatter probability (per data point)
    spatter_probability: float = 0.02  # 2% chance of spatter per point
    
    # Keyhole probability
    keyhole_probability: float = 0.01  # 1% chance of keyhole formation
    
    # Signal-to-noise ratio
    snr_mean: float = 30.0  # dB
    snr_std: float = 5.0  # dB
    
    # Sampling
    sampling_rate: float = 10000.0  # Hz (optical sensors typically faster than thermal)
    points_per_layer: int = 1000  # Number of data points per layer
    
    # Random seed
    random_seed: Optional[int] = None


class ISPMOpticalGenerator:
    """
    Generator for ISPM_Optical (In-Situ Process Monitoring - Optical) sensor data.
    
    Creates realistic optical melt pool monitoring data with temporal and spatial variations.
    This generator specifically handles ISPM_Optical (optical monitoring - photodiodes, cameras).
    Other ISPM types include: ISPM_Thermal, ISPM_Acoustic, ISPM_Strain, ISPM_Plume, etc.
    """
    
    def __init__(self, config: Optional[ISPMOpticalGeneratorConfig] = None):
        """
        Initialize ISPM_Optical generator.
        
        Args:
            config: Configuration for data generation
        """
        self.config = config or ISPMOpticalGeneratorConfig()
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
        
        logger.info(f"ISPMOpticalGenerator initialized with config: {self.config}")
    
    def generate_for_layer(self,
                             layer_index: int,
                             layer_z: float,
                             n_points: Optional[int] = None,
                             start_time: Optional[datetime] = None,
                             bounding_box: Optional[Dict[str, Tuple[float, float]]] = None) -> List[ISPMOpticalDataPoint]:
        """
        Generate ISPM_Optical data for a single layer.
        
        Args:
            layer_index: Layer number
            layer_z: Z position of the layer
            n_points: Number of data points (defaults to config.points_per_layer)
            start_time: Start timestamp (defaults to now)
            bounding_box: Optional bounding box dict with 'min' and 'max' keys, each containing (x, y, z) tuples
            
        Returns:
            List of ISPM_Optical data points
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
        spatter_count = 0  # Track spatter events in this layer
        
        for i in range(n_points):
            timestamp = start_time + timedelta(seconds=i * dt)
            
            # Generate spatial position (random within bounding box)
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)
            z = layer_z
            
            # Generate base photodiode intensity with variation
            base_intensity = np.random.normal(
                self.config.photodiode_intensity_mean,
                self.config.photodiode_intensity_std
            )
            base_intensity = max(0.0, base_intensity)  # Ensure non-negative
            
            # Generate frequency
            frequency = np.random.normal(
                self.config.photodiode_frequency_mean,
                self.config.photodiode_frequency_std
            )
            frequency = max(100.0, frequency)  # Ensure reasonable frequency
            
            # Coaxial and off-axis photodiodes (if available)
            # Coaxial typically has higher intensity
            photodiode_coaxial = base_intensity * np.random.uniform(1.0, 1.2) if np.random.random() < 0.7 else None
            photodiode_off_axis = base_intensity * np.random.uniform(0.7, 0.9) if np.random.random() < 0.5 else None
            
            # Melt pool brightness (correlated with photodiode intensity)
            melt_pool_brightness = np.random.normal(
                self.config.melt_pool_brightness_mean,
                self.config.melt_pool_brightness_std
            )
            melt_pool_brightness = max(0.0, melt_pool_brightness)
            
            # Melt pool intensity statistics
            melt_pool_intensity_mean = melt_pool_brightness
            melt_pool_intensity_max = melt_pool_brightness * np.random.uniform(1.1, 1.5)
            melt_pool_intensity_std = melt_pool_brightness * np.random.uniform(0.05, 0.15)
            
            # Process stability
            process_stability = np.random.normal(
                self.config.process_stability_mean,
                self.config.process_stability_std
            )
            process_stability = np.clip(process_stability, 0.0, 1.0)
            
            # Intensity variation (coefficient of variation)
            intensity_variation = np.random.uniform(0.05, 0.25) * (1.0 - process_stability)
            
            # Signal-to-noise ratio
            snr = np.random.normal(
                self.config.snr_mean,
                self.config.snr_std
            )
            snr = max(10.0, snr)  # Ensure reasonable SNR
            
            # Spatter detection
            spatter_detected = np.random.random() < self.config.spatter_probability
            spatter_intensity = None
            if spatter_detected:
                spatter_count += 1
                # Spatter causes intensity spike
                spatter_intensity = base_intensity * np.random.uniform(1.5, 3.0)
                # Increase intensity when spatter occurs
                base_intensity = spatter_intensity
            
            # Keyhole detection
            keyhole_detected = np.random.random() < self.config.keyhole_probability
            keyhole_intensity = None
            if keyhole_detected:
                # Keyhole causes very high intensity spike
                keyhole_intensity = base_intensity * np.random.uniform(2.0, 4.0)
                base_intensity = keyhole_intensity
            
            # Melt pool imaging (if camera-based system - 30% chance)
            melt_pool_image_available = np.random.random() < 0.3
            melt_pool_area_pixels = None
            melt_pool_centroid_x = None
            melt_pool_centroid_y = None
            if melt_pool_image_available:
                # Typical melt pool size: 10-50 pixels in diameter
                melt_pool_diameter_pixels = np.random.uniform(10, 50)
                melt_pool_area_pixels = int(np.pi * (melt_pool_diameter_pixels / 2.0) ** 2)
                # Centroid in image coordinates (assume 100x100 pixel image)
                melt_pool_centroid_x = np.random.uniform(20, 80)
                melt_pool_centroid_y = np.random.uniform(20, 80)
            
            # Frequency domain features
            dominant_frequency = frequency
            frequency_bandwidth = np.random.uniform(50, 200)  # Hz
            spectral_energy = base_intensity * np.random.uniform(0.8, 1.2)
            
            # Process events
            process_event = None
            if i == 0:
                process_event = "layer_start"
            elif np.random.random() < 0.01:  # 1% chance
                events = ["hatch_complete", "contour_complete", "layer_complete"]
                process_event = np.random.choice(events)
            elif spatter_detected:
                process_event = "spatter_event"
            elif keyhole_detected:
                process_event = "keyhole_event"
            
            data_point = ISPMOpticalDataPoint(
                timestamp=timestamp,
                layer_index=layer_index,
                x=x,
                y=y,
                z=z,
                photodiode_intensity=base_intensity,
                photodiode_frequency=frequency,
                photodiode_coaxial=photodiode_coaxial,
                photodiode_off_axis=photodiode_off_axis,
                melt_pool_brightness=melt_pool_brightness,
                melt_pool_intensity_mean=melt_pool_intensity_mean,
                melt_pool_intensity_max=melt_pool_intensity_max,
                melt_pool_intensity_std=melt_pool_intensity_std,
                spatter_detected=spatter_detected,
                spatter_intensity=spatter_intensity,
                spatter_count=spatter_count,
                process_stability=process_stability,
                intensity_variation=intensity_variation,
                signal_to_noise_ratio=snr,
                melt_pool_image_available=melt_pool_image_available,
                melt_pool_area_pixels=melt_pool_area_pixels,
                melt_pool_centroid_x=melt_pool_centroid_x,
                melt_pool_centroid_y=melt_pool_centroid_y,
                keyhole_detected=keyhole_detected,
                keyhole_intensity=keyhole_intensity,
                process_event=process_event,
                dominant_frequency=dominant_frequency,
                frequency_bandwidth=frequency_bandwidth,
                spectral_energy=spectral_energy
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
        Generate ISPM_Optical data for an entire build.
        
        Args:
            build_id: Build identifier
            n_layers: Number of layers
            layer_thickness: Thickness of each layer (mm)
            start_time: Build start time (defaults to now)
            bounding_box: Optional bounding box {'x': (min, max), 'y': (min, max), 'z': (min, max)}
            
        Returns:
            Dictionary containing build ISPM_Optical data
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
                'description': 'ISPM optical sensors are typically positioned at fixed locations on the machine (coaxial or off-axis)',
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
            'description': 'ISPM optical sensor coordinate system. Measurement coordinates are in build platform space. Sensors are typically positioned at fixed locations on the machine and monitor the process at specific spatial locations.'
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
                                     base_data_point: ISPMOpticalDataPoint,
                                     anomaly_type: str = "intensity_spike") -> ISPMOpticalDataPoint:
        """
        Generate an anomalous data point based on a normal one.
        
        Args:
            base_data_point: Base normal data point
            anomaly_type: Type of anomaly ("intensity_spike", "intensity_drop", "spatter_event", "keyhole_event")
            
        Returns:
            Anomalous data point
        """
        if anomaly_type == "intensity_spike":
            photodiode_intensity = base_data_point.photodiode_intensity * np.random.uniform(2.0, 4.0)
            melt_pool_brightness = base_data_point.melt_pool_brightness * np.random.uniform(1.5, 2.5)
            process_stability = max(0.0, base_data_point.process_stability - 0.3)
        elif anomaly_type == "intensity_drop":
            photodiode_intensity = base_data_point.photodiode_intensity * np.random.uniform(0.3, 0.6)
            melt_pool_brightness = base_data_point.melt_pool_brightness * np.random.uniform(0.4, 0.7)
            process_stability = max(0.0, base_data_point.process_stability - 0.2)
        elif anomaly_type == "spatter_event":
            photodiode_intensity = base_data_point.photodiode_intensity * np.random.uniform(1.5, 3.0)
            melt_pool_brightness = base_data_point.melt_pool_brightness * np.random.uniform(1.2, 2.0)
            spatter_detected = True
            spatter_intensity = photodiode_intensity
            process_event = "spatter_event"
        elif anomaly_type == "keyhole_event":
            photodiode_intensity = base_data_point.photodiode_intensity * np.random.uniform(2.0, 4.0)
            melt_pool_brightness = base_data_point.melt_pool_brightness * np.random.uniform(2.0, 3.5)
            keyhole_detected = True
            keyhole_intensity = photodiode_intensity
            process_event = "keyhole_event"
            process_stability = max(0.0, base_data_point.process_stability - 0.4)
        else:
            # Default: slight variation
            photodiode_intensity = base_data_point.photodiode_intensity
            melt_pool_brightness = base_data_point.melt_pool_brightness
            process_stability = base_data_point.process_stability
        
        # Copy other fields with potential modifications
        return ISPMOpticalDataPoint(
            timestamp=base_data_point.timestamp,
            layer_index=base_data_point.layer_index,
            x=base_data_point.x,
            y=base_data_point.y,
            z=base_data_point.z,
            photodiode_intensity=photodiode_intensity,
            photodiode_frequency=base_data_point.photodiode_frequency,
            photodiode_coaxial=base_data_point.photodiode_coaxial,
            photodiode_off_axis=base_data_point.photodiode_off_axis,
            melt_pool_brightness=melt_pool_brightness,
            melt_pool_intensity_mean=melt_pool_brightness,
            melt_pool_intensity_max=melt_pool_brightness * np.random.uniform(1.1, 1.5),
            melt_pool_intensity_std=melt_pool_brightness * np.random.uniform(0.1, 0.2),
            spatter_detected=spatter_detected if anomaly_type == "spatter_event" else base_data_point.spatter_detected,
            spatter_intensity=spatter_intensity if anomaly_type == "spatter_event" else base_data_point.spatter_intensity,
            spatter_count=base_data_point.spatter_count + (1 if anomaly_type == "spatter_event" else 0),
            process_stability=process_stability,
            intensity_variation=base_data_point.intensity_variation * np.random.uniform(1.2, 2.0),
            signal_to_noise_ratio=max(10.0, base_data_point.signal_to_noise_ratio - np.random.uniform(5, 15)),
            melt_pool_image_available=base_data_point.melt_pool_image_available,
            melt_pool_area_pixels=base_data_point.melt_pool_area_pixels,
            melt_pool_centroid_x=base_data_point.melt_pool_centroid_x,
            melt_pool_centroid_y=base_data_point.melt_pool_centroid_y,
            keyhole_detected=keyhole_detected if anomaly_type == "keyhole_event" else base_data_point.keyhole_detected,
            keyhole_intensity=keyhole_intensity if anomaly_type == "keyhole_event" else base_data_point.keyhole_intensity,
            process_event=process_event if anomaly_type in ["spatter_event", "keyhole_event"] else base_data_point.process_event,
            dominant_frequency=base_data_point.dominant_frequency,
            frequency_bandwidth=base_data_point.frequency_bandwidth,
            spectral_energy=base_data_point.spectral_energy
        )
