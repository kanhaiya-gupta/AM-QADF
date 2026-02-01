"""
ISPM_Acoustic (In-Situ Process Monitoring - Acoustic) Data Generator

Generates realistic acoustic ISPM sensor data for PBF-LB/M processes including:
- Acoustic emission signals
- Frequency spectra
- Event detection (spatter, defects, anomalies)
- Process noise patterns
- Acoustic energy and signal characteristics

Note: ISPM is a broad category. This generator specifically handles ISPM_Acoustic
(acoustic monitoring - acoustic emissions, sound sensors).
Other ISPM types include:
- ISPM_Thermal: Thermal monitoring
- ISPM_Optical: Photodiodes, cameras, melt pool imaging
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
class ISPMAcousticDataPoint:
    """Single ISPM_Acoustic data point."""
    timestamp: datetime
    layer_index: int
    x: float
    y: float
    z: float
    
    # Acoustic emission signals (primary acoustic monitoring in PBF-LB/M)
    acoustic_amplitude: float  # Arbitrary units (0-10000 typical range)
    acoustic_frequency: float  # Hz (dominant frequency component)
    acoustic_rms: float  # Root mean square amplitude
    acoustic_peak: float  # Peak amplitude value
    
    # Frequency domain features
    dominant_frequency: float  # Hz (primary frequency component)
    frequency_bandwidth: float  # Hz (spectral width)
    spectral_centroid: float  # Hz (weighted average frequency)
    spectral_energy: float  # Total spectral energy
    
    # Process stability metrics (required fields)
    process_stability: float  # 0-1 scale (1 = stable, 0 = unstable)
    acoustic_variation: float  # Coefficient of variation (std/mean)
    signal_to_noise_ratio: float  # SNR in dB
    
    # Acoustic energy metrics (required fields)
    acoustic_energy: float  # Total acoustic energy
    
    # Optional fields (with defaults - must come after required fields)
    spectral_rolloff: Optional[float] = None  # Hz (frequency below which 85% of energy is contained)
    
    # Event detection (critical for PBF-LB/M quality)
    spatter_event_detected: bool = False
    spatter_event_amplitude: Optional[float] = None  # Amplitude spike when spatter occurs
    defect_event_detected: bool = False
    defect_event_amplitude: Optional[float] = None  # Amplitude spike when defect forms
    anomaly_detected: bool = False
    anomaly_type: Optional[str] = None  # e.g., "lack_of_fusion", "keyhole_instability", "process_instability"
    
    # Time-domain features
    zero_crossing_rate: Optional[float] = None  # Zero crossings per second
    autocorrelation_peak: Optional[float] = None  # Peak autocorrelation value
    
    # Frequency-domain features (advanced analysis)
    harmonic_ratio: Optional[float] = None  # Ratio of harmonic to fundamental energy
    spectral_flatness: Optional[float] = None  # Measure of spectral uniformity (0-1)
    spectral_crest: Optional[float] = None  # Peak-to-average ratio in frequency domain
    
    # Process events
    process_event: Optional[str] = None  # e.g., "layer_start", "spatter_event", "defect_event", "anomaly_event"
    
    # Acoustic energy metrics (optional)
    energy_per_band: Optional[Dict[str, float]] = None  # Energy in different frequency bands (low, mid, high)


@dataclass
class ISPMAcousticGeneratorConfig:
    """Configuration for ISPM_Acoustic data generation."""
    # Acoustic amplitude ranges
    acoustic_amplitude_mean: float = 3000.0  # Typical amplitude value
    acoustic_amplitude_std: float = 400.0  # Variation
    
    # Frequency characteristics
    acoustic_frequency_mean: float = 5000.0  # Hz (typical acoustic emission frequency)
    acoustic_frequency_std: float = 500.0  # Hz
    
    # Dominant frequency
    dominant_frequency_mean: float = 5000.0  # Hz
    dominant_frequency_std: float = 500.0  # Hz
    
    # Process stability
    process_stability_mean: float = 0.80  # Typically 0.7-0.9 for good processes
    process_stability_std: float = 0.12
    
    # Event probabilities (per data point)
    spatter_event_probability: float = 0.015  # 1.5% chance of spatter per point
    defect_event_probability: float = 0.005  # 0.5% chance of defect per point
    anomaly_probability: float = 0.01  # 1% chance of anomaly per point
    
    # Signal-to-noise ratio
    snr_mean: float = 25.0  # dB (acoustic sensors typically lower SNR than optical)
    snr_std: float = 6.0  # dB
    
    # Frequency bandwidth
    frequency_bandwidth_mean: float = 2000.0  # Hz
    frequency_bandwidth_std: float = 300.0  # Hz
    
    # Sampling
    sampling_rate: float = 50000.0  # Hz (acoustic sensors typically very fast, 50kHz+)
    points_per_layer: int = 1000  # Number of data points per layer
    
    # Random seed
    random_seed: Optional[int] = None


class ISPMAcousticGenerator:
    """
    Generator for ISPM_Acoustic (In-Situ Process Monitoring - Acoustic) sensor data.
    
    Creates realistic acoustic emission monitoring data with temporal and spatial variations.
    This generator specifically handles ISPM_Acoustic (acoustic monitoring - acoustic emissions).
    Other ISPM types include: ISPM_Thermal, ISPM_Optical, ISPM_Strain, ISPM_Plume, etc.
    """
    
    def __init__(self, config: Optional[ISPMAcousticGeneratorConfig] = None):
        """
        Initialize ISPM_Acoustic generator.
        
        Args:
            config: Configuration for data generation
        """
        self.config = config or ISPMAcousticGeneratorConfig()
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
        
        logger.info(f"ISPMAcousticGenerator initialized with config: {self.config}")
    
    def generate_for_layer(self,
                             layer_index: int,
                             layer_z: float,
                             n_points: Optional[int] = None,
                             start_time: Optional[datetime] = None,
                             bounding_box: Optional[Dict[str, Tuple[float, float]]] = None) -> List[ISPMAcousticDataPoint]:
        """
        Generate ISPM_Acoustic data for a single layer.
        
        Args:
            layer_index: Layer number
            layer_z: Z position of the layer
            n_points: Number of data points (defaults to config.points_per_layer)
            start_time: Start timestamp (defaults to now)
            bounding_box: Optional bounding box dict with 'min' and 'max' keys, each containing (x, y, z) tuples
            
        Returns:
            List of ISPM_Acoustic data points
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
            
            # Generate base acoustic amplitude with variation
            base_amplitude = np.random.normal(
                self.config.acoustic_amplitude_mean,
                self.config.acoustic_amplitude_std
            )
            base_amplitude = max(0.0, base_amplitude)  # Ensure non-negative
            
            # Generate frequency
            frequency = np.random.normal(
                self.config.acoustic_frequency_mean,
                self.config.acoustic_frequency_std
            )
            frequency = max(100.0, frequency)  # Ensure reasonable frequency
            
            # Dominant frequency (may differ slightly from base frequency)
            dominant_frequency = np.random.normal(
                self.config.dominant_frequency_mean,
                self.config.dominant_frequency_std
            )
            dominant_frequency = max(100.0, dominant_frequency)
            
            # RMS and peak amplitude
            acoustic_rms = base_amplitude * np.random.uniform(0.7, 0.9)
            acoustic_peak = base_amplitude * np.random.uniform(1.2, 1.8)
            
            # Frequency bandwidth
            frequency_bandwidth = np.random.normal(
                self.config.frequency_bandwidth_mean,
                self.config.frequency_bandwidth_std
            )
            frequency_bandwidth = max(500.0, frequency_bandwidth)
            
            # Spectral centroid (weighted average frequency)
            spectral_centroid = dominant_frequency * np.random.uniform(0.9, 1.1)
            
            # Spectral energy
            spectral_energy = base_amplitude * np.random.uniform(0.8, 1.2)
            
            # Spectral rolloff (frequency below which 85% of energy is contained)
            spectral_rolloff = dominant_frequency + frequency_bandwidth * np.random.uniform(0.3, 0.7)
            
            # Process stability
            process_stability = np.random.normal(
                self.config.process_stability_mean,
                self.config.process_stability_std
            )
            process_stability = np.clip(process_stability, 0.0, 1.0)
            
            # Acoustic variation (coefficient of variation)
            acoustic_variation = np.random.uniform(0.08, 0.30) * (1.0 - process_stability)
            
            # Signal-to-noise ratio
            snr = np.random.normal(
                self.config.snr_mean,
                self.config.snr_std
            )
            snr = max(10.0, snr)  # Ensure reasonable SNR
            
            # Event detection
            spatter_event_detected = np.random.random() < self.config.spatter_event_probability
            spatter_event_amplitude = None
            if spatter_event_detected:
                # Spatter causes significant amplitude spike
                spatter_event_amplitude = base_amplitude * np.random.uniform(2.0, 4.0)
                base_amplitude = spatter_event_amplitude
                acoustic_peak = base_amplitude * np.random.uniform(1.3, 2.0)
            
            defect_event_detected = np.random.random() < self.config.defect_event_probability
            defect_event_amplitude = None
            if defect_event_detected:
                # Defect formation causes different acoustic signature
                defect_event_amplitude = base_amplitude * np.random.uniform(1.5, 3.0)
                base_amplitude = defect_event_amplitude
                # Defects often have different frequency characteristics
                dominant_frequency = dominant_frequency * np.random.uniform(0.7, 1.3)
            
            anomaly_detected = np.random.random() < self.config.anomaly_probability
            anomaly_type = None
            if anomaly_detected and not spatter_event_detected and not defect_event_detected:
                # Various anomaly types
                anomaly_types = ["lack_of_fusion", "keyhole_instability", "process_instability", "power_fluctuation"]
                anomaly_type = np.random.choice(anomaly_types)
                # Anomalies cause amplitude and frequency changes
                base_amplitude = base_amplitude * np.random.uniform(1.2, 2.5)
                dominant_frequency = dominant_frequency * np.random.uniform(0.8, 1.2)
                process_stability = max(0.0, process_stability - np.random.uniform(0.2, 0.4))
            
            # Time-domain features
            zero_crossing_rate = np.random.uniform(100, 1000)  # Zero crossings per second
            autocorrelation_peak = np.random.uniform(0.3, 0.9)
            
            # Frequency-domain features
            harmonic_ratio = np.random.uniform(0.1, 0.4)  # Ratio of harmonic to fundamental
            spectral_flatness = np.random.uniform(0.2, 0.8)  # Spectral uniformity
            spectral_crest = np.random.uniform(2.0, 8.0)  # Peak-to-average ratio
            
            # Acoustic energy
            acoustic_energy = base_amplitude * np.random.uniform(0.9, 1.1)
            
            # Energy per frequency band
            energy_per_band = {
                "low": acoustic_energy * np.random.uniform(0.2, 0.4),  # 0-2kHz
                "mid": acoustic_energy * np.random.uniform(0.3, 0.5),  # 2-10kHz
                "high": acoustic_energy * np.random.uniform(0.1, 0.3),  # 10kHz+
            }
            
            # Process events
            process_event = None
            if i == 0:
                process_event = "layer_start"
            elif np.random.random() < 0.01:  # 1% chance
                events = ["hatch_complete", "contour_complete", "layer_complete"]
                process_event = np.random.choice(events)
            elif spatter_event_detected:
                process_event = "spatter_event"
            elif defect_event_detected:
                process_event = "defect_event"
            elif anomaly_detected:
                process_event = f"anomaly_event_{anomaly_type}"
            
            data_point = ISPMAcousticDataPoint(
                timestamp=timestamp,
                layer_index=layer_index,
                x=x,
                y=y,
                z=z,
                acoustic_amplitude=base_amplitude,
                acoustic_frequency=frequency,
                acoustic_rms=acoustic_rms,
                acoustic_peak=acoustic_peak,
                dominant_frequency=dominant_frequency,
                frequency_bandwidth=frequency_bandwidth,
                spectral_centroid=spectral_centroid,
                spectral_energy=spectral_energy,
                spectral_rolloff=spectral_rolloff,
                spatter_event_detected=spatter_event_detected,
                spatter_event_amplitude=spatter_event_amplitude,
                defect_event_detected=defect_event_detected,
                defect_event_amplitude=defect_event_amplitude,
                anomaly_detected=anomaly_detected,
                anomaly_type=anomaly_type,
                process_stability=process_stability,
                acoustic_variation=acoustic_variation,
                signal_to_noise_ratio=snr,
                zero_crossing_rate=zero_crossing_rate,
                autocorrelation_peak=autocorrelation_peak,
                harmonic_ratio=harmonic_ratio,
                spectral_flatness=spectral_flatness,
                spectral_crest=spectral_crest,
                process_event=process_event,
                acoustic_energy=acoustic_energy,
                energy_per_band=energy_per_band
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
        Generate ISPM_Acoustic data for an entire build.
        
        Args:
            build_id: Build identifier
            n_layers: Number of layers
            layer_thickness: Thickness of each layer (mm)
            start_time: Build start time (defaults to now)
            bounding_box: Optional bounding box {'x': (min, max), 'y': (min, max), 'z': (min, max)}
            
        Returns:
            Dictionary containing build ISPM_Acoustic data
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
                'description': 'ISPM acoustic sensors are typically positioned at fixed locations on the machine (near build chamber)',
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
            'description': 'ISPM acoustic sensor coordinate system. Measurement coordinates are in build platform space. Sensors are typically positioned at fixed locations on the machine and monitor the process at specific spatial locations.'
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
                                     base_data_point: ISPMAcousticDataPoint,
                                     anomaly_type: str = "amplitude_spike") -> ISPMAcousticDataPoint:
        """
        Generate an anomalous data point based on a normal one.
        
        Args:
            base_data_point: Base normal data point
            anomaly_type: Type of anomaly ("amplitude_spike", "amplitude_drop", "spatter_event", "defect_event", "frequency_shift")
            
        Returns:
            Anomalous data point
        """
        if anomaly_type == "amplitude_spike":
            acoustic_amplitude = base_data_point.acoustic_amplitude * np.random.uniform(2.0, 4.0)
            acoustic_peak = acoustic_amplitude * np.random.uniform(1.3, 2.0)
            process_stability = max(0.0, base_data_point.process_stability - 0.3)
        elif anomaly_type == "amplitude_drop":
            acoustic_amplitude = base_data_point.acoustic_amplitude * np.random.uniform(0.3, 0.6)
            acoustic_peak = acoustic_amplitude * np.random.uniform(1.2, 1.5)
            process_stability = max(0.0, base_data_point.process_stability - 0.2)
        elif anomaly_type == "spatter_event":
            acoustic_amplitude = base_data_point.acoustic_amplitude * np.random.uniform(2.0, 4.0)
            acoustic_peak = acoustic_amplitude * np.random.uniform(1.5, 2.5)
            spatter_event_detected = True
            spatter_event_amplitude = acoustic_amplitude
            process_event = "spatter_event"
        elif anomaly_type == "defect_event":
            acoustic_amplitude = base_data_point.acoustic_amplitude * np.random.uniform(1.5, 3.0)
            acoustic_peak = acoustic_amplitude * np.random.uniform(1.3, 2.0)
            defect_event_detected = True
            defect_event_amplitude = acoustic_amplitude
            dominant_frequency = base_data_point.dominant_frequency * np.random.uniform(0.7, 1.3)
            process_event = "defect_event"
            process_stability = max(0.0, base_data_point.process_stability - 0.3)
        elif anomaly_type == "frequency_shift":
            acoustic_amplitude = base_data_point.acoustic_amplitude
            acoustic_peak = base_data_point.acoustic_peak
            dominant_frequency = base_data_point.dominant_frequency * np.random.uniform(0.5, 1.5)
            process_stability = max(0.0, base_data_point.process_stability - 0.2)
        else:
            # Default: slight variation
            acoustic_amplitude = base_data_point.acoustic_amplitude
            acoustic_peak = base_data_point.acoustic_peak
            process_stability = base_data_point.process_stability
        
        # Copy other fields with potential modifications
        return ISPMAcousticDataPoint(
            timestamp=base_data_point.timestamp,
            layer_index=base_data_point.layer_index,
            x=base_data_point.x,
            y=base_data_point.y,
            z=base_data_point.z,
            acoustic_amplitude=acoustic_amplitude,
            acoustic_frequency=base_data_point.acoustic_frequency,
            acoustic_rms=acoustic_amplitude * np.random.uniform(0.7, 0.9),
            acoustic_peak=acoustic_peak,
            dominant_frequency=dominant_frequency if anomaly_type in ["defect_event", "frequency_shift"] else base_data_point.dominant_frequency,
            frequency_bandwidth=base_data_point.frequency_bandwidth,
            spectral_centroid=base_data_point.spectral_centroid,
            spectral_energy=base_data_point.spectral_energy,
            spectral_rolloff=base_data_point.spectral_rolloff,
            spatter_event_detected=spatter_event_detected if anomaly_type == "spatter_event" else base_data_point.spatter_event_detected,
            spatter_event_amplitude=spatter_event_amplitude if anomaly_type == "spatter_event" else base_data_point.spatter_event_amplitude,
            defect_event_detected=defect_event_detected if anomaly_type == "defect_event" else base_data_point.defect_event_detected,
            defect_event_amplitude=defect_event_amplitude if anomaly_type == "defect_event" else base_data_point.defect_event_amplitude,
            anomaly_detected=base_data_point.anomaly_detected,
            anomaly_type=base_data_point.anomaly_type,
            process_stability=process_stability,
            acoustic_variation=base_data_point.acoustic_variation * np.random.uniform(1.2, 2.0),
            signal_to_noise_ratio=max(10.0, base_data_point.signal_to_noise_ratio - np.random.uniform(5, 15)),
            zero_crossing_rate=base_data_point.zero_crossing_rate,
            autocorrelation_peak=base_data_point.autocorrelation_peak,
            harmonic_ratio=base_data_point.harmonic_ratio,
            spectral_flatness=base_data_point.spectral_flatness,
            spectral_crest=base_data_point.spectral_crest,
            process_event=process_event if anomaly_type in ["spatter_event", "defect_event"] else base_data_point.process_event,
            acoustic_energy=acoustic_amplitude * np.random.uniform(0.9, 1.1),
            energy_per_band=base_data_point.energy_per_band
        )
