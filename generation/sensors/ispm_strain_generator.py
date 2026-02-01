"""
ISPM_Strain (In-Situ Process Monitoring - Strain) Data Generator

Generates realistic strain ISPM sensor data for PBF-LB/M processes including:
- Strain measurements (x, y, z components, principal strains)
- Deformation/displacement measurements
- Residual stress indicators
- Warping and distortion detection
- Layer-wise strain accumulation
- Process stability metrics

Note: ISPM is a broad category. This generator specifically handles ISPM_Strain
(strain monitoring - strain gauges, deformation sensors).
Other ISPM types include:
- ISPM_Thermal: Thermal monitoring
- ISPM_Optical: Photodiodes, cameras, melt pool imaging
- ISPM_Acoustic: Acoustic emissions, sound sensors
- ISPM_Plume: Vapor plume monitoring
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class ISPMStrainDataPoint:
    """Single ISPM_Strain data point."""
    timestamp: datetime
    layer_index: int
    x: float
    y: float
    z: float
    
    # Strain components (microstrain, typically -5000 to +5000 με)
    strain_xx: float  # Normal strain in x-direction (με)
    strain_yy: float  # Normal strain in y-direction (με)
    strain_zz: float  # Normal strain in z-direction (με)
    strain_xy: float  # Shear strain in xy-plane (με)
    strain_xz: float  # Shear strain in xz-plane (με)
    strain_yz: float  # Shear strain in yz-plane (με)
    
    # Principal strains (derived from strain tensor)
    principal_strain_max: float  # Maximum principal strain (με)
    principal_strain_min: float  # Minimum principal strain (με)
    principal_strain_intermediate: float  # Intermediate principal strain (με)
    
    # Equivalent/von Mises strain (important for yield/failure prediction)
    von_mises_strain: float  # Equivalent strain (με)
    
    # Deformation/displacement (mm)
    displacement_x: float  # Displacement in x-direction (mm)
    displacement_y: float  # Displacement in y-direction (mm)
    displacement_z: float  # Displacement in z-direction (mm)
    total_displacement: float  # Total displacement magnitude (mm)
    
    # Strain rate (με/s)
    strain_rate: float  # Rate of change of equivalent strain
    
    # Layer-wise strain accumulation (required fields)
    cumulative_strain: float  # Cumulative strain from build start (με)
    layer_strain_increment: float  # Strain increment for this layer (με)
    
    # Process stability metrics (required fields)
    process_stability: float  # 0-1 scale (1 = stable, 0 = unstable)
    strain_variation: float  # Coefficient of variation (std/mean)
    strain_uniformity: float  # 0-1 scale (1 = uniform, 0 = non-uniform)
    
    # Optional fields (with defaults - must come after required fields)
    # Residual stress indicators (MPa, estimated from strain)
    residual_stress_xx: Optional[float] = None  # Residual stress in x-direction (MPa)
    residual_stress_yy: Optional[float] = None  # Residual stress in y-direction (MPa)
    residual_stress_zz: Optional[float] = None  # Residual stress in z-direction (MPa)
    von_mises_stress: Optional[float] = None  # Equivalent von Mises stress (MPa)
    
    # Temperature-compensated strain (με)
    temperature_compensated_strain: Optional[float] = None  # Strain corrected for thermal expansion
    
    # Warping/distortion indicators
    warping_detected: bool = False  # Significant warping detected
    warping_magnitude: Optional[float] = None  # Warping magnitude (mm)
    distortion_angle: Optional[float] = None  # Distortion angle (degrees)
    
    # Event detection (critical for PBF-LB/M quality)
    excessive_strain_event: bool = False  # Strain exceeds threshold
    warping_event_detected: bool = False  # Significant warping event
    distortion_event_detected: bool = False  # Distortion event detected
    anomaly_detected: bool = False
    anomaly_type: Optional[str] = None  # e.g., "excessive_warping", "residual_stress_build-up", "layer_delamination"
    
    # Process events
    process_event: Optional[str] = None  # e.g., "layer_start", "warping_event", "distortion_event", "anomaly_event"
    
    # Strain energy metrics
    strain_energy_density: Optional[float] = None  # Strain energy per unit volume (J/m³)


@dataclass
class ISPMStrainGeneratorConfig:
    """Configuration for ISPM_Strain data generation."""
    # Strain ranges (microstrain, typical range -5000 to +5000 με)
    strain_mean: float = 500.0  # Typical strain value (με)
    strain_std: float = 300.0  # Variation (με)
    
    # Normal strain components
    strain_xx_mean: float = 500.0  # με
    strain_xx_std: float = 300.0  # με
    strain_yy_mean: float = 400.0  # με
    strain_yy_std: float = 250.0  # με
    strain_zz_mean: float = 600.0  # με (typically higher in z-direction due to layer stacking)
    strain_zz_std: float = 350.0  # με
    
    # Shear strain components (typically smaller)
    shear_strain_mean: float = 100.0  # με
    shear_strain_std: float = 80.0  # με
    
    # Displacement ranges (mm)
    displacement_mean: float = 0.05  # Typical displacement (mm)
    displacement_std: float = 0.03  # Variation (mm)
    
    # Strain rate
    strain_rate_mean: float = 10.0  # με/s
    strain_rate_std: float = 5.0  # με/s
    
    # Residual stress (MPa, estimated from strain using Young's modulus ~200 GPa for steel)
    residual_stress_mean: float = 100.0  # MPa
    residual_stress_std: float = 50.0  # MPa
    
    # Process stability
    process_stability_mean: float = 0.75  # Typically 0.7-0.85 for good processes
    process_stability_std: float = 0.15
    
    # Event probabilities (per data point)
    excessive_strain_probability: float = 0.02  # 2% chance of excessive strain per point
    warping_event_probability: float = 0.01  # 1% chance of warping event per point
    distortion_event_probability: float = 0.008  # 0.8% chance of distortion event per point
    anomaly_probability: float = 0.015  # 1.5% chance of anomaly per point
    
    # Strain thresholds
    excessive_strain_threshold: float = 3000.0  # με (strain above this is considered excessive)
    warping_threshold: float = 0.2  # mm (displacement above this indicates warping)
    
    # Sampling
    sampling_rate: float = 100.0  # Hz (strain sensors typically slower than acoustic, ~100Hz)
    points_per_layer: int = 500  # Number of data points per layer
    
    # Random seed
    random_seed: Optional[int] = None


class ISPMStrainGenerator:
    """
    Generator for ISPM_Strain (In-Situ Process Monitoring - Strain) sensor data.
    
    Creates realistic strain monitoring data with temporal and spatial variations.
    This generator specifically handles ISPM_Strain (strain monitoring - strain gauges).
    Other ISPM types include: ISPM_Thermal, ISPM_Optical, ISPM_Acoustic, ISPM_Plume, etc.
    """
    
    def __init__(self, config: Optional[ISPMStrainGeneratorConfig] = None):
        """
        Initialize ISPM_Strain generator.
        
        Args:
            config: Configuration for data generation
        """
        self.config = config or ISPMStrainGeneratorConfig()
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
        
        logger.info(f"ISPMStrainGenerator initialized with config: {self.config}")
    
    def generate_for_layer(self,
                             layer_index: int,
                             layer_z: float,
                             n_points: Optional[int] = None,
                             start_time: Optional[datetime] = None,
                             bounding_box: Optional[Dict[str, Tuple[float, float]]] = None) -> List[ISPMStrainDataPoint]:
        """
        Generate ISPM_Strain data for a single layer.
        
        Args:
            layer_index: Layer index (0-based)
            layer_z: Z-coordinate of the layer (mm)
            n_points: Number of data points to generate (default: config.points_per_layer)
            start_time: Start time for the layer (default: current time)
            bounding_box: Bounding box for spatial distribution {'x': (min, max), 'y': (min, max)}
        
        Returns:
            List of ISPM_Strain data points
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
        
        # Cumulative strain increases with layer (residual stress build-up)
        base_cumulative_strain = layer_index * 50.0  # Accumulation per layer
        
        for i in range(n_points):
            current_time = start_time + i * time_step
            
            # Spatial distribution (uniform within bounding box)
            x = np.random.uniform(bounding_box['x'][0], bounding_box['x'][1])
            y = np.random.uniform(bounding_box['y'][0], bounding_box['y'][1])
            z = layer_z
            
            # Generate normal strain components
            strain_xx = np.random.normal(self.config.strain_xx_mean, self.config.strain_xx_std)
            strain_yy = np.random.normal(self.config.strain_yy_mean, self.config.strain_yy_std)
            strain_zz = np.random.normal(self.config.strain_zz_mean, self.config.strain_zz_std)
            
            # Generate shear strain components
            strain_xy = np.random.normal(self.config.shear_strain_mean, self.config.shear_strain_std)
            strain_xz = np.random.normal(self.config.shear_strain_mean, self.config.shear_strain_std)
            strain_yz = np.random.normal(self.config.shear_strain_mean, self.config.shear_strain_std)
            
            # Calculate principal strains from strain tensor
            # For 3D: eigenvalues of strain tensor
            strain_tensor = np.array([
                [strain_xx, strain_xy, strain_xz],
                [strain_xy, strain_yy, strain_yz],
                [strain_xz, strain_yz, strain_zz]
            ])
            eigenvals = np.linalg.eigvals(strain_tensor)
            eigenvals_sorted = np.sort(eigenvals)[::-1]  # Descending order
            principal_strain_max = eigenvals_sorted[0]
            principal_strain_min = eigenvals_sorted[2]
            principal_strain_intermediate = eigenvals_sorted[1]
            
            # Calculate von Mises/equivalent strain
            # von Mises strain = sqrt(2/3 * sum of squared deviatoric strain components)
            mean_strain = (strain_xx + strain_yy + strain_zz) / 3.0
            dev_xx = strain_xx - mean_strain
            dev_yy = strain_yy - mean_strain
            dev_zz = strain_zz - mean_strain
            von_mises_strain = np.sqrt(
                (2.0/3.0) * (
                    dev_xx**2 + dev_yy**2 + dev_zz**2 +
                    0.5 * (strain_xy**2 + strain_xz**2 + strain_yz**2)
                )
            )
            
            # Generate displacement (related to strain)
            displacement_x = np.random.normal(self.config.displacement_mean, self.config.displacement_std)
            displacement_y = np.random.normal(self.config.displacement_mean, self.config.displacement_std)
            displacement_z = np.random.normal(self.config.displacement_mean * 1.2, self.config.displacement_std * 1.2)  # Higher in z
            total_displacement = np.sqrt(displacement_x**2 + displacement_y**2 + displacement_z**2)
            
            # Strain rate
            strain_rate = np.random.normal(self.config.strain_rate_mean, self.config.strain_rate_std)
            
            # Residual stress (estimated from strain using Young's modulus ~200 GPa for steel)
            # Stress = E * strain (simplified, ignoring Poisson's ratio for now)
            youngs_modulus = 200000.0  # MPa (200 GPa)
            residual_stress_xx = youngs_modulus * (strain_xx / 1e6)  # Convert με to strain
            residual_stress_yy = youngs_modulus * (strain_yy / 1e6)
            residual_stress_zz = youngs_modulus * (strain_zz / 1e6)
            
            # Von Mises stress
            mean_stress = (residual_stress_xx + residual_stress_yy + residual_stress_zz) / 3.0
            dev_stress_xx = residual_stress_xx - mean_stress
            dev_stress_yy = residual_stress_yy - mean_stress
            dev_stress_zz = residual_stress_zz - mean_stress
            von_mises_stress = np.sqrt(
                0.5 * (
                    (dev_stress_xx - dev_stress_yy)**2 +
                    (dev_stress_yy - dev_stress_zz)**2 +
                    (dev_stress_zz - dev_stress_xx)**2 +
                    6.0 * (strain_xy**2 + strain_xz**2 + strain_yz**2) * (youngs_modulus / 1e6)**2
                )
            )
            
            # Temperature-compensated strain (simplified: subtract thermal expansion component)
            thermal_expansion_coeff = 12e-6  # 1/K (typical for steel)
            # Assume temperature variation of ~100K during build
            temp_variation = 100.0  # K
            thermal_strain = thermal_expansion_coeff * temp_variation * 1e6  # Convert to με
            temperature_compensated_strain = von_mises_strain - thermal_strain * 0.3  # Partial compensation
            
            # Cumulative strain (increases with layer)
            layer_strain_increment = abs(von_mises_strain) * (1.0 + np.random.normal(0, 0.1))
            cumulative_strain = base_cumulative_strain + layer_strain_increment
            
            # Process stability
            process_stability = np.random.normal(self.config.process_stability_mean, self.config.process_stability_std)
            process_stability = np.clip(process_stability, 0.0, 1.0)
            
            # Strain variation (coefficient of variation)
            strain_variation = abs(np.random.normal(0.15, 0.05))  # 15% typical variation
            
            # Strain uniformity (how uniform strain is across the layer)
            strain_uniformity = np.random.normal(0.80, 0.10)
            strain_uniformity = np.clip(strain_uniformity, 0.0, 1.0)
            
            # Event detection
            excessive_strain_event = abs(von_mises_strain) > self.config.excessive_strain_threshold
            warping_detected = total_displacement > self.config.warping_threshold
            warping_magnitude = total_displacement if warping_detected else None
            distortion_angle = None
            warping_event_detected = False
            distortion_event_detected = False
            anomaly_detected = False
            anomaly_type = None
            
            # Random events
            if np.random.random() < self.config.excessive_strain_probability:
                excessive_strain_event = True
                von_mises_strain = self.config.excessive_strain_threshold * (1.0 + np.random.uniform(0.1, 0.5))
            
            if np.random.random() < self.config.warping_event_probability:
                warping_event_detected = True
                warping_detected = True
                warping_magnitude = self.config.warping_threshold * (1.0 + np.random.uniform(0.2, 1.0))
                distortion_angle = np.random.uniform(0.5, 5.0)  # degrees
                total_displacement = warping_magnitude
            
            if np.random.random() < self.config.distortion_event_probability:
                distortion_event_detected = True
                distortion_angle = np.random.uniform(1.0, 10.0)  # degrees
            
            if np.random.random() < self.config.anomaly_probability:
                anomaly_detected = True
                anomaly_types = ["excessive_warping", "residual_stress_build-up", "layer_delamination", "distortion"]
                anomaly_type = np.random.choice(anomaly_types)
            
            # Process event
            process_event = None
            if i == 0:
                process_event = "layer_start"
            elif warping_event_detected:
                process_event = "warping_event"
            elif distortion_event_detected:
                process_event = "distortion_event"
            elif anomaly_detected:
                process_event = "anomaly_event"
            
            # Strain energy density (J/m³)
            # Strain energy = 0.5 * stress * strain
            strain_energy_density = 0.5 * von_mises_stress * (von_mises_strain / 1e6) * 1e6  # Convert to J/m³
            
            data_point = ISPMStrainDataPoint(
                timestamp=current_time,
                layer_index=layer_index,
                x=x,
                y=y,
                z=z,
                strain_xx=strain_xx,
                strain_yy=strain_yy,
                strain_zz=strain_zz,
                strain_xy=strain_xy,
                strain_xz=strain_xz,
                strain_yz=strain_yz,
                principal_strain_max=principal_strain_max,
                principal_strain_min=principal_strain_min,
                principal_strain_intermediate=principal_strain_intermediate,
                von_mises_strain=von_mises_strain,
                displacement_x=displacement_x,
                displacement_y=displacement_y,
                displacement_z=displacement_z,
                total_displacement=total_displacement,
                strain_rate=strain_rate,
                residual_stress_xx=residual_stress_xx,
                residual_stress_yy=residual_stress_yy,
                residual_stress_zz=residual_stress_zz,
                von_mises_stress=von_mises_stress,
                temperature_compensated_strain=temperature_compensated_strain,
                warping_detected=warping_detected,
                warping_magnitude=warping_magnitude,
                distortion_angle=distortion_angle,
                cumulative_strain=cumulative_strain,
                layer_strain_increment=layer_strain_increment,
                excessive_strain_event=excessive_strain_event,
                warping_event_detected=warping_event_detected,
                distortion_event_detected=distortion_event_detected,
                anomaly_detected=anomaly_detected,
                anomaly_type=anomaly_type,
                process_stability=process_stability,
                strain_variation=strain_variation,
                strain_uniformity=strain_uniformity,
                process_event=process_event,
                strain_energy_density=strain_energy_density,
            )
            
            data_points.append(data_point)
        
        logger.info(f"Generated {len(data_points)} ISPM_Strain data points for layer {layer_index}")
        return data_points
    
    def generate_for_build(self,
                             build_id: str,
                             n_layers: int,
                             layer_thickness: float = 0.05,
                             start_time: Optional[datetime] = None,
                             bounding_box: Optional[Dict[str, Tuple[float, float]]] = None) -> Dict[str, Any]:
        """
        Generate ISPM_Strain data for an entire build.
        
        Args:
            build_id: Build identifier
            n_layers: Number of layers in the build
            layer_thickness: Thickness of each layer (mm)
            start_time: Start time for the build (default: current time)
            bounding_box: Bounding box for spatial distribution {'x': (min, max), 'y': (min, max), 'z': (min, max)}
        
        Returns:
            Dictionary containing build ISPM_Strain data with 'data_points' and 'coordinate_system'
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
                'description': 'ISPM strain sensors are typically positioned at fixed locations on the machine (near build platform)',
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
            'description': 'ISPM strain sensor coordinate system. Measurement coordinates are in build platform space. Sensors are typically positioned at fixed locations on the machine and monitor the process at specific spatial locations.'
        }
        
        logger.info(f"Generated {len(all_data_points)} total ISPM_Strain data points for {n_layers} layers")
        
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
                                        anomaly_type: str = "excessive_warping") -> ISPMStrainDataPoint:
        """
        Generate an anomalous ISPM_Strain data point (for testing anomaly detection).
        
        Args:
            layer_index: Layer index
            layer_z: Z-coordinate
            timestamp: Timestamp
            x: X-coordinate
            y: Y-coordinate
            anomaly_type: Type of anomaly ("excessive_warping", "residual_stress_build-up", "layer_delamination", "distortion")
        
        Returns:
            Anomalous ISPM_Strain data point
        """
        # Base strain values (high)
        strain_xx = np.random.normal(2000.0, 500.0)
        strain_yy = np.random.normal(1800.0, 400.0)
        strain_zz = np.random.normal(2500.0, 600.0)
        strain_xy = np.random.normal(300.0, 100.0)
        strain_xz = np.random.normal(300.0, 100.0)
        strain_yz = np.random.normal(300.0, 100.0)
        
        # Calculate principal strains
        strain_tensor = np.array([
            [strain_xx, strain_xy, strain_xz],
            [strain_xy, strain_yy, strain_yz],
            [strain_xz, strain_yz, strain_zz]
        ])
        eigenvals = np.linalg.eigvals(strain_tensor)
        eigenvals_sorted = np.sort(eigenvals)[::-1]
        principal_strain_max = eigenvals_sorted[0]
        principal_strain_min = eigenvals_sorted[2]
        principal_strain_intermediate = eigenvals_sorted[1]
        
        # High von Mises strain
        mean_strain = (strain_xx + strain_yy + strain_zz) / 3.0
        dev_xx = strain_xx - mean_strain
        dev_yy = strain_yy - mean_strain
        dev_zz = strain_zz - mean_strain
        von_mises_strain = np.sqrt(
            (2.0/3.0) * (
                dev_xx**2 + dev_yy**2 + dev_zz**2 +
                0.5 * (strain_xy**2 + strain_xz**2 + strain_yz**2)
            )
        )
        
        # Anomaly-specific modifications
        warping_event_detected = False
        distortion_event_detected = False
        distortion_angle = None
        
        if anomaly_type == "excessive_warping":
            von_mises_strain = 4000.0  # Very high strain
            displacement_x = np.random.uniform(0.5, 1.0)
            displacement_y = np.random.uniform(0.5, 1.0)
            displacement_z = np.random.uniform(0.3, 0.8)
            warping_detected = True
            warping_magnitude = np.sqrt(displacement_x**2 + displacement_y**2 + displacement_z**2)
            warping_event_detected = True
        elif anomaly_type == "residual_stress_build-up":
            von_mises_strain = 3500.0
            displacement_x = np.random.uniform(0.2, 0.4)
            displacement_y = np.random.uniform(0.2, 0.4)
            displacement_z = np.random.uniform(0.1, 0.3)
            warping_detected = False
            warping_magnitude = None
        elif anomaly_type == "layer_delamination":
            von_mises_strain = 4500.0  # Very high strain
            displacement_x = np.random.uniform(0.3, 0.6)
            displacement_y = np.random.uniform(0.3, 0.6)
            displacement_z = np.random.uniform(0.4, 0.9)  # High z-displacement
            warping_detected = True
            warping_magnitude = np.sqrt(displacement_x**2 + displacement_y**2 + displacement_z**2)
            warping_event_detected = True
        else:  # "distortion"
            von_mises_strain = 3800.0
            displacement_x = np.random.uniform(0.4, 0.7)
            displacement_y = np.random.uniform(0.4, 0.7)
            displacement_z = np.random.uniform(0.2, 0.5)
            warping_detected = True
            warping_magnitude = np.sqrt(displacement_x**2 + displacement_y**2 + displacement_z**2)
            distortion_event_detected = True
            distortion_angle = np.random.uniform(5.0, 15.0)
        
        total_displacement = np.sqrt(displacement_x**2 + displacement_y**2 + displacement_z**2)
        
        # Residual stress
        youngs_modulus = 200000.0  # MPa
        residual_stress_xx = youngs_modulus * (strain_xx / 1e6)
        residual_stress_yy = youngs_modulus * (strain_yy / 1e6)
        residual_stress_zz = youngs_modulus * (strain_zz / 1e6)
        
        mean_stress = (residual_stress_xx + residual_stress_yy + residual_stress_zz) / 3.0
        dev_stress_xx = residual_stress_xx - mean_stress
        dev_stress_yy = residual_stress_yy - mean_stress
        dev_stress_zz = residual_stress_zz - mean_stress
        von_mises_stress = np.sqrt(
            0.5 * (
                (dev_stress_xx - dev_stress_yy)**2 +
                (dev_stress_yy - dev_stress_zz)**2 +
                (dev_stress_zz - dev_stress_xx)**2 +
                6.0 * (strain_xy**2 + strain_xz**2 + strain_yz**2) * (youngs_modulus / 1e6)**2
            )
        )
        
        # Other fields
        strain_rate = np.random.normal(50.0, 10.0)  # High strain rate
        temperature_compensated_strain = von_mises_strain - 500.0  # Partial compensation
        cumulative_strain = layer_index * 50.0 + abs(von_mises_strain)
        layer_strain_increment = abs(von_mises_strain)
        process_stability = np.random.uniform(0.3, 0.5)  # Low stability
        strain_variation = np.random.uniform(0.3, 0.5)  # High variation
        strain_uniformity = np.random.uniform(0.4, 0.6)  # Low uniformity
        excessive_strain_event = True
        anomaly_detected = True
        process_event = "anomaly_event"
        strain_energy_density = 0.5 * von_mises_stress * (von_mises_strain / 1e6) * 1e6
        
        return ISPMStrainDataPoint(
            timestamp=timestamp,
            layer_index=layer_index,
            x=x,
            y=y,
            z=layer_z,
            strain_xx=strain_xx,
            strain_yy=strain_yy,
            strain_zz=strain_zz,
            strain_xy=strain_xy,
            strain_xz=strain_xz,
            strain_yz=strain_yz,
            principal_strain_max=principal_strain_max,
            principal_strain_min=principal_strain_min,
            principal_strain_intermediate=principal_strain_intermediate,
            von_mises_strain=von_mises_strain,
            displacement_x=displacement_x,
            displacement_y=displacement_y,
            displacement_z=displacement_z,
            total_displacement=total_displacement,
            strain_rate=strain_rate,
            residual_stress_xx=residual_stress_xx,
            residual_stress_yy=residual_stress_yy,
            residual_stress_zz=residual_stress_zz,
            von_mises_stress=von_mises_stress,
            temperature_compensated_strain=temperature_compensated_strain,
            warping_detected=warping_detected,
            warping_magnitude=warping_magnitude,
            distortion_angle=distortion_angle,
            cumulative_strain=cumulative_strain,
            layer_strain_increment=layer_strain_increment,
            excessive_strain_event=excessive_strain_event,
            warping_event_detected=warping_event_detected,
            distortion_event_detected=distortion_event_detected,
            anomaly_detected=anomaly_detected,
            anomaly_type=anomaly_type,
            process_stability=process_stability,
            strain_variation=strain_variation,
            strain_uniformity=strain_uniformity,
            process_event=process_event,
            strain_energy_density=strain_energy_density,
        )
