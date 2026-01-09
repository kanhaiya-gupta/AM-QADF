"""
Hatching Path Generator

Generates hatching paths using pyslm library:
- Layer slicing
- Contour generation
- Hatch pattern generation (raster, stripe, chessboard)
- Support structure generation

Uses the same approach as Complete_Integration_Example.ipynb
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Try to import pyslm
try:
    import pyslm
    from pyslm import hatching
    PYSLM_AVAILABLE = True
except ImportError:
    PYSLM_AVAILABLE = False
    logger.warning("pyslm not available. Hatching generation will not work.")


@dataclass
class HatchingConfig:
    """Configuration for hatching generation."""
    layer_thickness: float = 0.05  # mm
    hatch_spacing: float = 0.1  # mm
    hatch_angle: float = 10.0  # degrees (will rotate per layer)
    contour_offset: float = 0.0  # mm
    max_layers: Optional[int] = 50  # Limit layers for faster processing
    sampling_spacing: float = 1.0  # mm (for point extraction)
    laser_beam_width: float = 0.1  # mm - critical for overlap calculation


class HatchingGenerator:
    """
    Generator for hatching paths using pyslm.
    
    Generates layer-by-layer scan paths for PBF-LB/M processes.
    Uses the same approach as Complete_Integration_Example.ipynb.
    """
    
    def __init__(self, config: Optional[HatchingConfig] = None):
        """
        Initialize hatching generator.
        
        Args:
            config: Hatching configuration
        """
        self.config = config or HatchingConfig()
        
        if not PYSLM_AVAILABLE:
            logger.warning("pyslm not available. Hatching generation will not work.")
        else:
            # Initialize hatcher (same as notebook)
            self.hatcher = hatching.Hatcher()
            self.hatcher.hatchDistance = self.config.hatch_spacing
            self.hatcher.hatchAngle = self.config.hatch_angle
            self.hatcher.hatchSortMethod = hatching.AlternateSort()
        
        logger.info("HatchingGenerator initialized")
    
    def generate_hatching(self,
                         stl_part: Any,  # pyslm.Part object
                         config: Optional[HatchingConfig] = None) -> Dict[str, Any]:
        """
        Generate hatching paths for an STL model.
        
        Args:
            stl_part: pyslm.Part object (loaded STL)
            config: Optional hatching configuration (overrides instance config)
            
        Returns:
            Dictionary containing hatching layers and extracted points
        """
        if not PYSLM_AVAILABLE:
            raise ImportError("pyslm is required for hatching generation")
        
        if config is None:
            config = self.config
        
        # Update hatcher config
        self.hatcher.hatchDistance = config.hatch_spacing
        self.hatcher.hatchAngle = config.hatch_angle
        
        # Get bounding box
        bbox = stl_part.boundingBox
        z_max = bbox[5]  # Max Z
        
        # Generate layer heights
        z_heights = np.arange(0, z_max, config.layer_thickness)
        
        # Limit layers if specified
        if config.max_layers and len(z_heights) > config.max_layers:
            z_heights = z_heights[:config.max_layers]
            logger.info(f"Limited to {config.max_layers} layers (out of {len(z_heights)} total)")
        
        hatching_layers = []
        
        # Generate hatching for each layer
        for i, z in enumerate(z_heights):
            # Rotate hatch angle per layer (same as notebook)
            self.hatcher.hatchAngle = 10 + (i * 66.7) % 180
            
            # Slice the STL at this Z height
            geom_slice = stl_part.getVectorSlice(z)
            
            if geom_slice:
                # Generate hatching
                layer = self.hatcher.hatch(geom_slice)
                if layer is not None:
                    layer.z = int(z * 1000)  # Convert to microns
                    
                    # Check if layer has geometry
                    has_geometry = False
                    if hasattr(layer, 'geometry'):
                        if isinstance(layer.geometry, list) and len(layer.geometry) > 0:
                            has_geometry = True
                        elif hasattr(layer.geometry, '__iter__'):
                            try:
                                if len(list(layer.geometry)) > 0:
                                    has_geometry = True
                            except:
                                pass
                    
                    if has_geometry:
                        hatching_layers.append(layer)
        
        logger.info(f"✅ Generated {len(hatching_layers)} layers with hatching")
        
        # Extract scan points from hatching (same approach as notebook)
        all_points, all_power, all_velocity, all_energy = self._extract_scan_points(
            hatching_layers,
            config.sampling_spacing
        )
        
        # Calculate overlap percentage: (beam_width - hatch_spacing) / beam_width * 100
        # This is critical for lack of fusion analysis and part density
        if config.laser_beam_width > 0:
            overlap_percentage = max(0.0, (config.laser_beam_width - config.hatch_spacing) / config.laser_beam_width * 100.0)
            overlap_ratio = overlap_percentage / 100.0
        else:
            overlap_percentage = 0.0
            overlap_ratio = 0.0
        
        # Extract coordinate system information from stl_part (critical for merging with other data)
        coordinate_system = None
        if hasattr(stl_part, 'origin') and hasattr(stl_part, 'rotation'):
            origin = stl_part.origin
            rotation = stl_part.rotation
            scale_factor = getattr(stl_part, 'scaleFactor', np.array([1.0, 1.0, 1.0]))
            bbox = stl_part.boundingBox
            
            coordinate_system = {
                'type': 'build_platform',  # Machine coordinate system
                'origin': {
                    'x': float(origin[0]),
                    'y': float(origin[1]),
                    'z': float(origin[2])
                },
                'rotation': {
                    'x_deg': float(rotation[0]),
                    'y_deg': float(rotation[1]),
                    'z_deg': float(rotation[2])
                },
                'scale_factor': {
                    'x': float(scale_factor[0]),
                    'y': float(scale_factor[1]),
                    'z': float(scale_factor[2])
                },
                'bounding_box': {
                    'min': [float(bbox[0]), float(bbox[1]), float(bbox[2])],
                    'max': [float(bbox[3]), float(bbox[4]), float(bbox[5])]
                }
            }
        
        return {
            'model_id': getattr(stl_part, 'name', 'unknown'),
            'layers': hatching_layers,
            'points': all_points,
            'power': all_power,
            'velocity': all_velocity,
            'energy': all_energy,
            'metadata': {
                'n_layers': len(hatching_layers),
                'layer_thickness': config.layer_thickness,
                'hatch_spacing': config.hatch_spacing,
                'laser_beam_width': config.laser_beam_width,
                'overlap_percentage': overlap_percentage,
                'overlap_ratio': overlap_ratio,
                'n_points': len(all_points) if all_points is not None else 0,
                'coordinate_system': coordinate_system  # Critical for merging with ISPM/CT data
            }
        }
    
    def _extract_scan_points(self,
                            hatching_layers: List[Any],
                            sampling_spacing: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract scan points from hatching layers (same approach as notebook).
        
        Args:
            hatching_layers: List of hatching layer objects
            sampling_spacing: Spacing for point sampling (mm)
            
        Returns:
            Tuple of (points, power, velocity, energy) arrays
        """
        all_points = []
        all_power = []
        all_velocity = []
        all_energy = []
        
        base_power = 200.0  # W
        base_velocity = 500.0  # mm/s
        spot_size = 0.1  # mm
        
        for layer_idx, layer in enumerate(hatching_layers):
            z_height = layer.z / 1000.0  # Convert from microns to mm
            
            # Variable parameters (example: higher power in first 20 layers)
            if layer_idx < 20:
                power = base_power * 1.2
                velocity = base_velocity * 0.9
            else:
                power = base_power
                velocity = base_velocity
            
            # Extract geometries
            geometries = []
            if hasattr(layer, 'getHatchGeometry'):
                try:
                    geometries = layer.getHatchGeometry() or []
                except:
                    pass
            
            if not geometries and hasattr(layer, 'geometry'):
                if isinstance(layer.geometry, list):
                    geometries = layer.geometry
                else:
                    try:
                        geometries = list(layer.geometry) if hasattr(layer.geometry, '__iter__') else []
                    except:
                        pass
            
            # Process each geometry
            for geom in geometries:
                try:
                    if not hasattr(geom, 'coords') or geom.coords is None:
                        continue
                    
                    coords = geom.coords
                    
                    if isinstance(coords, np.ndarray) and len(coords.shape) == 2 and coords.shape[1] >= 2:
                        # Array of (x, y) points
                        for i in range(len(coords) - 1):
                            x1, y1 = coords[i][:2]
                            x2, y2 = coords[i + 1][:2]
                            
                            segment_length = np.linalg.norm([x2-x1, y2-y1])
                            if segment_length > 0:
                                num_samples = max(2, int(segment_length / sampling_spacing))
                                x_samples = np.linspace(x1, x2, num_samples)
                                y_samples = np.linspace(y1, y2, num_samples)
                                
                                for x, y in zip(x_samples, y_samples):
                                    all_points.append([x, y, z_height])
                                    all_power.append(power)
                                    all_velocity.append(velocity)
                                    energy = power / (velocity * spot_size) if velocity > 0 else 0
                                    all_energy.append(energy)
                except Exception as e:
                    logger.debug(f"Error processing geometry: {e}")
                    continue
        
        # Convert to numpy arrays
        points = np.array(all_points) if all_points else np.array([]).reshape(0, 3)
        power = np.array(all_power) if all_power else np.array([])
        velocity = np.array(all_velocity) if all_velocity else np.array([])
        energy = np.array(all_energy) if all_energy else np.array([])
        
        return points, power, velocity, energy

