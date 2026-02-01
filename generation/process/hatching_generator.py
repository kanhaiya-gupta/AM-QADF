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
    hatch_spacing: float = 0.15  # mm (0.15mm spacing allows gaps to be visible at 0.1mm voxel resolution)
    hatch_angle: float = 10.0  # degrees (will rotate per layer)
    contour_offset: float = 0.0  # mm
    max_layers: Optional[int] = None  # Limit layers for faster processing (None = generate all layers)
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
        total_layers = len(z_heights)
        if config.max_layers and total_layers > config.max_layers:
            z_heights = z_heights[:config.max_layers]
            logger.info(f"Limited to {config.max_layers} layers (out of {total_layers} total)")
        else:
            logger.info(f"Generating all {total_layers} layers")
        
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
    
    def extract_vectors(self,
                       hatching_layers: List[Any],
                       config: Optional[HatchingConfig] = None) -> List[Dict[str, Any]]:
        """
        Extract vectors directly from pyslm layers in vector-based format.
        
        This matches the structure used by pyslm visualization and provides
        accurate representation for voxelization.
        
        Args:
            hatching_layers: List of pyslm Layer objects
            config: Optional hatching configuration
            
        Returns:
            List of layer dictionaries, each containing:
            - layer_index: Layer index
            - z_position: Z height in mm
            - vectors: List of vector dicts {x1, y1, x2, y2, z, timestamp, dataindex}
            - vectordata: List of vector metadata dicts {dataindex, layer_index, type, laserpower, etc.}
        """
        if config is None:
            config = self.config
        
        base_power = 200.0  # W
        base_velocity = 500.0  # mm/s
        spot_size = 0.1  # mm
        scanner = 0  # Scanner ID (0 for single laser systems)
        
        layer_data = []
        global_dataindex = 0  # Global counter for dataindex across all layers
        
        for layer_idx, layer in enumerate(hatching_layers):
            z_height = layer.z / 1000.0  # Convert from microns to mm
            
            # Debug: Log layer attributes
            if layer_idx == 0:
                logger.info(f"Layer 0 attributes: {dir(layer)}")
                if hasattr(layer, 'geometry'):
                    logger.info(f"Layer 0 geometry type: {type(layer.geometry)}, value: {layer.geometry}")
            
            # Variable parameters (example: higher power in first 20 layers)
            if layer_idx < 20:
                power = base_power * 1.2
                velocity = base_velocity * 0.9
            else:
                power = base_power
                velocity = base_velocity
            
            vectors = []
            vectordata_dict = {}  # Map dataindex to vectordata
            
            # Extract hatch geometry (pairs of start/end points)
            # Try both methods: getHatchGeometry() and direct layer.geometry iteration
            hatch_geoms = []
            
            # Method 1: Use getHatchGeometry() if available
            if hasattr(layer, 'getHatchGeometry'):
                try:
                    hatch_geoms = layer.getHatchGeometry() or []
                    if len(hatch_geoms) == 0:
                        logger.debug(f"Layer {layer_idx}: getHatchGeometry() returned empty list")
                except Exception as e:
                    logger.warning(f"Layer {layer_idx}: Error calling getHatchGeometry(): {e}")
                    pass
            
            # Method 2: Iterate through layer.geometry and filter for HatchGeometry (like visualise.py line 172)
            if len(hatch_geoms) == 0 and hasattr(layer, 'geometry'):
                try:
                    # Import HatchGeometry type check if available
                    try:
                        from pyslm.geometry import HatchGeometry
                        hatch_geoms = [g for g in layer.geometry if isinstance(g, HatchGeometry)]
                    except ImportError:
                        # Fallback: check by attribute (HatchGeometry has coords as pairs)
                        hatch_geoms = []
                        for g in layer.geometry:
                            if hasattr(g, 'coords') and g.coords is not None:
                                coords = g.coords
                                if isinstance(coords, np.ndarray) and len(coords.shape) == 2:
                                    # HatchGeometry typically has even number of points (pairs)
                                    if coords.shape[0] % 2 == 0 and coords.shape[1] == 2:
                                        hatch_geoms.append(g)
                    
                    if len(hatch_geoms) > 0:
                        logger.debug(f"Layer {layer_idx}: Found {len(hatch_geoms)} hatch geometries via layer.geometry")
                except Exception as e:
                    logger.warning(f"Layer {layer_idx}: Error accessing layer.geometry for hatches: {e}")
            
            # Process hatch geometry (reshape to pairs like pyslm does)
            for geom in hatch_geoms:
                try:
                    if not hasattr(geom, 'coords') or geom.coords is None:
                        continue
                    
                    coords = geom.coords
                    # Reshape to pairs: (-1, 2, 2) -> [[start1, end1], [start2, end2], ...]
                    if isinstance(coords, np.ndarray):
                        if len(coords.shape) == 2 and coords.shape[1] == 2:
                            # HatchGeometry stores coords as pairs: [start1, end1, start2, end2, ...]
                            # numHatches() = coords.shape[0] / 2, so we need even number of points
                            if coords.shape[0] % 2 != 0:
                                logger.warning(f"HatchGeometry has odd number of points ({coords.shape[0]}), skipping")
                                continue
                            
                            # Reshape to pairs (like pyslm visualise.py line 286)
                            coords_reshaped = coords.reshape(-1, 2, 2)
                            
                            # Get geometry type and build style
                            geom_type = "(INFILL)"  # Default
                            if hasattr(geom, 'bid'):
                                bid = geom.bid
                                # Could map bid to type if needed
                            
                            # Create dataindex for this geometry group
                            dataindex = global_dataindex
                            global_dataindex += 1
                            
                            # Store vectordata for this geometry group
                            vectordata_dict[dataindex] = {
                                'dataindex': dataindex,
                                'partid': getattr(geom, 'mid', 0),  # Model ID
                                'type': geom_type,
                                'scanner': scanner,
                                'laserpower': float(power),
                                'scannerspeed': float(velocity),
                                'pulseperiod': 1.0,
                                'pulseontime': 1.0,
                                'pulseondelay': 0.0,
                                'expsetindex': 0,
                                'expstepindex': 1,  # Hatch infill
                                'layer_index': layer_idx,
                                'laser_beam_width': config.laser_beam_width,
                                'hatch_spacing': config.hatch_spacing,
                                'overlap_percentage': max(0.0, (config.laser_beam_width - config.hatch_spacing) / config.laser_beam_width * 100.0) if config.laser_beam_width > 0 else 0.0,
                            }
                            
                            # Extract vectors from pairs
                            timestamp = 0.0  # Could calculate based on scan speed
                            for pair in coords_reshaped:
                                x1, y1 = float(pair[0][0]), float(pair[0][1])
                                x2, y2 = float(pair[1][0]), float(pair[1][1])
                                
                                vectors.append({
                                    'x1': x1,
                                    'y1': y1,
                                    'x2': x2,
                                    'y2': y2,
                                    'z': z_height,
                                    'timestamp': timestamp,
                                    'dataindex': dataindex
                                })
                                
                                # Update timestamp based on vector length and scan speed
                                vector_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                                timestamp += vector_length / velocity if velocity > 0 else 0.0
                                
                except Exception as e:
                    logger.debug(f"Error processing hatch geometry: {e}")
                    continue
            
            # Extract contour geometry (continuous sequences)
            # Try both methods: getContourGeometry() and direct layer.geometry iteration
            contour_geoms = []
            
            # Method 1: Use getContourGeometry() if available
            if hasattr(layer, 'getContourGeometry'):
                try:
                    contour_geoms = layer.getContourGeometry() or []
                    if len(contour_geoms) == 0:
                        logger.debug(f"Layer {layer_idx}: getContourGeometry() returned empty list")
                except Exception as e:
                    logger.warning(f"Layer {layer_idx}: Error calling getContourGeometry(): {e}")
                    pass
            
            # Method 2: Iterate through layer.geometry and filter for ContourGeometry (like visualise.py line 176)
            if len(contour_geoms) == 0 and hasattr(layer, 'geometry'):
                try:
                    # Import ContourGeometry type check if available
                    try:
                        from pyslm.geometry import ContourGeometry
                        contour_geoms = [g for g in layer.geometry if isinstance(g, ContourGeometry)]
                    except ImportError:
                        # Fallback: check by attribute (ContourGeometry has coords as sequence, may have subType)
                        contour_geoms = []
                        for g in layer.geometry:
                            if hasattr(g, 'coords') and g.coords is not None:
                                coords = g.coords
                                if isinstance(coords, np.ndarray) and len(coords.shape) == 2:
                                    # ContourGeometry typically has any number of points (sequence)
                                    # Check if it has subType attribute (inner/outer)
                                    if hasattr(g, 'subType') or coords.shape[1] == 2:
                                        contour_geoms.append(g)
                    
                    if len(contour_geoms) > 0:
                        logger.debug(f"Layer {layer_idx}: Found {len(contour_geoms)} contour geometries via layer.geometry")
                except Exception as e:
                    logger.warning(f"Layer {layer_idx}: Error accessing layer.geometry for contours: {e}")
            
            # Process contour geometry (create pairs from continuous sequence)
            for geom in contour_geoms:
                try:
                    if not hasattr(geom, 'coords') or geom.coords is None:
                        continue
                    
                    coords = geom.coords
                    if isinstance(coords, np.ndarray) and len(coords.shape) == 2 and coords.shape[1] == 2:
                        # Create pairs from continuous sequence (like pyslm visualise.py line 177)
                        # For contours: [p0, p1, p2, ..., pn] -> [[p0, p1], [p1, p2], ..., [pn, p0]] (closed loop)
                        # Check if contour is already closed (first and last points are same)
                        is_closed = len(coords) > 0 and np.allclose(coords[0], coords[-1], atol=1e-6)
                        
                        if is_closed:
                            # Already closed, just pair consecutive points
                            coords_rolled = np.roll(coords, -1, axis=0)
                            coords_paired = np.hstack([coords, coords_rolled])[:-1, :].reshape(-1, 2, 2)
                        else:
                            # Not closed, add closing segment: [p0, p1, ..., pn, p0]
                            coords_closed = np.vstack([coords, coords[0:1]])  # Add first point at end
                            coords_rolled = np.roll(coords_closed, -1, axis=0)
                            coords_paired = np.hstack([coords_closed, coords_rolled])[:-1, :].reshape(-1, 2, 2)
                        
                        # Get contour type
                        geom_type = "(CONTOUR)"
                        if hasattr(geom, 'subType'):
                            if geom.subType == 'inner':
                                geom_type = "(INNERCONTOUR)"
                            elif geom.subType == 'outer':
                                geom_type = "(OUTERCONTOUR)"
                        
                        # Create dataindex for this contour
                        dataindex = global_dataindex
                        global_dataindex += 1
                        
                        # Store vectordata for contour
                        vectordata_dict[dataindex] = {
                            'dataindex': dataindex,
                            'partid': getattr(geom, 'mid', 0),
                            'type': geom_type,
                            'scanner': scanner,
                            'laserpower': float(power * 1.6),  # Contours typically higher power
                            'scannerspeed': float(velocity * 0.4),  # Contours typically slower
                            'pulseperiod': 1.0,
                            'pulseontime': 1.0,
                            'pulseondelay': 0.0,
                            'expsetindex': 0,
                            'expstepindex': 3,  # Contour
                            'layer_index': layer_idx,
                            'laser_beam_width': config.laser_beam_width,
                            'hatch_spacing': config.hatch_spacing,
                            'overlap_percentage': 0.0,  # Contours don't have overlap
                        }
                        
                        # Extract vectors from pairs
                        timestamp = 0.0
                        for pair in coords_paired:
                            x1, y1 = float(pair[0][0]), float(pair[0][1])
                            x2, y2 = float(pair[1][0]), float(pair[1][1])
                            
                            vectors.append({
                                'x1': x1,
                                'y1': y1,
                                'x2': x2,
                                'y2': y2,
                                'z': z_height,
                                'timestamp': timestamp,
                                'dataindex': dataindex
                            })
                            
                            # Update timestamp
                            vector_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                            contour_velocity = velocity * 0.4  # Contours are slower
                            timestamp += vector_length / contour_velocity if contour_velocity > 0 else 0.0
                            
                except Exception as e:
                    logger.debug(f"Error processing contour geometry: {e}")
                    continue
            
            # Convert vectordata_dict to list
            vectordata = list(vectordata_dict.values())
            
            if len(vectors) == 0:
                logger.warning(f"Layer {layer_idx}: No vectors extracted (hatch_geoms: {len(hatch_geoms)}, contour_geoms: {len(contour_geoms)})")
            
            layer_data.append({
                'layer_index': layer_idx,
                'z_position': z_height,
                'vectors': vectors,
                'vectordata': vectordata,
                'n_vectors': len(vectors),
                'n_vectordata': len(vectordata)
            })
        
        total_vectors = sum(layer['n_vectors'] for layer in layer_data)
        logger.info(f"Extracted {total_vectors} total vectors from {len(layer_data)} layers")
        
        return layer_data

