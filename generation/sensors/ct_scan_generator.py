"""
CT (Computed Tomography) Scan Data Generator

Generates realistic CT scan data including:
- 3D voxel grids
- Density maps
- Porosity maps
- Defect locations
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class CTScanVoxelGrid:
    """CT scan voxel grid data."""
    dimensions: Tuple[int, int, int]  # (nx, ny, nz)
    spacing: Tuple[float, float, float]  # (dx, dy, dz) in mm
    origin: Tuple[float, float, float]  # (x0, y0, z0) in mm
    density_values: np.ndarray  # Density in g/cm³
    porosity_map: np.ndarray  # Porosity (0-1)
    defect_map: np.ndarray  # Binary defect map (0=normal, 1=defect)
    rotation: Optional[Tuple[float, float, float]] = None  # (rx, ry, rz) in degrees - optional, for rotated scans


@dataclass
class CTScanGeneratorConfig:
    """Configuration for CT scan data generation."""
    # Voxel grid parameters
    grid_dimensions: Tuple[int, int, int] = (200, 200, 200)  # (nx, ny, nz)
    voxel_spacing: Tuple[float, float, float] = (0.1, 0.1, 0.1)  # mm
    
    # Material properties
    base_density: float = 8.0  # g/cm³ (Ti6Al4V)
    density_variation: float = 0.2  # g/cm³
    base_porosity: float = 0.01  # 1% porosity
    porosity_variation: float = 0.005  # 0.5% variation
    
    # Defect parameters
    defect_probability: float = 0.01  # 1% of voxels are defects
    defect_size_range: Tuple[int, int] = (2, 10)  # Voxels
    defect_density_reduction: float = 0.3  # Defects have 30% lower density
    
    # Scan quality
    noise_level: float = 0.05  # 5% noise
    
    # Random seed
    random_seed: Optional[int] = None


class CTScanGenerator:
    """
    Generator for CT (Computed Tomography) scan data.
    
    Creates realistic 3D voxel grids with density, porosity, and defects.
    """
    
    def __init__(self, config: Optional[CTScanGeneratorConfig] = None):
        """
        Initialize CT scan generator.
        
        Args:
            config: Configuration for data generation
        """
        self.config = config or CTScanGeneratorConfig()
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
        
        logger.info(f"CTScanGenerator initialized with config: {self.config}")
    
    def generate_voxel_grid(self,
                           dimensions: Optional[Tuple[int, int, int]] = None,
                           spacing: Optional[Tuple[float, float, float]] = None,
                           origin: Optional[Tuple[float, float, float]] = None,
                           rotation: Optional[Tuple[float, float, float]] = None) -> CTScanVoxelGrid:
        """
        Generate a CT scan voxel grid.
        
        Args:
            dimensions: Grid dimensions (nx, ny, nz)
            spacing: Voxel spacing (dx, dy, dz) in mm
            origin: Grid origin (x0, y0, z0) in mm
            rotation: Optional rotation angles (rx, ry, rz) in degrees
            
        Returns:
            CTScanVoxelGrid object
        """
        if dimensions is None:
            dimensions = self.config.grid_dimensions
        if spacing is None:
            spacing = self.config.voxel_spacing
        if origin is None:
            origin = (0.0, 0.0, 0.0)
        
        nx, ny, nz = dimensions
        
        # Generate base density map
        density = np.random.normal(
            self.config.base_density,
            self.config.density_variation,
            size=dimensions
        )
        
        # Generate base porosity map
        porosity = np.random.normal(
            self.config.base_porosity,
            self.config.porosity_variation,
            size=dimensions
        )
        porosity = np.clip(porosity, 0.0, 1.0)  # Ensure valid range
        
        # Generate defects
        defect_map = np.zeros(dimensions, dtype=bool)
        n_defects = int(np.prod(dimensions) * self.config.defect_probability)
        
        for _ in range(n_defects):
            # Random defect center
            cx = np.random.randint(0, nx)
            cy = np.random.randint(0, ny)
            cz = np.random.randint(0, nz)
            
            # Random defect size
            defect_size = np.random.randint(
                self.config.defect_size_range[0],
                self.config.defect_size_range[1] + 1
            )
            
            # Create spherical defect
            radius = defect_size / 2.0
            for i in range(max(0, int(cx - radius)), min(nx, int(cx + radius) + 1)):
                for j in range(max(0, int(cy - radius)), min(ny, int(cy + radius) + 1)):
                    for k in range(max(0, int(cz - radius)), min(nz, int(cz + radius) + 1)):
                        dist = np.sqrt((i - cx)**2 + (j - cy)**2 + (k - cz)**2)
                        if dist <= radius:
                            defect_map[i, j, k] = True
                            # Reduce density in defect region
                            density[i, j, k] *= (1.0 - self.config.defect_density_reduction)
                            # Increase porosity in defect region
                            porosity[i, j, k] = min(1.0, porosity[i, j, k] + 0.1)
        
        # Add noise
        density += np.random.normal(0, self.config.noise_level * self.config.base_density, size=dimensions)
        density = np.clip(density, 0.0, None)  # Ensure non-negative
        
        # Convert defect map to binary array
        defect_map_binary = defect_map.astype(int)
        
        return CTScanVoxelGrid(
            dimensions=dimensions,
            spacing=spacing,
            origin=origin,
            density_values=density,
            porosity_map=porosity,
            defect_map=defect_map_binary,
            rotation=rotation
        )
    
    def generate_for_build(self,
                          build_id: str,
                          bounding_box: Optional[Dict[str, Tuple[float, float]]] = None) -> Dict[str, Any]:
        """
        Generate CT scan data for a build.
        
        Args:
            build_id: Build identifier
            bounding_box: Optional bounding box {'x': (min, max), 'y': (min, max), 'z': (min, max)}
            
        Returns:
            Dictionary containing CT scan data
        """
        # Determine grid parameters from bounding box or use defaults
        if bounding_box:
            x_range = bounding_box['x'][1] - bounding_box['x'][0]
            y_range = bounding_box['y'][1] - bounding_box['y'][0]
            z_range = bounding_box['z'][1] - bounding_box['z'][0]
            
            dx, dy, dz = self.config.voxel_spacing
            dimensions = (
                int(x_range / dx),
                int(y_range / dx),
                int(z_range / dz)
            )
            origin = (bounding_box['x'][0], bounding_box['y'][0], bounding_box['z'][0])
        else:
            dimensions = self.config.grid_dimensions
            origin = (0.0, 0.0, 0.0)
        
        voxel_grid = self.generate_voxel_grid(
            dimensions=dimensions,
            origin=origin,
            rotation=None  # Typically CT scans are aligned with build platform, but can be set if needed
        )
        
        # Calculate statistics
        n_defects = np.sum(voxel_grid.defect_map)
        mean_density = np.mean(voxel_grid.density_values)
        mean_porosity = np.mean(voxel_grid.porosity_map)
        defect_locations = np.argwhere(voxel_grid.defect_map == 1).tolist()
        
        # Calculate bounding box from voxel grid
        nx, ny, nz = dimensions
        dx, dy, dz = self.config.voxel_spacing
        x0, y0, z0 = origin
        bbox_max = (x0 + nx * dx, y0 + ny * dy, z0 + nz * dz)
        
        # Create coordinate system information (similar to STL)
        coordinate_system = {
            'type': 'ct_scan',  # CT scan coordinate system
            'origin': {
                'x': float(origin[0]),
                'y': float(origin[1]),
                'z': float(origin[2])
            },
            'rotation': {
                'x_deg': float(voxel_grid.rotation[0]) if voxel_grid.rotation else 0.0,
                'y_deg': float(voxel_grid.rotation[1]) if voxel_grid.rotation else 0.0,
                'z_deg': float(voxel_grid.rotation[2]) if voxel_grid.rotation else 0.0
            },
            'scale_factor': {
                'x': 1.0,  # CT scans typically not scaled
                'y': 1.0,
                'z': 1.0
            },
            'voxel_spacing': {
                'x': float(dx),
                'y': float(dy),
                'z': float(dz)
            },
            'grid_dimensions': {
                'nx': int(nx),
                'ny': int(ny),
                'nz': int(nz)
            },
            'bounding_box': {
                'min': [float(x0), float(y0), float(z0)],
                'max': [float(bbox_max[0]), float(bbox_max[1]), float(bbox_max[2])]
            },
            'description': 'CT scan coordinate system. Origin is the corner of the voxel grid. Voxel spacing defines the physical size of each voxel. Typically aligned with build platform coordinates.'
        }
        
        return {
            'build_id': build_id,
            'scan_id': f"scan_{build_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'scan_timestamp': datetime.now(),
            'voxel_grid': voxel_grid,
            'coordinate_system': coordinate_system,  # Critical for merging with STL/hatching data
            'statistics': {
                'mean_density': float(mean_density),
                'mean_porosity': float(mean_porosity),
                'n_defects': int(n_defects),
                'defect_percentage': float(n_defects / np.prod(dimensions) * 100),
                'dimensions': dimensions,
                'spacing': self.config.voxel_spacing,
                'origin': origin
            },
            'defect_locations': defect_locations,
            'metadata': {
                'generator_config': self.config,
                'generated_at': datetime.now().isoformat()
            }
        }
    
    def add_porosity_cluster(self,
                            voxel_grid: CTScanVoxelGrid,
                            center: Tuple[int, int, int],
                            radius: int = 5,
                            porosity_increase: float = 0.2) -> CTScanVoxelGrid:
        """
        Add a porosity cluster to an existing voxel grid.
        
        Args:
            voxel_grid: Existing voxel grid
            center: Center of cluster (x, y, z) in voxel coordinates
            radius: Radius of cluster in voxels
            porosity_increase: Additional porosity to add
            
        Returns:
            Modified voxel grid
        """
        cx, cy, cz = center
        nx, ny, nz = voxel_grid.dimensions
        
        for i in range(max(0, cx - radius), min(nx, cx + radius + 1)):
            for j in range(max(0, cy - radius), min(ny, cy + radius + 1)):
                for k in range(max(0, cz - radius), min(nz, cz + radius + 1)):
                    dist = np.sqrt((i - cx)**2 + (j - cy)**2 + (k - cz)**2)
                    if dist <= radius:
                        # Increase porosity
                        voxel_grid.porosity_map[i, j, k] = min(
                            1.0,
                            voxel_grid.porosity_map[i, j, k] + porosity_increase
                        )
                        # Decrease density
                        voxel_grid.density_values[i, j, k] *= (1.0 - porosity_increase * 0.5)
                        # Mark as defect
                        voxel_grid.defect_map[i, j, k] = 1
        
        return voxel_grid

