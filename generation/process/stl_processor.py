"""
STL File Processor

Processes STL files and extracts metadata:
- Geometric properties (bounding box, volume, surface area)
- Complexity metrics
- Mesh validation

Uses pyslm for STL loading (same approach as Complete_Integration_Example.ipynb)
"""

from typing import Dict, Optional, Any, Tuple, List
from pathlib import Path
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Try to import models utilities
try:
    from ..models import get_stl_files, get_stl_file, MODELS_DIR
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    MODELS_DIR = None

# Try to import pyslm
try:
    import pyslm
    PYSLM_AVAILABLE = True
except ImportError:
    PYSLM_AVAILABLE = False
    logger.warning("pyslm not available. STL processing will be limited.")


class STLProcessor:
    """
    Processor for STL files.
    
    Processes STL files and extracts metadata for storage in data warehouse.
    Uses pyslm for STL loading (same approach as Complete_Integration_Example.ipynb).
    """
    
    def __init__(self, models_dir: Optional[Path] = None):
        """
        Initialize STL processor.
        
        Args:
            models_dir: Optional directory containing STL files (defaults to data_generation/models/)
        """
        if not PYSLM_AVAILABLE:
            logger.warning("pyslm not available. Some features may be limited.")
        
        self.models_dir = models_dir
        if self.models_dir is None and MODELS_AVAILABLE:
            self.models_dir = MODELS_DIR
        
        logger.info("STLProcessor initialized")
        if self.models_dir:
            logger.info(f"Models directory: {self.models_dir}")
    
    def find_stl_files(self, pattern: Optional[str] = None) -> List[Path]:
        """
        Find STL files in the models directory.
        
        Args:
            pattern: Optional pattern to filter filenames
            
        Returns:
            List of STL file paths
        """
        if self.models_dir is None:
            logger.warning("Models directory not set. Cannot find STL files.")
            return []
        
        stl_files = list(self.models_dir.glob("*.stl"))
        
        if pattern:
            stl_files = [f for f in stl_files if pattern in f.name]
        
        return sorted(stl_files)
    
    def get_stl_file(self, filename: str) -> Optional[Path]:
        """
        Get a specific STL file by filename.
        
        Args:
            filename: STL filename (with or without .stl extension)
            
        Returns:
            Path to STL file, or None if not found
        """
        if self.models_dir is None:
            logger.warning("Models directory not set. Cannot find STL file.")
            return None
        
        if not filename.endswith('.stl'):
            filename += '.stl'
        
        stl_path = self.models_dir / filename
        return stl_path if stl_path.exists() else None
    
    def process_stl_file(self, stl_path: str, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Process an STL file and extract metadata using pyslm.
        
        Args:
            stl_path: Path to STL file
            model_name: Optional model name (defaults to filename)
            
        Returns:
            Dictionary containing STL metadata
        """
        stl_file = Path(stl_path)
        if not stl_file.exists():
            raise FileNotFoundError(f"STL file not found: {stl_path}")
        
        if model_name is None:
            model_name = stl_file.stem
        
        metadata = {
            'filename': stl_file.name,
            'file_path': str(stl_file),
            'model_name': model_name,
            'bounding_box': None,
            'volume': None,
            'surface_area': None,
            'num_triangles': None,
            'complexity_score': None
        }
        
        if PYSLM_AVAILABLE:
            try:
                # Load STL using pyslm (same approach as notebook)
                stl_part = pyslm.Part(model_name)
                stl_part.setGeometry(str(stl_file))
                stl_part.origin = [0.0, 0.0, 0.0]
                stl_part.rotation = [0, 0, 0]
                stl_part.dropToPlatform()
                
                # Extract bounding box
                bbox = stl_part.boundingBox
                metadata['bounding_box'] = {
                    'min': [float(bbox[0]), float(bbox[1]), float(bbox[2])],
                    'max': [float(bbox[3]), float(bbox[4]), float(bbox[5])]
                }
                
                # Calculate dimensions
                dimensions = [
                    bbox[3] - bbox[0],  # x
                    bbox[4] - bbox[1],  # y
                    bbox[5] - bbox[2]   # z
                ]
                metadata['dimensions'] = dimensions
                
                # Extract coordinate system information (critical for merging with other data sources)
                # After dropToPlatform(), origin and rotation reflect the actual build position
                origin = stl_part.origin
                rotation = stl_part.rotation
                scale_factor = stl_part.scaleFactor
                
                metadata['coordinate_system'] = {
                    'type': 'build_platform',  # Machine coordinate system (after dropToPlatform)
                    'origin': {
                        'x': float(origin[0]),
                        'y': float(origin[1]),
                        'z': float(origin[2])
                    },
                    'rotation': {
                        'x_deg': float(rotation[0]),  # Rotation about X-axis (degrees)
                        'y_deg': float(rotation[1]),  # Rotation about Y-axis (degrees)
                        'z_deg': float(rotation[2])   # Rotation about Z-axis (degrees)
                    },
                    'scale_factor': {
                        'x': float(scale_factor[0]),
                        'y': float(scale_factor[1]),
                        'z': float(scale_factor[2])
                    },
                    'bounding_box': {
                        'min': [float(bbox[0]), float(bbox[1]), float(bbox[2])],
                        'max': [float(bbox[3]), float(bbox[4]), float(bbox[5])]
                    },
                    'description': 'Build platform coordinate system. Origin is the part position on the build platform after dropToPlatform(). Rotation is applied in order: X, Y, Z (degrees).'
                }
                
                # Try to get geometry info
                if hasattr(stl_part, 'geometry') and stl_part.geometry is not None:
                    try:
                        # Try trimesh if available
                        import trimesh
                        if isinstance(stl_part.geometry, trimesh.Trimesh):
                            mesh = stl_part.geometry
                            metadata['volume'] = float(mesh.volume)
                            metadata['surface_area'] = float(mesh.area)
                            metadata['num_triangles'] = len(mesh.faces)
                            
                            # Complexity score (simple heuristic)
                            if metadata['volume'] > 0:
                                complexity = metadata['num_triangles'] / (metadata['volume'] ** (2/3))
                                metadata['complexity_score'] = float(complexity)
                    except ImportError:
                        logger.warning("trimesh not available. Volume/surface area calculation skipped.")
                    except Exception as e:
                        logger.warning(f"Could not extract geometry info: {e}")
                
                logger.info(f"✅ STL processed: {model_name}")
                logger.info(f"   Bounding box: {metadata['bounding_box']}")
                
            except Exception as e:
                logger.error(f"Error processing STL with pyslm: {e}")
                raise
        else:
            logger.warning("pyslm not available. Using fallback metadata.")
            # Fallback: basic file info only
            metadata['bounding_box'] = {
                'min': [0.0, 0.0, 0.0],
                'max': [100.0, 100.0, 100.0]
            }
        
        return metadata

