"""
Data Generation Orchestrator

Orchestrates generation of all data types for a complete build dataset.
"""

from typing import Dict, Optional, Any, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Import generators
try:
    from ..process.stl_processor import STLProcessor
    from ..process.hatching_generator import HatchingGenerator, HatchingConfig
    from ..sensors.laser_parameter_generator import LaserParameterGenerator
    from ..sensors.ispm_thermal_generator import ISPMThermalGenerator
    from ..sensors.ct_scan_generator import CTScanGenerator
    GENERATORS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Could not import generators: {e}")
    GENERATORS_AVAILABLE = False


def generate_all_data(n_models: Optional[int] = None,
                     output_dir: str = "generated_data",
                     stl_files: Optional[List[str]] = None,
                     config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Generate all data types for multiple builds from STL files.
    
    Args:
        n_models: Number of STL models to process (None = process all available)
        output_dir: Output directory for generated data
        stl_files: Optional list of specific STL filenames to process
        config: Optional configuration dictionary
        
    Returns:
        Dictionary containing generation summary
    """
    if not GENERATORS_AVAILABLE:
        return {
            'error': 'Generators not available',
            'message': 'Could not import required generators'
        }
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize processors and generators
    stl_processor = STLProcessor()
    hatching_gen = HatchingGenerator()
    laser_gen = LaserParameterGenerator()
    ispm_gen = ISPMThermalGenerator()
    ct_gen = CTScanGenerator()
    
    # Find STL files
    if stl_files:
        # Use specified files
        stl_paths = [stl_processor.get_stl_file(f) for f in stl_files]
        stl_paths = [p for p in stl_paths if p is not None]
    else:
        # Find all STL files
        stl_paths = stl_processor.find_stl_files()
        if n_models:
            stl_paths = stl_paths[:n_models]
    
    if not stl_paths:
        return {
            'error': 'No STL files found',
            'message': 'No STL files available in models directory'
        }
    
    logger.info(f"Found {len(stl_paths)} STL files to process")
    
    results = []
    
    # Process each STL file
    for i, stl_path in enumerate(stl_paths):
        try:
            logger.info(f"Processing {i+1}/{len(stl_paths)}: {stl_path.name}")
            
            # 1. Process STL file
            stl_metadata = stl_processor.process_stl_file(str(stl_path))
            model_id = stl_metadata.get('model_name', stl_path.stem)
            
            # 2. Generate hatching (requires pyslm Part object)
            # Note: This would need the STL loaded as pyslm.Part first
            # For now, we'll skip hatching generation in the orchestrator
            # and let individual scripts handle it
            
            # 3. Generate sensor data based on STL metadata
            bbox = stl_metadata.get('bounding_box', {})
            if bbox:
                n_layers = int((bbox['max'][2] - bbox['min'][2]) / 0.05)  # Estimate layers
                
                # Generate ISPM data
                ispm_data = ispm_gen.generate_for_build(
                    build_id=model_id,
                    n_layers=min(n_layers, 100)  # Limit for demo
                )
                
                # Generate CT scan data
                ct_data = ct_gen.generate_for_build(
                    build_id=model_id,
                    bounding_box=bbox
                )
                
                # Generate laser parameters
                laser_data = laser_gen.generate_for_build(
                    build_id=model_id,
                    n_layers=min(n_layers, 100),
                    points_per_layer=1000
                )
            
            results.append({
                'model_id': model_id,
                'stl_file': stl_path.name,
                'stl_metadata': stl_metadata,
                'ispm_data': ispm_data if 'ispm_data' in locals() else None,
                'ct_data': ct_data if 'ct_data' in locals() else None,
                'laser_data': laser_data if 'laser_data' in locals() else None
            })
            
        except Exception as e:
            logger.error(f"Error processing {stl_path.name}: {e}")
            results.append({
                'model_id': stl_path.stem,
                'stl_file': stl_path.name,
                'error': str(e)
            })
    
    return {
        'n_models': len(stl_paths),
        'n_processed': len([r for r in results if 'error' not in r]),
        'output_dir': str(output_path),
        'results': results,
        'generated': True
    }

