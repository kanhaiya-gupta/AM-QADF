"""
Populate MongoDB Directly from Data Generation

This script generates data and directly populates MongoDB collections.
For demo/testing purposes - bypasses the ingestion layer.
"""

import sys
import os
from pathlib import Path
import logging
from typing import Dict, Optional, Any, List
from dataclasses import asdict, is_dataclass
import numpy as np
from datetime import datetime
import uuid
import io
import gzip
import time
import importlib.util

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
env_file = project_root / 'development.env'
if env_file.exists():
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                value = value.strip('"\'')
                os.environ[key] = value

# Import MongoDB client
try:
    from src.infrastructure.config import MongoDBConfig
    from src.infrastructure.database import MongoDBClient
    
    # Create a get_mongodb_config function for compatibility
    def get_mongodb_config():
        """Get MongoDB config from environment."""
        return MongoDBConfig.from_env()
except Exception as e:
    print(f"❌ Error loading MongoDB client: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Import data generators
try:
    # Add generation to path for absolute imports
    data_gen_dir = project_root / 'generation'
    sys.path.insert(0, str(project_root))
    
    # Set up package structure for relative imports to work
    import types
    if 'generation' not in sys.modules:
        data_gen_module = types.ModuleType('generation')
        data_gen_module.__path__ = [str(data_gen_dir)]
        sys.modules['generation'] = data_gen_module
    
    if 'generation.process' not in sys.modules:
        process_module = types.ModuleType('generation.process')
        process_module.__path__ = [str(data_gen_dir / 'process')]
        process_module.__package__ = 'generation.process'
        sys.modules['generation.process'] = process_module
    
    if 'generation.sensors' not in sys.modules:
        sensors_module = types.ModuleType('generation.sensors')
        sensors_module.__path__ = [str(data_gen_dir / 'sensors')]
        sensors_module.__package__ = 'generation.sensors'
        sys.modules['generation.sensors'] = sensors_module
    
    if 'generation.models' not in sys.modules:
        models_module = types.ModuleType('generation.models')
        models_module.__path__ = [str(data_gen_dir / 'models')]
        models_module.__package__ = 'generation.models'
        sys.modules['generation.models'] = models_module
    
    # Try absolute imports
    try:
        from generation.process.stl_processor import STLProcessor
        from generation.process.hatching_generator import HatchingGenerator
        from generation.sensors.laser_parameter_generator import LaserParameterGenerator
        from generation.sensors.ispm_generator import ISPMGenerator
        from generation.sensors.ct_scan_generator import CTScanGenerator
    except ImportError as e:
        # Fallback to direct module loading with proper package setup
        stl_processor_path = data_gen_dir / 'process' / 'stl_processor.py'
        hatching_gen_path = data_gen_dir / 'process' / 'hatching_generator.py'
        laser_gen_path = data_gen_dir / 'sensors' / 'laser_parameter_generator.py'
        ispm_gen_path = data_gen_dir / 'sensors' / 'ispm_generator.py'
        ct_gen_path = data_gen_dir / 'sensors' / 'ct_scan_generator.py'
        
        # Load models module first (needed by stl_processor)
        models_path = data_gen_dir / 'models' / '__init__.py'
        if models_path.exists():
            spec_models = importlib.util.spec_from_file_location("generation.models", models_path)
            models_module = importlib.util.module_from_spec(spec_models)
            spec_models.loader.exec_module(models_module)
            sys.modules['generation.models'] = models_module
        
        spec_stl = importlib.util.spec_from_file_location("generation.process.stl_processor", stl_processor_path)
        stl_module = importlib.util.module_from_spec(spec_stl)
        stl_module.__package__ = 'generation.process'
        spec_stl.loader.exec_module(stl_module)
        STLProcessor = stl_module.STLProcessor
        
        spec_hatch = importlib.util.spec_from_file_location("generation.process.hatching_generator", hatching_gen_path)
        hatch_module = importlib.util.module_from_spec(spec_hatch)
        hatch_module.__package__ = 'generation.process'
        spec_hatch.loader.exec_module(hatch_module)
        HatchingGenerator = hatch_module.HatchingGenerator
        
        spec_laser = importlib.util.spec_from_file_location("generation.sensors.laser_parameter_generator", laser_gen_path)
        laser_module = importlib.util.module_from_spec(spec_laser)
        laser_module.__package__ = 'generation.sensors'
        spec_laser.loader.exec_module(laser_module)
        LaserParameterGenerator = laser_module.LaserParameterGenerator
        
        spec_ispm = importlib.util.spec_from_file_location("generation.sensors.ispm_generator", ispm_gen_path)
        ispm_module = importlib.util.module_from_spec(spec_ispm)
        ispm_module.__package__ = 'generation.sensors'
        spec_ispm.loader.exec_module(ispm_module)
        ISPMGenerator = ispm_module.ISPMGenerator
        
        spec_ct = importlib.util.spec_from_file_location("generation.sensors.ct_scan_generator", ct_gen_path)
        ct_module = importlib.util.module_from_spec(spec_ct)
        ct_module.__package__ = 'generation.sensors'
        spec_ct.loader.exec_module(ct_module)
        CTScanGenerator = ct_module.CTScanGenerator
    
    GENERATORS_AVAILABLE = True
except Exception as e:
    print(f"⚠️  Could not import generators: {e}")
    import traceback
    traceback.print_exc()
    GENERATORS_AVAILABLE = False

logger = logging.getLogger(__name__)


def delete_existing_data(mongo_client: Any,
                         model_id: Optional[str] = None,
                         stl_filenames: Optional[List[str]] = None,
                         collections: Optional[List[str]] = None,
                         delete_all: bool = False) -> Dict[str, Any]:
    """
    Delete existing data from MongoDB collections.
    
    Args:
        mongo_client: MongoDB client instance
        model_id: Optional model ID to delete
        stl_filenames: Optional list of STL filenames to delete (matches by filename or original_stem)
        collections: Optional list of collections to clean (None = all)
        delete_all: If True, delete all data (use with caution!)
        
    Returns:
        Dictionary with deletion summary
    """
    collections_to_clean = collections or ['stl_models', 'hatching_layers', 'laser_parameters', 
                                          'ct_scan_data', 'ispm_monitoring_data']
    
    results = {}
    
    if delete_all:
        print("🗑️  Deleting ALL data from collections...")
        filter_query = {}
    elif model_id:
        print(f"🗑️  Deleting data for model_id: {model_id}...")
        filter_query = {'model_id': model_id}
    elif stl_filenames:
        # Find model_ids for the given STL filenames
        print(f"🗑️  Finding models for STL files: {', '.join(stl_filenames)}...")
        stl_models_collection = mongo_client.get_collection('stl_models')
        
        # Build query to match by filename or original_stem
        filename_queries = []
        for filename in stl_filenames:
            stem = Path(filename).stem  # Remove .stl extension if present
            filename_queries.append({'filename': filename})
            filename_queries.append({'original_stem': stem})
            filename_queries.append({'metadata.original_filename': filename})
            filename_queries.append({'metadata.original_stem': stem})
        
        # Find all matching model_ids
        matching_models = stl_models_collection.find({'$or': filename_queries})
        model_ids = [doc.get('model_id') for doc in matching_models if doc.get('model_id')]
        
        if not model_ids:
            print(f"   ℹ️  No existing models found for STL files: {', '.join(stl_filenames)}")
            return {'deleted': 0, 'collections': {}}
        
        print(f"   Found {len(model_ids)} existing model(s) to delete")
        filter_query = {'model_id': {'$in': model_ids}}
    else:
        print("⚠️  No deletion criteria specified. Skipping deletion.")
        return {'deleted': 0, 'collections': {}}
    
    for collection_name in collections_to_clean:
        try:
            collection = mongo_client.get_collection(collection_name)
            count_before = collection.count_documents(filter_query)
            
            if count_before > 0:
                result = collection.delete_many(filter_query)
                deleted_count = result.deleted_count
                results[collection_name] = deleted_count
                print(f"   ✅ Deleted {deleted_count} document(s) from {collection_name}")
            else:
                results[collection_name] = 0
                print(f"   ℹ️  No documents to delete from {collection_name}")
        except Exception as e:
            print(f"   ❌ Error deleting from {collection_name}: {e}")
            results[collection_name] = 0
    
    total_deleted = sum(results.values())
    print(f"\n✅ Deletion complete: {total_deleted} total document(s) deleted")
    
    return {
        'deleted': total_deleted,
        'collections': results
    }


def populate_mongodb(n_models: Optional[int] = None,
                    stl_files: Optional[List[str]] = None,
                    collections: Optional[List[str]] = None,
                    delete_existing: bool = False,
                    delete_all: bool = False,
                    delete_only: bool = False,
                    resume_from: Optional[int] = None) -> Dict[str, Any]:
    """
    Generate data and directly populate MongoDB collections.
    
    Args:
        n_models: Number of STL models to process
        stl_files: Optional list of specific STL filenames
        collections: Optional list of collections to populate (None = all)
        delete_existing: Delete existing data for specified STL files before populating
        delete_all: Delete ALL data from collections before populating
        delete_only: If True, only delete data without populating
        resume_from: Resume processing from model index (0-based). Useful if script was interrupted.
        
    Returns:
        Dictionary with population summary
    """
    if not GENERATORS_AVAILABLE:
        return {'error': 'Generators not available'}
    
    # Connect to MongoDB
    print("🔌 Connecting to MongoDB...")
    config = get_mongodb_config()
    
    # Ensure credentials are set
    if not config.username:
        config.username = os.getenv('MONGO_ROOT_USERNAME', 'admin')
    if not config.password:
        config.password = os.getenv('MONGO_ROOT_PASSWORD', 'password')
    
    # Create MongoDB client wrapper with convenience methods
    class MongoClientWrapper:
        """Wrapper around MongoDBClient with convenience methods."""
        def __init__(self, client: MongoDBClient):
            self._client = client
            self._db = client.database
        
        def get_collection(self, collection_name: str):
            """Get a collection."""
            return self._db[collection_name]
        
        def insert_document(self, collection_name: str, document: dict):
            """Insert a single document."""
            collection = self.get_collection(collection_name)
            collection.insert_one(document)
        
        def insert_documents(self, collection_name: str, documents: list):
            """Insert multiple documents."""
            if documents:
                collection = self.get_collection(collection_name)
                collection.insert_many(documents)
        
        def store_file(self, data: bytes, filename: str, metadata: Optional[dict] = None) -> str:
            """Store a file in GridFS."""
            from gridfs import GridFS
            fs = GridFS(self._db)
            file_id = fs.put(data, filename=filename, metadata=metadata or {})
            return str(file_id)
        
        def disconnect(self):
            """Disconnect from MongoDB."""
            self._client.close()
    
    try:
        mongo_client_wrapper = MongoClientWrapper(MongoDBClient(config=config))
        mongo_client = mongo_client_wrapper
        print(f"✅ Connected to MongoDB: {config.database}")
    except Exception as e:
        return {'error': f'Failed to connect to MongoDB: {e}'}
    
    # Set up models directory
    models_dir = project_root / 'generation' / 'models'
    if not models_dir.exists():
        return {'error': f'Models directory not found: {models_dir}'}
    
    # Initialize generators (needed for finding STL files)
    stl_processor = STLProcessor(models_dir=models_dir)
    
    # Find STL files first (needed for deletion by filename)
    if stl_files:
        stl_paths = []
        for filename in stl_files:
            # Try to find in models directory
            stl_path = stl_processor.get_stl_file(filename)
            if stl_path is None:
                # Try direct path
                direct_path = Path(filename)
                if direct_path.exists() and direct_path.suffix.lower() == '.stl':
                    stl_paths.append(direct_path)
                else:
                    print(f"⚠️  STL file not found: {filename}")
            else:
                stl_paths.append(stl_path)
    else:
        stl_paths = stl_processor.find_stl_files()
        if n_models:
            stl_paths = stl_paths[:n_models]
    
    if not stl_paths:
        return {'error': f'No STL files found. Models directory: {models_dir}'}
    
    # Delete existing data if requested (now we have stl_paths)
    if delete_existing or delete_all or delete_only:
        # Get STL filenames for deletion
        stl_filenames_for_deletion = None
        if (delete_existing or delete_only) and stl_paths:
            stl_filenames_for_deletion = [p.name for p in stl_paths]
        
        delete_result = delete_existing_data(
            mongo_client, 
            model_id=None, 
            stl_filenames=stl_filenames_for_deletion,
            collections=collections, 
            delete_all=delete_all
        )
        print()  # Empty line for readability
        
        # If delete_only, return early without populating
        if delete_only:
            mongo_client.disconnect()
            return {
                'deleted': delete_result.get('deleted', 0),
                'collections': delete_result.get('collections', {}),
                'message': 'Deletion only - no data populated'
            }
    
    # Initialize remaining generators
    hatching_gen = HatchingGenerator()
    laser_gen = LaserParameterGenerator()
    ispm_gen = ISPMGenerator()
    ct_gen = CTScanGenerator()
    
    print(f"📦 Processing {len(stl_paths)} STL file(s)...")
    
    results = []
    collections_to_populate = collections or ['stl_models', 'hatching_layers', 'laser_parameters', 
                                               'ct_scan_data', 'ispm_monitoring_data']
    
    for i, stl_path in enumerate(stl_paths):
        model_id = None  # Initialize in case of early error
        model_name = None
        try:
            print(f"\n📄 Processing {i+1}/{len(stl_paths)}: {stl_path.name}")
            
            # Generate unique UUID for this model instance
            model_id = str(uuid.uuid4())
            model_name = f"{stl_path.stem}_{uuid.uuid4().hex[:8]}"  # Keep original name + short UUID for readability
            
            # Process STL metadata first (needed for bounding box in all collections)
            stl_metadata = stl_processor.process_stl_file(str(stl_path))
            
            # Update metadata with UUID-based identifiers
            stl_metadata['model_id'] = model_id
            stl_metadata['model_name'] = model_name
            stl_metadata['original_filename'] = stl_path.name
            stl_metadata['original_stem'] = stl_path.stem
            
            # 1. Store STL model in MongoDB
            if 'stl_models' in collections_to_populate:
                
                stl_doc = {
                    'model_id': model_id,  # UUID
                    'model_name': model_name,  # Readable name with UUID suffix
                    'filename': stl_path.name,  # Original filename
                    'file_path': str(stl_path),  # Original file path
                    'original_stem': stl_path.stem,  # Original filename without extension
                    'metadata': stl_metadata,
                    'tags': [stl_path.suffix.lower(), 'generated'],
                    'created_at': datetime.now().isoformat()
                }
                
                mongo_client.insert_document('stl_models', stl_doc)
                print(f"   ✅ Stored STL model: {model_name} (ID: {model_id})")
            
            # 2. Generate and store hatching data
            if 'hatching_layers' in collections_to_populate:
                try:
                    # Load STL as pyslm.Part (required for hatching)
                    import pyslm
                    stl_part = pyslm.Part(model_id)
                    stl_part.setGeometry(str(stl_path))
                    stl_part.origin = [0.0, 0.0, 0.0]
                    stl_part.rotation = [0, 0, 0]
                    stl_part.dropToPlatform()
                    
                    print(f"   🔄 Generating hatching paths...")
                    
                    # Generate hatching - HatchingGenerator does all the work
                    hatching_result = hatching_gen.generate_hatching(stl_part)
                    
                    # Extract points, power, velocity, energy from result
                    all_points = hatching_result.get('points', np.array([]))
                    all_power = hatching_result.get('power', np.array([]))
                    all_velocity = hatching_result.get('velocity', np.array([]))
                    all_energy = hatching_result.get('energy', np.array([]))
                    
                    # Get laser beam parameters from metadata (calculated by HatchingGenerator)
                    metadata = hatching_result['metadata']
                    laser_beam_width = metadata.get('laser_beam_width', 0.1)
                    hatch_spacing = metadata.get('hatch_spacing', 0.1)
                    overlap_percentage = metadata.get('overlap_percentage', 0.0)
                    overlap_ratio = metadata.get('overlap_ratio', 0.0)
                    
                    # Get coordinate system information (critical for merging with other data sources)
                    coordinate_system = metadata.get('coordinate_system', None)
                    
                    # Check if we have points
                    if len(all_points) == 0:
                        print(f"   ⚠️  No points extracted from hatching")
                    else:
                        # Group points by layer (z-coordinate)
                        hatching_docs = []
                        for layer_idx, layer in enumerate(hatching_result['layers']):
                            z_height = layer.z / 1000.0  # Convert from microns to mm
                            z_tolerance = 0.001  # 1 micron tolerance
                            
                            # Find all points for this layer (matching z-coordinate)
                            layer_mask = np.abs(all_points[:, 2] - z_height) < z_tolerance
                            layer_points = all_points[layer_mask]
                            layer_power = all_power[layer_mask] if len(all_power) > 0 else np.array([])
                            layer_velocity = all_velocity[layer_mask] if len(all_velocity) > 0 else np.array([])
                            layer_energy = all_energy[layer_mask] if len(all_energy) > 0 else np.array([])
                            
                            # Group points into hatch paths (sequential points form a hatch)
                            hatches = []
                            if len(layer_points) > 0:
                                # Group consecutive points into hatch segments
                                current_hatch = []
                                current_power = layer_power[0] if len(layer_power) > 0 else 200.0
                                current_velocity = layer_velocity[0] if len(layer_velocity) > 0 else 500.0
                                
                                for i, point in enumerate(layer_points):
                                    if len(current_hatch) == 0:
                                        current_hatch = [point.tolist()]
                                        if i < len(layer_power):
                                            current_power = float(layer_power[i])
                                        if i < len(layer_velocity):
                                            current_velocity = float(layer_velocity[i])
                                    else:
                                        # Check if point continues the current hatch (within tolerance)
                                        last_point = current_hatch[-1]
                                        distance = np.linalg.norm(np.array(point[:2]) - np.array(last_point[:2]))
                                        
                                        if distance < 2.0:  # Points within 2mm are part of same hatch
                                            current_hatch.append(point.tolist())
                                        else:
                                            # End current hatch, start new one
                                            if len(current_hatch) > 1:
                                                hatches.append({
                                                    'start_point': current_hatch[0],
                                                    'end_point': current_hatch[-1],
                                                    'points': current_hatch,  # Full path coordinates
                                                    'laser_power': current_power,
                                                    'scan_speed': current_velocity,
                                                    'energy_density': float(layer_energy[i-1]) if i-1 < len(layer_energy) else current_power / (current_velocity * 0.1),
                                                    'laser_beam_width': laser_beam_width,  # mm - critical for overlap calculation
                                                    'hatch_spacing': hatch_spacing,  # mm - spacing between hatch lines
                                                    'overlap_percentage': overlap_percentage,  # % - overlap between consecutive hatches
                                                    'hatch_type': 'raster',
                                                    'scan_order': len(hatches)
                                                })
                                            current_hatch = [point.tolist()]
                                            if i < len(layer_power):
                                                current_power = float(layer_power[i])
                                            if i < len(layer_velocity):
                                                current_velocity = float(layer_velocity[i])
                                
                                # Add final hatch
                                if len(current_hatch) > 1:
                                    hatches.append({
                                        'start_point': current_hatch[0],
                                        'end_point': current_hatch[-1],
                                        'points': current_hatch,  # Full path coordinates
                                        'laser_power': current_power,
                                        'scan_speed': current_velocity,
                                        'energy_density': float(layer_energy[-1]) if len(layer_energy) > 0 else current_power / (current_velocity * 0.1),
                                        'laser_beam_width': laser_beam_width,  # mm - critical for overlap calculation
                                        'hatch_spacing': hatch_spacing,  # mm - spacing between hatch lines
                                        'overlap_percentage': overlap_percentage,  # % - overlap between consecutive hatches
                                        'hatch_type': 'raster',
                                        'scan_order': len(hatches)
                                    })
                        
                            hatching_docs.append({
                                'model_id': model_id,
                                'layer_index': layer_idx,
                                'layer_height': hatching_result['metadata']['layer_thickness'],
                                'z_position': z_height,
                                'contours': [],  # Contours can be extracted separately if needed
                                'hatches': hatches,  # Laser path coordinates with parameters
                                'processing_time': datetime.now().isoformat(),
                                'coordinate_system': coordinate_system,  # Critical for merging with ISPM/CT data
                                'metadata': {
                                    'n_contours': 0,
                                    'n_hatches': len(hatches),
                                    'n_points': len(layer_points),
                                    'hatch_spacing': hatch_spacing,
                                    'laser_beam_width': laser_beam_width,
                                    'overlap_percentage': overlap_percentage,
                                    'overlap_ratio': overlap_ratio  # 0.0 to 1.0 for calculations
                                }
                            })
                        
                        if hatching_docs:
                            mongo_client.insert_documents('hatching_layers', hatching_docs)
                            print(f"   ✅ Stored {len(hatching_docs)} hatching layers")
                        else:
                            print(f"   ⚠️  No hatching layers generated")
                        
                except ImportError:
                    print(f"   ⚠️  pyslm not available - skipping hatching generation")
                except Exception as e:
                    print(f"   ⚠️  Error generating hatching: {e}")
                    logger.error(f"Hatching generation error: {e}", exc_info=True)
            
            # 3. Generate and store laser parameters
            if 'laser_parameters' in collections_to_populate:
                bbox = stl_metadata.get('bounding_box', {})
                if bbox:
                    n_layers = int((bbox['max'][2] - bbox['min'][2]) / 0.05)
                    # Convert bbox to format expected by laser generator: {'min': (x, y, z), 'max': (x, y, z)}
                    laser_bbox = {
                        'min': tuple(bbox['min']),
                        'max': tuple(bbox['max'])
                    }
                    laser_data = laser_gen.generate_for_build(
                        build_id=model_id,
                        n_layers=n_layers,
                        points_per_layer=1000,
                        bounding_box=laser_bbox
                    )
                    
                    # Convert to MongoDB documents
                    # laser_data['points'] is a list of LaserParameterPoint objects
                    laser_docs = []
                    all_laser_points = laser_data.get('points', [])
                    for point in all_laser_points:
                        # point is a LaserParameterPoint dataclass
                        laser_docs.append({
                            'model_id': model_id,
                            'layer_index': point.layer_index,
                            'point_id': f"{model_id}_lp_{len(laser_docs)}",
                            'spatial_coordinates': [point.x, point.y, point.z],
                            'laser_power': point.laser_power,
                            'scan_speed': point.scan_speed,
                            'hatch_spacing': point.hatch_spacing,
                            'energy_density': point.energy_density,
                            'exposure_time': point.exposure_time,  # Time to scan one hatch spacing
                            'timestamp': point.timestamp.isoformat() if isinstance(point.timestamp, datetime) else str(point.timestamp),
                            'region_type': point.region_type
                        })
                    
                    if laser_docs:
                        mongo_client.insert_documents('laser_parameters', laser_docs)
                        print(f"   ✅ Stored {len(laser_docs)} laser parameter points")
            
            # 4. Generate and store CT scan data
            if 'ct_scan_data' in collections_to_populate:
                bbox = stl_metadata.get('bounding_box', {})
                if bbox:
                    # Convert bounding box format for CT scan generator
                    # CT generator expects: {'x': (min, max), 'y': (min, max), 'z': (min, max)}
                    bbox_dict = {
                        'x': (bbox['min'][0], bbox['max'][0]),
                        'y': (bbox['min'][1], bbox['max'][1]),
                        'z': (bbox['min'][2], bbox['max'][2])
                    }
                    # Use much smaller grid for demo to avoid memory issues
                    # In production, use full resolution (200, 200, 200) with GridFS
                    from generation.sensors.ct_scan_generator import CTScanGeneratorConfig
                    ct_config = CTScanGeneratorConfig(
                        grid_dimensions=(30, 30, 30),  # Reduced from (200, 200, 200) for demo - 30³ = 27K voxels vs 8M
                        voxel_spacing=(0.67, 0.67, 0.67)  # Larger spacing to cover same area
                    )
                    ct_gen_demo = CTScanGenerator(config=ct_config)
                    
                    ct_data = ct_gen_demo.generate_for_build(
                        build_id=model_id,
                        bounding_box=bbox_dict
                    )
                    
                    # Extract CT scan data
                    voxel_grid_obj = ct_data.get('voxel_grid')
                    defect_locations_full = ct_data.get('defect_locations', [])  # Already calculated by generator
                    
                    # Limit defect_locations for storage (store only first 1000 for demo, or just count)
                    # In production, consider storing in GridFS if very large
                    if len(defect_locations_full) > 1000:
                        defect_locations = defect_locations_full[:1000]  # Store sample
                        defect_count = len(defect_locations_full)
                    else:
                        defect_locations = defect_locations_full
                        defect_count = len(defect_locations_full)
                    
                    # Extract voxel grid data
                    # For large arrays, use GridFS to avoid memory issues
                    density_values_gridfs_id = None
                    porosity_map_gridfs_id = None
                    
                    if voxel_grid_obj:
                        # Store large arrays in GridFS instead of document to avoid memory issues
                        if hasattr(voxel_grid_obj, 'density_values') and voxel_grid_obj.density_values is not None:
                            if isinstance(voxel_grid_obj.density_values, np.ndarray):
                                try:
                                    # Save as compressed numpy array in GridFS
                                    buffer = io.BytesIO()
                                    np.save(buffer, voxel_grid_obj.density_values)
                                    compressed_data = gzip.compress(buffer.getvalue())
                                    
                                    # Store in GridFS
                                    filename = f"{model_id}_density_values.npy.gz"
                                    density_values_gridfs_id = mongo_client.store_file(
                                        compressed_data,
                                        filename,
                                        metadata={'model_id': model_id, 'data_type': 'density_values', 'format': 'numpy_gzip'}
                                    )
                                except Exception as e:
                                    logger.warning(f"Failed to store density_values in GridFS: {e}")
                                    density_values_gridfs_id = None
                        
                        if hasattr(voxel_grid_obj, 'porosity_map') and voxel_grid_obj.porosity_map is not None:
                            if isinstance(voxel_grid_obj.porosity_map, np.ndarray):
                                try:
                                    # Save as compressed numpy array in GridFS
                                    buffer = io.BytesIO()
                                    np.save(buffer, voxel_grid_obj.porosity_map)
                                    compressed_data = gzip.compress(buffer.getvalue())
                                    
                                    # Store in GridFS
                                    filename = f"{model_id}_porosity_map.npy.gz"
                                    porosity_map_gridfs_id = mongo_client.store_file(
                                        compressed_data,
                                        filename,
                                        metadata={'model_id': model_id, 'data_type': 'porosity_map', 'format': 'numpy_gzip'}
                                    )
                                except Exception as e:
                                    logger.warning(f"Failed to store porosity_map in GridFS: {e}")
                                    porosity_map_gridfs_id = None
                    
                    # Get coordinate system information (critical for merging with STL/hatching data)
                    coordinate_system = ct_data.get('coordinate_system', None)
                    
                    # Process metadata - convert dataclass objects to dicts for MongoDB serialization
                    raw_metadata = ct_data.get('metadata', {})
                    processed_metadata = {}
                    for key, value in raw_metadata.items():
                        if hasattr(value, '__dict__') and not isinstance(value, (dict, list, str, int, float, bool, type(None))):
                            # Convert dataclass to dict
                            if is_dataclass(value):
                                processed_metadata[key] = asdict(value)
                            else:
                                # Skip non-serializable objects
                                processed_metadata[key] = str(value)
                        else:
                            processed_metadata[key] = value
                    
                    ct_doc = {
                        'model_id': model_id,
                        'scan_id': ct_data.get('scan_id', f"{model_id}_ct_scan"),
                        'scan_timestamp': ct_data.get('scan_timestamp'),
                        'coordinate_system': coordinate_system,  # Critical for merging with STL/hatching data
                        'voxel_grid': {
                            'dimensions': list(voxel_grid_obj.dimensions) if voxel_grid_obj and hasattr(voxel_grid_obj, 'dimensions') else None,
                            'spacing': list(voxel_grid_obj.spacing) if voxel_grid_obj and hasattr(voxel_grid_obj, 'spacing') else None,
                            'origin': list(voxel_grid_obj.origin) if voxel_grid_obj and hasattr(voxel_grid_obj, 'origin') else None,
                        },
                        'data_storage': {
                            'density_values_gridfs_id': str(density_values_gridfs_id) if density_values_gridfs_id else None,
                            'porosity_map_gridfs_id': str(porosity_map_gridfs_id) if porosity_map_gridfs_id else None,
                            'storage_type': 'gridfs' if (density_values_gridfs_id or porosity_map_gridfs_id) else 'none'
                        },
                        'defect_locations': defect_locations,  # Sample of defect coordinates (limited to 1000 for demo)
                        'defect_count': defect_count,  # Total number of defects
                        'metadata': {
                            **processed_metadata,
                            'statistics': ct_data.get('statistics', {})  # Include statistics from generator
                        }
                    }
                    
                    mongo_client.insert_document('ct_scan_data', ct_doc)
                    print(f"   ✅ Stored CT scan data (arrays in GridFS: density={density_values_gridfs_id is not None}, porosity={porosity_map_gridfs_id is not None})")
            
            # 5. Generate and store ISPM data
            if 'ispm_monitoring_data' in collections_to_populate:
                bbox = stl_metadata.get('bounding_box', {})
                if bbox:
                    # Convert bounding box format for ISPM generator
                    bbox_dict = {
                        'x': (bbox['min'][0], bbox['max'][0]),
                        'y': (bbox['min'][1], bbox['max'][1]),
                        'z': (bbox['min'][2], bbox['max'][2])
                    }
                    n_layers = int((bbox['max'][2] - bbox['min'][2]) / 0.05)
                    # Reduce data generation for demo to avoid memory issues
                    # In production, use batching or GridFS for large datasets
                    from generation.sensors.ispm_generator import ISPMGeneratorConfig
                    ispm_config = ISPMGeneratorConfig(points_per_layer=25)  # Reduced from 1000 to 25 for demo
                    ispm_gen_demo = ISPMGenerator(config=ispm_config)
                    
                    ispm_data = ispm_gen_demo.generate_for_build(
                        build_id=model_id,
                        n_layers=n_layers,
                        bounding_box=bbox_dict
                    )
                    
                    # Get coordinate system information (critical for merging with STL/hatching/CT data)
                    coordinate_system = ispm_data.get('coordinate_system', None)
                    
                    # Convert to MongoDB documents in batches to avoid memory issues
                    # ispm_data['data_points'] is a list of ISPMDataPoint objects
                    all_data_points = ispm_data.get('data_points', [])
                    
                    if all_data_points:
                        # Batch size for insertion (to avoid memory issues)
                        batch_size = 10000
                        total_inserted = 0
                        
                        for batch_start in range(0, len(all_data_points), batch_size):
                            batch_points = all_data_points[batch_start:batch_start + batch_size]
                            ispm_docs = []
                            
                            for point in batch_points:
                                # point is an ISPMDataPoint dataclass
                                ispm_docs.append({
                                    'model_id': model_id,
                                    'layer_index': point.layer_index,
                                    'timestamp': point.timestamp.isoformat() if isinstance(point.timestamp, datetime) else str(point.timestamp),
                                    'spatial_coordinates': [point.x, point.y, point.z],
                                    'melt_pool_temperature': point.melt_pool_temperature,
                                    'melt_pool_size': point.melt_pool_size,  # Dict with width, length, depth
                                    'peak_temperature': point.peak_temperature,
                                    'cooling_rate': point.cooling_rate,
                                    'temperature_gradient': point.temperature_gradient,
                                    'process_event': point.process_event,
                                    'coordinate_system': coordinate_system
                                })
                            
                            if ispm_docs:
                                mongo_client.insert_documents('ispm_monitoring_data', ispm_docs)
                                total_inserted += len(ispm_docs)
                                print(f"   📊 Inserted batch: {total_inserted}/{len(all_data_points)} ISPM records...", end='\r')
                        
                        print(f"\n   ✅ Stored {total_inserted} ISPM monitoring records")
                    else:
                        print(f"   ⚠️  No ISPM data points generated")
            
            results.append({
                'model_id': model_id,
                'stl_file': stl_path.name,
                'status': 'success'
            })
            
        except Exception as e:
            logger.error(f"Error processing {stl_path.name}: {e}")
            results.append({
                'model_id': model_id if model_id else f"error_{stl_path.stem}",  # Use UUID if available, else error prefix
                'model_name': model_name if model_name else None,
                'stl_file': stl_path.name,
                'status': 'error',
                'error': str(e)
            })
    
    # Disconnect
    mongo_client.disconnect()
    
    print(f"\n✅ Completed! Processed {len([r for r in results if r['status'] == 'success'])}/{len(results)} models")
    
    return {
        'n_models': len(stl_paths),
        'n_success': len([r for r in results if r['status'] == 'success']),
        'results': results
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Populate MongoDB from data generation')
    parser.add_argument('--all', action='store_true',
                       help='Process all STL files in the models directory')
    parser.add_argument('--n-models', type=int, help='Number of models to process')
    parser.add_argument('--stl-files', nargs='+', help='Specific STL files to process')
    parser.add_argument('--collections', nargs='+', 
                       choices=['stl_models', 'hatching_layers', 'laser_parameters', 
                               'ct_scan_data', 'ispm_monitoring_data'],
                       help='Collections to populate')
    parser.add_argument('--delete-existing', action='store_true',
                       help='Delete existing data for the same STL files before populating')
    parser.add_argument('--delete-all', action='store_true',
                       help='Delete ALL data from collections before populating (use with caution!)')
    parser.add_argument('--delete-only', action='store_true',
                       help='Only delete existing data without populating (requires --delete-existing or --delete-all)')
    parser.add_argument('--resume-from', type=int, 
                       help='Resume processing from model index (0-based). Useful if script was interrupted.')
    
    args = parser.parse_args()
    
    # Handle --all flag
    if args.all:
        args.stl_files = None
        args.n_models = None
    
    # Validate delete-only usage
    if args.delete_only and not (args.delete_existing or args.delete_all):
        print("❌ Error: --delete-only requires --delete-existing or --delete-all")
        sys.exit(1)
    
    # Safety check for delete-all
    if args.delete_all:
        response = input("⚠️  WARNING: This will delete ALL data from collections. Continue? (yes/no): ")
        if response.lower() != 'yes':
            print("❌ Aborted. No data deleted.")
            sys.exit(0)
    
    result = populate_mongodb(
        n_models=args.n_models,
        stl_files=args.stl_files,
        collections=args.collections,
        delete_existing=args.delete_existing,
        delete_all=args.delete_all,
        delete_only=args.delete_only,
        resume_from=args.resume_from
    )
    
    if 'error' in result:
        print(f"❌ Error: {result['error']}")
        sys.exit(1)
    else:
        print(f"\n📊 Summary:")
        if 'message' in result and 'Deletion only' in result['message']:
            # Deletion only mode
            print(f"   Documents deleted: {result.get('deleted', 0)}")
            if 'collections' in result:
                print(f"   Collections affected:")
                for coll_name, count in result['collections'].items():
                    if count > 0:
                        print(f"      {coll_name}: {count} document(s)")
        else:
            # Normal population mode
            print(f"   Models processed: {result.get('n_success', 0)}/{result.get('n_models', 0)}")
        sys.exit(0)

