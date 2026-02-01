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
        from generation.sensors.ispm_thermal_generator import ISPMThermalGenerator
        from generation.sensors.ispm_optical_generator import ISPMOpticalGenerator
        from generation.sensors.ispm_acoustic_generator import ISPMAcousticGenerator
        from generation.sensors.ispm_strain_generator import ISPMStrainGenerator
        from generation.sensors.ispm_plume_generator import ISPMPlumeGenerator
        from generation.sensors.ct_scan_generator import CTScanGenerator
    except ImportError as e:
        # Fallback to direct module loading with proper package setup
        stl_processor_path = data_gen_dir / 'process' / 'stl_processor.py'
        hatching_gen_path = data_gen_dir / 'process' / 'hatching_generator.py'
        laser_gen_path = data_gen_dir / 'sensors' / 'laser_parameter_generator.py'
        ispm_thermal_gen_path = data_gen_dir / 'sensors' / 'ispm_thermal_generator.py'
        ispm_optical_gen_path = data_gen_dir / 'sensors' / 'ispm_optical_generator.py'
        ispm_acoustic_gen_path = data_gen_dir / 'sensors' / 'ispm_acoustic_generator.py'
        ispm_strain_gen_path = data_gen_dir / 'sensors' / 'ispm_strain_generator.py'
        ispm_plume_gen_path = data_gen_dir / 'sensors' / 'ispm_plume_generator.py'
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
        
        spec_ispm_thermal = importlib.util.spec_from_file_location("generation.sensors.ispm_thermal_generator", ispm_thermal_gen_path)
        ispm_thermal_module = importlib.util.module_from_spec(spec_ispm_thermal)
        ispm_thermal_module.__package__ = 'generation.sensors'
        spec_ispm_thermal.loader.exec_module(ispm_thermal_module)
        ISPMThermalGenerator = ispm_thermal_module.ISPMThermalGenerator
        
        spec_ispm_optical = importlib.util.spec_from_file_location("generation.sensors.ispm_optical_generator", ispm_optical_gen_path)
        ispm_optical_module = importlib.util.module_from_spec(spec_ispm_optical)
        ispm_optical_module.__package__ = 'generation.sensors'
        spec_ispm_optical.loader.exec_module(ispm_optical_module)
        ISPMOpticalGenerator = ispm_optical_module.ISPMOpticalGenerator
        
        spec_ispm_acoustic = importlib.util.spec_from_file_location("generation.sensors.ispm_acoustic_generator", ispm_acoustic_gen_path)
        ispm_acoustic_module = importlib.util.module_from_spec(spec_ispm_acoustic)
        ispm_acoustic_module.__package__ = 'generation.sensors'
        spec_ispm_acoustic.loader.exec_module(ispm_acoustic_module)
        ISPMAcousticGenerator = ispm_acoustic_module.ISPMAcousticGenerator
        
        spec_ispm_strain = importlib.util.spec_from_file_location("generation.sensors.ispm_strain_generator", ispm_strain_gen_path)
        ispm_strain_module = importlib.util.module_from_spec(spec_ispm_strain)
        ispm_strain_module.__package__ = 'generation.sensors'
        spec_ispm_strain.loader.exec_module(ispm_strain_module)
        ISPMStrainGenerator = ispm_strain_module.ISPMStrainGenerator
        
        spec_ispm_plume = importlib.util.spec_from_file_location("generation.sensors.ispm_plume_generator", ispm_plume_gen_path)
        ispm_plume_module = importlib.util.module_from_spec(spec_ispm_plume)
        ispm_plume_module.__package__ = 'generation.sensors'
        spec_ispm_plume.loader.exec_module(ispm_plume_module)
        ISPMPlumeGenerator = ispm_plume_module.ISPMPlumeGenerator
        
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
    collections_to_clean = collections or ['stl_models', 'hatching_layers', 'laser_monitoring_data', 
                                          'ct_scan_data', 'ispm_thermal_monitoring_data', 'ispm_optical_monitoring_data', 'ispm_acoustic_monitoring_data', 'ispm_strain_monitoring_data', 'ispm_plume_monitoring_data']
    
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
    ispm_thermal_gen = ISPMThermalGenerator()
    ispm_optical_gen = ISPMOpticalGenerator()
    ispm_acoustic_gen = ISPMAcousticGenerator()
    ispm_strain_gen = ISPMStrainGenerator()
    ispm_plume_gen = ISPMPlumeGenerator()
    ct_gen = CTScanGenerator()
    
    print(f"📦 Processing {len(stl_paths)} STL file(s)...")
    
    results = []
    collections_to_populate = collections or ['stl_models', 'hatching_layers', 'laser_monitoring_data', 
                                               'ct_scan_data', 'ispm_thermal_monitoring_data', 'ispm_optical_monitoring_data', 'ispm_acoustic_monitoring_data', 'ispm_strain_monitoring_data', 'ispm_plume_monitoring_data']
    
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
                    
                    # Get laser beam parameters from metadata (calculated by HatchingGenerator)
                    metadata = hatching_result['metadata']
                    laser_beam_width = metadata.get('laser_beam_width', 0.1)
                    hatch_spacing = metadata.get('hatch_spacing', 0.1)
                    overlap_percentage = metadata.get('overlap_percentage', 0.0)
                    overlap_ratio = metadata.get('overlap_ratio', 0.0)
                    layer_thickness = metadata.get('layer_thickness', 0.05)
                    
                    # Get coordinate system information (critical for merging with other data sources)
                    coordinate_system = metadata.get('coordinate_system', None)
                    
                    # Extract vectors directly from pyslm layers (new vector-based format)
                    # This preserves the exact structure from pyslm (pairs for hatches, sequences for contours)
                    layers = hatching_result['layers']
                    vector_layers = hatching_gen.extract_vectors(layers, hatching_gen.config)
                    
                    if len(vector_layers) == 0:
                        print(f"   ⚠️  No vectors extracted from hatching")
                    else:
                        # Prepare documents in vector-based format
                        hatching_docs = []
                        
                        for layer_data in vector_layers:
                            layer_idx = layer_data['layer_index']
                            z_height = layer_data['z_position']
                            vectors = layer_data['vectors']
                            vectordata = layer_data['vectordata']
                            
                            # Create document in vector-based format (matches the JSON structure you provided)
                            doc = {
                                'model_id': model_id,
                                'layer_index': layer_idx,
                                'layer_height': layer_thickness,
                                'z_position': z_height,
                                'length': len(vectors),  # Total number of vectors
                                'vectors': vectors,  # Array of {x1, y1, x2, y2, z, timestamp, dataindex}
                                'vectordata': vectordata,  # Array of {dataindex, partid, type, scanner, laserpower, scannerspeed, laser_beam_width, hatch_spacing, layer_index, etc.}
                                'processing_time': datetime.now().isoformat(),
                                'coordinate_system': coordinate_system,  # Critical for merging with ISPM/CT data
                                'metadata': {
                                    'n_vectors': len(vectors),
                                    'n_vectordata': len(vectordata),
                                    'hatch_spacing': hatch_spacing,
                                    'laser_beam_width': laser_beam_width,
                                    'overlap_percentage': overlap_percentage,
                                    'overlap_ratio': overlap_ratio,  # 0.0 to 1.0 for calculations
                                    'layer_thickness': layer_thickness
                                }
                            }
                            
                            hatching_docs.append(doc)
                        
                        if hatching_docs:
                            mongo_client.insert_documents('hatching_layers', hatching_docs)
                            total_vectors = sum(doc['length'] for doc in hatching_docs)
                            print(f"   ✅ Stored {len(hatching_docs)} hatching layers with {total_vectors} total vectors")
                        else:
                            print(f"   ⚠️  No hatching layers generated")
                        
                except ImportError:
                    print(f"   ⚠️  pyslm not available - skipping hatching generation")
                except Exception as e:
                    print(f"   ⚠️  Error generating hatching: {e}")
                    logger.error(f"Hatching generation error: {e}", exc_info=True)
            
            # 3. Generate and store laser parameters
            if 'laser_monitoring_data' in collections_to_populate:
                bbox = stl_metadata.get('bounding_box', {})
                if bbox:
                    n_layers = int((bbox['max'][2] - bbox['min'][2]) / 0.05)
                    # Convert bbox to format expected by laser generator: {'min': (x, y, z), 'max': (x, y, z)}
                    laser_bbox = {
                        'min': tuple(bbox['min']),
                        'max': tuple(bbox['max'])
                    }
                    # 100 points per layer: each point gets correct layer_index 0..n_layers-1 (no fallback in query)
                    laser_data = laser_gen.generate_for_build(
                        build_id=model_id,
                        n_layers=n_layers,
                        points_per_layer=100,
                        bounding_box=laser_bbox
                    )
                    
                    # Convert to MongoDB documents
                    # laser_data['points'] is a list of LaserParameterPoint objects
                    laser_docs = []
                    all_laser_points = laser_data.get('points', [])
                    for point in all_laser_points:
                        # point is a LaserParameterPoint dataclass
                        # layer_index and spatial_coordinates are required for query client and 2D Analysis Panel 4 (signal vs layer)
                        doc = {
                            'model_id': model_id,
                            'layer_index': point.layer_index,
                            'point_id': f"{model_id}_lp_{len(laser_docs)}",
                            'spatial_coordinates': [point.x, point.y, point.z],
                            # Process Parameters (setpoints/commanded values)
                            'commanded_power': point.commanded_power,  # Setpoint/commanded power
                            'commanded_scan_speed': point.commanded_scan_speed,  # Setpoint/commanded speed
                            'hatch_spacing': point.hatch_spacing,
                            'energy_density': point.energy_density,
                            'exposure_time': point.exposure_time,  # Time to scan one hatch spacing
                            'region_type': point.region_type,
                            'timestamp': _timestamp_to_float(point.timestamp),
                            'timestamp_iso': point.timestamp.isoformat() if isinstance(point.timestamp, datetime) else str(point.timestamp),
                        }
                        
                        # Temporal Sensors - Laser Power (Category 3.1)
                        if point.actual_power is not None:
                            doc['actual_power'] = point.actual_power
                        if point.power_setpoint is not None:
                            doc['power_setpoint'] = point.power_setpoint
                        if point.power_error is not None:
                            doc['power_error'] = point.power_error
                        if point.power_stability is not None:
                            doc['power_stability'] = point.power_stability
                        if point.power_fluctuation_amplitude is not None:
                            doc['power_fluctuation_amplitude'] = point.power_fluctuation_amplitude
                        if point.power_fluctuation_frequency is not None:
                            doc['power_fluctuation_frequency'] = point.power_fluctuation_frequency
                        
                        # Temporal Sensors - Beam Temporal Characteristics (Category 3.2)
                        if point.pulse_frequency is not None:
                            doc['pulse_frequency'] = point.pulse_frequency
                        if point.pulse_duration is not None:
                            doc['pulse_duration'] = point.pulse_duration
                        if point.pulse_energy is not None:
                            doc['pulse_energy'] = point.pulse_energy
                        if point.duty_cycle is not None:
                            doc['duty_cycle'] = point.duty_cycle
                        if point.beam_modulation_frequency is not None:
                            doc['beam_modulation_frequency'] = point.beam_modulation_frequency
                        
                        # Temporal Sensors - Laser System Health (Category 3.3)
                        if point.laser_temperature is not None:
                            doc['laser_temperature'] = point.laser_temperature
                        if point.laser_cooling_water_temp is not None:
                            doc['laser_cooling_water_temp'] = point.laser_cooling_water_temp
                        if point.laser_cooling_flow_rate is not None:
                            doc['laser_cooling_flow_rate'] = point.laser_cooling_flow_rate
                        if point.laser_power_supply_voltage is not None:
                            doc['laser_power_supply_voltage'] = point.laser_power_supply_voltage
                        if point.laser_power_supply_current is not None:
                            doc['laser_power_supply_current'] = point.laser_power_supply_current
                        if point.laser_diode_current is not None:
                            doc['laser_diode_current'] = point.laser_diode_current
                        if point.laser_diode_temperature is not None:
                            doc['laser_diode_temperature'] = point.laser_diode_temperature
                        if point.laser_operating_hours is not None:
                            doc['laser_operating_hours'] = point.laser_operating_hours
                        if point.laser_pulse_count is not None:
                            doc['laser_pulse_count'] = point.laser_pulse_count
                        
                        laser_docs.append(doc)
                    
                    if laser_docs:
                        mongo_client.insert_documents('laser_monitoring_data', laser_docs)
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
            if 'ispm_thermal_monitoring_data' in collections_to_populate:
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
                    from generation.sensors.ispm_thermal_generator import ISPMThermalGeneratorConfig
                    ispm_config = ISPMThermalGeneratorConfig(points_per_layer=100)  # 100 points per layer; layer_index 0..n_layers-1
                    ispm_gen_demo = ISPMThermalGenerator(config=ispm_config)
                    
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
                                # point is an ISPMThermalDataPoint dataclass
                                ispm_doc = {
                                    'model_id': model_id,
                                    'layer_index': point.layer_index,
                                    'timestamp': _timestamp_to_float(point.timestamp),
                                    'timestamp_iso': point.timestamp.isoformat() if isinstance(point.timestamp, datetime) else str(point.timestamp),
                                    'spatial_coordinates': [point.x, point.y, point.z],
                                    'melt_pool_temperature': point.melt_pool_temperature,
                                    'melt_pool_size': point.melt_pool_size,  # Dict with width, length, depth
                                    'peak_temperature': point.peak_temperature,
                                    'cooling_rate': point.cooling_rate,
                                    'temperature_gradient': point.temperature_gradient,
                                    'process_event': point.process_event,
                                    'coordinate_system': coordinate_system
                                }
                                
                                # Add additional ISPM fields from research (if available)
                                if point.melt_pool_area is not None:
                                    ispm_doc['melt_pool_area'] = point.melt_pool_area
                                if point.melt_pool_eccentricity is not None:
                                    ispm_doc['melt_pool_eccentricity'] = point.melt_pool_eccentricity
                                if point.melt_pool_perimeter is not None:
                                    ispm_doc['melt_pool_perimeter'] = point.melt_pool_perimeter
                                if point.time_over_threshold_1200K is not None:
                                    ispm_doc['time_over_threshold_1200K'] = point.time_over_threshold_1200K
                                if point.time_over_threshold_1680K is not None:
                                    ispm_doc['time_over_threshold_1680K'] = point.time_over_threshold_1680K
                                if point.time_over_threshold_2400K is not None:
                                    ispm_doc['time_over_threshold_2400K'] = point.time_over_threshold_2400K
                                
                                ispm_docs.append(ispm_doc)
                            
                            if ispm_docs:
                                mongo_client.insert_documents('ispm_thermal_monitoring_data', ispm_docs)
                                total_inserted += len(ispm_docs)
                                print(f"   📊 Inserted batch: {total_inserted}/{len(all_data_points)} ISPM records...", end='\r')
                        
                        print(f"\n   ✅ Stored {total_inserted} ISPM thermal monitoring records")
                    else:
                        print(f"   ⚠️  No ISPM thermal data points generated")
            
            # 6. Generate and store ISPM_Optical data
            if 'ispm_optical_monitoring_data' in collections_to_populate:
                bbox = stl_metadata.get('bounding_box', {})
                if bbox:
                    # Convert bounding box format for ISPM optical generator
                    bbox_dict = {
                        'x': (bbox['min'][0], bbox['max'][0]),
                        'y': (bbox['min'][1], bbox['max'][1]),
                        'z': (bbox['min'][2], bbox['max'][2])
                    }
                    n_layers = int((bbox['max'][2] - bbox['min'][2]) / 0.05)
                    # Reduce data generation for demo to avoid memory issues
                    # In production, use batching or GridFS for large datasets
                    from generation.sensors.ispm_optical_generator import ISPMOpticalGeneratorConfig
                    ispm_optical_config = ISPMOpticalGeneratorConfig(points_per_layer=100)  # 100 points per layer; layer_index 0..n_layers-1
                    ispm_optical_gen_demo = ISPMOpticalGenerator(config=ispm_optical_config)
                    
                    ispm_optical_data = ispm_optical_gen_demo.generate_for_build(
                        build_id=model_id,
                        n_layers=n_layers,
                        bounding_box=bbox_dict
                    )
                    
                    # Get coordinate system information (critical for merging with STL/hatching/CT data)
                    coordinate_system = ispm_optical_data.get('coordinate_system', None)
                    
                    # Convert to MongoDB documents in batches to avoid memory issues
                    # ispm_optical_data['data_points'] is a list of ISPMOpticalDataPoint objects
                    all_data_points = ispm_optical_data.get('data_points', [])
                    
                    if all_data_points:
                        # Batch size for insertion (to avoid memory issues)
                        batch_size = 10000
                        total_inserted = 0
                        
                        for batch_start in range(0, len(all_data_points), batch_size):
                            batch_points = all_data_points[batch_start:batch_start + batch_size]
                            ispm_optical_docs = []
                            
                            for point in batch_points:
                                # point is an ISPMOpticalDataPoint dataclass
                                ispm_optical_doc = {
                                    'model_id': model_id,
                                    'layer_index': point.layer_index,
                                    'timestamp': _timestamp_to_float(point.timestamp),
                                    'timestamp_iso': point.timestamp.isoformat() if isinstance(point.timestamp, datetime) else str(point.timestamp),
                                    'spatial_coordinates': [point.x, point.y, point.z],
                                    # Photodiode signals
                                    'photodiode_intensity': point.photodiode_intensity,
                                    'photodiode_frequency': point.photodiode_frequency,
                                    'photodiode_coaxial': point.photodiode_coaxial,
                                    'photodiode_off_axis': point.photodiode_off_axis,
                                    # Melt pool brightness/intensity
                                    'melt_pool_brightness': point.melt_pool_brightness,
                                    'melt_pool_intensity_mean': point.melt_pool_intensity_mean,
                                    'melt_pool_intensity_max': point.melt_pool_intensity_max,
                                    'melt_pool_intensity_std': point.melt_pool_intensity_std,
                                    # Spatter detection
                                    'spatter_detected': point.spatter_detected,
                                    'spatter_intensity': point.spatter_intensity,
                                    'spatter_count': point.spatter_count,
                                    # Process stability
                                    'process_stability': point.process_stability,
                                    'intensity_variation': point.intensity_variation,
                                    'signal_to_noise_ratio': point.signal_to_noise_ratio,
                                    # Melt pool imaging
                                    'melt_pool_image_available': point.melt_pool_image_available,
                                    'melt_pool_area_pixels': point.melt_pool_area_pixels,
                                    'melt_pool_centroid_x': point.melt_pool_centroid_x,
                                    'melt_pool_centroid_y': point.melt_pool_centroid_y,
                                    # Keyhole detection
                                    'keyhole_detected': point.keyhole_detected,
                                    'keyhole_intensity': point.keyhole_intensity,
                                    # Process events
                                    'process_event': point.process_event,
                                    # Frequency domain features
                                    'dominant_frequency': point.dominant_frequency,
                                    'frequency_bandwidth': point.frequency_bandwidth,
                                    'spectral_energy': point.spectral_energy,
                                    'coordinate_system': coordinate_system
                                }
                                
                                ispm_optical_docs.append(ispm_optical_doc)
                            
                            if ispm_optical_docs:
                                mongo_client.insert_documents('ispm_optical_monitoring_data', ispm_optical_docs)
                                total_inserted += len(ispm_optical_docs)
                                print(f"   📊 Inserted batch: {total_inserted}/{len(all_data_points)} ISPM optical records...", end='\r')
                        
                        print(f"\n   ✅ Stored {total_inserted} ISPM optical monitoring records")
                    else:
                        print(f"   ⚠️  No ISPM optical data points generated")
            
            # 7. Generate and store ISPM_Acoustic data
            if 'ispm_acoustic_monitoring_data' in collections_to_populate:
                bbox = stl_metadata.get('bounding_box', {})
                if bbox:
                    # Convert bounding box format for ISPM acoustic generator
                    bbox_dict = {
                        'x': (bbox['min'][0], bbox['max'][0]),
                        'y': (bbox['min'][1], bbox['max'][1]),
                        'z': (bbox['min'][2], bbox['max'][2])
                    }
                    n_layers = int((bbox['max'][2] - bbox['min'][2]) / 0.05)
                    # Reduce data generation for demo to avoid memory issues
                    # In production, use batching or GridFS for large datasets
                    from generation.sensors.ispm_acoustic_generator import ISPMAcousticGeneratorConfig
                    ispm_acoustic_config = ISPMAcousticGeneratorConfig(points_per_layer=100)  # 100 points per layer; layer_index 0..n_layers-1
                    ispm_acoustic_gen_demo = ISPMAcousticGenerator(config=ispm_acoustic_config)
                    
                    ispm_acoustic_data = ispm_acoustic_gen_demo.generate_for_build(
                        build_id=model_id,
                        n_layers=n_layers,
                        bounding_box=bbox_dict
                    )
                    
                    # Get coordinate system information (critical for merging with STL/hatching/CT data)
                    coordinate_system = ispm_acoustic_data.get('coordinate_system', None)
                    
                    # Convert to MongoDB documents in batches to avoid memory issues
                    # ispm_acoustic_data['data_points'] is a list of ISPMAcousticDataPoint objects
                    all_data_points = ispm_acoustic_data.get('data_points', [])
                    
                    if all_data_points:
                        # Batch size for insertion (to avoid memory issues)
                        batch_size = 10000
                        total_inserted = 0
                        
                        for batch_start in range(0, len(all_data_points), batch_size):
                            batch_points = all_data_points[batch_start:batch_start + batch_size]
                            ispm_acoustic_docs = []
                            
                            for point in batch_points:
                                # point is an ISPMAcousticDataPoint dataclass
                                ispm_acoustic_doc = {
                                    'model_id': model_id,
                                    'layer_index': point.layer_index,
                                    'timestamp': _timestamp_to_float(point.timestamp),
                                    'timestamp_iso': point.timestamp.isoformat() if isinstance(point.timestamp, datetime) else str(point.timestamp),
                                    'spatial_coordinates': [point.x, point.y, point.z],
                                    # Acoustic emission signals
                                    'acoustic_amplitude': point.acoustic_amplitude,
                                    'acoustic_frequency': point.acoustic_frequency,
                                    'acoustic_rms': point.acoustic_rms,
                                    'acoustic_peak': point.acoustic_peak,
                                    # Frequency domain features
                                    'dominant_frequency': point.dominant_frequency,
                                    'frequency_bandwidth': point.frequency_bandwidth,
                                    'spectral_centroid': point.spectral_centroid,
                                    'spectral_energy': point.spectral_energy,
                                    'spectral_rolloff': point.spectral_rolloff,
                                    # Event detection
                                    'spatter_event_detected': point.spatter_event_detected,
                                    'spatter_event_amplitude': point.spatter_event_amplitude,
                                    'defect_event_detected': point.defect_event_detected,
                                    'defect_event_amplitude': point.defect_event_amplitude,
                                    'anomaly_detected': point.anomaly_detected,
                                    'anomaly_type': point.anomaly_type,
                                    # Process stability
                                    'process_stability': point.process_stability,
                                    'acoustic_variation': point.acoustic_variation,
                                    'signal_to_noise_ratio': point.signal_to_noise_ratio,
                                    # Time-domain features
                                    'zero_crossing_rate': point.zero_crossing_rate,
                                    'autocorrelation_peak': point.autocorrelation_peak,
                                    # Frequency-domain features
                                    'harmonic_ratio': point.harmonic_ratio,
                                    'spectral_flatness': point.spectral_flatness,
                                    'spectral_crest': point.spectral_crest,
                                    # Process events
                                    'process_event': point.process_event,
                                    # Acoustic energy
                                    'acoustic_energy': point.acoustic_energy,
                                    'energy_per_band': point.energy_per_band,
                                    'coordinate_system': coordinate_system
                                }
                                
                                ispm_acoustic_docs.append(ispm_acoustic_doc)
                            
                            if ispm_acoustic_docs:
                                mongo_client.insert_documents('ispm_acoustic_monitoring_data', ispm_acoustic_docs)
                                total_inserted += len(ispm_acoustic_docs)
                                print(f"   📊 Inserted batch: {total_inserted}/{len(all_data_points)} ISPM acoustic records...", end='\r')
                        
                        print(f"\n   ✅ Stored {total_inserted} ISPM acoustic monitoring records")
                    else:
                        print(f"   ⚠️  No ISPM acoustic data points generated")
            
            # 8. Generate and store ISPM_Strain data
            if 'ispm_strain_monitoring_data' in collections_to_populate:
                bbox = stl_metadata.get('bounding_box', {})
                if bbox:
                    # Convert bounding box format for ISPM strain generator
                    bbox_dict = {
                        'x': (bbox['min'][0], bbox['max'][0]),
                        'y': (bbox['min'][1], bbox['max'][1]),
                        'z': (bbox['min'][2], bbox['max'][2])
                    }
                    n_layers = int((bbox['max'][2] - bbox['min'][2]) / 0.05)
                    # Reduce data generation for demo to avoid memory issues
                    # In production, use batching or GridFS for large datasets
                    from generation.sensors.ispm_strain_generator import ISPMStrainGeneratorConfig
                    ispm_strain_config = ISPMStrainGeneratorConfig(points_per_layer=100)  # 100 points per layer; layer_index 0..n_layers-1
                    ispm_strain_gen_demo = ISPMStrainGenerator(config=ispm_strain_config)
                    
                    ispm_strain_data = ispm_strain_gen_demo.generate_for_build(
                        build_id=model_id,
                        n_layers=n_layers,
                        layer_thickness=0.05,
                        start_time=datetime.now(),
                        bounding_box=bbox_dict
                    )
                    
                    # Get coordinate system information (critical for merging with STL/hatching/CT data)
                    coordinate_system = ispm_strain_data.get('coordinate_system', None)
                    
                    # Convert to MongoDB documents in batches to avoid memory issues
                    # ispm_strain_data['data_points'] is a list of ISPMStrainDataPoint objects
                    all_data_points = ispm_strain_data.get('data_points', [])
                    
                    if all_data_points:
                        # Batch size for insertion (to avoid memory issues)
                        batch_size = 10000
                        total_inserted = 0
                        
                        for batch_start in range(0, len(all_data_points), batch_size):
                            batch_points = all_data_points[batch_start:batch_start + batch_size]
                            ispm_strain_docs = []
                            
                            for point in batch_points:
                                # point is an ISPMStrainDataPoint dataclass
                                ispm_strain_doc = {
                                    'model_id': model_id,
                                    'layer_index': point.layer_index,
                                    'timestamp': _timestamp_to_float(point.timestamp),
                                    'timestamp_iso': point.timestamp.isoformat() if isinstance(point.timestamp, datetime) else str(point.timestamp),
                                    'spatial_coordinates': [point.x, point.y, point.z],
                                    # Strain components
                                    'strain_xx': point.strain_xx,
                                    'strain_yy': point.strain_yy,
                                    'strain_zz': point.strain_zz,
                                    'strain_xy': point.strain_xy,
                                    'strain_xz': point.strain_xz,
                                    'strain_yz': point.strain_yz,
                                    # Principal strains
                                    'principal_strain_max': point.principal_strain_max,
                                    'principal_strain_min': point.principal_strain_min,
                                    'principal_strain_intermediate': point.principal_strain_intermediate,
                                    # Von Mises strain
                                    'von_mises_strain': point.von_mises_strain,
                                    # Displacement
                                    'displacement_x': point.displacement_x,
                                    'displacement_y': point.displacement_y,
                                    'displacement_z': point.displacement_z,
                                    'total_displacement': point.total_displacement,
                                    # Strain rate
                                    'strain_rate': point.strain_rate,
                                    # Residual stress
                                    'residual_stress_xx': point.residual_stress_xx,
                                    'residual_stress_yy': point.residual_stress_yy,
                                    'residual_stress_zz': point.residual_stress_zz,
                                    'von_mises_stress': point.von_mises_stress,
                                    # Temperature-compensated strain
                                    'temperature_compensated_strain': point.temperature_compensated_strain,
                                    # Warping/distortion
                                    'warping_detected': point.warping_detected,
                                    'warping_magnitude': point.warping_magnitude,
                                    'distortion_angle': point.distortion_angle,
                                    # Layer-wise strain accumulation
                                    'cumulative_strain': point.cumulative_strain,
                                    'layer_strain_increment': point.layer_strain_increment,
                                    # Event detection
                                    'excessive_strain_event': point.excessive_strain_event,
                                    'warping_event_detected': point.warping_event_detected,
                                    'distortion_event_detected': point.distortion_event_detected,
                                    'anomaly_detected': point.anomaly_detected,
                                    'anomaly_type': point.anomaly_type,
                                    # Process stability
                                    'process_stability': point.process_stability,
                                    'strain_variation': point.strain_variation,
                                    'strain_uniformity': point.strain_uniformity,
                                    # Process events
                                    'process_event': point.process_event,
                                    # Strain energy
                                    'strain_energy_density': point.strain_energy_density,
                                    'coordinate_system': coordinate_system
                                }
                                
                                ispm_strain_docs.append(ispm_strain_doc)
                            
                            if ispm_strain_docs:
                                mongo_client.insert_documents('ispm_strain_monitoring_data', ispm_strain_docs)
                                total_inserted += len(ispm_strain_docs)
                                print(f"   📊 Inserted batch: {total_inserted}/{len(all_data_points)} ISPM strain records...", end='\r')
                        
                        print(f"\n   ✅ Stored {total_inserted} ISPM strain monitoring records")
                    else:
                        print(f"   ⚠️  No ISPM strain data points generated")
            
            # 9. Generate and store ISPM_Plume data
            if 'ispm_plume_monitoring_data' in collections_to_populate:
                bbox = stl_metadata.get('bounding_box', {})
                if bbox:
                    # Convert bounding box format for ISPM plume generator
                    bbox_dict = {
                        'x': (bbox['min'][0], bbox['max'][0]),
                        'y': (bbox['min'][1], bbox['max'][1]),
                        'z': (bbox['min'][2], bbox['max'][2])
                    }
                    n_layers = int((bbox['max'][2] - bbox['min'][2]) / 0.05)
                    # Reduce data generation for demo to avoid memory issues
                    # In production, use batching or GridFS for large datasets
                    from generation.sensors.ispm_plume_generator import ISPMPlumeGeneratorConfig
                    ispm_plume_config = ISPMPlumeGeneratorConfig(points_per_layer=100)  # 100 points per layer; layer_index 0..n_layers-1
                    ispm_plume_gen_demo = ISPMPlumeGenerator(config=ispm_plume_config)
                    
                    ispm_plume_data = ispm_plume_gen_demo.generate_for_build(
                        build_id=model_id,
                        n_layers=n_layers,
                        layer_thickness=0.05,
                        start_time=datetime.now(),
                        bounding_box=bbox_dict
                    )
                    
                    # Get coordinate system information (critical for merging with STL/hatching/CT data)
                    coordinate_system = ispm_plume_data.get('coordinate_system', None)
                    
                    # Convert to MongoDB documents in batches to avoid memory issues
                    # ispm_plume_data['data_points'] is a list of ISPMPlumeDataPoint objects
                    all_data_points = ispm_plume_data.get('data_points', [])
                    
                    if all_data_points:
                        # Batch size for insertion (to avoid memory issues)
                        batch_size = 10000
                        total_inserted = 0
                        
                        for batch_start in range(0, len(all_data_points), batch_size):
                            batch_points = all_data_points[batch_start:batch_start + batch_size]
                            ispm_plume_docs = []
                            
                            for point in batch_points:
                                # point is an ISPMPlumeDataPoint dataclass
                                ispm_plume_doc = {
                                    'model_id': model_id,
                                    'layer_index': point.layer_index,
                                    'timestamp': _timestamp_to_float(point.timestamp),
                                    'timestamp_iso': point.timestamp.isoformat() if isinstance(point.timestamp, datetime) else str(point.timestamp),
                                    'spatial_coordinates': [point.x, point.y, point.z],
                                    # Plume characteristics
                                    'plume_intensity': point.plume_intensity,
                                    'plume_density': point.plume_density,
                                    'plume_temperature': point.plume_temperature,
                                    'plume_velocity': point.plume_velocity,
                                    'plume_velocity_x': point.plume_velocity_x,
                                    'plume_velocity_y': point.plume_velocity_y,
                                    # Plume geometry
                                    'plume_height': point.plume_height,
                                    'plume_width': point.plume_width,
                                    'plume_angle': point.plume_angle,
                                    'plume_spread': point.plume_spread,
                                    'plume_area': point.plume_area,
                                    # Plume composition
                                    'particle_concentration': point.particle_concentration,
                                    'metal_vapor_concentration': point.metal_vapor_concentration,
                                    'gas_composition_ratio': point.gas_composition_ratio,
                                    # Plume dynamics
                                    'plume_fluctuation_rate': point.plume_fluctuation_rate,
                                    'plume_instability_index': point.plume_instability_index,
                                    'plume_turbulence': point.plume_turbulence,
                                    # Process quality indicators
                                    'process_stability': point.process_stability,
                                    'plume_stability': point.plume_stability,
                                    'intensity_variation': point.intensity_variation,
                                    # Event detection
                                    'excessive_plume_event': point.excessive_plume_event,
                                    'unstable_plume_event': point.unstable_plume_event,
                                    'contamination_event': point.contamination_event,
                                    'anomaly_detected': point.anomaly_detected,
                                    'anomaly_type': point.anomaly_type,
                                    # Plume energy
                                    'plume_energy': point.plume_energy,
                                    'energy_density': point.energy_density,
                                    # Process events
                                    'process_event': point.process_event,
                                    # Signal quality
                                    'signal_to_noise_ratio': point.signal_to_noise_ratio,
                                    # Additional plume features
                                    'plume_momentum': point.plume_momentum,
                                    'plume_pressure': point.plume_pressure,
                                    'coordinate_system': coordinate_system
                                }
                                
                                ispm_plume_docs.append(ispm_plume_doc)
                            
                            if ispm_plume_docs:
                                mongo_client.insert_documents('ispm_plume_monitoring_data', ispm_plume_docs)
                                total_inserted += len(ispm_plume_docs)
                                print(f"   📊 Inserted batch: {total_inserted}/{len(all_data_points)} ISPM plume records...", end='\r')
                        
                        print(f"\n   ✅ Stored {total_inserted} ISPM plume monitoring records")
                    else:
                        print(f"   ⚠️  No ISPM plume data points generated")
            
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
                       choices=['stl_models', 'hatching_layers', 'laser_monitoring_data', 
                               'ct_scan_data', 'ispm_thermal_monitoring_data', 'ispm_optical_monitoring_data', 'ispm_acoustic_monitoring_data', 'ispm_strain_monitoring_data', 'ispm_plume_monitoring_data'],
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

