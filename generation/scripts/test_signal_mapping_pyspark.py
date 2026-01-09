#!/usr/bin/env python3
"""
Test Signal Mapping with PySpark Debug Script

This script tests signal mapping using PySpark with detailed debug output.
Useful for debugging large-scale signal mapping operations.
"""

import sys
from pathlib import Path
import numpy as np
import time
import logging
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
src_dir = project_root / 'src'
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_dir))

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
import os
env_file = project_root / 'development.env'
if env_file.exists():
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                value = value.strip('"\'')
                os.environ[key] = value

def test_sequential_mapping(model_id: str, grid_id: str, method: str = 'nearest'):
    """Test sequential (non-Spark) signal mapping."""
    logger.info("=" * 80)
    logger.info("Testing Sequential Signal Mapping")
    logger.info("=" * 80)
    
    try:
        # Import required modules
        from src.infrastructure.database import get_connection_manager
        from am_qadf.query import UnifiedQueryClient
        from am_qadf.voxelization.voxel_grid import VoxelGrid
        from am_qadf.signal_mapping.execution.sequential import interpolate_to_voxels
        import importlib.util
        
        # Connect to MongoDB
        logger.info("Connecting to MongoDB...")
        manager = get_connection_manager(env_name="development")
        mongo_client = manager.get_mongodb_client()
        
        if not mongo_client or not mongo_client.is_connected():
            logger.error("Failed to connect to MongoDB")
            return False
        
        logger.info("✅ MongoDB connected")
        
        # Load grid
        logger.info(f"Loading grid: {grid_id[:8]}...")
        voxel_storage_path = src_dir / 'am_qadf' / 'voxel_domain' / 'voxel_storage.py'
        spec = importlib.util.spec_from_file_location("voxel_storage", voxel_storage_path)
        voxel_storage_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(voxel_storage_module)
        VoxelGridStorage = voxel_storage_module.VoxelGridStorage
        voxel_storage = VoxelGridStorage(mongo_client)
        
        loaded_grid_data = voxel_storage.load_voxel_grid(grid_id)
        if not loaded_grid_data:
            logger.error(f"Failed to load grid {grid_id}")
            return False
        
        metadata = loaded_grid_data.get('metadata', {})
        bbox_min = tuple(metadata.get('bbox_min', [-50, -50, 0]))
        bbox_max = tuple(metadata.get('bbox_max', [50, 50, 100]))
        resolution = metadata.get('resolution', 2.0)
        
        logger.info(f"Grid: bbox={bbox_min} to {bbox_max}, resolution={resolution}mm")
        
        # Create voxel grid
        voxel_grid = VoxelGrid(
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            resolution=resolution,
            aggregation='mean'
        )
        logger.info(f"Voxel grid created: {voxel_grid.dims} dimensions")
        
        # Load signal data
        logger.info(f"Loading signal data for model: {model_id[:8]}...")
        unified_client = UnifiedQueryClient(mongo_client=mongo_client)
        all_data = unified_client.get_all_data(model_id)
        
        # Extract points and signals
        points_list = []
        signals_dict = {}
        
        def extract_data(data, source_name):
            if data is None:
                return
            if hasattr(data, 'points') and data.points is not None:
                points = data.points
                if isinstance(points, list):
                    points = np.array(points)
                if len(points) > 0:
                    points_list.append(points)
                    logger.info(f"  {source_name}: {len(points):,} points")
                    
                    if hasattr(data, 'signals') and data.signals:
                        for signal_name, signal_values in data.signals.items():
                            if isinstance(signal_values, list):
                                signal_values = np.array(signal_values)
                            signal_key = f"{source_name}_{signal_name}" if source_name else signal_name
                            if signal_key not in signals_dict:
                                signals_dict[signal_key] = []
                            signals_dict[signal_key].append(signal_values)
        
        extract_data(all_data.get('laser_parameters'), 'laser')
        extract_data(all_data.get('ispm_monitoring'), 'ispm')
        extract_data(all_data.get('hatching_layers'), 'hatching')
        extract_data(all_data.get('ct_scan'), 'ct')
        
        if not points_list:
            logger.error("No signal data found")
            return False
        
        # Combine points
        if len(points_list) > 1:
            current_points = np.concatenate(points_list, axis=0)
        else:
            current_points = points_list[0]
        
        # Combine signals
        current_signals = {}
        for signal_name, signal_arrays in signals_dict.items():
            arrays = [np.array(arr) if not isinstance(arr, np.ndarray) else arr for arr in signal_arrays]
            if len(arrays) > 1:
                current_signals[signal_name] = np.concatenate(arrays)
            else:
                current_signals[signal_name] = arrays[0] if arrays else np.array([])
        
        logger.info(f"Total points: {len(current_points):,}")
        logger.info(f"Total signals: {len(current_signals)}")
        logger.info(f"Signal names: {list(current_signals.keys())}")
        
        # Perform mapping
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Starting {method} interpolation...")
        logger.info(f"{'=' * 80}")
        
        start_time = time.time()
        
        mapped_grid = interpolate_to_voxels(
            current_points,
            current_signals,
            voxel_grid,
            method=method,
            use_parallel=False,
            use_spark=False
        )
        
        elapsed_time = time.time() - start_time
        
        logger.info(f"\n{'=' * 80}")
        logger.info(f"✅ Mapping completed in {elapsed_time:.2f} seconds")
        logger.info(f"{'=' * 80}")
        logger.info(f"Mapped signals: {sorted(list(mapped_grid.available_signals))}")
        logger.info(f"Grid dimensions: {mapped_grid.dims}")
        logger.info(f"Total voxels: {np.prod(mapped_grid.dims):,}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        return False


def test_spark_mapping(model_id: str, grid_id: str, method: str = 'nearest'):
    """Test Spark-based signal mapping."""
    logger.info("=" * 80)
    logger.info("Testing Spark Signal Mapping")
    logger.info("=" * 80)
    
    try:
        # Check PySpark availability
        try:
            from pyspark.sql import SparkSession
            from am_qadf.signal_mapping.utils.spark_utils import create_spark_session
        except ImportError as e:
            logger.error(f"PySpark not available: {e}")
            return False
        
        # Create Spark session
        logger.info("Creating Spark session...")
        spark = create_spark_session(app_name="SignalMappingTest")
        if spark is None:
            logger.error("Failed to create Spark session")
            return False
        
        logger.info(f"✅ Spark session created: {spark.sparkContext.appName}")
        logger.info(f"Spark version: {spark.version}")
        logger.info(f"Master: {spark.sparkContext.master}")
        
        try:
            # Import required modules
            from src.infrastructure.database import get_connection_manager
            from am_qadf.query import UnifiedQueryClient
            from am_qadf.voxelization.voxel_grid import VoxelGrid
            from am_qadf.signal_mapping.execution.sequential import interpolate_to_voxels
            import importlib.util
            
            # Connect to MongoDB
            logger.info("Connecting to MongoDB...")
            manager = get_connection_manager(env_name="development")
            mongo_client = manager.get_mongodb_client()
            
            if not mongo_client or not mongo_client.is_connected():
                logger.error("Failed to connect to MongoDB")
                return False
            
            logger.info("✅ MongoDB connected")
            
            # Load grid
            logger.info(f"Loading grid: {grid_id[:8]}...")
            voxel_storage_path = src_dir / 'am_qadf' / 'voxel_domain' / 'voxel_storage.py'
            spec = importlib.util.spec_from_file_location("voxel_storage", voxel_storage_path)
            voxel_storage_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(voxel_storage_module)
            VoxelGridStorage = voxel_storage_module.VoxelGridStorage
            voxel_storage = VoxelGridStorage(mongo_client)
            
            loaded_grid_data = voxel_storage.load_voxel_grid(grid_id)
            if not loaded_grid_data:
                logger.error(f"Failed to load grid {grid_id}")
                return False
            
            metadata = loaded_grid_data.get('metadata', {})
            bbox_min = tuple(metadata.get('bbox_min', [-50, -50, 0]))
            bbox_max = tuple(metadata.get('bbox_max', [50, 50, 100]))
            resolution = metadata.get('resolution', 2.0)
            
            logger.info(f"Grid: bbox={bbox_min} to {bbox_max}, resolution={resolution}mm")
            
            # Create voxel grid
            voxel_grid = VoxelGrid(
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                resolution=resolution,
                aggregation='mean'
            )
            logger.info(f"Voxel grid created: {voxel_grid.dims} dimensions")
            
            # Load signal data
            logger.info(f"Loading signal data for model: {model_id[:8]}...")
            unified_client = UnifiedQueryClient(mongo_client=mongo_client)
            all_data = unified_client.get_all_data(model_id)
            
            # Extract points and signals
            points_list = []
            signals_dict = {}
            
            def extract_data(data, source_name):
                if data is None:
                    return
                if hasattr(data, 'points') and data.points is not None:
                    points = data.points
                    if isinstance(points, list):
                        points = np.array(points)
                    if len(points) > 0:
                        points_list.append(points)
                        logger.info(f"  {source_name}: {len(points):,} points")
                        
                        if hasattr(data, 'signals') and data.signals:
                            for signal_name, signal_values in data.signals.items():
                                if isinstance(signal_values, list):
                                    signal_values = np.array(signal_values)
                                signal_key = f"{source_name}_{signal_name}" if source_name else signal_name
                                if signal_key not in signals_dict:
                                    signals_dict[signal_key] = []
                                signals_dict[signal_key].append(signal_values)
            
            extract_data(all_data.get('laser_parameters'), 'laser')
            extract_data(all_data.get('ispm_monitoring'), 'ispm')
            extract_data(all_data.get('hatching_layers'), 'hatching')
            extract_data(all_data.get('ct_scan'), 'ct')
            
            if not points_list:
                logger.error("No signal data found")
                return False
            
            # Combine points
            if len(points_list) > 1:
                current_points = np.concatenate(points_list, axis=0)
            else:
                current_points = points_list[0]
            
            # Combine signals
            current_signals = {}
            for signal_name, signal_arrays in signals_dict.items():
                arrays = [np.array(arr) if not isinstance(arr, np.ndarray) else arr for arr in signal_arrays]
                if len(arrays) > 1:
                    current_signals[signal_name] = np.concatenate(arrays)
                else:
                    current_signals[signal_name] = arrays[0] if arrays else np.array([])
            
            logger.info(f"Total points: {len(current_points):,}")
            logger.info(f"Total signals: {len(current_signals)}")
            logger.info(f"Signal names: {list(current_signals.keys())}")
            
            # Perform mapping with Spark
            logger.info(f"\n{'=' * 80}")
            logger.info(f"Starting {method} interpolation with Spark...")
            logger.info(f"{'=' * 80}")
            
            start_time = time.time()
            
            mapped_grid = interpolate_to_voxels(
                current_points,
                current_signals,
                voxel_grid,
                method=method,
                use_parallel=False,
                use_spark=True,
                spark_session=spark
            )
            
            elapsed_time = time.time() - start_time
            
            logger.info(f"\n{'=' * 80}")
            logger.info(f"✅ Spark mapping completed in {elapsed_time:.2f} seconds")
            logger.info(f"{'=' * 80}")
            logger.info(f"Mapped signals: {sorted(list(mapped_grid.available_signals))}")
            logger.info(f"Grid dimensions: {mapped_grid.dims}")
            logger.info(f"Total voxels: {np.prod(mapped_grid.dims):,}")
            
            return True
            
        finally:
            # Close Spark session
            logger.info("Closing Spark session...")
            spark.stop()
            logger.info("✅ Spark session closed")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test signal mapping with debug output")
    parser.add_argument('--model-id', type=str, required=True, help='Model ID')
    parser.add_argument('--grid-id', type=str, required=True, help='Grid ID')
    parser.add_argument('--method', type=str, default='nearest', 
                       choices=['nearest', 'linear', 'idw', 'gaussian_kde'],
                       help='Interpolation method')
    parser.add_argument('--use-spark', action='store_true', help='Use Spark for mapping')
    
    args = parser.parse_args()
    
    if args.use_spark:
        success = test_spark_mapping(args.model_id, args.grid_id, args.method)
    else:
        success = test_sequential_mapping(args.model_id, args.grid_id, args.method)
    
    sys.exit(0 if success else 1)

