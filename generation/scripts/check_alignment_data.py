"""
Check Alignment Data in MongoDB

This script checks alignment results stored in MongoDB, including
transformation matrices, metrics, configuration, and aligned data references.
It verifies completeness of data including GridFS storage.
"""

import sys
import os
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

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
    # Note: AlignmentStorage removed - alignment results now stored in OpenVDB format with grid metadata
    
    def get_mongodb_config():
        """Get MongoDB config from environment."""
        return MongoDBConfig.from_env()
except Exception as e:
    print(f"❌ Error loading MongoDB client: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


def format_matrix(matrix: List[List[float]], precision: int = 4) -> str:
    """Format transformation matrix for display."""
    if not matrix:
        return "N/A"
    lines = []
    for row in matrix:
        formatted_row = [f"{val:.{precision}f}" for val in row]
        lines.append("  [" + ", ".join(formatted_row) + "]")
    return "\n".join(lines)


def check_gridfs_file(mongo_client: MongoDBClient, file_id: str, file_type: str) -> Dict[str, Any]:
    """Check if a GridFS file exists and get its metadata."""
    try:
        file_data = mongo_client.get_file(file_id)
        if file_data:
            # Try to decompress and load to verify it's valid
            import gzip
            import io
            try:
                decompressed = gzip.decompress(file_data)
                data = np.load(io.BytesIO(decompressed))
                return {
                    'exists': True,
                    'valid': True,
                    'size_bytes': len(file_data),
                    'data_shape': data.shape if hasattr(data, 'shape') else 'N/A',
                    'data_dtype': str(data.dtype) if hasattr(data, 'dtype') else 'N/A'
                }
            except Exception as e:
                return {
                    'exists': True,
                    'valid': False,
                    'error': str(e),
                    'size_bytes': len(file_data)
                }
        else:
            return {'exists': False, 'valid': False}
    except Exception as e:
        return {'exists': False, 'valid': False, 'error': str(e)}


def check_alignment_data(
    model_id: Optional[str] = None,
    alignment_id: Optional[str] = None,
    summary_only: bool = False,
    verify_gridfs: bool = True
):
    """
    Check alignment data in MongoDB.
    
    Args:
        model_id: Optional model ID to filter
        alignment_id: Optional alignment ID to check
        summary_only: If True, only show summary statistics
        verify_gridfs: If True, verify GridFS files exist and are valid
    """
    print("=" * 80)
    print("🔍 Checking Alignment Data in MongoDB")
    print("=" * 80)
    
    # Connect to MongoDB
    print("\n🔌 Connecting to MongoDB...")
    config = get_mongodb_config()
    
    # Ensure credentials are set
    if not config.username:
        config.username = os.getenv('MONGO_ROOT_USERNAME', 'admin')
    if not config.password:
        config.password = os.getenv('MONGO_ROOT_PASSWORD', 'password')
    
    try:
        mongo_client = MongoDBClient(config=config)
        alignment_storage = AlignmentStorage(mongo_client)
        print(f"✅ Connected to MongoDB: {config.database}\n")
    except Exception as e:
        print(f"❌ Failed to connect to MongoDB: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # List alignments
    try:
        if alignment_id:
            alignments = [alignment_storage.load_alignment(alignment_id, load_aligned_data=False)]
            alignments = [a for a in alignments if a is not None]
        elif model_id:
            alignments = alignment_storage.list_alignments(model_id=model_id, limit=100)
        else:
            alignments = alignment_storage.list_alignments(limit=100)
    except Exception as e:
        print(f"❌ Error listing alignments: {e}")
        import traceback
        traceback.print_exc()
        mongo_client.close()
        return
    
    if not alignments:
        print("\n⚠️ No alignments found in database.")
        mongo_client.close()
        return
    
    # Summary statistics
    print(f"\n📊 Summary Statistics")
    print("-" * 80)
    print(f"Total Alignments: {len(alignments)}")
    
    # Count by mode
    mode_counts = {}
    model_ids = set()
    alignments_with_data = 0
    alignments_with_signals = 0
    alignments_complete = 0  # All 4 sources present
    expected_sources_summary = ['hatching', 'laser', 'ct', 'ispm']
    
    for align in alignments:
        mode = align.get('alignment_mode', 'unknown')
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        model_ids.add(align.get('model_id', 'unknown'))
        
        refs = align.get('aligned_data_references', {})
        if refs:
            alignments_with_data += 1
            # Check if all expected sources are present
            present_sources = set(refs.keys())
            if all(src in present_sources for src in expected_sources_summary):
                alignments_complete += 1
            # Check if any source has signals
            for source_refs in refs.values():
                if 'signals_gridfs_id' in source_refs or 'signals_gridfs_ids' in source_refs:
                    alignments_with_signals += 1
                    break
    
    print(f"Unique Models: {len(model_ids)}")
    print(f"Alignments by Mode:")
    for mode, count in sorted(mode_counts.items()):
        print(f"  {mode}: {count}")
    print(f"Alignments with Aligned Data: {alignments_with_data}/{len(alignments)}")
    print(f"Alignments with Signals: {alignments_with_signals}/{len(alignments)}")
    print(f"Alignments with All Sources (Hatching, Laser, CT, ISPM): {alignments_complete}/{len(alignments)}")
    
    if summary_only:
        mongo_client.close()
        return
    
    # Detailed check for each alignment
    print(f"\n{'=' * 80}")
    print(f"📋 Detailed Alignment Information")
    print(f"{'=' * 80}")
    
    for idx, alignment in enumerate(alignments, 1):
        print(f"\n{'─' * 80}")
        print(f"Alignment {idx}/{len(alignments)}")
        print(f"{'─' * 80}")
        
        # Basic info
        alignment_id = alignment.get('alignment_id', 'N/A')
        model_id_align = alignment.get('model_id', 'N/A')
        model_name = alignment.get('model_name', 'N/A')
        alignment_mode = alignment.get('alignment_mode', 'N/A')
        created_at = alignment.get('created_at', 'N/A')
        updated_at = alignment.get('updated_at', 'N/A')
        
        print(f"\n📌 Basic Information:")
        print(f"  Alignment ID: {alignment_id[:36]}..." if len(alignment_id) > 36 else f"  Alignment ID: {alignment_id}")
        print(f"  Model ID: {model_id_align[:36]}..." if len(model_id_align) > 36 else f"  Model ID: {model_id_align}")
        print(f"  Model Name: {model_name}")
        print(f"  Alignment Mode: {alignment_mode}")
        print(f"  Created: {created_at}")
        print(f"  Updated: {updated_at}")
        
        # Configuration
        config = alignment.get('configuration', {})
        if config:
            print(f"\n⚙️ Configuration:")
            print(f"  Temporal Reference: {config.get('temporal_reference', 'N/A')}")
            print(f"  Spatial Transform Type: {config.get('spatial_transform_type', 'N/A')}")
            print(f"  Data Sources: {', '.join(config.get('data_sources', []))}")
            
            translation = config.get('translation', {})
            if translation:
                print(f"  Translation: X={translation.get('x', 0):.3f}, Y={translation.get('y', 0):.3f}, Z={translation.get('z', 0):.3f}")
            
            rotation = config.get('rotation', {})
            if rotation:
                print(f"  Rotation: X={rotation.get('x', 0):.3f}°, Y={rotation.get('y', 0):.3f}°, Z={rotation.get('z', 0):.3f}°")
            
            scaling = config.get('scaling', {})
            if scaling:
                print(f"  Scaling: X={scaling.get('x', 1):.3f}, Y={scaling.get('y', 1):.3f}, Z={scaling.get('z', 1):.3f}")
            
            print(f"  Temporal Tolerance: {config.get('temporal_tolerance', 'N/A')}")
            print(f"  Temporal Interpolation: {config.get('temporal_interpolation', 'N/A')}")
        
        # Transformation Matrix
        transform_matrix = alignment.get('transformation_matrix')
        if transform_matrix:
            print(f"\n🔄 Transformation Matrix:")
            if isinstance(transform_matrix, dict) and 'matrix' in transform_matrix:
                matrix = transform_matrix['matrix']
                print(format_matrix(matrix))
            elif isinstance(transform_matrix, list):
                print(format_matrix(transform_matrix))
            else:
                print(f"  {transform_matrix}")
        else:
            print(f"\n🔄 Transformation Matrix: N/A")
        
        # Alignment Metrics
        metrics = alignment.get('alignment_metrics', {})
        if metrics:
            print(f"\n📊 Alignment Metrics:")
            
            # Basic metrics
            if 'temporal_accuracy' in metrics:
                print(f"  Temporal Accuracy: {metrics.get('temporal_accuracy', 'N/A')} s")
            if 'spatial_accuracy' in metrics:
                print(f"  Spatial Accuracy: {metrics.get('spatial_accuracy', 'N/A')} mm")
            if 'alignment_score' in metrics:
                print(f"  Alignment Score: {metrics.get('alignment_score', 'N/A')}")
            if 'coverage' in metrics:
                print(f"  Coverage: {metrics.get('coverage', 'N/A')}%")
            
            # Validation
            validation = metrics.get('validation', {})
            if validation:
                print(f"  Validation Status: {validation.get('status', 'N/A')}")
                print(f"  Mean Error: {validation.get('mean_error_mm', 'N/A')} mm")
                print(f"  Max Error: {validation.get('max_error_mm', 'N/A')} mm")
                print(f"  RMS Error: {validation.get('rms_error_mm', 'N/A')} mm")
            
            # Matrix properties
            matrix_props = metrics.get('transformation_matrix_properties', {})
            if matrix_props:
                print(f"  Matrix Determinant: {matrix_props.get('determinant', 'N/A')}")
                print(f"  Matrix Trace: {matrix_props.get('trace', 'N/A')}")
                print(f"  Is Orthogonal: {matrix_props.get('is_orthogonal', 'N/A')}")
                print(f"  Translation Magnitude: {matrix_props.get('translation_magnitude', 'N/A')} mm")
            
            # Data statistics
            data_stats = metrics.get('data_statistics', {})
            if data_stats:
                print(f"  Data Statistics:")
                for source_name, stats in data_stats.items():
                    print(f"    {source_name}:")
                    print(f"      Point Count: {stats.get('point_count', 'N/A'):,}")
                    bbox = stats.get('bounding_box', {})
                    if bbox:
                        bbox_min = bbox.get('min', [])
                        bbox_max = bbox.get('max', [])
                        if bbox_min and bbox_max:
                            print(f"      BBox: [{bbox_min[0]:.2f}, {bbox_min[1]:.2f}, {bbox_min[2]:.2f}] to [{bbox_max[0]:.2f}, {bbox_max[1]:.2f}, {bbox_max[2]:.2f}]")
        
        # Expected data sources
        expected_sources = ['hatching', 'laser', 'ct', 'ispm']
        config_sources = config.get('data_sources', [])
        if config_sources:
            # Use sources from configuration if available
            expected_sources = [s.lower() for s in config_sources]
        
        # Aligned Data References
        refs = alignment.get('aligned_data_references', {})
        if refs:
            print(f"\n💾 Aligned Data References:")
            
            # Check which sources are present
            present_sources = set(refs.keys())
            expected_sources_set = set(expected_sources)
            missing_sources = expected_sources_set - present_sources
            extra_sources = present_sources - expected_sources_set
            
            # Data completeness summary
            print(f"\n  📊 Data Source Completeness:")
            print(f"    Expected Sources: {', '.join(expected_sources)}")
            print(f"    Present Sources: {', '.join(sorted(present_sources))}")
            if missing_sources:
                print(f"    ⚠️ Missing Sources: {', '.join(sorted(missing_sources))}")
            if extra_sources:
                print(f"    ℹ️ Extra Sources: {', '.join(sorted(extra_sources))}")
            
            completeness_score = 0
            total_checks = len(expected_sources)
            
            # Check each expected source
            for source_name in expected_sources:
                source_complete = False  # Initialize for each source
                if source_name in refs:
                    source_refs = refs[source_name]
                    print(f"\n  📦 {source_name.upper()}:")
                    source_complete = True  # Start as complete, will be set to False if issues found
                    
                    # Check points
                    if 'points_gridfs_id' in source_refs:
                        points_id = source_refs['points_gridfs_id']
                        print(f"    ✅ Points: {points_id[:36]}...")
                        if verify_gridfs:
                            gridfs_check = check_gridfs_file(mongo_client, points_id, 'points')
                            if gridfs_check.get('exists'):
                                print(f"      Size: {gridfs_check.get('size_bytes', 0):,} bytes")
                                if gridfs_check.get('valid'):
                                    print(f"      Shape: {gridfs_check.get('data_shape', 'N/A')}")
                                    print(f"      Dtype: {gridfs_check.get('data_dtype', 'N/A')}")
                                else:
                                    print(f"      ⚠️ Invalid data: {gridfs_check.get('error', 'Unknown error')}")
                                    source_complete = False
                            else:
                                print(f"      ❌ File not found in GridFS")
                                source_complete = False
                    else:
                        print(f"    ❌ Points: Not found")
                        source_complete = False
                    
                    # Check signals
                    if 'signals_gridfs_ids' in source_refs:
                        signals_dict = source_refs['signals_gridfs_ids']
                        print(f"    ✅ Signals (multiple): {len(signals_dict)} signal(s)")
                        for signal_name, signal_id in signals_dict.items():
                            print(f"      - {signal_name}: {signal_id[:36]}...")
                            if verify_gridfs:
                                gridfs_check = check_gridfs_file(mongo_client, signal_id, f'signal_{signal_name}')
                                if gridfs_check.get('exists'):
                                    if gridfs_check.get('valid'):
                                        print(f"        Size: {gridfs_check.get('size_bytes', 0):,} bytes, Shape: {gridfs_check.get('data_shape', 'N/A')}")
                                    else:
                                        print(f"        ⚠️ Invalid: {gridfs_check.get('error', 'Unknown error')}")
                                        source_complete = False
                                else:
                                    print(f"        ❌ File not found in GridFS")
                                    source_complete = False
                    elif 'signals_gridfs_id' in source_refs:
                        signals_id = source_refs['signals_gridfs_id']
                        print(f"    ✅ Signals (single): {signals_id[:36]}...")
                        if verify_gridfs:
                            gridfs_check = check_gridfs_file(mongo_client, signals_id, 'signals')
                            if gridfs_check.get('exists'):
                                if gridfs_check.get('valid'):
                                    print(f"      Size: {gridfs_check.get('size_bytes', 0):,} bytes, Shape: {gridfs_check.get('data_shape', 'N/A')}")
                                else:
                                    print(f"      ⚠️ Invalid: {gridfs_check.get('error', 'Unknown error')}")
                                    source_complete = False
                            else:
                                print(f"      ❌ File not found in GridFS")
                                source_complete = False
                    else:
                        print(f"    ⚠️ Signals: Not found")
                        # Signals are optional, so don't mark as incomplete
                    
                    # Check times
                    if 'times' in source_refs:
                        times = source_refs['times']
                        if times and (isinstance(times, list) and len(times) > 0 or not isinstance(times, list)):
                            print(f"    ✅ Times: {len(times) if isinstance(times, list) else 'Present'} values")
                        else:
                            print(f"    ⚠️ Times: Empty")
                    
                    # Check layers
                    if 'layers' in source_refs:
                        layers = source_refs['layers']
                        if layers and (isinstance(layers, list) and len(layers) > 0 or not isinstance(layers, list)):
                            print(f"    ✅ Layers: {len(layers) if isinstance(layers, list) else 'Present'} values")
                        else:
                            print(f"    ⚠️ Layers: Empty")
                
                    # Update completeness score for this source
                    if source_complete:
                        completeness_score += 1
                else:
                    print(f"\n  📦 {source_name.upper()}:")
                    print(f"    ❌ Source not found in aligned data")
                    source_complete = False
            
            print(f"\n  📈 Data Completeness: {completeness_score}/{total_checks} sources complete")
            if completeness_score == total_checks:
                print(f"    ✅ All expected data sources are present!")
            else:
                print(f"    ⚠️ Missing {total_checks - completeness_score} source(s)")
        else:
            print(f"\n💾 Aligned Data References: ⚠️ No aligned data references found")
        
        # Tags and Description
        tags = alignment.get('tags', [])
        if tags:
            print(f"\n🏷️ Tags: {', '.join(tags)}")
        
        description = alignment.get('description', '')
        if description:
            print(f"\n📝 Description: {description}")
    
    print(f"\n{'=' * 80}")
    print("✅ Check complete!")
    print(f"{'=' * 80}\n")
    
    mongo_client.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Check alignment data in MongoDB')
    parser.add_argument('--model-id', type=str, help='Filter by model ID')
    parser.add_argument('--alignment-id', type=str, help='Check specific alignment ID')
    parser.add_argument('--summary-only', action='store_true', help='Show only summary statistics')
    parser.add_argument('--no-verify-gridfs', action='store_true', help='Skip GridFS file verification')
    parser.add_argument('--list-models', action='store_true', help='List all models with alignments')
    
    args = parser.parse_args()
    
    if args.list_models:
        # Quick list mode
        config = get_mongodb_config()
        if not config.username:
            config.username = os.getenv('MONGO_ROOT_USERNAME', 'admin')
        if not config.password:
            config.password = os.getenv('MONGO_ROOT_PASSWORD', 'password')
        
        try:
            mongo_client = MongoDBClient(config=config)
            alignment_storage = AlignmentStorage(mongo_client)
            alignments = alignment_storage.list_alignments(limit=1000)
            
            model_ids = set()
            for align in alignments:
                model_ids.add(align.get('model_id', 'unknown'))
            
            print("Models with alignments:")
            for model_id in sorted(model_ids):
                count = sum(1 for a in alignments if a.get('model_id') == model_id)
                print(f"  {model_id[:36]}... ({count} alignment(s))")
            
            mongo_client.close()
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        check_alignment_data(
            model_id=args.model_id,
            alignment_id=args.alignment_id,
            summary_only=args.summary_only,
            verify_gridfs=not args.no_verify_gridfs
        )


if __name__ == '__main__':
    main()
