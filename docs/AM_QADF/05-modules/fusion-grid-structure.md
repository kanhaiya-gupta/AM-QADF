# Fused Grid Structure Reference

## Overview

This document provides a complete reference for the structure of fused grids created by the `MultiSourceFusion` module. The fused grid follows industry best practices for multi-modal data fusion, preserving all original data while creating fused versions for analysis.

## Complete Structure

```python
Fused Grid = {
    'signal_arrays': {
        # ============================================
        # ORIGINAL SIGNALS (13 signals - preserved as-is from sources)
        # ============================================
        # Laser signals (original)
        'laser_power': np.ndarray,
        'laser_velocity': np.ndarray,
        'laser_energy': np.ndarray,
        
        # ISPM signals (original)
        'ispm_cooling_rate': np.ndarray,
        'ispm_peak_temperature': np.ndarray,
        'ispm_temperature': np.ndarray,
        'ispm_temperature_gradient': np.ndarray,
        
        # Hatching signals (original)
        'hatching_temperature': np.ndarray,
        'hatching_power': np.ndarray,
        'hatching_density': np.ndarray,
        
        # CT signals (original)
        'ct_temperature': np.ndarray,
        'ct_power': np.ndarray,
        'ct_density': np.ndarray,
        
        # ============================================
        # SOURCE-SPECIFIC FUSED SIGNALS (13 signals - all with _fused suffix)
        # ============================================
        # Laser signals (fused)
        'laser_power_fused': np.ndarray,
        'laser_velocity_fused': np.ndarray,
        'laser_energy_fused': np.ndarray,
        
        # ISPM signals (fused)
        'ispm_cooling_rate_fused': np.ndarray,
        'ispm_peak_temperature_fused': np.ndarray,
        'ispm_temperature_fused': np.ndarray,
        'ispm_temperature_gradient_fused': np.ndarray,
        
        # Hatching signals (fused)
        'hatching_temperature_fused': np.ndarray,
        'hatching_power_fused': np.ndarray,
        'hatching_density_fused': np.ndarray,
        
        # CT signals (fused)
        'ct_temperature_fused': np.ndarray,
        'ct_power_fused': np.ndarray,
        'ct_density_fused': np.ndarray,
        
        # ============================================
        # MULTI-SOURCE FUSED SIGNALS (3 signals - fused from matching types)
        # ============================================
        'temperature_fused': np.ndarray,  # ispm + hatching + ct
        'power_fused': np.ndarray,        # hatching + ct
        'density_fused': np.ndarray,      # hatching + ct
    },
    
    'metadata': {
        # ============================================
        # GRID METADATA
        # ============================================
        'grid_name': str,              # e.g., 'fused_uniform_100_weighted_avg_fused_20260116_111506'
        'grid_id': str,                 # Unique grid identifier
        'grid_shape': List[int],       # e.g., [9, 9, 9]
        'resolution': float,            # e.g., 1.0
        'bbox_min': List[float],       # e.g., [-50, -50, 0]
        'bbox_max': List[float],       # e.g., [50, 50, 100]
        'creation_timestamp': str,      # ISO format timestamp
        
        # ============================================
        # FUSION METADATA
        # ============================================
        'fusion_applied': bool,         # True
        'fusion_timestamp': str,        # ISO format timestamp
        'fusion_strategy': str,         # e.g., 'weighted_average'
        'fusion_sources': List[str],    # e.g., ['laser', 'ispm', 'hatching', 'ct']
        'fusion_method': str,           # 'voxel_domain_fusion'
        
        # ============================================
        # SIGNAL CATEGORIZATION
        # ============================================
        'signal_categories': {
            'original': List[str],                    # All original signal names
            'source_specific_fused': List[str],       # All source-specific fused signal names
            'multi_source_fused': List[str],          # All multi-source fused signal names
            'unique_signals': List[str],              # Signals unique to one source
            'shared_signals': List[str]               # Signal types shared across sources
        },
        
        # ============================================
        # SOURCE MAPPING
        # ============================================
        'source_mapping': {
            'laser': {
                'original_signals': List[str],
                'fused_signals': List[str],
                'grid_id': str,
                'grid_name': str,
                'source_type': str,
                'quality_score': float,
                'coverage': float
            },
            'ispm': {...},
            'hatching': {...},
            'ct': {...}
        },
        
        # ============================================
        # MULTI-SOURCE FUSION METADATA
        # ============================================
        'multi_source_fusion': {
            'temperature_fused': {
                'signal_type': str,                   # 'temperature'
                'sources': List[str],                 # ['ispm', 'hatching', 'ct']
                'source_signals': List[str],          # ['ispm_temperature_fused', ...]
                'strategy': str,                      # 'weighted_average'
                'weights': Dict[str, float],          # {'ispm': 0.4, 'hatching': 0.3, 'ct': 0.3}
                'quality_scores': Dict[str, float],  # Quality scores per source
                'fusion_timestamp': str,
                'fusion_score': float,                # 0-1, higher is better
                'consistency_score': float             # 0-1, higher is better
            },
            'power_fused': {...},
            'density_fused': {...}
        },
        
        # ============================================
        # SIGNAL STATISTICS
        # ============================================
        'signal_statistics': {
            'laser_power': {
                'shape': List[int],
                'dtype': str,
                'min': float,
                'max': float,
                'mean': float,
                'std': float,
                'non_zero_count': int,
                'total_count': int,
                'coverage': float                     # 0-1
            },
            # ... similar for all signals
        },
        
        # ============================================
        # FUSION QUALITY METRICS
        # ============================================
        'fusion_metrics': {
            'overall_fusion_score': float,            # 0-1, higher is better
            'coverage': float,                        # 0-1, spatial coverage
            'consistency_score': float,               # 0-1, consistency across sources
            'quality_score': float,                   # 0-1, average quality
            'signal_count': {
                'original': int,                      # 13
                'source_specific_fused': int,         # 13
                'multi_source_fused': int,            # 3
                'total': int                          # 29
            }
        },
        
        # ============================================
        # CONFIGURATION METADATA
        # ============================================
        'configuration_metadata': {
            'fusion_applied': bool,
            'fusion_strategy': str,
            'normalize_weights': bool,
            'use_quality_scores': bool,
            'source_weights': Dict[str, float],
            'grid_parameters': {
                'resolution': float,
                'bbox_min': List[float],
                'bbox_max': List[float]
            }
        },
        
        # ============================================
        # PROVENANCE & LINEAGE
        # ============================================
        'provenance': {
            'created_by': str,                        # e.g., 'multi_source_fusion'
            'creation_method': str,                   # e.g., 'comprehensive_fusion'
            'source_grids': [
                {
                    'source': str,
                    'grid_id': str,
                    'grid_name': str,
                    'stage': str,                     # 'corrected', 'processed', etc.
                    'signals_contributed': List[str]
                },
                # ... for each source
            ],
            'processing_chain': List[str]            # ['signal_mapping', 'alignment', ...]
        }
    }
}
```

## Signal Categories

### 1. Original Signals

**Purpose**: Preserve all original signals from source grids for traceability and analysis.

**Naming Convention**: `{source}_{signal_name}`

**Examples**:
- `laser_power`, `laser_velocity`, `laser_energy`
- `ispm_temperature`, `ispm_cooling_rate`
- `hatching_temperature`, `hatching_power`
- `ct_temperature`, `ct_power`

**Count**: 13 signals (varies based on sources)

### 2. Source-Specific Fused Signals

**Purpose**: Create fused versions of each signal with `_fused` suffix for consistency and future-proofing.

**Naming Convention**: `{source}_{signal_name}_fused`

**Examples**:
- `laser_power_fused`, `laser_velocity_fused`
- `ispm_temperature_fused`, `ispm_cooling_rate_fused`
- `hatching_temperature_fused`, `hatching_power_fused`
- `ct_temperature_fused`, `ct_power_fused`

**Count**: 13 signals (one per original signal)

**Note**: Currently, these are copies of the original signals. In the future, they can be processed independently (e.g., noise reduction, smoothing) while preserving originals.

### 3. Multi-Source Fused Signals

**Purpose**: Fuse signals of the same type from multiple sources into unified signals.

**Naming Convention**: `{signal_type}_fused`

**Examples**:
- `temperature_fused`: Fused from `ispm_temperature_fused` + `hatching_temperature_fused` + `ct_temperature_fused`
- `power_fused`: Fused from `hatching_power_fused` + `ct_power_fused`
- `density_fused`: Fused from `hatching_density_fused` + `ct_density_fused`

**Count**: 3 signals (varies based on shared signal types)

**Fusion Process**:
1. Identify signal types shared across multiple sources
2. Extract base type (remove source prefix and `_fused` suffix)
3. Group signals by type
4. Fuse using configured strategy (weighted average, median, etc.)
5. Store with `{signal_type}_fused` name

## Metadata Sections

### Grid Metadata

Basic grid information:
- Grid identification (name, ID)
- Spatial properties (shape, resolution, bounding box)
- Timestamps

### Fusion Metadata

Fusion operation information:
- Strategy used
- Sources involved
- Timestamps
- Method

### Signal Categorization

Organized lists of signals by category for easy access:
- Original signals
- Source-specific fused signals
- Multi-source fused signals
- Unique signals (only in one source)
- Shared signals (in multiple sources)

### Source Mapping

Complete mapping of each source to:
- Original signals contributed
- Fused signals created
- Source grid information (ID, name)
- Quality metrics (score, coverage)

### Multi-Source Fusion Metadata

Detailed information for each multi-source fused signal:
- Sources involved
- Fusion strategy
- Weights used
- Quality scores
- Fusion and consistency scores

### Signal Statistics

Statistics for all signals:
- Shape and dtype
- Min, max, mean, std
- Coverage (non-zero ratio)
- Counts

### Fusion Quality Metrics

Overall fusion quality:
- Fusion score
- Coverage
- Consistency
- Quality score
- Signal counts

### Configuration Metadata

Fusion configuration:
- Strategy
- Weights
- Quality score usage
- Grid parameters

### Provenance & Lineage

Complete traceability:
- Creation method
- Source grids
- Processing chain
- Signal contributions

## Usage Examples

### Accessing Signals

```python
# Original signals
laser_power = fused_grid['signal_arrays']['laser_power']
ispm_temperature = fused_grid['signal_arrays']['ispm_temperature']

# Source-specific fused signals
laser_power_fused = fused_grid['signal_arrays']['laser_power_fused']

# Multi-source fused signals
temperature_fused = fused_grid['signal_arrays']['temperature_fused']
power_fused = fused_grid['signal_arrays']['power_fused']
```

### Accessing Metadata

```python
metadata = fused_grid['metadata']

# Signal categories
original_signals = metadata['signal_categories']['original']
fused_signals = metadata['signal_categories']['multi_source_fused']

# Source information
laser_info = metadata['source_mapping']['laser']
laser_quality = laser_info['quality_score']

# Fusion quality
fusion_score = metadata['fusion_metrics']['overall_fusion_score']

# Multi-source fusion details
temp_fusion_info = metadata['multi_source_fusion']['temperature_fused']
temp_sources = temp_fusion_info['sources']
temp_weights = temp_fusion_info['weights']
```

### Querying Signals

```python
# Get all signals from a source
source_signals = [
    sig for sig in fused_grid['signal_arrays'].keys()
    if sig.startswith('laser_')
]

# Get all fused signals
fused_signals = [
    sig for sig in fused_grid['signal_arrays'].keys()
    if sig.endswith('_fused')
]

# Get multi-source fused signals
multi_source = metadata['signal_categories']['multi_source_fused']
```

## Benefits

1. **Complete Data Preservation**: All original signals are preserved
2. **Future-Proof**: New sources can be added without breaking existing code
3. **Full Traceability**: Complete metadata for audit and reproducibility
4. **Flexible Analysis**: Use original, source-specific, or multi-source fused signals
5. **Industry Standard**: Follows best practices for multi-modal data fusion
6. **Consistent Naming**: All fused signals use `_fused` suffix for clarity

## Related Documentation

- [Fusion Module](fusion.md) - Module overview
- [Fusion API Reference](../06-api-reference/fusion-api.md) - API documentation
- [Notebook 06: Multi-Source Data Fusion](../../Notebook/04-notebooks/06-fusion.md) - Notebook guide

---

**Last Updated**: 2026-01-16
